import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy

logger = logging.getLogger(__name__)


class BatCLIPLoss(nn.Module):
    """Minimize similarity between mean features of different classes."""

    def forward(self, img_features, labels):
        unique_labels = torch.unique(labels, sorted=True)
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=img_features.device)

        mean_feats = []
        for l in unique_labels.tolist():
            class_mean = img_features[labels == l].mean(dim=0)
            mean_feats.append(F.normalize(class_mean, dim=0))

        mean_feats = torch.stack(mean_feats)
        cosine_sim_matrix = torch.mm(mean_feats, mean_feats.t())
        mask = torch.triu(torch.ones_like(cosine_sim_matrix), diagonal=1).bool()
        if mask.sum() > 0:
            return cosine_sim_matrix[mask].mean()
        return torch.tensor(0.0, device=img_features.device)


class MintLoss(nn.Module):
    """Batch-wise (intra_var - total_var) separation loss."""

    def forward(self, features, labels):
        unique_labels = torch.unique(labels)
        if features.shape[0] < 2 or len(unique_labels) < 2:
            return torch.tensor(0.0, device=features.device)

        total_mean = features.mean(dim=0)
        class_means = torch.zeros_like(features)
        for l in unique_labels:
            mask = (labels == l)
            class_means[mask] = features[mask].mean(dim=0)

        diff_intra = ((features - class_means) ** 2).sum(dim=1).mean()
        diff_total = ((features - total_mean) ** 2).sum(dim=1).mean()
        return diff_intra - diff_total


@ADAPTATION_REGISTRY.register()
class SUBTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.remove_first_pc = cfg.SUBTTA.REMOVE_FIRST_PC
        if self.remove_first_pc:
            self.subspace_rank = min(num_classes - 1, cfg.SUBTTA.SUBSPACE_RANK)
        else:
            self.subspace_rank = min(num_classes, cfg.SUBTTA.SUBSPACE_RANK)

        self.alpha = cfg.SUBTTA.COV_MOMENTUM
        self.ent_scale = cfg.SUBTTA.ENT_SCALE

        self.loss_type = cfg.SUBTTA.LOSS_TYPE.lower()
        if self.loss_type == 'tent':
            self.separation_loss = Entropy()
        elif self.loss_type == 'batclip':
            self.separation_loss = BatCLIPLoss()
        elif self.loss_type == 'mint':
            self.separation_loss = MintLoss()
        elif self.loss_type == 'source':
            self.separation_loss = None
        else:
            raise ValueError(f"Unsupported LOSS_TYPE: {self.loss_type}. Choose from: source, tent, mint, batclip")

        self.text_basis = None
        self.text_projector = None
        self.text_mean = None
        self.register_buffer('global_cov', None)
        self.num_samples_seen = 0

        self.t_align = 0
        self.t_tta = 0
        self.total_batch = 0

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                m.requires_grad_(True)

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def get_text_subspace(self, text_features):
        self.text_mean = text_features.mean(dim=0, keepdim=True)
        Y_centered = text_features - self.text_mean
        K = text_features.shape[0]
        self.text_cov_init = (Y_centered.t() @ Y_centered) / (K - 1 + 1e-6)
        self.text_cov_init = self.text_cov_init.detach()

        try:
            _, S, Vh = torch.linalg.svd(Y_centered, full_matrices=False)
        except RuntimeError:
            _, S, Vh = torch.linalg.svd(Y_centered.float(), full_matrices=False)
            Vh = Vh.to(Y_centered.dtype)

        start_idx = 1 if self.remove_first_pc else 0
        end_idx = start_idx + self.subspace_rank
        self.text_basis = Vh[start_idx:end_idx, :].detach()
        self.text_projector = self.text_basis.T @ self.text_basis

    def get_image_subspace(self, img_features):
        B, D = img_features.shape
        curr_mean = img_features.mean(dim=0, keepdim=True)
        X_centered = img_features - curr_mean
        curr_cov = (X_centered.t() @ X_centered) / (B - 1 + 1e-6) + 1e-6 * torch.eye(
            D, device=img_features.device, dtype=img_features.dtype
        )

        if self.global_cov is None:
            self.global_cov = self.text_cov_init.clone()

        cov_for_svd = self.alpha * self.global_cov.detach() + (1 - self.alpha) * curr_cov
        self.global_cov = cov_for_svd.detach()
        self.num_samples_seen += B

        start_idx = 1 if self.remove_first_pc else 0
        valid_rank = min(self.subspace_rank, D - start_idx)

        try:
            U, S, Vh = torch.linalg.svd(cov_for_svd, full_matrices=False)
        except RuntimeError:
            U, S, Vh = torch.linalg.svd(cov_for_svd.float(), full_matrices=False)
            Vh = Vh.to(cov_for_svd.dtype)

        return Vh[start_idx: start_idx + valid_rank, :]

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        scaler = torch.cuda.amp.GradScaler()

        with torch.cuda.amp.autocast():
            logits_1, _, text_features, img_pre_features_1, _ = self.model(imgs_test, return_features=True)

        img_features_1 = img_pre_features_1.float()
        text_features_f = text_features.float()

        if self.text_basis is None:
            self.get_text_subspace(text_features_f)

        t1 = time.time()
        if self.alpha != 0:
            img_basis = self.get_image_subspace(img_features_1)
            alignment_matrix = img_basis @ self.text_basis.t()
            loss_align = -torch.sum(alignment_matrix ** 2) / (self.subspace_rank * max(1 - self.alpha, 1e-6))
            scaler.scale(loss_align).backward()
            scaler.step(self.optimizer)
            scaler.update()
        t2 = time.time()

        if self.loss_type == 'source':
            with torch.cuda.amp.autocast():
                logits_inference = self.model(imgs_test)
            return logits_inference.detach()

        with torch.cuda.amp.autocast():
            _, _, _, img_pre_features_2, _ = self.model(imgs_test, return_features=True)
        img_features_2 = img_pre_features_2.float()
        img_proj = img_features_2 @ self.text_projector
        img_proj_norm = F.normalize(img_proj, dim=-1)
        text_proj_norm = F.normalize(text_features_f, dim=-1)

        logit_scale = self.model.model.logit_scale.exp()
        logits_proj = logit_scale * (img_proj_norm @ text_proj_norm.t())

        with torch.no_grad():
            _, pseudo_labels = torch.max(torch.softmax(logits_proj, dim=1), dim=1)

        self.optimizer.zero_grad()
        if self.loss_type == 'batclip':
            loss_sep = self.separation_loss(img_proj_norm, pseudo_labels)
        elif self.loss_type == 'tent':
            loss_sep = self.separation_loss(self.ent_scale * img_proj_norm).mean(0)
        elif self.loss_type == 'mint':
            loss_sep = self.separation_loss(img_proj_norm, pseudo_labels)
        else:
            raise ValueError(f"Unsupported LOSS_TYPE: {self.loss_type}")

        scaler.scale(loss_sep).backward()
        scaler.step(self.optimizer)
        scaler.update()

        t3 = time.time()
        self.t_align += t2 - t1
        self.t_tta += t3 - t2
        self.total_batch += 1

        with torch.cuda.amp.autocast():
            logits_inference = self.model(imgs_test)
        return logits_inference.detach()

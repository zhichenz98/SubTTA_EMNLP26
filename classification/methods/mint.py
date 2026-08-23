import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY


class GradientAccumulator:
    def __init__(self, dim, dtype=torch.float32, device="cpu"):
        self.n = 0
        self.avg = torch.zeros(dim, dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, grad):
        self.avg = (self.n / (self.n + 1)) * self.avg + (1 / (self.n + 1)) * grad
        self.n += 1

    def get(self):
        return self.avg

    def clear(self):
        self.n = 0
        self.avg.zero_()


class MeanAccumulator:
    def __init__(self, num_classes, dim, dtype=torch.float32, device="cpu"):
        self.class_sums = torch.zeros(num_classes, dim, dtype=dtype, device=device)
        self.total_sum = torch.zeros(dim, dtype=dtype, device=device)
        self.class_n = torch.zeros(num_classes, dtype=dtype, device=device)
        self.total_n = 0

        self.dtype = dtype
        self.device = device

    def update(self, embeds, preds):
        embeds, preds = embeds.detach(), preds.detach()
        self.class_sums.index_add_(dim=0, index=preds, source=embeds)
        self.total_sum += embeds.sum(dim=0)
        self.class_n.index_add_(dim=0, index=preds,
                                source=torch.ones(preds.size(0), dtype=self.dtype, device=self.device))
        self.total_n += preds.size(0)

    def get_class_means(self):
        class_means = self.class_sums / self.class_n.unsqueeze(1)
        class_means[self.class_n == 0] = 0
        return class_means

    def get_total_mean(self):
        return self.total_sum / self.total_n

    def clear(self):
        self.class_sums.zero_()
        self.total_sum.zero_()
        self.class_n.zero_()
        self.total_n = 0


@ADAPTATION_REGISTRY.register()
class MINT(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        
        # Get device from model
        self.device = next(model.parameters()).device
        
        # Get feature dimension from text features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            _, _, text_features, _, _ = self.model(dummy_input, return_features=True)
            self.feat_dim = text_features.shape[1]
        
        # Calculate gradient dimension - only count params that will actually have gradients
        # Store parameter names that will be updated
        self.trainable_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)
        
        self.grad_dim = sum(param.numel() for param in self.trainable_params)
        
        # Initialize accumulators
        self.pre_feat_accumulator = MeanAccumulator(
            num_classes=self.num_classes, 
            dim=self.feat_dim,
            dtype=torch.float32, 
            device=self.device
        )
        
        self.post_feat_accumulator = MeanAccumulator(
            num_classes=self.num_classes, 
            dim=self.feat_dim,
            dtype=torch.float32, 
            device=self.device
        )
        
        self.grad_accumulator = GradientAccumulator(
            dim=self.grad_dim, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Get prior parameter from config (default to 1000 as in Mint paper)
        self.prior = cfg.MINT.PRIOR_N
        
        # Store initial state
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        
        # # Restore model state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        imgs_test = x[0]

        ######## ######## ######## ######## ######## ######## ######## ########
        # Adaptation
        ######## ######## ######## ######## ######## ######## ######## ########

        with torch.cuda.amp.autocast():
            logits, _, text_features, img_features, _ = self.model(imgs_test, return_features=True)
            # logits: [B, num_classes], img_features: [B, D], text_features: [num_classes, D]
            
            # Normalize image features
            img_features = F.normalize(img_features, dim=1)
            text_features_norm = F.normalize(text_features, dim=1)
            
            # Get predictions
            logits_for_pred = 100.0 * img_features @ text_features_norm.T
            preds = torch.argmax(logits_for_pred, dim=1)

            # Update embedding mean accumulator
            self.pre_feat_accumulator.update(img_features, preds)

            # Calculate PL-inter variance within this batch
            present_classes = torch.unique(preds)
            C = len(present_classes)

            class_means = self.pre_feat_accumulator.get_class_means()
            total_mean = self.pre_feat_accumulator.get_total_mean()

            counts = torch.bincount(preds, minlength=self.num_classes)
            class_weights = 1.0 / C / counts

            sample_weights = class_weights[preds]

            diff_intra = ((img_features - class_means[preds]) ** 2).sum(dim=1)
            intra_var = (diff_intra * sample_weights).sum()

            diff_total = ((img_features - total_mean) ** 2).sum(dim=1)
            total_var = (diff_total * sample_weights).sum()

            inter_var = total_var - intra_var
            loss = -inter_var  # to use the built-in optimizer

        loss.backward()

        with torch.no_grad():
            # Get gradient (only from trainable params to ensure consistent size)
            grads = []
            for param in self.trainable_params:
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
                else:
                    # If a param should have grad but doesn't, use zeros
                    grads.append(torch.zeros(param.numel(), device=self.device))
            
            flat_grad = torch.cat(grads)
            
            # Update grad accumulator and get aggregated gradient
            self.grad_accumulator.update(flat_grad)
            agg_grad = self.grad_accumulator.get()

            # Set gradient
            offset = 0
            for param in self.trainable_params:
                numel = param.numel()
                if param.grad is not None:
                    param.grad.copy_(agg_grad[offset:offset + numel].view_as(param))
                else:
                    # Create gradient if it doesn't exist
                    param.grad = agg_grad[offset:offset + numel].view_as(param).clone()
                offset += numel

        self.optimizer.step()
        self.optimizer.zero_grad()

        ######## ######## ######## ######## ######## ######## ######## ########
        # Inference
        ######## ######## ######## ######## ######## ######## ######## ########

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                logits, _, text_features, img_features, _ = self.model(imgs_test, return_features=True)
                
                # Normalize features
                img_features = F.normalize(img_features, dim=1)
                text_features_norm = F.normalize(text_features, dim=1)
                
                # Get predictions
                logits_for_pred = 100.0 * img_features @ text_features_norm.T
                preds = torch.argmax(logits_for_pred, dim=1)

                # Update embedding mean accumulator
                self.post_feat_accumulator.update(img_features, preds)

                class_means = self.post_feat_accumulator.get_class_means()

                n = self.post_feat_accumulator.total_n
                ratio = n / (self.prior + n)

                mixed_weights = (1 - ratio) * text_features_norm + ratio * class_means
                mixed_weights = F.normalize(mixed_weights, dim=1)

                logits = 100.0 * img_features @ mixed_weights.T

        return logits.detach()

    def configure_model(self):
        """Configure model for Mint adaptation (adapt normalization layers)"""
        self.model.eval()
        self.model.requires_grad_(False)

        # Enable normalization layers for adaptation
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, nn.BatchNorm2d):
                m.train()
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def collect_params(self):
        """Collect parameters to optimize (normalization layers)"""
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    def reset(self):
        """Reset model and accumulators to initial state"""
        super().reset()
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.pre_feat_accumulator.clear()
        self.post_feat_accumulator.clear()
        self.grad_accumulator.clear()
        


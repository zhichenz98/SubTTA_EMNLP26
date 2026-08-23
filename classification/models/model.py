import logging
import torch
import torch.nn as nn

from open_clip import create_model_and_transforms, get_tokenizer
from typing import Union
from torch import Tensor
from datasets.cls_names import get_class_names


logger = logging.getLogger(__name__)


class ImageNormalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


class ZeroShotCLIP(nn.Module):
    def __init__(self, cfg, model, device, normalize):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.device = device
        self.normalize = normalize
        self.freeze_text_encoder = cfg.CLIP.FREEZE_TEXT_ENCODER
        self.class_names = get_class_names(cfg.CORRUPTION.DATASET)
        self.tokenize = get_tokenizer(cfg.MODEL.ARCH)
        self.logit_scale = self.model.logit_scale.data

        prompt_templates = cfg.CLIP.PROMPT_TEMPLATE

        with torch.no_grad():
            self.text_features = []
            self.text_pre_features = []
            for c_name in self.class_names:
                texts = [template.format(c_name) for template in prompt_templates]
                texts = self.tokenize(texts).to(self.device)
                class_embeddings = model.encode_text(texts)
                self.text_pre_features.append(class_embeddings)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                self.text_features.append(class_embedding)
            self.text_pre_features = torch.stack(self.text_pre_features, dim=0).to(self.device)
            self.text_features = torch.stack(self.text_features, dim=0).to(self.device)

        if self.freeze_text_encoder:
            self.model.transformer = None

    @property
    def dtype(self):
        return next(self.model.visual.parameters()).dtype

    def forward(self, imgs_test, return_features=False):
        imgs_test = self.normalize(imgs_test.type(self.dtype))

        with torch.cuda.amp.autocast():
            img_pre_features = self.model.encode_image(imgs_test.half())
        img_features = img_pre_features / img_pre_features.norm(dim=1, keepdim=True)

        text_pre_features = self.text_pre_features.squeeze(1)
        text_features = self.text_features

        with torch.cuda.amp.autocast():
            logits_per_image = self.logit_scale.exp() * img_features @ text_features.T

        if return_features:
            return logits_per_image, img_features, text_features, img_pre_features, text_pre_features
        return logits_per_image


def get_model(cfg, num_classes: int, device: Union[str, torch.device]):
    """Load an OpenCLIP model and wrap it as ZeroShotCLIP."""
    assert cfg.MODEL.USE_CLIP, "This codebase only supports CLIP models (MODEL.USE_CLIP=True)."

    base_model, _, preprocess = create_model_and_transforms(
        cfg.MODEL.ARCH,
        pretrained=cfg.MODEL.WEIGHTS,
        device=device,
        precision=cfg.CLIP.PRECISION,
    )
    normalization = preprocess.transforms[-1]
    preprocess.transforms = preprocess.transforms[:-1]
    base_model = ZeroShotCLIP(cfg, base_model, device, normalize=normalization)

    return base_model, preprocess

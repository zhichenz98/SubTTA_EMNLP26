import logging
import torch
import torch.nn as nn
from torchvision import transforms

from copy import deepcopy
from functools import wraps


logger = logging.getLogger(__name__)


class TTAMethod(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.episodic = cfg.MODEL.EPISODIC
        self.dataset_name = cfg.CORRUPTION.DATASET
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.performed_updates = 0
        self.reset_after_num_updates = cfg.MODEL.RESET_AFTER_NUM_UPDATES

        self.img_size = (32, 32) if "cifar" in self.dataset_name else (224, 224)
        if hasattr(self.model, "model_preprocess") and isinstance(self.model.model_preprocess, transforms.Compose):
            for transf in self.model.model_preprocess.transforms[::-1]:
                if hasattr(transf, "size"):
                    self.img_size = getattr(transf, "size")
                    if self.dataset_name == "imagenet_c" and max(self.img_size) > 224:
                        raise ValueError(
                            f"The specified model with pre-processing {model.model_preprocess} "
                            f"is not suited in combination with ImageNet-C! "
                            f"This dataset is already resized and center cropped to 224"
                        )
                    break

        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.num_trainable_params, self.num_total_params = self.get_number_trainable_params()

        self.input_buffer = None
        self.window_length = cfg.TEST.WINDOW_LENGTH
        self.pointer = torch.tensor([0], dtype=torch.long).to(self.device)
        self.has_bn = any([isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()])

        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        self.mixed_precision = cfg.MIXED_PRECISION
        self.scaler = torch.cuda.amp.GradScaler() if cfg.MIXED_PRECISION else None

    def forward(self, x):
        if self.episodic:
            self.reset()

        x = x if isinstance(x, list) else [x]

        if x[0].shape[0] == 1:  # single sample test-time adaptation
            if self.input_buffer is None:
                self.input_buffer = [x_item for x_item in x]
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=False)
            elif self.input_buffer[0].shape[0] < self.window_length:
                self.input_buffer = [torch.cat([self.input_buffer[i], x_item], dim=0) for i, x_item in enumerate(x)]
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=True)
            else:
                for i, x_item in enumerate(x):
                    self.input_buffer[i][self.pointer] = x_item

            if self.pointer == (self.window_length - 1):
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(self.input_buffer)
                    self.performed_updates += 1
                    if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                        self.reset()
                outputs = outputs[self.pointer.long().to(outputs.device)]
            else:
                if self.has_bn:
                    outputs = self.forward_sliding_window(self.input_buffer)
                    outputs = outputs[self.pointer.long()]
                else:
                    outputs = self.forward_sliding_window(x)

            self.pointer += 1
            self.pointer %= self.window_length

        else:  # common batch adaptation setting
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)
                self.performed_updates += 1
                if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                    self.reset()

        return outputs

    def forward_and_adapt(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def forward_sliding_window(self, x):
        imgs_test = x[0]
        return self.model(imgs_test)

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        if self.cfg.OPTIM.METHOD == 'Adam':
            return torch.optim.Adam(self.params,
                                    lr=self.cfg.OPTIM.LR,
                                    betas=(self.cfg.OPTIM.BETA, 0.999),
                                    weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'AdamW':
            return torch.optim.AdamW(self.params,
                                     lr=self.cfg.OPTIM.LR,
                                     betas=(self.cfg.OPTIM.BETA, 0.999),
                                     weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'SGD':
            return torch.optim.SGD(self.params,
                                   lr=self.cfg.OPTIM.LR,
                                   momentum=self.cfg.OPTIM.MOMENTUM,
                                   dampening=self.cfg.OPTIM.DAMPENING,
                                   weight_decay=self.cfg.OPTIM.WD,
                                   nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    def get_number_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"#Trainable/total parameters: {trainable:,}/{total:,} \t Ratio: {trainable / total * 100:.3f}% ")
        return trainable, total

    def reset(self):
        if self.model_states is None:
            raise Exception("cannot reset without saved model state")
        self.load_model_and_optimizer()

    def copy_model_and_optimizer(self):
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

    @staticmethod
    def copy_model(model):
        return deepcopy(model)

    @staticmethod
    def change_mode_of_batchnorm1d(model_list, to_train_mode=True):
        for model in model_list:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    if to_train_mode:
                        m.train()
                    else:
                        m.eval()


def forward_decorator(fn):
    @wraps(fn)
    def decorator(self, *args, **kwargs):
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = fn(self, *args, **kwargs)
        else:
            outputs = fn(self, *args, **kwargs)
        return outputs
    return decorator

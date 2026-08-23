# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #
_C.SETTING = "reset_each_shift"
_C.DATA_DIR = "./data"
_C.CKPT_DIR = "./ckpt"
_C.SAVE_DIR = "./output"
_C.LOG_DEST = "log.txt"
_C.LOG_TIME = ''
_C.PRINT_EVERY = -1
_C.RNG_SEED = 1
_C.DETERMINISM = False
_C.MIXED_PRECISION = False
_C.DESC = ""

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.ARCH = 'ViT-B-16'
_C.MODEL.WEIGHTS = "openai"
_C.MODEL.USE_CLIP = True
_C.MODEL.CKPT_PATH = ""
_C.MODEL.ADAPTATION = 'source'
_C.MODEL.EPISODIC = False
_C.MODEL.RESET_AFTER_NUM_UPDATES = 0

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()
_C.CORRUPTION.DATASET = 'cifar10_c'
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]
_C.CORRUPTION.NUM_EX = -1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()
_C.OPTIM.STEPS = 1
_C.OPTIM.LR = 1e-3
_C.OPTIM.METHOD = 'Adam'
_C.OPTIM.BETA = 0.9
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.DAMPENING = 0.0
_C.OPTIM.NESTEROV = True
_C.OPTIM.WD = 0.0

# ------------------------------- CLIP options ---------------------------- #
_C.CLIP = CfgNode()
_C.CLIP.PROMPT_TEMPLATE = ["a photo of a {}."]
_C.CLIP.PRECISION = "fp16"
_C.CLIP.FREEZE_TEXT_ENCODER = True

# --------------------------------- MINT options ---------------------------- #
_C.MINT = CfgNode()
_C.MINT.PRIOR_N = 10000

# ------------------------------- SubTTA options -------------------------- #
_C.SUBTTA = CfgNode()
_C.SUBTTA.SUBSPACE_RANK = 64
_C.SUBTTA.REMOVE_FIRST_PC = True
_C.SUBTTA.COV_MOMENTUM = 0.9
_C.SUBTTA.LOSS_TYPE = "mint"  # source | tent | mint | batclip
_C.SUBTTA.ENT_SCALE = 0.05

# ------------------------------- Testing options ------------------------- #
_C.TEST = CfgNode()
_C.TEST.NUM_WORKERS = 4
_C.TEST.BATCH_SIZE = 128
_C.TEST.N_CLASSES = -1
_C.TEST.WINDOW_LENGTH = 1
_C.TEST.N_AUGMENTATIONS = 32
_C.TEST.DELTA_DIRICHLET = 0.0
_C.TEST.DEBUG = False

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True

_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def reset_cfg():
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    if cfg.MODEL.ADAPTATION == 'subtta':
        cfg.SAVE_DIR = os.path.join(
            cfg.SAVE_DIR,
            f"{cfg.CORRUPTION.DATASET}/{cfg.MODEL.ARCH}/{cfg.MODEL.ADAPTATION}_{cfg.SUBTTA.LOSS_TYPE}_{cfg.CORRUPTION.DATASET}_{current_time}",
        )
    else:
        cfg.SAVE_DIR = os.path.join(
            cfg.SAVE_DIR,
            f"{cfg.CORRUPTION.DATASET}/{cfg.MODEL.ARCH}/{cfg.MODEL.ADAPTATION}_{cfg.CORRUPTION.DATASET}_{current_time}",
        )
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)


def complete_data_dir_path(data_root_dir: str, dataset_name: str):
    mapping = {
        "imagenet_c": "ImageNet-C",
        "cifar10_c": "",
        "cifar100_c": "",
    }
    assert dataset_name in mapping.keys(), \
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(mapping.keys())}"
    return os.path.join(data_root_dir, mapping[dataset_name])


def get_num_classes(dataset_name: str):
    dataset_name2num_classes = {
        "cifar10_c": 10,
        "cifar100_c": 100,
        "imagenet_c": 1000,
    }
    assert dataset_name in dataset_name2num_classes.keys(), \
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(dataset_name2num_classes.keys())}"
    return dataset_name2num_classes[dataset_name]

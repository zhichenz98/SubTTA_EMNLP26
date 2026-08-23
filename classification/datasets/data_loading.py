import os
import logging
import random
import numpy as np
import time

import torch
import torchvision.transforms as transforms

from typing import Union
from conf import complete_data_dir_path
from datasets.corruptions_datasets import create_cifarc_dataset, create_imagenetc_dataset


logger = logging.getLogger(__name__)


def get_transform(dataset_name: str, preprocess: Union[transforms.Compose, None], use_clip: bool):
    """Build the input transform. Normalization is applied inside the model."""
    if use_clip:
        return preprocess
    if dataset_name in ["cifar10_c", "cifar100_c"]:
        return None
    if dataset_name == "imagenet_c":
        return transforms.Compose([transforms.ToTensor()])
    raise ValueError(f"Dataset '{dataset_name}' is not supported!")


def get_test_loader(setting: str, adaptation: str, dataset_name: str, preprocess: Union[transforms.Compose, None],
                    data_root_dir: str, domain_name: str, domain_names_all: list, severity: int, num_examples: int,
                    rng_seed: int, use_clip: bool, n_views: int = 64, delta_dirichlet: float = 0.,
                    batch_size: int = 128, shuffle: bool = False, workers: int = 4):
    """Create the test data loader for cifar10_c / cifar100_c / imagenet_c."""
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    data_dir = complete_data_dir_path(data_root_dir, dataset_name)
    transform = get_transform(dataset_name, preprocess, use_clip)

    if dataset_name in ["cifar10_c", "cifar100_c"]:
        test_dataset = create_cifarc_dataset(
            dataset_name=dataset_name,
            severity=severity,
            data_dir=data_dir,
            corruption=domain_name,
            corruptions_seq=domain_names_all,
            transform=transform,
            setting=setting,
        )
    elif dataset_name == "imagenet_c":
        test_dataset = create_imagenetc_dataset(
            n_examples=num_examples,
            severity=severity,
            data_dir=data_dir,
            corruption=domain_name,
            corruptions_seq=domain_names_all,
            transform=transform,
            setting=setting,
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported! Choose from: cifar10_c, cifar100_c, imagenet_c")

    try:
        random.shuffle(test_dataset.samples)

        if num_examples != -1:
            num_samples_orig = len(test_dataset)
            test_dataset.samples = random.sample(test_dataset.samples, k=min(num_examples, num_samples_orig))

        if "mixed_domains" in setting:
            logger.info(f"Successfully mixed the file paths of the following domains: {domain_names_all}")

        if "correlated" in setting:
            if delta_dirichlet > 0.:
                logger.info(f"Using Dirichlet distribution with delta={delta_dirichlet} to temporally correlated samples by class labels...")
                test_dataset.samples = sort_by_dirichlet(delta_dirichlet, samples=test_dataset.samples)
            else:
                logger.info(f"Sorting the file paths by class labels...")
                test_dataset.samples.sort(key=lambda x: x[1])
    except AttributeError:
        logger.warning("Attribute 'samples' is missing. Continuing without shuffling, sorting or subsampling the files...")

    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def sort_by_dirichlet(delta_dirichlet: float, samples: list):
    """Sort classes according to a Dirichlet distribution (NOTE-style)."""
    N = len(samples)
    samples_sorted = []
    class_labels = np.array([val[1] for val in samples])
    num_classes = int(np.max(class_labels) + 1)
    dirichlet_numchunks = num_classes

    time_start = time.time()
    time_duration = 120

    min_size = -1
    min_size_thresh = 10
    while min_size < min_size_thresh:
        idx_batch = [[] for _ in range(dirichlet_numchunks)]
        idx_batch_cls = [[] for _ in range(dirichlet_numchunks)]
        for k in range(num_classes):
            idx_k = np.where(class_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(delta_dirichlet, dirichlet_numchunks))
            proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                idx_j.append(idx)

        if time.time() > time_start + time_duration:
            raise ValueError(f"Could not correlate sequence using dirichlet value '{delta_dirichlet}'. Try other value!")

    for chunk in idx_batch_cls:
        cls_seq = list(range(num_classes))
        np.random.shuffle(cls_seq)
        for cls in cls_seq:
            idx = chunk[cls]
            samples_sorted.extend([samples[i] for i in idx])

    return samples_sorted

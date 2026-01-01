"""
Collapse Model
Dataset loading and feature extraction module

Supported datasets (all auto-downloadable via torchvision):
    Handwritten:    mnist, kmnist, fashion_mnist
    Natural images: cifar10, cifar100, stl10
    Digit scenes:   svhn
    Satellite:      eurosat
    Fine-grained:   oxford_pets, flowers102, food101
    Texture:        dtd

Usage:
    loader = make_loader("cifar10", split="train", batch_size=256)
    H = probe.extract(loader, max_samples=2000)
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.datasets as dsets
import numpy as np

from config import DATA_DIR, BATCH_SIZE, SEED


# ─────────────────────────────────────────────────────────────
# Standard transforms (224x224 for large models; 32x32 for fast CNN experiments)
# ─────────────────────────────────────────────────────────────

def _tf_224(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def _tf_224_vit():
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _tf_32():
    """Lightweight 32x32 transform; no large model required; used for E1 real-data pairs."""
    return T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ─────────────────────────────────────────────────────────────
# Single-channel -> three-channel conversion (MNIST family)
# ─────────────────────────────────────────────────────────────

class _GrayToRGB:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


def _tf_gray_224():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        _GrayToRGB(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────
# Dataset factory
# ─────────────────────────────────────────────────────────────

def _load(name: str, split: str, transform) -> torch.utils.data.Dataset:
    """Internal: return a Dataset object for the given dataset name and split."""
    root = str(DATA_DIR)
    train = (split == "train")

    if name == "mnist":
        return dsets.MNIST(root, train=train, transform=transform, download=True)

    elif name == "kmnist":
        return dsets.KMNIST(root, train=train, transform=transform, download=True)

    elif name == "fashion_mnist":
        return dsets.FashionMNIST(root, train=train, transform=transform, download=True)

    elif name == "cifar10":
        return dsets.CIFAR10(root, train=train, transform=transform, download=True)

    elif name == "cifar100":
        return dsets.CIFAR100(root, train=train, transform=transform, download=True)

    elif name == "stl10":
        stl_split = "train" if train else "test"
        return dsets.STL10(root, split=stl_split, transform=transform, download=True)

    elif name == "svhn":
        svhn_split = "train" if train else "test"
        return dsets.SVHN(root, split=svhn_split, transform=transform, download=True)

    elif name == "eurosat":
        # EuroSAT has no official train/test split; use the full dataset
        try:
            ds = dsets.EuroSAT(root, transform=transform, download=True)
        except Exception:
            ds = dsets.EuroSAT(root, transform=transform, download=False)
        return ds

    elif name == "oxford_pets":
        split_str = "trainval" if train else "test"
        return dsets.OxfordIIITPet(
            root, split=split_str, transform=transform, download=True)

    elif name == "flowers102":
        split_str = "train" if train else "test"
        return dsets.Flowers102(
            root, split=split_str, transform=transform, download=True)

    elif name == "food101":
        split_str = "train" if train else "test"
        return dsets.Food101(
            root, split=split_str, transform=transform, download=True)

    elif name == "dtd":
        split_str = "train" if train else "test"
        return dsets.DTD(root, split=split_str, transform=transform, download=True)

    else:
        raise ValueError(f"Unknown dataset: '{name}'")


def load_dataset(
    name: str,
    split: str = "train",
    transform_size: int = 224,
) -> torch.utils.data.Dataset:
    """
    Load a dataset (full version, for feature extraction).

    Args:
        name:           dataset name (see module docstring)
        split:          "train" or "test"
        transform_size: target resize dimension (224 for ViT/ResNet, 32 for fast CNN)

    Returns:
        torch.utils.data.Dataset
    """
    gray_datasets = {"mnist", "kmnist", "fashion_mnist"}
    if transform_size == 32:
        tf = _tf_32()
    elif name in gray_datasets:
        tf = _tf_gray_224()
    else:
        tf = _tf_224()

    return _load(name, split, tf)


def make_loader(
    name: str,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    n_samples: Optional[int] = None,
    transform_size: int = 224,
    seed: int = SEED,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader with optional random subsampling to n_samples.

    Args:
        name:           dataset name
        split:          "train" or "test"
        batch_size:     batch size
        n_samples:      if set, randomly subsample to this count (reproducible)
        transform_size: image size
        seed:           random seed (used for subsampling)
        num_workers:    DataLoader worker processes

    Returns:
        DataLoader
    """
    ds = load_dataset(name, split, transform_size)

    if n_samples is not None and n_samples < len(ds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=n_samples, replace=False)
        ds = Subset(ds, idx.tolist())

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def dataset_info(name: str) -> dict:
    """Return static metadata for the dataset (number of classes, image size, etc.)."""
    INFO = {
        "mnist":         {"n_classes": 10,  "img_size": 28,  "domain": "handwritten"},
        "kmnist":        {"n_classes": 10,  "img_size": 28,  "domain": "handwritten"},
        "fashion_mnist": {"n_classes": 10,  "img_size": 28,  "domain": "handwritten"},
        "cifar10":       {"n_classes": 10,  "img_size": 32,  "domain": "natural"},
        "cifar100":      {"n_classes": 100, "img_size": 32,  "domain": "natural"},
        "stl10":         {"n_classes": 10,  "img_size": 96,  "domain": "natural"},
        "svhn":          {"n_classes": 10,  "img_size": 32,  "domain": "digit_scene"},
        "eurosat":       {"n_classes": 10,  "img_size": 64,  "domain": "satellite"},
        "oxford_pets":   {"n_classes": 37,  "img_size": 224, "domain": "finegrained"},
        "flowers102":    {"n_classes": 102, "img_size": 224, "domain": "finegrained"},
        "food101":       {"n_classes": 101, "img_size": 224, "domain": "finegrained"},
        "dtd":           {"n_classes": 47,  "img_size": 224, "domain": "texture"},
    }
    if name not in INFO:
        raise ValueError(f"Unknown dataset: '{name}'")
    return INFO[name]


# ─────────────────────────────────────────────────────────────
# Feature matrix extraction utilities (used with ProbeModel)
# ─────────────────────────────────────────────────────────────

def extract_features(
    probe,
    dataset_name: str,
    split: str = "train",
    n_samples: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the feature matrix and labels for a dataset using a probe model.

    Args:
        probe:        ProbeModel instance
        dataset_name: dataset name
        split:        "train" or "test"
        n_samples:    subsample count (None = all)
        batch_size:   batch size

    Returns:
        H:      (N, d) float32 feature matrix
        labels: (N,) int64 label array
    """
    loader = make_loader(
        dataset_name, split=split,
        n_samples=n_samples, batch_size=batch_size,
    )

    feats, labs = [], []
    import torch
    with torch.no_grad():
        for batch in loader:
            images, targets = batch[0], batch[1]
            images = images.to(probe.device)
            f = probe._forward(images)
            feats.append(f.cpu().numpy())
            labs.append(targets.numpy() if hasattr(targets, "numpy") else np.array(targets))

    H      = np.concatenate(feats, axis=0).astype(np.float32)
    labels = np.concatenate(labs,  axis=0).astype(np.int64)
    return H, labels


# ─────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("datasets.py -- unit tests (DataLoader construction only, no download)")

    for ds_name in ["cifar10", "mnist"]:
        info = dataset_info(ds_name)
        print(f"  {ds_name}: n_classes={info['n_classes']}, domain={info['domain']}")

    print("dataset_info tests passed")

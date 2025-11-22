"""Dataset and dataloader helpers for the cat breed classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataSettings:
    image_size: int = 256
    center_crop: int = 224
    augment: bool = True
    auto_augment: bool = False
    color_jitter: float = 0.1


def build_transforms(settings: DataSettings) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train/eval torchvision transforms."""

    train_tfms = [
        transforms.Resize(int(settings.image_size * 1.1)),
        transforms.RandomResizedCrop(settings.center_crop, scale=(0.85, 1.0)),
    ]
    if settings.augment:
        train_tfms.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ColorJitter(settings.color_jitter, settings.color_jitter, settings.color_jitter / 2, 0.05),
            ]
        )
        if settings.auto_augment:
            train_tfms.append(transforms.AutoAugment())
    train_tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(settings.image_size),
            transforms.CenterCrop(settings.center_crop),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return transforms.Compose(train_tfms), eval_tfms


def create_datasets(data_dir: Path, settings: DataSettings) -> Dict[str, datasets.ImageFolder]:
    """Instantiate ImageFolder datasets for available splits."""

    train_tfms, eval_tfms = build_transforms(settings)
    datasets_per_split: Dict[str, datasets.ImageFolder] = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        transform = train_tfms if split == "train" else eval_tfms
        datasets_per_split[split] = datasets.ImageFolder(split_dir, transform=transform)
    if "train" not in datasets_per_split:
        raise FileNotFoundError(f"Train split missing under {data_dir}. Ensure data/processed/train exists.")
    return datasets_per_split


def create_dataloaders(
    datasets_per_split: Dict[str, datasets.ImageFolder],
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Wrap datasets into PyTorch dataloaders."""

    dataloaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets_per_split.items():
        shuffle = split == "train"
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return dataloaders


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Return inverse-frequency class weights for imbalanced datasets."""

    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / (counts + 1e-9)
    weights /= weights.mean()
    return weights

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_loaders(data_dir, batch_size=32, val_split=0.2, seed=42):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {data_dir}")

    train_ds = datasets.ImageFolder(data_dir, transform=train_transform())
    val_ds = datasets.ImageFolder(data_dir, transform=val_transform())
    if len(train_ds.classes) < 2:
        raise ValueError(
            f"Achei só as pastas {train_ds.classes} em {data_dir}. "
            "Aponte --data-dir para a pasta que contém direto as pastas das classes."
        )

    # Mesmo split para os dois datasets, com transforms diferentes
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_ds), generator=generator).tolist()
    n_val = int(len(indices) * val_split)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size)
    return train_loader, val_loader, train_ds.classes
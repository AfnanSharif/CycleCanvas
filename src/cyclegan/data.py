from __future__ import annotations

import random
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: str | Path) -> list[Path]:
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"image directory does not exist: {path}")
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in EXTENSIONS)


def create_dataloader(domain_a: str | Path, domain_b: str | Path, image_size: int, batch_size: int, workers: int, seed: int):
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Data loading requires torch, torchvision, and Pillow") from exc
    images_a, images_b = list_images(domain_a), list_images(domain_b)
    if not images_a or not images_b:
        raise ValueError("both domains need at least one supported image")
    transform = transforms.Compose([
        transforms.Resize((image_size + 30, image_size + 30), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(image_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    class UnpairedDataset(Dataset):
        def __len__(self): return max(len(images_a), len(images_b))
        def __getitem__(self, index):
            a = Image.open(images_a[index % len(images_a)]).convert("RGB")
            b = Image.open(images_b[random.randrange(len(images_b))]).convert("RGB")
            return {"A": transform(a), "B": transform(b), "path_A": str(images_a[index % len(images_a)])}

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(UnpairedDataset(), batch_size=batch_size, shuffle=True, num_workers=workers, generator=generator, drop_last=True)

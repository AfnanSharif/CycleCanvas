from __future__ import annotations

import random


class ImagePool:
    """Replay previously generated images to reduce discriminator oscillation."""

    def __init__(self, size: int, seed: int = 42):
        if size < 0: raise ValueError("pool size cannot be negative")
        self.size, self.images, self.random = size, [], random.Random(seed)

    def query(self, batch):
        if self.size == 0: return batch.detach()
        returned = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(image.clone()); returned.append(image)
            elif self.random.random() > 0.5:
                index = self.random.randrange(self.size)
                old = self.images[index].clone(); self.images[index] = image.clone(); returned.append(old)
            else: returned.append(image)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("ImagePool requires PyTorch tensors") from exc
        return torch.cat(returned, dim=0)

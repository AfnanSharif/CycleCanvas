from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CycleGANConfig:
    image_size: int = 256
    batch_size: int = 1
    epochs: int = 200
    decay_start_epoch: int = 100
    learning_rate: float = 0.0002
    beta1: float = 0.5
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    residual_blocks: int = 9
    pool_size: int = 50
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    output_dir: str = "outputs"

    def validate(self) -> "CycleGANConfig":
        if self.image_size < 64 or self.image_size % 4:
            raise ValueError("image_size must be at least 64 and divisible by 4")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if not 0 <= self.decay_start_epoch < self.epochs:
            raise ValueError("decay_start_epoch must be in [0, epochs)")
        if not 0 < self.learning_rate <= 0.01:
            raise ValueError("learning_rate must be in (0, 0.01]")
        if self.lambda_cycle < 0 or self.lambda_identity < 0:
            raise ValueError("loss weights cannot be negative")
        if self.residual_blocks < 1 or self.pool_size < 0:
            raise ValueError("residual_blocks must be positive and pool_size non-negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def output_path(self, root: str | Path = ".") -> Path:
        path = Path(root) / self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


class LinearDecay:
    """Epoch multiplier: constant first, then linearly decay to zero."""

    def __init__(self, total_epochs: int, decay_start: int):
        if not 0 <= decay_start < total_epochs:
            raise ValueError("invalid decay schedule")
        self.total_epochs, self.decay_start = total_epochs, decay_start

    def __call__(self, epoch: int) -> float:
        if epoch < self.decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - self.decay_start) / max(1, self.total_epochs - self.decay_start))

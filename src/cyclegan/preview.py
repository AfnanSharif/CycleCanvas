from __future__ import annotations

import random
from pathlib import Path


def write_synthetic_domains(directory: str | Path, seed: int = 42, size: int = 96) -> tuple[Path, Path]:
    """Create a dependency-free palette demo. Files are not training/model output."""
    if size < 32: raise ValueError("size must be at least 32")
    rng, root = random.Random(seed), Path(directory); root.mkdir(parents=True, exist_ok=True)
    paths = (root / "synthetic_domain_a.ppm", root / "synthetic_domain_b.ppm")
    for domain, path in enumerate(paths):
        pixels = []
        for y in range(size):
            row = []
            for x in range(size):
                wave = int(25 * ((x + y) % 13) / 12)
                if domain == 0:
                    color = (150 + wave, 92 + rng.randrange(8), 48 + rng.randrange(8))
                else:
                    stripe = 230 if (x // 8 + y // 18) % 2 else 30
                    color = (stripe, stripe, min(255, stripe + wave))
                row.extend(color)
            pixels.append(row)
        with path.open("w", encoding="ascii") as target:
            target.write(f"P3\n# Synthetic domain preview; NOT CycleGAN output\n{size} {size}\n255\n")
            target.writelines(" ".join(map(str, row)) + "\n" for row in pixels)
    return paths

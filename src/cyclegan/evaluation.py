from __future__ import annotations

import json
from pathlib import Path

from .data import list_images
from .models import build_generator
from .training import resolve_device


def test_split_counts(test_a: str | Path, test_b: str | Path) -> dict[str, int]:
    return {"testA": len(list_images(test_a)), "testB": len(list_images(test_b))}


def evaluate_checkpoint(checkpoint: str | Path, test_a: str | Path, test_b: str | Path, destination: str | Path, device: str = "auto") -> dict[str, object]:
    """Translate held-out domains and measure forward/backward cycle L1."""
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.utils import save_image
    except ImportError as exc:
        raise RuntimeError("Evaluation requires torch, torchvision, and Pillow") from exc
    images = {"AtoB": list_images(test_a), "BtoA": list_images(test_b)}
    if not images["AtoB"] and not images["BtoA"]:
        raise ValueError("add at least one image to testA or testB")
    target_device = resolve_device(device)
    payload = torch.load(checkpoint, map_location=target_device, weights_only=True)
    config = payload["config"]
    size, blocks = int(config["image_size"]), int(config["residual_blocks"])
    generators = {"AtoB": build_generator(blocks).to(target_device), "BtoA": build_generator(blocks).to(target_device)}
    generators["AtoB"].load_state_dict(payload["G_AB"]); generators["BtoA"].load_state_dict(payload["G_BA"])
    for model in generators.values(): model.eval()
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    output = Path(destination); output.mkdir(parents=True, exist_ok=True)
    records, losses = [], []
    with torch.inference_mode():
        for direction, paths in images.items():
            reverse = "BtoA" if direction == "AtoB" else "AtoB"
            direction_dir = output / direction; direction_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(paths, 1):
                real = transform(Image.open(source).convert("RGB")).unsqueeze(0).to(target_device)
                translated = generators[direction](real)
                reconstructed = generators[reverse](translated)
                cycle_l1 = float(torch.mean(torch.abs(reconstructed - real)))
                losses.append(cycle_l1)
                translated_path = direction_dir / f"{index:04d}_{source.stem}_translated.jpg"
                cycle_path = direction_dir / f"{index:04d}_{source.stem}_cycle.jpg"
                save_image(translated, translated_path, normalize=True)
                save_image(reconstructed, cycle_path, normalize=True)
                records.append({"source": str(source), "direction": direction, "translated": str(translated_path), "cycle": str(cycle_path), "cycle_l1": round(cycle_l1, 6)})
    report = {
        "checkpoint": str(checkpoint),
        "testA": len(images["AtoB"]),
        "testB": len(images["BtoA"]),
        "images": len(records),
        "mean_cycle_l1": round(sum(losses) / len(losses), 6),
        "records": records,
        "note": "Cycle L1 measures reconstruction consistency; it does not measure target-domain realism.",
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


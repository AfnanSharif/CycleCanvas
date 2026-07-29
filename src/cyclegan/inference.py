from __future__ import annotations

from pathlib import Path

from .models import build_generator
from .training import resolve_device


def translate(checkpoint: str | Path, source: str | Path, destination: str | Path, direction: str = "AtoB", device: str = "auto") -> Path:
    if direction not in {"AtoB", "BtoA"}: raise ValueError("direction must be AtoB or BtoA")
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.utils import save_image
    except ImportError as exc:
        raise RuntimeError("Inference requires torch, torchvision, and Pillow") from exc
    target_device = resolve_device(device)
    payload = torch.load(checkpoint, map_location=target_device, weights_only=True)
    config = payload["config"]; size = int(config["image_size"])
    generator = build_generator(int(config["residual_blocks"])).to(target_device)
    generator.load_state_dict(payload["G_AB" if direction == "AtoB" else "G_BA"]); generator.eval()
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    tensor = transform(Image.open(source).convert("RGB")).unsqueeze(0).to(target_device)
    with torch.no_grad(): output_tensor = generator(tensor)
    output = Path(destination); output.parent.mkdir(parents=True, exist_ok=True)
    save_image(output_tensor, output, normalize=True)
    return output

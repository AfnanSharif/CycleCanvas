from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ArchitectureProfile:
    architecture: str
    image_size: int
    parameter_estimate: int
    skip_connections: int
    spatial_inductive_bias: bool
    longest_nonlinear_path: int
    learning_implication: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def architecture_profiles(image_size: int = 32, hidden_features: int = 256, dense_depth: int = 3, residual_blocks: int = 3, resnet_features: int = 16) -> list[ArchitectureProfile]:
    if image_size < 16 or image_size > 64 or image_size % 4:
        raise ValueError("comparison image_size must be 16..64 and divisible by 4")
    if hidden_features < 16 or not 2 <= dense_depth <= 12:
        raise ValueError("use hidden_features >= 16 and dense_depth between 2 and 12")
    if residual_blocks < 1 or resnet_features < 4:
        raise ValueError("residual_blocks and resnet_features must be positive")
    pixels = 3 * image_size * image_size
    dense_params = pixels * hidden_features + hidden_features
    dense_params += max(0, dense_depth - 2) * (hidden_features * hidden_features + hidden_features)
    dense_params += hidden_features * pixels + pixels

    # Exact convolution weight/bias count for build_generator at this width;
    # InstanceNorm layers use affine=False and therefore add no parameters.
    f = resnet_features
    resnet_params = (3 * f * 7 * 7 + f)
    resnet_params += (f * (2 * f) * 3 * 3 + 2 * f) + ((2 * f) * (4 * f) * 3 * 3 + 4 * f)
    resnet_params += residual_blocks * 2 * ((4 * f) * (4 * f) * 3 * 3 + 4 * f)
    resnet_params += ((4 * f) * (2 * f) * 3 * 3 + 2 * f) + ((2 * f) * f * 3 * 3 + f)
    resnet_params += f * 3 * 7 * 7 + 3
    return [
        ArchitectureProfile("Dense DNN", image_size, dense_params, 0, False, dense_depth, "Global fully connected mappings can memorize a small benchmark but discard locality and grow rapidly with image resolution."),
        ArchitectureProfile("Residual CNN", image_size, resnet_params, residual_blocks, True, 6 + 2 * residual_blocks, "Convolutions share spatial features while residual identity paths shorten gradient routes and preserve image content."),
    ]


def run_learning_comparison(*, steps: int = 20, image_size: int = 32, batch_size: int = 4, hidden_features: int = 256, dense_depth: int = 3, residual_blocks: int = 3, features: int = 16, learning_rate: float = 1e-3, seed: int = 42, device: str = "auto") -> dict[str, object]:
    """Train both generators on the same deterministic synthetic flip task."""
    if not 1 <= steps <= 500 or not 1 <= batch_size <= 32:
        raise ValueError("steps must be 1..500 and batch_size 1..32")
    if not 0 < learning_rate <= 0.1:
        raise ValueError("learning_rate must be in (0, 0.1]")
    profiles = architecture_profiles(image_size, hidden_features, dense_depth, residual_blocks, features)
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("The learning comparison requires PyTorch") from exc
    from .models import build_dense_generator, build_generator
    from .training import resolve_device

    torch.manual_seed(seed)
    target_device = resolve_device(device)
    inputs = torch.rand(batch_size, 3, image_size, image_size, device=target_device) * 2 - 1
    targets = torch.flip(inputs, dims=(3,))
    models = {
        "Dense DNN": build_dense_generator(image_size, hidden_features, dense_depth).to(target_device),
        "Residual CNN": build_generator(residual_blocks, features).to(target_device),
    }
    results: dict[str, object] = {}
    criterion = nn.MSELoss()
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        losses, gradients = [], []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            gradient = math.sqrt(sum(float(parameter.grad.detach().pow(2).sum()) for parameter in model.parameters() if parameter.grad is not None))
            optimizer.step()
            losses.append(float(loss.detach()))
            gradients.append(gradient)
        results[name] = {
            "initial_loss": round(losses[0], 6),
            "final_loss": round(losses[-1], 6),
            "relative_improvement": round((losses[0] - losses[-1]) / max(losses[0], 1e-12), 6),
            "first_gradient_norm": round(gradients[0], 6),
            "final_gradient_norm": round(gradients[-1], 6),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    return {
        "task": "learn a horizontal image flip from one fixed synthetic mini-batch",
        "warning": "This controlled learning demonstration is not a CycleGAN quality benchmark.",
        "config": {"steps": steps, "image_size": image_size, "batch_size": batch_size, "seed": seed},
        "profiles": [profile.to_dict() for profile in profiles],
        "results": results,
    }

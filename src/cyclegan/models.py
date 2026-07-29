from __future__ import annotations


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for CycleGAN models; install requirements.txt") from exc
    return torch, nn


def build_generator(residual_blocks: int = 9, features: int = 64):
    _, nn = _torch()

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels),
            )

        def forward(self, value):
            return value + self.block(value)

    layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, features, 7), nn.InstanceNorm2d(features), nn.ReLU(inplace=True)]
    channels = features
    for _ in range(2):
        layers += [nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(channels * 2), nn.ReLU(inplace=True)]
        channels *= 2
    layers += [ResidualBlock(channels) for _ in range(residual_blocks)]
    for _ in range(2):
        layers += [nn.ConvTranspose2d(channels, channels // 2, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(channels // 2), nn.ReLU(inplace=True)]
        channels //= 2
    layers += [nn.ReflectionPad2d(3), nn.Conv2d(channels, 3, 7), nn.Tanh()]
    return nn.Sequential(*layers)


def build_discriminator(features: int = 64):
    _, nn = _torch()
    layers = [nn.Conv2d(3, features, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
    channels = features
    for multiplier, stride in ((2, 2), (4, 2), (8, 1)):
        layers += [nn.Conv2d(channels, features * multiplier, 4, stride=stride, padding=1, bias=False), nn.InstanceNorm2d(features * multiplier), nn.LeakyReLU(0.2, inplace=True)]
        channels = features * multiplier
    layers += [nn.Conv2d(channels, 1, 4, padding=1)]
    return nn.Sequential(*layers)


def build_dense_generator(image_size: int = 32, hidden_features: int = 256, depth: int = 3):
    """Educational fully connected image translator used only for comparison.

    The production CycleGAN continues to use ``build_generator``. Flattening an
    image discards spatial locality and scales parameter count quadratically,
    which is precisely why this DNN baseline is constrained to small images.
    """
    torch, nn = _torch()
    if image_size < 16 or image_size > 64:
        raise ValueError("dense comparison image_size must be between 16 and 64")
    if hidden_features < 16 or depth < 2:
        raise ValueError("dense comparison needs hidden_features >= 16 and depth >= 2")
    pixels = 3 * image_size * image_size

    class DenseGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Linear(pixels, hidden_features), nn.GELU()]
            for _ in range(depth - 2):
                layers += [nn.Linear(hidden_features, hidden_features), nn.GELU()]
            layers += [nn.Linear(hidden_features, pixels), nn.Tanh()]
            self.network = nn.Sequential(*layers)

        def forward(self, images):
            return self.network(images.reshape(images.size(0), pixels)).reshape(-1, 3, image_size, image_size)

    return DenseGenerator()


def initialize_weights(module) -> None:
    _, nn = _torch()
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None: nn.init.zeros_(module.bias.data)

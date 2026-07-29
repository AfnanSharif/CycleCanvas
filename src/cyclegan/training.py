from __future__ import annotations

import json
import random
from pathlib import Path

from .config import CycleGANConfig, LinearDecay
from .data import create_dataloader
from .image_pool import ImagePool
from .models import build_discriminator, build_generator, initialize_weights


def resolve_device(requested: str):
    try: import torch
    except ImportError as exc: raise RuntimeError("Training requires PyTorch") from exc
    if requested != "auto": return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def train(config: CycleGANConfig, domain_a: str | Path, domain_b: str | Path, root: str | Path = ".") -> list[dict[str, float | int]]:
    config.validate()
    try:
        import torch
        from torch import nn
        from torchvision.utils import save_image
    except ImportError as exc:
        raise RuntimeError("Training requires torch and torchvision; install requirements.txt") from exc
    random.seed(config.seed); torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    device, output = resolve_device(config.device), config.output_path(root)
    loader = create_dataloader(domain_a, domain_b, config.image_size, config.batch_size, config.num_workers, config.seed)
    g_ab, g_ba = build_generator(config.residual_blocks).to(device), build_generator(config.residual_blocks).to(device)
    d_a, d_b = build_discriminator().to(device), build_discriminator().to(device)
    for model in (g_ab, g_ba, d_a, d_b): model.apply(initialize_weights)
    optimizer_g = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizer_d = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=config.learning_rate, betas=(config.beta1, 0.999))
    schedule = LinearDecay(config.epochs, config.decay_start_epoch)
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=schedule)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=schedule)
    gan_loss, cycle_loss, identity_loss = nn.MSELoss(), nn.L1Loss(), nn.L1Loss()
    pool_a, pool_b = ImagePool(config.pool_size, config.seed), ImagePool(config.pool_size, config.seed + 1)
    history = []
    for epoch in range(config.epochs):
        sums = {"g": 0.0, "d": 0.0, "cycle": 0.0, "identity": 0.0}
        for batch in loader:
            real_a, real_b = batch["A"].to(device), batch["B"].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            identity_a, identity_b = g_ba(real_a), g_ab(real_b)
            loss_identity = (identity_loss(identity_a, real_a) + identity_loss(identity_b, real_b)) * config.lambda_identity
            fake_b, fake_a = g_ab(real_a), g_ba(real_b)
            loss_gan = gan_loss(d_b(fake_b), torch.ones_like(d_b(fake_b))) + gan_loss(d_a(fake_a), torch.ones_like(d_a(fake_a)))
            recovered_a, recovered_b = g_ba(fake_b), g_ab(fake_a)
            loss_cycle = (cycle_loss(recovered_a, real_a) + cycle_loss(recovered_b, real_b)) * config.lambda_cycle
            loss_g = loss_identity + loss_gan + loss_cycle; loss_g.backward(); optimizer_g.step()
            optimizer_d.zero_grad(set_to_none=True)
            pooled_a, pooled_b = pool_a.query(fake_a), pool_b.query(fake_b)
            loss_da = (gan_loss(d_a(real_a), torch.ones_like(d_a(real_a))) + gan_loss(d_a(pooled_a), torch.zeros_like(d_a(pooled_a)))) * 0.5
            loss_db = (gan_loss(d_b(real_b), torch.ones_like(d_b(real_b))) + gan_loss(d_b(pooled_b), torch.zeros_like(d_b(pooled_b)))) * 0.5
            loss_d = loss_da + loss_db; loss_d.backward(); optimizer_d.step()
            sums["g"] += loss_g.item(); sums["d"] += loss_d.item(); sums["cycle"] += loss_cycle.item(); sums["identity"] += loss_identity.item()
        scheduler_g.step(); scheduler_d.step(); count = len(loader)
        row = {"epoch": epoch + 1, **{key: round(value / count, 6) for key, value in sums.items()}, "lr": scheduler_g.get_last_lr()[0]}; history.append(row)
        with torch.no_grad():
            preview = torch.cat((real_a[:1], fake_b[:1], recovered_a[:1], real_b[:1], fake_a[:1], recovered_b[:1]))
            save_image(preview, output / f"epoch_{epoch + 1:03d}.jpg", normalize=True, nrow=3)
        torch.save({"config": config.to_dict(), "epoch": epoch + 1, "G_AB": g_ab.state_dict(), "G_BA": g_ba.state_dict(), "D_A": d_a.state_dict(), "D_B": d_b.state_dict(), "history": history}, output / "checkpoint.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch {epoch + 1:03d}/{config.epochs} G={row['g']:.4f} D={row['d']:.4f} cycle={row['cycle']:.4f}")
    return history

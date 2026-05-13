"""Synthetic smoke-training script for NeCK-DiffReg."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from neck_diffreg.data.dataset import build_dataloader
from neck_diffreg.losses.total_loss import NeckDiffRegLoss
from neck_diffreg.models.neck_diffreg import NeCKDiffReg
from neck_diffreg.training.config import load_config
from neck_diffreg.utils.io import ensure_dir
from neck_diffreg.utils.seed import set_seed


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cpu"))
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train(config_path: str | Path) -> Path:
    """Run a short synthetic training job and save a checkpoint."""

    config = load_config(config_path)
    set_seed(int(config.get("seed", 1234)))
    device = _device_from_config(config.get("training", {}))

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})

    loader = build_dataloader(
        data_cfg,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
    )
    model = NeCKDiffReg(
        image_channels=int(model_cfg.get("image_channels", 1)),
        tissue_channels=int(model_cfg.get("tissue_channels", 0)),
        base_channels=int(model_cfg.get("base_channels", 8)),
        num_skeleton_nodes=int(model_cfg.get("num_skeleton_nodes", 10)),
        integration_steps=int(model_cfg.get("velocity_integration_steps", model_cfg.get("integration_steps", 4))),
        residual_velocity_scale=float(model_cfg.get("residual_velocity_scale", 0.15)),
    ).to(device)
    criterion = NeckDiffRegLoss(
        weights=loss_cfg.get("weights", {}),
        similarity=str(loss_cfg.get("similarity", "weighted_mse")),
        ncc_window_size=int(loss_cfg.get("ncc_window_size", 5)),
        tissue_aware_smoothness=bool(loss_cfg.get("tissue_aware_smoothness", False)),
        tissue_channel_weights=loss_cfg.get("tissue_channel_weights"),
        reliability_mean_target=float(loss_cfg.get("reliability_mean_target", 0.75)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1.0e-3)))

    max_iters = int(train_cfg.get("max_iters", 2))
    log_interval = max(1, int(train_cfg.get("log_interval", 1)))
    model.train()
    step = 0
    while step < max_iters:
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(
                moving=batch["moving"],
                fixed=batch["fixed"],
                moving_seg=batch.get("moving_seg"),
                fixed_seg=batch.get("fixed_seg"),
                tissue_map=batch.get("tissue_map"),
                bone_masks=batch.get("bone_masks"),
                reliability_prior=batch.get("reliability_prior"),
            )
            loss, components = criterion(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if step == 1 or step % log_interval == 0 or step >= max_iters:
                component_text = ", ".join(
                    f"{name}={value.detach().cpu().item():.5f}"
                    for name, value in components.items()
                    if name != "total"
                )
                print(f"iter={step:03d} total={loss.detach().cpu().item():.5f} {component_text}", flush=True)
            if step >= max_iters:
                break

    output_dir = Path(train_cfg.get("output_dir", "outputs/checkpoints"))
    ensure_dir(output_dir)
    checkpoint_name = str(train_cfg.get("checkpoint_name", "neck_diffreg_synthetic.pt"))
    checkpoint_path = output_dir / checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "step": step,
        },
        checkpoint_path,
    )
    print(f"saved_checkpoint={checkpoint_path}", flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeCK-DiffReg on synthetic 3D data.")
    parser.add_argument("--config", type=str, default="configs/neck_diffreg_base.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

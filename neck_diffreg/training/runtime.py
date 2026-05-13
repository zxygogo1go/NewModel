"""Shared runtime helpers for training and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neck_diffreg.losses.total_loss import NeckDiffRegLoss
from neck_diffreg.models.neck_diffreg import NeCKDiffReg
from neck_diffreg.utils.io import ensure_dir


def device_from_config(config: dict[str, Any]) -> torch.device:
    """Resolve the requested device with safe CPU fallback."""

    requested = str(config.get("device", "cpu"))
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch to the target device."""

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_model_from_config(model_cfg: dict[str, Any]) -> NeCKDiffReg:
    """Construct ``NeCKDiffReg`` from a config mapping."""

    return NeCKDiffReg(
        image_channels=int(model_cfg.get("image_channels", 1)),
        tissue_channels=int(model_cfg.get("tissue_channels", 0)),
        base_channels=int(model_cfg.get("base_channels", 8)),
        num_skeleton_nodes=int(model_cfg.get("num_skeleton_nodes", 10)),
        integration_steps=int(model_cfg.get("velocity_integration_steps", model_cfg.get("integration_steps", 4))),
        residual_velocity_scale=float(model_cfg.get("residual_velocity_scale", 0.15)),
    )


def build_criterion_from_config(loss_cfg: dict[str, Any]) -> NeckDiffRegLoss:
    """Construct the composite loss from a config mapping."""

    return NeckDiffRegLoss(
        weights=loss_cfg.get("weights", {}),
        similarity=str(loss_cfg.get("similarity", "weighted_mse")),
        ncc_window_size=int(loss_cfg.get("ncc_window_size", 5)),
        tissue_aware_smoothness=bool(loss_cfg.get("tissue_aware_smoothness", False)),
        tissue_channel_weights=loss_cfg.get("tissue_channel_weights"),
        reliability_mean_target=float(loss_cfg.get("reliability_mean_target", 0.75)),
        jacobian_margin=float(loss_cfg.get("jacobian_margin", 0.0)),
        jacobian_penalty=str(loss_cfg.get("jacobian_penalty", "squared")),
    )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    step: int,
    best_metric: float | None = None,
) -> Path:
    """Save a training checkpoint."""

    checkpoint_path = Path(path)
    ensure_dir(checkpoint_path.parent)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if best_metric is not None:
        payload["best_metric"] = best_metric
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into model and optionally optimizer."""

    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

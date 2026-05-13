"""Validation entry points and evaluation loop."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from neck_diffreg.data.dataset import build_dataloader
from neck_diffreg.evaluation.metrics import registration_metrics
from neck_diffreg.training.config import load_config
from neck_diffreg.training.runtime import (
    build_criterion_from_config,
    build_model_from_config,
    device_from_config,
    load_checkpoint,
    to_device,
)


def _forward_model(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    return model(
        moving=batch["moving"],
        fixed=batch["fixed"],
        moving_seg=batch.get("moving_seg"),
        fixed_seg=batch.get("fixed_seg"),
        tissue_map=batch.get("tissue_map"),
        bone_masks=batch.get("bone_masks"),
        reliability_prior=batch.get("reliability_prior"),
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate a model on a validation loader.

    Returns averaged loss components and registration metrics. The function
    preserves the caller's training/eval mode.
    """

    was_training = model.training
    model.eval()
    accum: dict[str, list[float]] = defaultdict(list)
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        outputs = _forward_model(model, batch)
        loss, components = criterion(outputs, batch)
        accum["loss"].append(float(loss.detach().cpu()))
        for name, value in components.items():
            accum[f"loss_{name}"].append(float(value.detach().cpu()))
        for name, value in registration_metrics(outputs, batch).items():
            accum[name].append(float(value.detach().cpu()))
    if was_training:
        model.train()
    if not accum:
        raise ValueError("Validation loader produced no batches")
    return {name: sum(values) / len(values) for name, values in accum.items()}


def build_validation_loader(config: dict[str, Any]) -> DataLoader:
    """Build validation loader from ``val_data`` or infer it from ``data``."""

    train_cfg = config.get("training", {})
    if "val_data" in config:
        val_data_cfg = dict(config["val_data"])
    else:
        val_data_cfg = dict(config.get("data", {}))
        val_data_cfg["split"] = "val"
        val_data_cfg.pop("list_path", None)
        val_data_cfg.pop("pairs_per_epoch", None)
    return build_dataloader(
        val_data_cfg,
        batch_size=int(train_cfg.get("val_batch_size", train_cfg.get("batch_size", 1))),
        shuffle=False,
    )


def validate(config_path: str | Path, checkpoint_path: str | Path | None = None) -> dict[str, float]:
    """Run standalone validation from a config and optional checkpoint."""

    config = load_config(config_path)
    device = device_from_config(config.get("training", {}))
    model = build_model_from_config(config.get("model", {})).to(device)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, map_location=device)
    criterion = build_criterion_from_config(config.get("loss", {}))
    loader = build_validation_loader(config)
    metrics = evaluate_model(
        model,
        loader,
        criterion,
        device,
        max_batches=config.get("training", {}).get("val_max_batches"),
    )
    metric_text = ", ".join(f"{name}={value:.5f}" for name, value in sorted(metrics.items()))
    print(f"validation {metric_text}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NeCK-DiffReg.")
    parser.add_argument("--config", type=str, default="configs/neck_diffreg_segrap_train.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    validate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()

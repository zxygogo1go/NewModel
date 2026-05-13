"""Synthetic smoke-training script for NeCK-DiffReg."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from neck_diffreg.data.dataset import build_dataloader
from neck_diffreg.training.config import load_config
from neck_diffreg.training.runtime import (
    build_criterion_from_config,
    build_model_from_config,
    device_from_config,
    load_checkpoint,
    save_checkpoint,
    to_device,
)
from neck_diffreg.training.validate import build_validation_loader, evaluate_model
from neck_diffreg.utils.seed import set_seed


def train(config_path: str | Path) -> Path:
    """Run training and save checkpoints."""

    config = load_config(config_path)
    set_seed(int(config.get("seed", 1234)))
    device = device_from_config(config.get("training", {}))

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})

    loader = build_dataloader(
        data_cfg,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
    )
    val_loader = None
    val_interval = int(train_cfg.get("val_interval", 0))
    if val_interval > 0:
        val_loader = build_validation_loader(config)

    model = build_model_from_config(model_cfg).to(device)
    criterion = build_criterion_from_config(loss_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1.0e-3)))

    max_iters = int(train_cfg.get("max_iters", 2))
    log_interval = max(1, int(train_cfg.get("log_interval", 1)))
    save_every = int(train_cfg.get("save_every", 0))
    output_dir = Path(train_cfg.get("output_dir", "outputs/checkpoints"))
    checkpoint_name = str(train_cfg.get("checkpoint_name", "neck_diffreg_synthetic.pt"))
    checkpoint_path = output_dir / checkpoint_name
    best_checkpoint_name = str(train_cfg.get("best_checkpoint_name", f"best_{checkpoint_name}"))
    best_checkpoint_path = output_dir / best_checkpoint_name
    resume_from = train_cfg.get("resume_from")
    val_metric_name = str(train_cfg.get("val_metric", "loss"))
    val_metric_mode = str(train_cfg.get("val_metric_mode", "min"))
    best_metric = float("inf") if val_metric_mode == "min" else -float("inf")
    step = 0

    if resume_from:
        checkpoint = load_checkpoint(resume_from, model, optimizer=optimizer, map_location=device)
        step = int(checkpoint.get("step", 0))
        if "best_metric" in checkpoint:
            best_metric = float(checkpoint["best_metric"])
        print(f"resumed_checkpoint={resume_from} step={step}", flush=True)

    model.train()
    while step < max_iters:
        for batch in loader:
            batch = to_device(batch, device)
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
            if val_loader is not None and step % val_interval == 0:
                val_metrics = evaluate_model(
                    model,
                    val_loader,
                    criterion,
                    device,
                    max_batches=train_cfg.get("val_max_batches"),
                )
                metric_text = ", ".join(f"{name}={value:.5f}" for name, value in sorted(val_metrics.items()))
                print(f"val_iter={step:03d} {metric_text}", flush=True)
                current_metric = float(val_metrics[val_metric_name])
                is_better = current_metric < best_metric if val_metric_mode == "min" else current_metric > best_metric
                if is_better:
                    best_metric = current_metric
                    saved = save_checkpoint(best_checkpoint_path, model, optimizer, config, step, best_metric=best_metric)
                    print(f"saved_best_checkpoint={saved} {val_metric_name}={best_metric:.5f}", flush=True)
            if save_every > 0 and step % save_every == 0:
                periodic_path = output_dir / f"iter_{step:06d}_{checkpoint_name}"
                saved = save_checkpoint(periodic_path, model, optimizer, config, step, best_metric=best_metric)
                print(f"saved_periodic_checkpoint={saved}", flush=True)
            if step >= max_iters:
                break

    save_checkpoint(checkpoint_path, model, optimizer, config, step, best_metric=best_metric)
    print(f"saved_checkpoint={checkpoint_path}", flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeCK-DiffReg on synthetic 3D data.")
    parser.add_argument("--config", type=str, default="configs/neck_diffreg_base.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

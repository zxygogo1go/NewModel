from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from neck_diffreg.training.train import train


def _write_case(root: Path, case_id: str, shape: tuple[int, int, int]) -> None:
    for subdir in ("images", "seg_o", "seg_b", "metadata"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(case_id)) % (2**32))
    image = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
    seg_o = (image > 0.7).astype(np.int16)
    seg_b = np.zeros(shape, dtype=np.int16)
    seg_b[2:6, 2:6, 2:6] = 1
    np.save(root / "images" / f"{case_id}.npy", image)
    np.save(root / "seg_o" / f"{case_id}.npy", seg_o)
    np.save(root / "seg_b" / f"{case_id}.npy", seg_b)
    with (root / "metadata" / f"{case_id}.json").open("w", encoding="utf-8") as f:
        json.dump({"case_id": case_id, "target_shape": list(shape)}, f)


def test_training_loop_saves_latest_and_best_checkpoints(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    shape = (8, 8, 8)
    for idx in range(3):
        _write_case(data_root, f"segrap_000{idx}", shape)
    (data_root / "lists").mkdir(parents=True, exist_ok=True)
    (data_root / "lists" / "trn_list_inter.txt").write_text(
        "segrap_0000\nsegrap_0001\nsegrap_0002\n",
        encoding="utf-8",
    )
    (data_root / "lists" / "val_pairs.csv").write_text(
        "segrap_0000,segrap_0001\n",
        encoding="utf-8",
    )
    config = {
        "seed": 123,
        "model": {
            "image_channels": 1,
            "tissue_channels": 0,
            "base_channels": 2,
            "num_skeleton_nodes": 1,
            "velocity_integration_steps": 2,
            "residual_velocity_scale": 0.05,
        },
        "data": {
            "mode": "segrap",
            "root": str(data_root),
            "split": "train",
            "list_path": str(data_root / "lists" / "trn_list_inter.txt"),
            "expected_shape": list(shape),
            "pairs_per_epoch": 2,
        },
        "val_data": {
            "mode": "segrap",
            "root": str(data_root),
            "split": "val",
            "pairs_path": str(data_root / "lists" / "val_pairs.csv"),
            "expected_shape": list(shape),
        },
        "training": {
            "device": "cpu",
            "batch_size": 1,
            "learning_rate": 1.0e-3,
            "max_iters": 2,
            "log_interval": 1,
            "val_interval": 1,
            "val_max_batches": 1,
            "save_every": 0,
            "output_dir": str(tmp_path / "checkpoints"),
            "checkpoint_name": "latest.pt",
            "best_checkpoint_name": "best.pt",
        },
        "loss": {
            "similarity": "weighted_mse",
            "weights": {
                "similarity": 1.0,
                "smoothness": 0.01,
                "jacobian": 0.0,
                "reliability_tv": 0.0,
                "reliability_sparsity": 0.0,
                "reliability_mean_prior": 0.0,
                "segmentation": 0.01,
                "skeleton": 0.0,
                "uncertainty": 0.0,
            },
        },
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    latest_path = train(config_path)
    assert latest_path.exists()
    assert (tmp_path / "checkpoints" / "best.pt").exists()

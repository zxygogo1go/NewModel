from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neck_diffreg.data.dataset import build_dataloader
from neck_diffreg.data.segrap import (
    SegRapCaseStore,
    SegRapInterSubjectDataset,
    read_val_pairs_inter_list,
)
from neck_diffreg.losses.total_loss import NeckDiffRegLoss
from neck_diffreg.models.neck_diffreg import NeCKDiffReg


def _write_case(root: Path, case_id: str, shape: tuple[int, int, int]) -> None:
    for subdir in ("images", "seg_o", "seg_b", "metadata"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    image = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    seg_o = np.zeros(shape, dtype=np.int16)
    seg_o[1:3, 1:3, 1:3] = 1
    seg_b = np.zeros(shape, dtype=np.int16)
    seg_b[2:4, 2:4, 2:4] = 1
    np.save(root / "images" / f"{case_id}.npy", image)
    np.save(root / "seg_o" / f"{case_id}.npy", seg_o)
    np.save(root / "seg_b" / f"{case_id}.npy", seg_b)
    with (root / "metadata" / f"{case_id}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": case_id,
                "target_shape": list(shape),
                "target_spacing": [2.0, 2.0, 2.0],
                "ct_clip": [-1024.0, 3000.0],
                "label_map": {"mock_oar": 1},
                "bone_structures_used": ["mock_bone"],
            },
            f,
        )


def test_segrap_case_store_loads_expected_files(tmp_path: Path) -> None:
    shape = (6, 7, 8)
    _write_case(tmp_path, "segrap_0000", shape)
    store = SegRapCaseStore(root=tmp_path, expected_shape=shape)
    case = store.load_case("segrap_0000")
    assert case["image"].shape == (1, *shape)
    assert case["image"].dtype.is_floating_point
    assert case["seg_o"].shape == (1, *shape)
    assert case["seg_b"].shape == (1, *shape)
    assert case["metadata"]["case_id"] == "segrap_0000"


def test_train_list_builds_inter_subject_pairs(tmp_path: Path) -> None:
    shape = (6, 7, 8)
    for idx in range(3):
        _write_case(tmp_path, f"segrap_000{idx}", shape)
    (tmp_path / "lists").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lists" / "trn_list_inter.txt").write_text(
        "segrap_0000\nsegrap_0001\nsegrap_0002\n",
        encoding="utf-8",
    )
    dataset = SegRapInterSubjectDataset(
        root=tmp_path,
        split="train",
        expected_shape=shape,
        pairs_per_epoch=4,
    )
    sample = dataset[0]
    assert len(dataset) == 4
    assert sample["moving"].shape == (1, *shape)
    assert sample["fixed"].shape == (1, *shape)
    assert sample["moving_seg"].shape == (1, *shape)
    assert sample["bone_masks"].shape == (1, *shape)
    assert sample["moving_case_id"] != sample["fixed_case_id"]


def test_val_list_inter_matches_dir_musa_pairing(tmp_path: Path) -> None:
    list_path = tmp_path / "val_list_inter.txt"
    list_path.write_text(
        "segrap_0000\nsegrap_0002\nsegrap_0001\nsegrap_0003\n",
        encoding="utf-8",
    )
    assert read_val_pairs_inter_list(list_path) == [
        ("segrap_0000", "segrap_0001"),
        ("segrap_0002", "segrap_0003"),
    ]


def test_segrap_dataloader_collates_case_ids_and_tensors(tmp_path: Path) -> None:
    shape = (6, 7, 8)
    for idx in range(2):
        _write_case(tmp_path, f"segrap_000{idx}", shape)
    (tmp_path / "lists").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lists" / "trn_list_inter.txt").write_text("segrap_0000\nsegrap_0001\n", encoding="utf-8")
    loader = build_dataloader(
        {
            "mode": "segrap",
            "root": str(tmp_path),
            "split": "train",
            "expected_shape": list(shape),
            "pairs_per_epoch": 2,
        },
        batch_size=1,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert batch["moving"].shape == (1, 1, *shape)
    assert batch["fixed"].shape == (1, 1, *shape)
    assert batch["bone_masks"].shape == (1, 1, *shape)
    assert batch["moving_case_id"] == ["segrap_0000"]


def test_segrap_batch_runs_model_loss_backward(tmp_path: Path) -> None:
    shape = (8, 8, 8)
    for idx in range(2):
        _write_case(tmp_path, f"segrap_000{idx}", shape)
    (tmp_path / "lists").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lists" / "trn_list_inter.txt").write_text("segrap_0000\nsegrap_0001\n", encoding="utf-8")
    loader = build_dataloader(
        {
            "mode": "segrap",
            "root": str(tmp_path),
            "split": "train",
            "expected_shape": list(shape),
            "pairs_per_epoch": 1,
        },
        batch_size=1,
        shuffle=False,
    )
    batch = next(iter(loader))
    model = NeCKDiffReg(base_channels=2, num_skeleton_nodes=1, integration_steps=2)
    outputs = model(
        moving=batch["moving"],
        fixed=batch["fixed"],
        moving_seg=batch["moving_seg"],
        fixed_seg=batch["fixed_seg"],
        bone_masks=batch["bone_masks"],
    )
    criterion = NeckDiffRegLoss(weights={"segmentation": 0.1})
    loss, components = criterion(outputs, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(components["reliability_mean"])

"""SegRap/DIR-MUSA preprocessed NPY datasets.

Expected root layout:
    data/
      images/{case_id}.npy
      seg_o/{case_id}.npy
      seg_b/{case_id}.npy
      metadata/{case_id}.json
      lists/trn_list_inter.txt
      lists/val_list_inter.txt
      lists/val_pairs.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


DEFAULT_TARGET_SHAPE = (160, 160, 192)


def read_case_ids(path: str | Path) -> list[str]:
    """Read one case id per non-empty, non-comment line."""

    list_path = Path(path)
    with list_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def read_val_pairs_csv(path: str | Path) -> list[tuple[str, str]]:
    """Read ``moving_case_id,fixed_case_id`` validation pairs."""

    pairs: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"Validation pair row must contain two ids: {row}")
            moving, fixed = row[0].strip(), row[1].strip()
            if moving.lower() == "moving_case_id" and fixed.lower() == "fixed_case_id":
                continue
            pairs.append((moving, fixed))
    return pairs


def read_val_pairs_inter_list(path: str | Path) -> list[tuple[str, str]]:
    """Read DIR-MUSA-style validation list with moving half then fixed half."""

    ids = read_case_ids(path)
    if len(ids) % 2 != 0:
        raise ValueError("DIR-MUSA val_list_inter.txt must contain an even number of case ids")
    half = len(ids) // 2
    return list(zip(ids[:half], ids[half:]))


class SegRapCaseStore:
    """Load and validate preprocessed SegRap/DIR-MUSA case arrays."""

    def __init__(
        self,
        root: str | Path = "data",
        expected_shape: tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
        validate_shapes: bool = True,
        validate_image_range: bool = True,
    ) -> None:
        self.root = Path(root)
        self.expected_shape = tuple(expected_shape)
        self.validate_shapes = validate_shapes
        self.validate_image_range = validate_image_range

    def _case_path(self, subdir: str, case_id: str, suffix: str) -> Path:
        return self.root / subdir / f"{case_id}{suffix}"

    def _load_image(self, case_id: str) -> Tensor:
        path = self._case_path("images", case_id, ".npy")
        image = np.load(path)
        if self.validate_shapes and tuple(image.shape) != self.expected_shape:
            raise ValueError(f"{path} shape {image.shape} != expected {self.expected_shape}")
        image = image.astype(np.float32, copy=False)
        if self.validate_image_range and (float(image.min()) < -1.0e-4 or float(image.max()) > 1.0 + 1.0e-4):
            raise ValueError(f"{path} must be normalized to [0, 1], got [{image.min()}, {image.max()}]")
        return torch.from_numpy(image).unsqueeze(0)

    def _load_seg_o(self, case_id: str) -> Tensor:
        path = self._case_path("seg_o", case_id, ".npy")
        seg = np.load(path)
        if self.validate_shapes and tuple(seg.shape) != self.expected_shape:
            raise ValueError(f"{path} shape {seg.shape} != expected {self.expected_shape}")
        if not np.issubdtype(seg.dtype, np.integer):
            raise ValueError(f"{path} must contain integer labels, got {seg.dtype}")
        return torch.from_numpy(seg.astype(np.int64, copy=False)).unsqueeze(0)

    def _load_seg_b(self, case_id: str) -> Tensor:
        path = self._case_path("seg_b", case_id, ".npy")
        seg = np.load(path)
        if self.validate_shapes and tuple(seg.shape) != self.expected_shape:
            raise ValueError(f"{path} shape {seg.shape} != expected {self.expected_shape}")
        if not np.issubdtype(seg.dtype, np.integer):
            raise ValueError(f"{path} must contain integer binary mask, got {seg.dtype}")
        unique = np.unique(seg)
        if not np.isin(unique, [0, 1]).all():
            raise ValueError(f"{path} must be binary with values {{0, 1}}, got {unique.tolist()}")
        return torch.from_numpy(seg.astype(np.float32, copy=False)).unsqueeze(0)

    def load_metadata(self, case_id: str) -> dict[str, Any]:
        """Load metadata JSON when present, otherwise return an empty dict."""

        path = self._case_path("metadata", case_id, ".json")
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return metadata

    def load_case(self, case_id: str) -> dict[str, Tensor | dict[str, Any] | str]:
        """Load one case according to the project data convention."""

        return {
            "case_id": case_id,
            "image": self._load_image(case_id),
            "seg_o": self._load_seg_o(case_id),
            "seg_b": self._load_seg_b(case_id),
            "metadata": self.load_metadata(case_id),
        }


class SegRapInterSubjectDataset(Dataset):
    """Inter-subject moving/fixed pair dataset for preprocessed cases.

    Training mode reads ``trn_list_inter.txt`` and forms deterministic
    cross-case pairs. Validation mode can read explicit ``val_pairs.csv`` or
    DIR-MUSA ``val_list_inter.txt``.

    Returned tensor shapes:
        moving/fixed: ``[1, D, H, W]`` float32 in ``[0, 1]``
        moving_seg/fixed_seg: ``[1, D, H, W]`` foreground masks from ``seg_o > 0``
        bone_masks: ``[1, D, H, W]`` binary union from ``seg_b``
    """

    def __init__(
        self,
        root: str | Path = "data",
        split: Literal["train", "val"] = "train",
        list_path: str | Path | None = None,
        pairs_path: str | Path | None = None,
        expected_shape: tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
        validate_shapes: bool = True,
        validate_image_range: bool = True,
        pairs_per_epoch: int | None = None,
        pair_stride: int = 1,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.store = SegRapCaseStore(
            root=self.root,
            expected_shape=expected_shape,
            validate_shapes=validate_shapes,
            validate_image_range=validate_image_range,
        )
        self.pair_stride = max(1, int(pair_stride))

        if split == "train":
            ids_path = Path(list_path) if list_path is not None else self.root / "lists" / "trn_list_inter.txt"
            self.case_ids = read_case_ids(ids_path)
            if len(self.case_ids) < 2:
                raise ValueError("Training inter-subject pairing requires at least two case ids")
            total_pairs = pairs_per_epoch if pairs_per_epoch is not None else len(self.case_ids)
            self.pairs = [self._pair_for_index(i) for i in range(int(total_pairs))]
        else:
            if pairs_path is not None:
                self.pairs = read_val_pairs_csv(pairs_path)
            else:
                csv_path = self.root / "lists" / "val_pairs.csv"
                inter_path = self.root / "lists" / "val_list_inter.txt"
                if csv_path.exists():
                    self.pairs = read_val_pairs_csv(csv_path)
                else:
                    self.pairs = read_val_pairs_inter_list(inter_path)
            self.case_ids = sorted({case_id for pair in self.pairs for case_id in pair})

        if not self.pairs:
            raise ValueError("No moving/fixed pairs were found")

    def _pair_for_index(self, index: int) -> tuple[str, str]:
        moving_idx = index % len(self.case_ids)
        fixed_idx = (moving_idx + self.pair_stride + index // len(self.case_ids)) % len(self.case_ids)
        if fixed_idx == moving_idx:
            fixed_idx = (fixed_idx + 1) % len(self.case_ids)
        return self.case_ids[moving_idx], self.case_ids[fixed_idx]

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _seg_foreground(seg_o: Tensor) -> Tensor:
        return (seg_o > 0).float()

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        moving_id, fixed_id = self.pairs[index]
        moving = self.store.load_case(moving_id)
        fixed = self.store.load_case(fixed_id)
        moving_image = moving["image"]
        fixed_image = fixed["image"]
        moving_seg_o = moving["seg_o"]
        fixed_seg_o = fixed["seg_o"]
        moving_bone = moving["seg_b"]
        fixed_bone = fixed["seg_b"]
        assert isinstance(moving_image, Tensor)
        assert isinstance(fixed_image, Tensor)
        assert isinstance(moving_seg_o, Tensor)
        assert isinstance(fixed_seg_o, Tensor)
        assert isinstance(moving_bone, Tensor)
        assert isinstance(fixed_bone, Tensor)
        return {
            "moving": moving_image.float(),
            "fixed": fixed_image.float(),
            "moving_seg": self._seg_foreground(moving_seg_o),
            "fixed_seg": self._seg_foreground(fixed_seg_o),
            "bone_masks": moving_bone.float(),
            "fixed_bone_mask": fixed_bone.float(),
            "moving_case_id": moving_id,
            "fixed_case_id": fixed_id,
        }

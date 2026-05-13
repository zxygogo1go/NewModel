"""Dataset factory helpers."""

from __future__ import annotations

from torch.utils.data import DataLoader

from neck_diffreg.data.segrap import SegRapInterSubjectDataset
from neck_diffreg.data.synthetic import SyntheticNeckDataset


def build_synthetic_dataloader(config: dict, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    """Build a dataloader backed by deterministic synthetic tensors."""

    dataset = SyntheticNeckDataset(
        num_samples=int(config.get("num_samples", 8)),
        spatial_shape=tuple(config.get("spatial_shape", (16, 16, 16))),
        num_blobs=int(config.get("num_blobs", 5)),
        displacement_scale=float(config.get("displacement_scale", 0.06)),
        seed=int(config.get("seed", 1234)),
        include_segmentation=bool(config.get("include_segmentation", True)),
        enable_corruption=bool(config.get("enable_corruption", False)),
        corruption_probability=float(config.get("corruption_probability", 0.5)),
        num_corruptions=int(config.get("num_corruptions", 2)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_segrap_dataloader(config: dict, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    """Build a SegRap/DIR-MUSA dataloader from preprocessed NPY files."""

    split = str(config.get("split", "train"))
    dataset = SegRapInterSubjectDataset(
        root=config.get("root", "data"),
        split="val" if split == "val" else "train",
        list_path=config.get("list_path"),
        pairs_path=config.get("pairs_path"),
        expected_shape=tuple(config.get("expected_shape", (160, 160, 192))),
        validate_shapes=bool(config.get("validate_shapes", True)),
        validate_image_range=bool(config.get("validate_image_range", True)),
        pairs_per_epoch=config.get("pairs_per_epoch"),
        pair_stride=int(config.get("pair_stride", 1)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
    )


def build_dataloader(config: dict, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    """Build a dataloader based on ``data.mode``."""

    mode = str(config.get("mode", "synthetic")).lower()
    if mode in {"segrap", "dir-musa", "dirmusa", "real"}:
        return build_segrap_dataloader(config, batch_size=batch_size, shuffle=shuffle)
    if mode == "synthetic":
        return build_synthetic_dataloader(config, batch_size=batch_size, shuffle=shuffle)
    raise ValueError(f"Unsupported data.mode: {mode}")

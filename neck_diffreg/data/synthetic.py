"""Synthetic 3D data generation for NeCK-DiffReg tests and smoke training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from neck_diffreg.data.transforms import random_smooth_displacement
from neck_diffreg.models.spatial import make_identity_grid, warp_image


@dataclass
class SyntheticConfig:
    num_samples: int = 8
    spatial_shape: tuple[int, int, int] = (16, 16, 16)
    num_blobs: int = 5
    displacement_scale: float = 0.06
    seed: int = 1234
    include_segmentation: bool = True
    enable_corruption: bool = False
    corruption_probability: float = 0.5
    num_corruptions: int = 2


class SyntheticNeckDataset(Dataset[dict[str, Tensor]]):
    """Generate deterministic synthetic moving/fixed 3D registration pairs.

    Each sample contains smooth Gaussian-like blobs. The moving image is
    produced by warping the fixed image with a known smooth random displacement.

    Returned tensors:
        moving/fixed: ``[1, D, H, W]``
        moving_seg/fixed_seg: ``[1, D, H, W]`` when enabled
        target_displacement: ``[3, D, H, W]``
        noncorrespondence_mask: ``[1, D, H, W]`` marking synthetic corruptions
    """

    def __init__(
        self,
        num_samples: int = 8,
        spatial_shape: tuple[int, int, int] = (16, 16, 16),
        num_blobs: int = 5,
        displacement_scale: float = 0.06,
        seed: int = 1234,
        include_segmentation: bool = True,
        enable_corruption: bool = False,
        corruption_probability: float = 0.5,
        num_corruptions: int = 2,
    ) -> None:
        self.config = SyntheticConfig(
            num_samples=num_samples,
            spatial_shape=spatial_shape,
            num_blobs=num_blobs,
            displacement_scale=displacement_scale,
            seed=seed,
            include_segmentation=include_segmentation,
            enable_corruption=enable_corruption,
            corruption_probability=corruption_probability,
            num_corruptions=num_corruptions,
        )

    def __len__(self) -> int:
        return self.config.num_samples

    def _make_fixed(self, generator: torch.Generator) -> Tensor:
        depth, height, width = self.config.spatial_shape
        grid = make_identity_grid((depth, height, width), batch_size=1).permute(0, 4, 1, 2, 3)
        image = torch.zeros(1, 1, depth, height, width)
        for _ in range(self.config.num_blobs):
            center = torch.empty(1, 3, 1, 1, 1).uniform_(-0.65, 0.65, generator=generator)
            radii = torch.empty(1, 3, 1, 1, 1).uniform_(0.16, 0.42, generator=generator)
            amplitude = torch.empty(1, 1, 1, 1, 1).uniform_(0.35, 1.0, generator=generator)
            exponent = -(((grid - center) / radii).pow(2).sum(dim=1, keepdim=True))
            image = image + amplitude * torch.exp(exponent)
        image = image - image.amin(dim=(-3, -2, -1), keepdim=True)
        image = image / image.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0e-6)
        return image

    def _random_ellipsoid_mask(self, generator: torch.Generator) -> Tensor:
        depth, height, width = self.config.spatial_shape
        grid = make_identity_grid((depth, height, width), batch_size=1).permute(0, 4, 1, 2, 3)
        center = torch.empty(1, 3, 1, 1, 1).uniform_(-0.55, 0.55, generator=generator)
        radii = torch.empty(1, 3, 1, 1, 1).uniform_(0.12, 0.32, generator=generator)
        distance = ((grid - center) / radii).pow(2).sum(dim=1, keepdim=True)
        return (distance <= 1.0).float()

    def _apply_corruptions(
        self,
        fixed: Tensor,
        moving: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Inject local non-correspondence into a synthetic pair.

        Corruption types:
            1. air-cavity-like holes: set local moving intensity to air-like 0;
            2. intensity artifacts: add bright/dark local intensity shifts;
            3. missing correspondence: replace a local moving region with
               independent random texture.
        """

        noncorrespondence_mask = torch.zeros_like(moving)
        if not self.config.enable_corruption:
            return fixed, moving, noncorrespondence_mask

        apply = torch.rand((), generator=generator).item() < self.config.corruption_probability
        if not apply:
            return fixed, moving, noncorrespondence_mask

        for _ in range(max(1, self.config.num_corruptions)):
            mask = self._random_ellipsoid_mask(generator)
            kind = int(torch.randint(0, 3, (), generator=generator).item())
            noncorrespondence_mask = torch.maximum(noncorrespondence_mask, mask)
            if kind == 0:
                moving = moving * (1.0 - mask)
            elif kind == 1:
                shift = torch.empty(()).uniform_(-0.7, 0.7, generator=generator).item()
                moving = (moving + shift * mask).clamp(0.0, 1.0)
            else:
                texture = torch.rand(moving.shape, generator=generator, dtype=moving.dtype)
                moving = moving * (1.0 - mask) + texture * mask
        return fixed, moving, noncorrespondence_mask

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(self.config.seed + int(index))
        fixed = self._make_fixed(generator)
        displacement = random_smooth_displacement(
            self.config.spatial_shape,
            scale=self.config.displacement_scale,
            generator=generator,
        )
        moving = warp_image(fixed, displacement)
        fixed, moving, noncorrespondence_mask = self._apply_corruptions(fixed, moving, generator)
        sample = {
            "moving": moving.squeeze(0),
            "fixed": fixed.squeeze(0),
            "target_displacement": displacement.squeeze(0),
            "noncorrespondence_mask": noncorrespondence_mask.squeeze(0),
        }
        if self.config.include_segmentation:
            fixed_seg = (fixed > 0.35).float()
            moving_seg = warp_image(fixed_seg, displacement, mode="nearest")
            sample["fixed_seg"] = fixed_seg.squeeze(0)
            sample["moving_seg"] = moving_seg.squeeze(0)
        return sample

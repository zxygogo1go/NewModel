"""Voxel-wise deformation uncertainty head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neck_diffreg.models.layers import SmallUNet3d


class UncertaintyHead(nn.Module):
    """Predict log variance for the deformation field.

    Inputs are fixed image, warped moving image, absolute difference, and
    optional tissue channels. Output shape is ``[B, C, D, H, W]`` where
    ``C`` is usually 3 for per-axis deformation log variance.
    """

    def __init__(
        self,
        image_channels: int = 1,
        tissue_channels: int = 0,
        out_channels: int = 3,
        base_channels: int = 8,
        logvar_min: float = -6.0,
        logvar_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.tissue_channels = tissue_channels
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.net = SmallUNet3d(image_channels * 3 + tissue_channels, out_channels, base_channels=base_channels)
        nn.init.zeros_(self.net.out.weight)
        nn.init.constant_(self.net.out.bias, -2.0)

    def _prepare_tissue(self, reference: Tensor, tissue_map: Tensor | None) -> Tensor:
        batch, _, depth, height, width = reference.shape
        if self.tissue_channels == 0:
            return reference.new_zeros(batch, 0, depth, height, width)
        if tissue_map is None:
            return reference.new_zeros(batch, self.tissue_channels, depth, height, width)
        if tissue_map.shape[-3:] != reference.shape[-3:]:
            tissue_map = F.interpolate(tissue_map, size=reference.shape[-3:], mode="trilinear", align_corners=True)
        if tissue_map.shape[1] == self.tissue_channels:
            return tissue_map
        if tissue_map.shape[1] > self.tissue_channels:
            return tissue_map[:, : self.tissue_channels]
        zeros = tissue_map.new_zeros(batch, self.tissue_channels - tissue_map.shape[1], depth, height, width)
        return torch.cat((tissue_map, zeros), dim=1)

    def forward(self, fixed: Tensor, warped_moving: Tensor, tissue_map: Tensor | None = None) -> Tensor:
        diff = (fixed - warped_moving).abs()
        tissue = self._prepare_tissue(fixed, tissue_map)
        log_variance = self.net(torch.cat((fixed, warped_moving, diff, tissue), dim=1))
        return torch.clamp(log_variance, min=self.logvar_min, max=self.logvar_max)

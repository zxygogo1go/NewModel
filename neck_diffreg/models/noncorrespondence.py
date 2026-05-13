"""Non-correspondence reliability head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neck_diffreg.models.layers import SmallUNet3d


class ReliabilityHead(nn.Module):
    """Predict a voxel-wise correspondence reliability map.

    Inputs are fixed image, currently warped moving image, absolute difference,
    and optional tissue channels. The output is ``R(x)`` in ``[0, 1]`` with
    shape ``[B, 1, D, H, W]``.
    """

    def __init__(self, image_channels: int = 1, tissue_channels: int = 0, base_channels: int = 8) -> None:
        super().__init__()
        self.tissue_channels = tissue_channels
        self.net = SmallUNet3d(image_channels * 3 + tissue_channels, 1, base_channels=base_channels)
        nn.init.zeros_(self.net.out.weight)
        nn.init.constant_(self.net.out.bias, 2.0)

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
        logits = self.net(torch.cat((fixed, warped_moving, diff, tissue), dim=1))
        return torch.sigmoid(logits)

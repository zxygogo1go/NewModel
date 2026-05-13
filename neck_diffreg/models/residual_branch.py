"""Residual diffeomorphic branch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neck_diffreg.models.layers import SmallUNet3d
from neck_diffreg.models.spatial import integrate_velocity_scaling_squaring


class ResidualDiffeomorphicBranch(nn.Module):
    """Predict a stationary velocity field and integrate it.

    Inputs:
        fixed: fixed image ``[B, 1, D, H, W]``
        warped_moving: skeleton-warped moving image ``[B, 1, D, H, W]``
        tissue_map: optional tissue channels ``[B, K, D, H, W]``
        reliability_prior: optional prior map ``[B, 1, D, H, W]``

    Returns:
        Dictionary with ``velocity`` and integrated ``displacement``, both
        shaped ``[B, 3, D, H, W]``.
    """

    def __init__(
        self,
        image_channels: int = 1,
        tissue_channels: int = 0,
        base_channels: int = 8,
        integration_steps: int = 5,
        velocity_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.tissue_channels = tissue_channels
        self.integration_steps = integration_steps
        self.velocity_scale = velocity_scale
        in_channels = image_channels * 2 + tissue_channels + 1
        self.unet = SmallUNet3d(in_channels, 3, base_channels=base_channels)
        nn.init.zeros_(self.unet.out.weight)
        nn.init.zeros_(self.unet.out.bias)

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
        pad = self.tissue_channels - tissue_map.shape[1]
        zeros = tissue_map.new_zeros(batch, pad, depth, height, width)
        return torch.cat((tissue_map, zeros), dim=1)

    @staticmethod
    def _prepare_reliability_prior(reference: Tensor, reliability_prior: Tensor | None) -> Tensor:
        batch, _, depth, height, width = reference.shape
        if reliability_prior is None:
            return reference.new_zeros(batch, 1, depth, height, width)
        if reliability_prior.shape[-3:] != reference.shape[-3:]:
            reliability_prior = F.interpolate(
                reliability_prior.float(),
                size=reference.shape[-3:],
                mode="trilinear",
                align_corners=True,
            )
        if reliability_prior.shape[1] != 1:
            reliability_prior = reliability_prior[:, :1]
        return reliability_prior.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)

    def forward(
        self,
        fixed: Tensor,
        warped_moving: Tensor,
        tissue_map: Tensor | None = None,
        reliability_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Estimate residual soft-tissue diffeomorphic displacement."""

        tissue = self._prepare_tissue(fixed, tissue_map)
        reliability = self._prepare_reliability_prior(fixed, reliability_prior)
        x = torch.cat((fixed, warped_moving, tissue, reliability), dim=1)
        velocity = torch.tanh(self.unet(x)) * self.velocity_scale
        displacement = integrate_velocity_scaling_squaring(velocity, n_steps=self.integration_steps)
        return {"velocity": velocity, "displacement": displacement}

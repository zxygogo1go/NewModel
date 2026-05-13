"""Reusable 3D network layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock3d(nn.Module):
    """Two convolutional layers preserving spatial shape.

    Input shape:
        ``[B, C_in, D, H, W]``
    Output shape:
        ``[B, C_out, D, H, W]``
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = max(1, min(8, out_channels))
        while out_channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SmallUNet3d(nn.Module):
    """Small 3D UNet for prototype registration heads.

    Input shape:
        ``[B, C_in, D, H, W]``
    Output shape:
        ``[B, C_out, D, H, W]``
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 8) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.enc1 = ConvBlock3d(in_channels, c1)
        self.enc2 = ConvBlock3d(c1, c2)
        self.enc3 = ConvBlock3d(c2, c3)
        self.dec2 = ConvBlock3d(c3 + c2, c2)
        self.dec1 = ConvBlock3d(c2 + c1, c1)
        self.out = nn.Conv3d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, kernel_size=2, stride=2))
        e3 = self.enc3(F.avg_pool3d(e2, kernel_size=2, stride=2))
        d2 = F.interpolate(e3, size=e2.shape[-3:], mode="trilinear", align_corners=True)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=True)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        return self.out(d1)

"""Synthetic spatial transforms."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def random_smooth_displacement(
    spatial_shape: tuple[int, int, int],
    scale: float = 0.08,
    control_point_shape: tuple[int, int, int] = (4, 4, 4),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a random smooth normalized displacement field.

    Args:
        spatial_shape: ``(D, H, W)``.
        scale: Maximum approximate normalized displacement magnitude.
        control_point_shape: Low-resolution noise grid shape.
        generator: Optional deterministic generator.

    Returns:
        Tensor ``[1, 3, D, H, W]``.
    """

    noise = torch.randn(
        1,
        3,
        *control_point_shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    disp = F.interpolate(noise, size=spatial_shape, mode="trilinear", align_corners=True)
    rms = disp.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1.0e-6)
    return disp / rms * scale

"""Spatial utilities for 3D deformable registration.

Coordinate convention:
    Dense displacement fields have shape ``[B, 3, D, H, W]`` and use
    normalized ``grid_sample`` coordinates in channel order ``x, y, z``.
    ``warp_image(image, disp)`` returns ``image`` sampled at
    ``identity_grid + disp``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def make_identity_grid(
    spatial_shape: Sequence[int],
    batch_size: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a normalized identity grid for ``grid_sample``.

    Args:
        spatial_shape: Either ``(D, H, W)`` or an image-like shape ending in
            ``(D, H, W)``.
        batch_size: Number of grid batches to create.
        device: Target device.
        dtype: Target floating point dtype.

    Returns:
        Tensor with shape ``[B, D, H, W, 3]`` and last-channel order
        ``x, y, z``.
    """

    if len(spatial_shape) < 3:
        raise ValueError("spatial_shape must contain at least D, H, W")
    depth, height, width = [int(v) for v in spatial_shape[-3:]]

    z = torch.linspace(-1.0, 1.0, depth, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1)
    return grid.unsqueeze(0).expand(batch_size, depth, height, width, 3)


def displacement_to_grid(displacement: Tensor) -> Tensor:
    """Convert a normalized displacement field to a sampling grid.

    Args:
        displacement: Tensor ``[B, 3, D, H, W]`` in normalized coordinates.

    Returns:
        Grid tensor ``[B, D, H, W, 3]``.
    """

    if displacement.ndim != 5 or displacement.shape[1] != 3:
        raise ValueError("displacement must have shape [B, 3, D, H, W]")
    batch = displacement.shape[0]
    grid = make_identity_grid(
        displacement.shape[-3:],
        batch_size=batch,
        device=displacement.device,
        dtype=displacement.dtype,
    )
    return grid + displacement.permute(0, 2, 3, 4, 1)


def warp_image(
    image: Tensor,
    displacement_or_grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
) -> Tensor:
    """Warp a 3D image or field with ``torch.nn.functional.grid_sample``.

    Args:
        image: Tensor ``[B, C, D, H, W]``.
        displacement_or_grid: Either a displacement ``[B, 3, D, H, W]`` or a
            sampling grid ``[B, D, H, W, 3]``.
        mode: ``"bilinear"`` for images/fields or ``"nearest"`` for labels.
        padding_mode: Passed to ``grid_sample``.
        align_corners: Passed to ``grid_sample``.

    Returns:
        Warped tensor with shape ``[B, C, D, H, W]``.
    """

    if image.ndim != 5:
        raise ValueError("image must have shape [B, C, D, H, W]")
    if displacement_or_grid.ndim == 5 and displacement_or_grid.shape[1] == 3:
        grid = displacement_to_grid(displacement_or_grid)
    elif displacement_or_grid.ndim == 5 and displacement_or_grid.shape[-1] == 3:
        grid = displacement_or_grid
    else:
        raise ValueError("displacement_or_grid must be [B, 3, D, H, W] or [B, D, H, W, 3]")
    return F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def compose_displacements(disp_a: Tensor, disp_b: Tensor) -> Tensor:
    """Compose two normalized displacement fields.

    The returned field applies ``disp_a`` first and then ``disp_b``:
    ``phi_total(x) = phi_b(phi_a(x))``. With displacement fields this is
    ``disp_a(x) + disp_b(x + disp_a(x))``.

    Args:
        disp_a: Tensor ``[B, 3, D, H, W]``.
        disp_b: Tensor ``[B, 3, D, H, W]``.

    Returns:
        Composed displacement ``[B, 3, D, H, W]``.
    """

    if disp_a.shape != disp_b.shape:
        raise ValueError("disp_a and disp_b must have matching shapes")
    warped_b = warp_image(disp_b, disp_a, mode="bilinear", padding_mode="border")
    return disp_a + warped_b


def integrate_velocity_scaling_squaring(velocity: Tensor, n_steps: int = 5) -> Tensor:
    """Integrate a stationary velocity field by scaling and squaring.

    Args:
        velocity: Normalized stationary velocity field ``[B, 3, D, H, W]``.
        n_steps: Number of scaling and squaring steps.

    Returns:
        Approximate diffeomorphic displacement ``[B, 3, D, H, W]``.
    """

    if velocity.ndim != 5 or velocity.shape[1] != 3:
        raise ValueError("velocity must have shape [B, 3, D, H, W]")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    displacement = velocity / float(2**n_steps)
    for _ in range(n_steps):
        displacement = compose_displacements(displacement, displacement)
    return displacement


def _finite_difference(tensor: Tensor, dim: int, spacing: float) -> Tensor:
    """Finite difference derivative along one spatial dimension."""

    if tensor.shape[dim] <= 1:
        return torch.zeros_like(tensor)

    out = torch.zeros_like(tensor)
    center = [slice(None)] * tensor.ndim
    before = [slice(None)] * tensor.ndim
    after = [slice(None)] * tensor.ndim
    center[dim] = slice(1, -1)
    before[dim] = slice(0, -2)
    after[dim] = slice(2, None)
    out[tuple(center)] = (tensor[tuple(after)] - tensor[tuple(before)]) / (2.0 * spacing)

    first = [slice(None)] * tensor.ndim
    first_next = [slice(None)] * tensor.ndim
    first[dim] = 0
    first_next[dim] = 1
    out[tuple(first)] = (tensor[tuple(first_next)] - tensor[tuple(first)]) / spacing

    last = [slice(None)] * tensor.ndim
    last_prev = [slice(None)] * tensor.ndim
    last[dim] = -1
    last_prev[dim] = -2
    out[tuple(last)] = (tensor[tuple(last)] - tensor[tuple(last_prev)]) / spacing
    return out


def displacement_gradient(displacement: Tensor) -> Tensor:
    """Compute gradients of a normalized displacement field.

    Args:
        displacement: Tensor ``[B, 3, D, H, W]`` with channels ``x, y, z``.

    Returns:
        Tensor ``[B, 3, 3, D, H, W]`` where ``grad[:, i, j]`` is the
        derivative of displacement channel ``i`` with respect to coordinate
        ``j`` in order ``x, y, z``.
    """

    if displacement.ndim != 5 or displacement.shape[1] != 3:
        raise ValueError("displacement must have shape [B, 3, D, H, W]")
    depth, height, width = displacement.shape[-3:]
    spacing_x = 2.0 / max(width - 1, 1)
    spacing_y = 2.0 / max(height - 1, 1)
    spacing_z = 2.0 / max(depth - 1, 1)
    grad_x = _finite_difference(displacement, dim=4, spacing=spacing_x)
    grad_y = _finite_difference(displacement, dim=3, spacing=spacing_y)
    grad_z = _finite_difference(displacement, dim=2, spacing=spacing_z)
    return torch.stack((grad_x, grad_y, grad_z), dim=2)


def jacobian_determinant(displacement: Tensor) -> Tensor:
    """Approximate determinant of ``d(identity + displacement) / dx``.

    Args:
        displacement: Normalized displacement ``[B, 3, D, H, W]``.

    Returns:
        Jacobian determinant map ``[B, 1, D, H, W]``. This finite-difference
        implementation is intended for diagnostics and folding penalties in the
        prototype, not for high-precision numerical analysis.
    """

    grad = displacement_gradient(displacement)
    batch, _, _, depth, height, width = grad.shape
    eye = torch.eye(3, device=displacement.device, dtype=displacement.dtype)
    jac = grad.permute(0, 3, 4, 5, 1, 2).contiguous()
    jac = jac + eye.view(1, 1, 1, 1, 3, 3)
    det = torch.linalg.det(jac.view(-1, 3, 3))
    return det.view(batch, depth, height, width).unsqueeze(1)

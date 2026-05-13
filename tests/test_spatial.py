from __future__ import annotations

import torch

from neck_diffreg.losses.regularization import jacobian_folding_penalty
from neck_diffreg.models.spatial import (
    integrate_velocity_scaling_squaring,
    jacobian_determinant,
    make_identity_grid,
    warp_image,
)


def test_identity_warp_returns_same_image() -> None:
    torch.manual_seed(1)
    image = torch.rand(1, 1, 8, 9, 10)
    zero = torch.zeros(1, 3, 8, 9, 10)
    warped = warp_image(image, zero)
    assert torch.allclose(warped, image, atol=1.0e-5)


def test_zero_velocity_integration_shape_and_value() -> None:
    velocity = torch.zeros(2, 3, 8, 8, 8)
    displacement = integrate_velocity_scaling_squaring(velocity, n_steps=3)
    assert displacement.shape == velocity.shape
    assert torch.allclose(displacement, velocity)


def test_identity_jacobian_is_one() -> None:
    displacement = torch.zeros(1, 3, 8, 8, 8)
    det_j = jacobian_determinant(displacement)
    assert det_j.shape == (1, 1, 8, 8, 8)
    assert torch.allclose(det_j, torch.ones_like(det_j), atol=1.0e-5)


def test_jacobian_folding_penalty_detects_reflection() -> None:
    grid = make_identity_grid((8, 8, 8), batch_size=1).permute(0, 4, 1, 2, 3)
    displacement = torch.zeros_like(grid)
    displacement[:, 0] = -2.2 * grid[:, 0]
    identity_penalty = jacobian_folding_penalty(torch.zeros_like(displacement), margin=0.05)
    folding_penalty = jacobian_folding_penalty(displacement, margin=0.05)
    assert torch.isfinite(folding_penalty)
    assert identity_penalty < 1.0e-6
    assert folding_penalty > identity_penalty

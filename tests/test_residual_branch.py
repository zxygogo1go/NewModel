from __future__ import annotations

import torch

from neck_diffreg.losses.regularization import tissue_aware_smoothness_loss
from neck_diffreg.models.residual_branch import ResidualDiffeomorphicBranch


def test_residual_branch_velocity_integration_output_shape() -> None:
    torch.manual_seed(21)
    fixed = torch.rand(2, 1, 8, 8, 8)
    warped_moving = torch.rand(2, 1, 8, 8, 8)
    tissue_map = torch.rand(2, 2, 8, 8, 8)
    reliability_prior = torch.rand(2, 1, 8, 8, 8)
    branch = ResidualDiffeomorphicBranch(
        image_channels=1,
        tissue_channels=2,
        base_channels=2,
        integration_steps=2,
        velocity_scale=0.1,
    )
    outputs = branch(
        fixed=fixed,
        warped_moving=warped_moving,
        tissue_map=tissue_map,
        reliability_prior=reliability_prior,
    )
    assert outputs["velocity"].shape == (2, 3, 8, 8, 8)
    assert outputs["displacement"].shape == (2, 3, 8, 8, 8)
    assert torch.isfinite(outputs["velocity"]).all()
    assert torch.isfinite(outputs["displacement"]).all()


def test_zero_velocity_gives_zero_residual_displacement() -> None:
    fixed = torch.rand(1, 1, 8, 8, 8)
    warped_moving = torch.rand(1, 1, 8, 8, 8)
    branch = ResidualDiffeomorphicBranch(base_channels=2, integration_steps=3)
    outputs = branch(fixed=fixed, warped_moving=warped_moving)
    assert torch.allclose(outputs["velocity"], torch.zeros_like(outputs["velocity"]), atol=1.0e-6)
    assert torch.allclose(outputs["displacement"], torch.zeros_like(outputs["displacement"]), atol=1.0e-6)


def test_tissue_aware_smoothness_loss_returns_finite_scalar() -> None:
    torch.manual_seed(22)
    velocity = torch.rand(2, 3, 8, 8, 8) * 0.01
    tissue_map = torch.rand(2, 3, 8, 8, 8)
    loss = tissue_aware_smoothness_loss(
        velocity,
        tissue_map=tissue_map,
        channel_weights=[2.0, 1.0, 0.5],
        penalty="l2",
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)

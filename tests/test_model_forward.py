from __future__ import annotations

import torch

from neck_diffreg.models.neck_diffreg import NeCKDiffReg


def _synthetic_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(2)
    moving = torch.rand(1, 1, 8, 8, 8)
    fixed = torch.rand(1, 1, 8, 8, 8)
    return moving, fixed


def test_forward_pass_shapes_and_finite_values() -> None:
    moving, fixed = _synthetic_pair()
    tissue_map = torch.rand(1, 2, 8, 8, 8)
    reliability_prior = torch.rand(1, 1, 8, 8, 8)
    model = NeCKDiffReg(base_channels=2, tissue_channels=2, integration_steps=2, residual_velocity_scale=0.1)
    outputs = model(moving, fixed, tissue_map=tissue_map, reliability_prior=reliability_prior)
    expected = {
        "warped_moving",
        "phi_skeleton",
        "phi_soft",
        "phi_total",
        "velocity",
        "reliability",
        "uncertainty",
        "aux",
    }
    assert expected.issubset(outputs.keys())
    assert outputs["warped_moving"].shape == moving.shape
    assert outputs["phi_total"].shape == (1, 3, 8, 8, 8)
    assert outputs["reliability"].shape == moving.shape
    assert outputs["uncertainty"].shape == (1, 3, 8, 8, 8)
    assert outputs["aux"]["skeleton_rotation_vector"].shape == (1, 10, 3)
    assert outputs["aux"]["skeleton_translation"].shape == (1, 10, 3)
    for key in ("warped_moving", "phi_total", "velocity", "reliability", "uncertainty"):
        assert torch.isfinite(outputs[key]).all()
    assert outputs["reliability"].amin() >= 0
    assert outputs["reliability"].amax() <= 1


def test_tiny_backward_pass() -> None:
    moving, fixed = _synthetic_pair()
    model = NeCKDiffReg(base_channels=2, integration_steps=2)
    outputs = model(moving, fixed)
    loss = (outputs["warped_moving"] - fixed).pow(2).mean() + outputs["phi_total"].pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)

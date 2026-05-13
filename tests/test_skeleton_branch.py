from __future__ import annotations

import torch

from neck_diffreg.models.skeleton_branch import (
    NeuroKinematicSkeletonBranch,
    rotation_vector_to_matrix,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    moving = torch.rand(2, 1, 8, 8, 8)
    fixed = torch.rand(2, 1, 8, 8, 8)
    return moving, fixed


def test_skeleton_branch_shape_correctness_with_masks() -> None:
    moving, fixed = _inputs()
    masks = torch.zeros(2, 4, 8, 8, 8)
    masks[:, 0, :, :, :4] = 1.0
    masks[:, 1, :, :, 4:] = 1.0
    branch = NeuroKinematicSkeletonBranch(num_nodes=4, hidden_channels=8)
    displacement, aux = branch(moving, fixed, bone_masks=masks)
    assert displacement.shape == (2, 3, 8, 8, 8)
    assert aux["rotation_vector"].shape == (2, 4, 3)
    assert aux["translation"].shape == (2, 4, 3)
    assert aux["rotation_matrix"].shape == (2, 4, 3, 3)
    assert aux["node_displacements"].shape == (2, 4, 3, 8, 8, 8)
    assert aux["blend_weights"].shape == (2, 4, 8, 8, 8)
    assert torch.isfinite(displacement).all()


def test_zero_transform_gives_near_zero_displacement() -> None:
    moving, fixed = _inputs()
    masks = torch.ones(2, 4, 8, 8, 8)
    branch = NeuroKinematicSkeletonBranch(num_nodes=4, hidden_channels=8, zero_init=True)
    displacement, aux = branch(moving, fixed, bone_masks=masks)
    assert torch.allclose(displacement, torch.zeros_like(displacement), atol=1.0e-6)
    assert torch.allclose(aux["rotation_vector"], torch.zeros_like(aux["rotation_vector"]), atol=1.0e-6)
    assert torch.allclose(aux["translation"], torch.zeros_like(aux["translation"]), atol=1.0e-6)


def test_mask_normalization_sums_safely_for_empty_voxels() -> None:
    branch = NeuroKinematicSkeletonBranch(num_nodes=4, hidden_channels=8)
    masks = torch.zeros(1, 2, 5, 6, 7)
    masks[:, 0, :, :, :3] = 2.0
    weights = branch.normalize_bone_masks(masks, (5, 6, 7))
    assert weights.shape == (1, 4, 5, 6, 7)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(1, 5, 6, 7), atol=1.0e-6)
    assert torch.all(weights >= 0)


def test_rotation_vector_to_matrix_is_stable_at_zero() -> None:
    rotation_vector = torch.zeros(2, 4, 3, requires_grad=True)
    matrix = rotation_vector_to_matrix(rotation_vector)
    eye = torch.eye(3).view(1, 1, 3, 3)
    assert torch.allclose(matrix, eye.expand_as(matrix), atol=1.0e-6)
    matrix.sum().backward()
    assert rotation_vector.grad is not None
    assert torch.isfinite(rotation_vector.grad).all()


def test_skeleton_forward_backward_has_no_nans() -> None:
    moving, fixed = _inputs()
    masks = torch.rand(2, 4, 8, 8, 8)
    branch = NeuroKinematicSkeletonBranch(num_nodes=4, hidden_channels=8, zero_init=False)
    displacement, aux = branch(moving, fixed, bone_masks=masks)
    loss = displacement.pow(2).mean() + aux["rotation_matrix"].pow(2).mean()
    loss.backward()
    grads = [p.grad for p in branch.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert torch.isfinite(displacement).all()
    assert all(torch.isfinite(grad).all() for grad in grads)

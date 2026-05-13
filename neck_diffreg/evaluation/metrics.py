"""Prototype evaluation metrics."""

from __future__ import annotations

import torch
from torch import Tensor

from neck_diffreg.losses.anatomy import dice_loss
from neck_diffreg.models.spatial import compose_displacements, jacobian_determinant


def dice_score(prediction: Tensor, target: Tensor) -> Tensor:
    """Return soft Dice score for segmentation tensors."""

    return 1.0 - dice_loss(prediction, target)


def folding_rate(displacement: Tensor) -> Tensor:
    """Fraction of voxels with non-positive Jacobian determinant."""

    det_j = jacobian_determinant(displacement)
    return (det_j <= 0).float().mean()


def displacement_magnitude(displacement: Tensor) -> Tensor:
    """Mean vector magnitude of a displacement field."""

    return displacement.pow(2).sum(dim=1).sqrt().mean()


def inverse_consistency_error(forward: Tensor, backward: Tensor) -> Tensor:
    """Approximate inverse consistency error for forward/backward fields."""

    return compose_displacements(forward, backward).pow(2).sum(dim=1).sqrt().mean()


def bone_rigidity_error(displacement: Tensor, bone_masks: Tensor | None = None) -> Tensor:
    """Placeholder bone rigidity metric.

    A future version should measure within-bone deviation from a best-fit rigid
    transform. The current prototype reports zero when no masks are supplied.
    """

    if bone_masks is None:
        return displacement.new_zeros(())
    masked = displacement.unsqueeze(1) * bone_masks.unsqueeze(2)
    return masked.var(dim=(-3, -2, -1)).mean()

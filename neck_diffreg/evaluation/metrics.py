"""Prototype evaluation metrics."""

from __future__ import annotations

import torch
from torch import Tensor

from neck_diffreg.losses.anatomy import dice_loss
from neck_diffreg.models.spatial import compose_displacements, jacobian_determinant, warp_image


def mse_metric(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error for image tensors."""

    return (prediction - target).pow(2).mean()


def dice_score(prediction: Tensor, target: Tensor) -> Tensor:
    """Return soft Dice score for segmentation tensors."""

    return 1.0 - dice_loss(prediction, target)


def folding_rate(displacement: Tensor) -> Tensor:
    """Fraction of voxels with non-positive Jacobian determinant."""

    det_j = jacobian_determinant(displacement)
    return (det_j <= 0).float().mean()


def jacobian_det_mean(displacement: Tensor) -> Tensor:
    """Mean Jacobian determinant."""

    return jacobian_determinant(displacement).mean()


def jacobian_det_min(displacement: Tensor) -> Tensor:
    """Minimum Jacobian determinant."""

    return jacobian_determinant(displacement).amin()


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


def registration_metrics(outputs: dict, batch: dict) -> dict[str, Tensor]:
    """Compute validation metrics for one registration batch."""

    metrics = {
        "mse": mse_metric(outputs["warped_moving"], batch["fixed"]),
        "folding_rate": folding_rate(outputs["phi_total"]),
        "jacobian_mean": jacobian_det_mean(outputs["phi_total"]),
        "jacobian_min": jacobian_det_min(outputs["phi_total"]),
        "disp_mean": displacement_magnitude(outputs["phi_total"]),
        "reliability_mean": outputs["reliability"].mean(),
    }
    if "moving_seg" in batch and "fixed_seg" in batch:
        warped_seg = warp_image(batch["moving_seg"].float(), outputs["phi_total"], mode="bilinear")
        metrics["dice"] = dice_score(warped_seg, batch["fixed_seg"].float())
    if "bone_masks" in batch:
        metrics["bone_rigidity"] = bone_rigidity_error(outputs["phi_total"], batch["bone_masks"].float())
    return metrics

"""Deformation and reliability regularization losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from neck_diffreg.models.spatial import jacobian_determinant


def gradient_smoothness_loss(field: Tensor, penalty: str = "l2") -> Tensor:
    """Penalize spatial gradients of a displacement or velocity field.

    Args:
        field: Tensor ``[B, C, D, H, W]``.
        penalty: ``"l1"`` or ``"l2"``.

    Returns:
        Scalar smoothness loss.
    """

    diffs = [
        field[..., 1:, :, :] - field[..., :-1, :, :],
        field[..., :, 1:, :] - field[..., :, :-1, :],
        field[..., :, :, 1:] - field[..., :, :, :-1],
    ]
    losses = []
    for diff in diffs:
        if penalty == "l1":
            losses.append(diff.abs().mean())
        elif penalty == "l2":
            losses.append(diff.pow(2).mean())
        else:
            raise ValueError("penalty must be 'l1' or 'l2'")
    return sum(losses) / len(losses)


def _penalize_difference(diff: Tensor, penalty: str) -> Tensor:
    if penalty == "l1":
        return diff.abs()
    if penalty == "l2":
        return diff.pow(2)
    raise ValueError("penalty must be 'l1' or 'l2'")


def tissue_aware_smoothness_loss(
    field: Tensor,
    tissue_map: Tensor | None = None,
    channel_weights: list[float] | tuple[float, ...] | Tensor | None = None,
    penalty: str = "l2",
    eps: float = 1.0e-6,
) -> Tensor:
    """Penalize spatial gradients with tissue-dependent voxel weights.

    Args:
        field: Displacement or velocity tensor ``[B, C, D, H, W]``.
        tissue_map: Optional tissue tensor ``[B, K, D, H, W]``. Typical
            channels might represent bone, soft tissue, air/tumor/artifact, and
            external body. All-zero tissue voxels fall back to unit weight.
        channel_weights: Optional per-tissue scalar weights. If omitted, every
            tissue channel receives weight 1.
        penalty: ``"l1"`` or ``"l2"``.
        eps: Numerical stabilizer for weighted means.

    Returns:
        Finite scalar smoothness loss. If ``tissue_map`` is absent, this is
        identical to ordinary ``gradient_smoothness_loss``.
    """

    if tissue_map is None:
        return gradient_smoothness_loss(field, penalty=penalty)

    tissue = tissue_map.to(device=field.device, dtype=field.dtype)
    if tissue.shape[-3:] != field.shape[-3:]:
        tissue = F.interpolate(tissue, size=field.shape[-3:], mode="trilinear", align_corners=True)

    channels = tissue.shape[1]
    if channel_weights is None:
        weights = torch.ones(channels, device=field.device, dtype=field.dtype)
    else:
        weights = torch.as_tensor(channel_weights, device=field.device, dtype=field.dtype)
        if weights.numel() < channels:
            pad = torch.ones(channels - weights.numel(), device=field.device, dtype=field.dtype)
            weights = torch.cat((weights.flatten(), pad), dim=0)
        weights = weights.flatten()[:channels]

    tissue = tissue.clamp_min(0.0)
    support = tissue.sum(dim=1, keepdim=True)
    weighted = (tissue * weights.view(1, channels, 1, 1, 1)).sum(dim=1, keepdim=True)
    voxel_weight = torch.where(support > eps, weighted / support.clamp_min(eps), torch.ones_like(weighted))
    voxel_weight = voxel_weight.clamp_min(0.0)

    diffs_and_weights = [
        (
            field[..., 1:, :, :] - field[..., :-1, :, :],
            0.5 * (voxel_weight[..., 1:, :, :] + voxel_weight[..., :-1, :, :]),
        ),
        (
            field[..., :, 1:, :] - field[..., :, :-1, :],
            0.5 * (voxel_weight[..., :, 1:, :] + voxel_weight[..., :, :-1, :]),
        ),
        (
            field[..., :, :, 1:] - field[..., :, :, :-1],
            0.5 * (voxel_weight[..., :, :, 1:] + voxel_weight[..., :, :, :-1]),
        ),
    ]
    losses = []
    for diff, weight in diffs_and_weights:
        error = _penalize_difference(diff, penalty=penalty)
        losses.append((error * weight).mean() / weight.mean().clamp_min(eps))
    return sum(losses) / len(losses)


def total_variation_loss(field: Tensor) -> Tensor:
    """Anisotropic total variation loss for ``[B, C, D, H, W]`` tensors."""

    return gradient_smoothness_loss(field, penalty="l1")


def jacobian_folding_penalty(
    displacement: Tensor,
    margin: float = 0.0,
    penalty: str = "squared",
) -> Tensor:
    """Penalize small or negative Jacobian determinants.

    Args:
        displacement: Dense displacement field ``[B, 3, D, H, W]``.
        margin: Lower determinant margin. ``0`` penalizes only folding, while
            values such as ``0.05`` start discouraging near-singular local
            volume changes before actual folding appears.
        penalty: ``"l1"`` for linear hinge or ``"squared"`` for a stronger
            hinge penalty.

    Returns:
        Scalar anti-folding penalty.
    """

    det_j = jacobian_determinant(displacement)
    violation = torch.relu(displacement.new_tensor(float(margin)) - det_j)
    if penalty == "l1":
        return violation.mean()
    if penalty == "squared":
        return violation.pow(2).mean()
    raise ValueError("penalty must be 'l1' or 'squared'")


def reliability_sparsity_loss(reliability: Tensor) -> Tensor:
    """Encourage reliability to stay high unless evidence suggests otherwise."""

    return (1.0 - reliability).mean()


def mean_reliability_prior_loss(reliability: Tensor, target_mean: float = 0.75) -> Tensor:
    """Penalize collapse to globally low reliability.

    Args:
        reliability: Map ``R`` with shape ``[B, 1, D, H, W]``.
        target_mean: Soft lower bound for the average reliability.

    Returns:
        Scalar ``ReLU(target_mean - mean(R))^2``.
    """

    target = reliability.new_tensor(float(target_mean))
    return torch.relu(target - reliability.mean()).pow(2)

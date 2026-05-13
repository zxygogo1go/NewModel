"""Anatomy-aware losses and segmentation helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def dice_loss(prediction: Tensor, target: Tensor, eps: float = 1.0e-6) -> Tensor:
    """Soft Dice loss for segmentation tensors.

    Args:
        prediction: Tensor ``[B, C, D, H, W]``.
        target: Tensor ``[B, C, D, H, W]``.

    Returns:
        Scalar ``1 - mean Dice``.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    dims = tuple(range(2, prediction.ndim))
    intersection = (prediction * target).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def skeleton_parameter_penalty(node_params: Tensor) -> Tensor:
    """Small placeholder penalty on excessive skeleton transform parameters."""

    return node_params.pow(2).mean()

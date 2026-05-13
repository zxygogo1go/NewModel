"""Image similarity losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def mse_error_map(prediction: Tensor, target: Tensor) -> Tensor:
    """Return voxel-wise squared error ``[B, C, D, H, W]``."""

    return (prediction - target).pow(2)


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error scalar for image tensors ``[B, C, D, H, W]``."""

    return mse_error_map(prediction, target).mean()


def reliability_weighted_mse(
    prediction: Tensor,
    target: Tensor,
    reliability: Tensor,
    eps: float = 1.0e-6,
) -> Tensor:
    """MSE weighted by correspondence reliability.

    Args:
        prediction: Warped moving image ``[B, C, D, H, W]``.
        target: Fixed image ``[B, C, D, H, W]``.
        reliability: Reliability map ``[B, 1, D, H, W]`` in ``[0, 1]``.

    Returns:
        Finite scalar ``sum(R * error) / (sum(R) + eps)``. This intentionally
        separates correspondence weighting from reliability regularization;
        the latter is handled by non-correspondence losses.
    """

    error = mse_error_map(prediction, target)
    weighted = reliability * error
    return weighted.sum() / reliability.sum().clamp_min(eps)


def local_ncc_loss(prediction: Tensor, target: Tensor, window_size: int = 5, eps: float = 1.0e-5) -> Tensor:
    """Local normalized cross-correlation loss for 3D images.

    Args:
        prediction: Tensor ``[B, C, D, H, W]``.
        target: Tensor ``[B, C, D, H, W]``.
        window_size: Odd local window side length.

    Returns:
        Scalar ``1 - NCC`` averaged over local windows.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    channels = prediction.shape[1]
    kernel = torch.ones(
        channels,
        1,
        window_size,
        window_size,
        window_size,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    padding = window_size // 2
    stride = 1
    win_volume = float(window_size**3)

    pred_sum = F.conv3d(prediction, kernel, padding=padding, stride=stride, groups=channels)
    tgt_sum = F.conv3d(target, kernel, padding=padding, stride=stride, groups=channels)
    pred2_sum = F.conv3d(prediction * prediction, kernel, padding=padding, stride=stride, groups=channels)
    tgt2_sum = F.conv3d(target * target, kernel, padding=padding, stride=stride, groups=channels)
    cross_sum = F.conv3d(prediction * target, kernel, padding=padding, stride=stride, groups=channels)

    pred_mean = pred_sum / win_volume
    tgt_mean = tgt_sum / win_volume
    cross = cross_sum - tgt_mean * pred_sum - pred_mean * tgt_sum + pred_mean * tgt_mean * win_volume
    pred_var = pred2_sum - 2.0 * pred_mean * pred_sum + pred_mean * pred_mean * win_volume
    tgt_var = tgt2_sum - 2.0 * tgt_mean * tgt_sum + tgt_mean * tgt_mean * win_volume
    ncc = cross.pow(2) / (pred_var * tgt_var + eps)
    return 1.0 - ncc.mean()

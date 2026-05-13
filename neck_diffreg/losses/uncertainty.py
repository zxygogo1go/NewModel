"""Uncertainty losses."""

from __future__ import annotations

from torch import Tensor


def gaussian_displacement_nll(prediction: Tensor, target: Tensor, log_variance: Tensor) -> Tensor:
    """Gaussian negative log likelihood for displacement supervision.

    Args:
        prediction: Predicted displacement ``[B, 3, D, H, W]``.
        target: Teacher or synthetic target displacement ``[B, 3, D, H, W]``.
        log_variance: Predicted log variance ``[B, 3, D, H, W]``.

    Returns:
        Scalar NLL-style loss.
    """

    error2 = (prediction - target).pow(2)
    return (error2 * (-log_variance).exp() + log_variance).mean()

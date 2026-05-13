from __future__ import annotations

import torch

from neck_diffreg.losses.total_loss import NeckDiffRegLoss
from neck_diffreg.models.neck_diffreg import NeCKDiffReg


def test_losses_return_finite_scalars() -> None:
    torch.manual_seed(3)
    batch = {
        "moving": torch.rand(1, 1, 8, 8, 8),
        "fixed": torch.rand(1, 1, 8, 8, 8),
        "moving_seg": (torch.rand(1, 1, 8, 8, 8) > 0.5).float(),
        "fixed_seg": (torch.rand(1, 1, 8, 8, 8) > 0.5).float(),
        "tissue_map": torch.rand(1, 2, 8, 8, 8),
        "target_displacement": torch.zeros(1, 3, 8, 8, 8),
    }
    model = NeCKDiffReg(base_channels=2, tissue_channels=2, integration_steps=2)
    outputs = model(
        batch["moving"],
        batch["fixed"],
        batch["moving_seg"],
        batch["fixed_seg"],
        tissue_map=batch["tissue_map"],
    )
    criterion = NeckDiffRegLoss(
        weights={"segmentation": 0.1, "uncertainty": 0.01},
        tissue_aware_smoothness=True,
        tissue_channel_weights=[2.0, 0.5],
    )
    total, components = criterion(outputs, batch)
    assert total.ndim == 0
    assert torch.isfinite(total)
    assert components
    for value in components.values():
        assert value.ndim == 0
        assert torch.isfinite(value)

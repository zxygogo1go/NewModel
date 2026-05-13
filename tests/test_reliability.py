from __future__ import annotations

import torch

from neck_diffreg.data.synthetic import SyntheticNeckDataset
from neck_diffreg.losses.similarity import reliability_weighted_mse
from neck_diffreg.losses.total_loss import NeckDiffRegLoss
from neck_diffreg.models.neck_diffreg import NeCKDiffReg
from neck_diffreg.models.noncorrespondence import ReliabilityHead


def test_reliability_head_outputs_values_in_unit_interval() -> None:
    torch.manual_seed(31)
    fixed = torch.rand(1, 1, 8, 8, 8)
    warped_moving = torch.rand(1, 1, 8, 8, 8)
    tissue_map = torch.rand(1, 2, 8, 8, 8)
    head = ReliabilityHead(image_channels=1, tissue_channels=2, base_channels=2)
    reliability = head(fixed=fixed, warped_moving=warped_moving, tissue_map=tissue_map)
    assert reliability.shape == (1, 1, 8, 8, 8)
    assert torch.isfinite(reliability).all()
    assert reliability.amin() >= 0.0
    assert reliability.amax() <= 1.0


def test_all_zero_reliability_collapse_is_penalized() -> None:
    fixed = torch.rand(1, 1, 8, 8, 8)
    outputs = {
        "warped_moving": fixed.clone(),
        "phi_total": torch.zeros(1, 3, 8, 8, 8),
        "phi_skeleton": torch.zeros(1, 3, 8, 8, 8),
        "velocity": torch.zeros(1, 3, 8, 8, 8),
        "reliability": torch.zeros(1, 1, 8, 8, 8),
        "aux": {},
    }
    batch = {"fixed": fixed}
    criterion = NeckDiffRegLoss(
        weights={
            "similarity": 0.0,
            "smoothness": 0.0,
            "jacobian": 0.0,
            "reliability_tv": 0.0,
            "reliability_sparsity": 1.0,
            "reliability_mean_prior": 1.0,
        },
        reliability_mean_target=0.75,
    )
    total, components = criterion(outputs, batch)
    assert components["noncorrespondence_sparsity"] > 0.9
    assert components["reliability_mean_prior"] > 0.0
    assert total > 0.0


def test_weighted_similarity_returns_finite_scalar() -> None:
    prediction = torch.rand(1, 1, 8, 8, 8)
    target = torch.rand(1, 1, 8, 8, 8)
    reliability = torch.zeros(1, 1, 8, 8, 8)
    loss = reliability_weighted_mse(prediction, target, reliability)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_corrupted_synthetic_sample_runs_training_step() -> None:
    dataset = SyntheticNeckDataset(
        num_samples=1,
        spatial_shape=(8, 8, 8),
        num_blobs=3,
        displacement_scale=0.04,
        seed=41,
        include_segmentation=True,
        enable_corruption=True,
        corruption_probability=1.0,
        num_corruptions=2,
    )
    sample = dataset[0]
    assert sample["noncorrespondence_mask"].shape == (1, 8, 8, 8)
    assert sample["noncorrespondence_mask"].sum() > 0

    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = NeCKDiffReg(base_channels=2, integration_steps=2)
    criterion = NeckDiffRegLoss(weights={"segmentation": 0.1, "reliability_mean_prior": 0.01})
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    outputs = model(
        moving=batch["moving"],
        fixed=batch["fixed"],
        moving_seg=batch["moving_seg"],
        fixed_seg=batch["fixed_seg"],
    )
    loss, components = criterion(outputs, batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert torch.isfinite(components["reliability_mean"])

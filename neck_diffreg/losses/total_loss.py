"""Composite NeCK-DiffReg training objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from neck_diffreg.losses.anatomy import dice_loss, skeleton_parameter_penalty
from neck_diffreg.losses.regularization import (
    gradient_smoothness_loss,
    jacobian_folding_penalty,
    mean_reliability_prior_loss,
    reliability_sparsity_loss,
    tissue_aware_smoothness_loss,
    total_variation_loss,
)
from neck_diffreg.losses.similarity import local_ncc_loss, reliability_weighted_mse
from neck_diffreg.losses.uncertainty import gaussian_displacement_nll
from neck_diffreg.models.spatial import warp_image


@dataclass
class LossWeights:
    similarity: float = 1.0
    smoothness: float = 0.05
    jacobian: float = 0.01
    reliability_tv: float = 0.02
    reliability_sparsity: float = 0.01
    reliability_mean_prior: float = 0.01
    segmentation: float = 0.0
    skeleton: float = 0.001
    uncertainty: float = 0.0


class NeckDiffRegLoss(nn.Module):
    """Compute total loss and named components for a model output dict."""

    def __init__(
        self,
        weights: LossWeights | dict[str, float] | None = None,
        similarity: str = "weighted_mse",
        ncc_window_size: int = 5,
        tissue_aware_smoothness: bool = False,
        tissue_channel_weights: list[float] | None = None,
        reliability_mean_target: float = 0.75,
        jacobian_margin: float = 0.0,
        jacobian_penalty: str = "squared",
    ) -> None:
        super().__init__()
        if weights is None:
            self.weights = LossWeights()
        elif isinstance(weights, dict):
            self.weights = LossWeights(**{k: v for k, v in weights.items() if hasattr(LossWeights, k)})
        else:
            self.weights = weights
        self.similarity = similarity
        self.ncc_window_size = ncc_window_size
        self.tissue_aware_smoothness = tissue_aware_smoothness
        self.tissue_channel_weights = tissue_channel_weights
        self.reliability_mean_target = reliability_mean_target
        self.jacobian_margin = jacobian_margin
        self.jacobian_penalty = jacobian_penalty

    def _similarity(self, outputs: dict[str, Any], batch: dict[str, Tensor]) -> Tensor:
        if self.similarity == "ncc":
            return local_ncc_loss(outputs["warped_moving"], batch["fixed"], window_size=self.ncc_window_size)
        return reliability_weighted_mse(outputs["warped_moving"], batch["fixed"], outputs["reliability"])

    def forward(self, outputs: dict[str, Any], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        components: dict[str, Tensor] = {}
        components["similarity"] = self._similarity(outputs, batch)
        if self.tissue_aware_smoothness:
            components["smoothness"] = tissue_aware_smoothness_loss(
                outputs["velocity"],
                tissue_map=batch.get("tissue_map"),
                channel_weights=self.tissue_channel_weights,
                penalty="l2",
            )
        else:
            components["smoothness"] = gradient_smoothness_loss(outputs["velocity"], penalty="l2")
        components["jacobian"] = jacobian_folding_penalty(
            outputs["phi_total"],
            margin=self.jacobian_margin,
            penalty=self.jacobian_penalty,
        )
        components["reliability_tv"] = total_variation_loss(outputs["reliability"])
        components["noncorrespondence_sparsity"] = reliability_sparsity_loss(outputs["reliability"])
        components["reliability_mean_prior"] = mean_reliability_prior_loss(
            outputs["reliability"],
            target_mean=self.reliability_mean_target,
        )
        components["reliability_mean"] = outputs["reliability"].mean()

        node_params = outputs.get("aux", {}).get("skeleton_node_params")
        if node_params is None:
            components["skeleton"] = outputs["phi_skeleton"].new_zeros(())
        else:
            components["skeleton"] = skeleton_parameter_penalty(node_params)

        if "moving_seg" in batch and "fixed_seg" in batch and self.weights.segmentation > 0:
            warped_seg = warp_image(batch["moving_seg"].float(), outputs["phi_total"], mode="bilinear")
            components["segmentation"] = dice_loss(warped_seg, batch["fixed_seg"].float())
        else:
            components["segmentation"] = outputs["phi_total"].new_zeros(())

        if "target_displacement" in batch and self.weights.uncertainty > 0:
            components["uncertainty"] = gaussian_displacement_nll(
                outputs["phi_total"],
                batch["target_displacement"].to(outputs["phi_total"].device),
                outputs["log_variance"],
            )
        else:
            components["uncertainty"] = outputs["phi_total"].new_zeros(())

        total = outputs["phi_total"].new_zeros(())
        total = total + self.weights.similarity * components["similarity"]
        total = total + self.weights.smoothness * components["smoothness"]
        total = total + self.weights.jacobian * components["jacobian"]
        total = total + self.weights.reliability_tv * components["reliability_tv"]
        total = total + self.weights.reliability_sparsity * components["noncorrespondence_sparsity"]
        total = total + self.weights.reliability_mean_prior * components["reliability_mean_prior"]
        total = total + self.weights.segmentation * components["segmentation"]
        total = total + self.weights.skeleton * components["skeleton"]
        total = total + self.weights.uncertainty * components["uncertainty"]

        if not torch.isfinite(total):
            raise FloatingPointError("Total loss is not finite")
        components["total"] = total
        return total, components

"""Minimal NeCK-DiffReg model."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from neck_diffreg.models.noncorrespondence import ReliabilityHead
from neck_diffreg.models.residual_branch import ResidualDiffeomorphicBranch
from neck_diffreg.models.skeleton_branch import NeuroKinematicSkeletonBranch
from neck_diffreg.models.spatial import compose_displacements, warp_image
from neck_diffreg.models.uncertainty import UncertaintyHead


class NeCKDiffReg(nn.Module):
    """Head-and-neck deformable registration prototype.

    Forward inputs:
        moving: ``[B, 1, D, H, W]``
        fixed: ``[B, 1, D, H, W]``
        moving_seg/fixed_seg: optional segmentation tensors
        tissue_map: optional ``[B, K, D, H, W]``
        bone_masks: optional ``[B, N, D, H, W]``

    Returns:
        Dictionary containing warped image, skeleton/soft/total displacement
        fields, stationary velocity, reliability map, uncertainty log variance,
        and auxiliary diagnostics.
    """

    def __init__(
        self,
        image_channels: int = 1,
        tissue_channels: int = 0,
        base_channels: int = 8,
        num_skeleton_nodes: int = 10,
        integration_steps: int = 5,
        residual_velocity_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.skeleton_branch = NeuroKinematicSkeletonBranch(
            image_channels=image_channels,
            num_nodes=num_skeleton_nodes,
            hidden_channels=max(8, base_channels * 2),
        )
        self.residual_branch = ResidualDiffeomorphicBranch(
            image_channels=image_channels,
            tissue_channels=tissue_channels,
            base_channels=base_channels,
            integration_steps=integration_steps,
            velocity_scale=residual_velocity_scale,
        )
        self.reliability_head = ReliabilityHead(
            image_channels=image_channels,
            tissue_channels=tissue_channels,
            base_channels=base_channels,
        )
        self.uncertainty_head = UncertaintyHead(
            image_channels=image_channels,
            tissue_channels=tissue_channels,
            out_channels=3,
            base_channels=base_channels,
        )

    def forward(
        self,
        moving: Tensor,
        fixed: Tensor,
        moving_seg: Tensor | None = None,
        fixed_seg: Tensor | None = None,
        tissue_map: Tensor | None = None,
        bone_masks: Tensor | None = None,
        reliability_prior: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor | None]]:
        phi_skeleton, skeleton_aux = self.skeleton_branch(
            moving=moving,
            fixed=fixed,
            bone_masks=bone_masks,
        )
        warped_skeleton = warp_image(moving, phi_skeleton)
        residual = self.residual_branch(
            fixed=fixed,
            warped_moving=warped_skeleton,
            tissue_map=tissue_map,
            reliability_prior=reliability_prior,
        )
        phi_soft = residual["displacement"]
        phi_total = compose_displacements(phi_skeleton, phi_soft)
        warped_moving = warp_image(moving, phi_total)
        reliability = self.reliability_head(fixed, warped_moving, tissue_map=tissue_map)
        uncertainty = self.uncertainty_head(fixed, warped_moving, tissue_map=tissue_map)

        return {
            "warped_moving": warped_moving,
            "phi_skeleton": phi_skeleton,
            "phi_soft": phi_soft,
            "phi_total": phi_total,
            "velocity": residual["velocity"],
            "reliability": reliability,
            "uncertainty": uncertainty,
            "log_variance": uncertainty,
            "aux": {
                "skeleton": skeleton_aux,
                "skeleton_node_params": skeleton_aux["skeleton_node_params"],
                "skeleton_rotation_vector": skeleton_aux["rotation_vector"],
                "skeleton_translation": skeleton_aux["translation"],
                "warped_skeleton_moving": warped_skeleton,
                "residual": residual,
                "moving_seg": moving_seg,
                "fixed_seg": fixed_seg,
            },
        }

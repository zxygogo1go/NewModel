"""Neuro-kinematic skeleton branch for coarse head-and-neck motion."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neck_diffreg.models.spatial import make_identity_grid


def rotation_vector_to_matrix(rotation_vector: Tensor, eps: float = 1.0e-6) -> Tensor:
    """Convert rotation vectors to rotation matrices with stable Rodrigues.

    Args:
        rotation_vector: Tensor ``[..., 3]`` in radians.
        eps: Small-angle threshold.

    Returns:
        Rotation matrices with shape ``[..., 3, 3]``.
    """

    if rotation_vector.shape[-1] != 3:
        raise ValueError("rotation_vector must end with channel dimension 3")

    rx, ry, rz = rotation_vector.unbind(dim=-1)
    zeros = torch.zeros_like(rx)
    skew = torch.stack(
        (
            zeros,
            -rz,
            ry,
            rz,
            zeros,
            -rx,
            -ry,
            rx,
            zeros,
        ),
        dim=-1,
    ).reshape(*rotation_vector.shape[:-1], 3, 3)

    theta2 = rotation_vector.pow(2).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(eps * eps))

    a_series = 1.0 - theta2 / 6.0 + theta2.pow(2) / 120.0
    b_series = 0.5 - theta2 / 24.0 + theta2.pow(2) / 720.0
    a = torch.where(theta2 < eps * eps, a_series, torch.sin(theta) / theta)
    b = torch.where(theta2 < eps * eps, b_series, (1.0 - torch.cos(theta)) / theta2.clamp_min(eps * eps))

    eye = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    eye = eye.view(*((1,) * (rotation_vector.ndim - 1)), 3, 3)
    return eye + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * torch.matmul(skew, skew)


class NeuroKinematicSkeletonBranch(nn.Module):
    """Predict and blend rigid node transforms for head-and-neck anatomy.

    Inputs:
        moving: ``[B, 1, D, H, W]``
        fixed: ``[B, 1, D, H, W]``
        bone_masks: optional ``[B, N, D, H, W]``

    Returns:
        ``(dense_displacement, aux)`` where dense displacement is
        ``[B, 3, D, H, W]``. ``aux`` contains rotation vectors
        ``[B, N, 3]``, translations ``[B, N, 3]``, rotation matrices
        ``[B, N, 3, 3]``, node-specific displacement fields
        ``[B, N, 3, D, H, W]``, and blend weights when masks are supplied.

    If ``bone_masks`` is ``None``, the branch returns zero dense skeleton
    displacement. This is the safest fallback for the first prototype because
    learned global blending without anatomical masks can invent bulk rigid
    motion in unsupported regions.
    """

    def __init__(
        self,
        image_channels: int = 1,
        num_nodes: int = 10,
        hidden_channels: int = 16,
        max_translation: float = 0.08,
        max_rotation: float = 0.25,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        in_channels = image_channels * 2
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self._num_groups(hidden_channels), hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(self._num_groups(hidden_channels), hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(self._num_groups(hidden_channels * 2), hidden_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(hidden_channels * 2, num_nodes * 6)
        if zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _num_groups(channels: int) -> int:
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return max(groups, 1)

    def predict_node_parameters(self, moving: Tensor, fixed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict rotation vectors, translations, and packed parameters.

        Args:
            moving: Tensor ``[B, 1, D, H, W]``.
            fixed: Tensor ``[B, 1, D, H, W]``.

        Returns:
            ``(rotation_vector, translation, node_params)`` with shapes
            ``[B, N, 3]``, ``[B, N, 3]``, and ``[B, N, 6]``.
        """

        features = self.encoder(torch.cat((moving, fixed), dim=1)).flatten(1)
        raw_params = self.head(features).view(moving.shape[0], self.num_nodes, 6)
        rotation_vector = torch.tanh(raw_params[..., :3]) * self.max_rotation
        translation = torch.tanh(raw_params[..., 3:]) * self.max_translation
        node_params = torch.cat((rotation_vector, translation), dim=-1)
        return rotation_vector, translation, node_params

    def normalize_bone_masks(self, bone_masks: Tensor, spatial_shape: tuple[int, int, int]) -> Tensor:
        """Normalize bone masks into safe per-voxel blend weights.

        Args:
            bone_masks: Tensor ``[B, N_mask, D, H, W]``.
            spatial_shape: Target ``(D, H, W)``.

        Returns:
            Tensor ``[B, N, D, H, W]``. Weights sum to one at every voxel.
            Voxels with all-zero masks use uniform weights to avoid NaNs.
        """

        if bone_masks.ndim != 5:
            raise ValueError("bone_masks must have shape [B, N, D, H, W]")
        if bone_masks.shape[-3:] != spatial_shape:
            bone_masks = F.interpolate(bone_masks.float(), size=spatial_shape, mode="nearest")
        masks = bone_masks.float().clamp_min(0.0)
        batch, channels, depth, height, width = masks.shape
        if channels > self.num_nodes:
            masks = masks[:, : self.num_nodes]
        elif channels < self.num_nodes:
            pad = self.num_nodes - channels
            zeros = masks.new_zeros(batch, pad, depth, height, width)
            masks = torch.cat((masks, zeros), dim=1)

        mask_sum = masks.sum(dim=1, keepdim=True)
        normalized = masks / mask_sum.clamp_min(1.0e-6)
        uniform = masks.new_full(masks.shape, 1.0 / float(self.num_nodes))
        return torch.where(mask_sum > 1.0e-6, normalized, uniform)

    def node_displacements(
        self,
        rotation_matrix: Tensor,
        translation: Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> Tensor:
        """Apply each rigid node transform to the identity grid.

        Args:
            rotation_matrix: Tensor ``[B, N, 3, 3]``.
            translation: Tensor ``[B, N, 3]`` in normalized coordinates.
            spatial_shape: ``(D, H, W)``.

        Returns:
            Node displacement tensor ``[B, N, 3, D, H, W]``.
        """

        batch = rotation_matrix.shape[0]
        depth, height, width = spatial_shape
        grid = make_identity_grid(
            spatial_shape,
            batch_size=batch,
            device=rotation_matrix.device,
            dtype=rotation_matrix.dtype,
        ).permute(0, 4, 1, 2, 3)
        coords = grid.unsqueeze(1)
        transformed = torch.einsum("bnij,bnjdhw->bnidhw", rotation_matrix, coords)
        transformed = transformed + translation.view(batch, self.num_nodes, 3, 1, 1, 1)
        return transformed - coords.expand(batch, self.num_nodes, 3, depth, height, width)

    def forward(
        self,
        moving: Tensor,
        fixed: Tensor,
        bone_masks: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        rotation_vector, translation, node_params = self.predict_node_parameters(moving, fixed)
        rotation_matrix = rotation_vector_to_matrix(rotation_vector)
        node_disp = self.node_displacements(rotation_matrix, translation, moving.shape[-3:])

        if bone_masks is None:
            dense_displacement = moving.new_zeros(moving.shape[0], 3, *moving.shape[-3:])
            blend_weights = None
            fallback = "zero_displacement_without_bone_masks"
        else:
            blend_weights = self.normalize_bone_masks(bone_masks, moving.shape[-3:])
            dense_displacement = (blend_weights.unsqueeze(2) * node_disp).sum(dim=1)
            fallback = "bone_mask_blending"

        aux = {
            "rotation_vector": rotation_vector,
            "translation": translation,
            "rotation_matrix": rotation_matrix,
            "node_params": node_params,
            "skeleton_node_params": node_params,
            "node_displacements": node_disp,
            "blend_weights": blend_weights,
            "fallback": fallback,
        }
        return dense_displacement, aux


SkeletonBranch = NeuroKinematicSkeletonBranch

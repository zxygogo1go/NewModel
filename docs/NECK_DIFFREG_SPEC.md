# NeCK-DiffReg Specification

## 1. Research goal

Implement NeCK-DiffReg: a head-and-neck-specific deformable registration framework for adaptive radiotherapy.

The model should estimate anatomically plausible deformation between a moving image and a fixed image. Unlike generic dense registration, this model decomposes head-and-neck deformation into:

1. skeleton-driven bulk motion;
2. soft-tissue residual diffeomorphic motion;
3. non-correspondence-aware matching;
4. optional uncertainty estimation.

Mathematically:

phi_total = phi_soft o phi_skeleton

where:
- phi_skeleton models skull, mandible, cervical vertebrae, and shoulder-like rigid or quasi-rigid motion;
- phi_soft is a residual diffeomorphic deformation generated from a stationary velocity field;
- R(x) is a reliability map indicating whether local image matching should be trusted;
- U(x) or log_sigma2(x) represents voxel-wise deformation uncertainty.

## 2. Inputs

Required:

moving image:
  Tensor [B, 1, D, H, W]

fixed image:
  Tensor [B, 1, D, H, W]

Optional:

moving segmentation:
  Tensor [B, C, D, H, W] or [B, 1, D, H, W]

fixed segmentation:
  Tensor [B, C, D, H, W] or [B, 1, D, H, W]

tissue map:
  Tensor [B, K, D, H, W]
  Example channels: bone, soft tissue, parotid, airway, tumor, external body, artifact.

bone masks:
  Tensor [B, N, D, H, W]
  Example nodes: skull, mandible, C1, C2, C3, C4, C5, C6, C7, shoulders.

## 3. Outputs

The model forward pass should return a dictionary:

{
  "warped_moving": Tensor [B, 1, D, H, W],
  "phi_skeleton": Tensor [B, 3, D, H, W] or displacement field,
  "phi_soft": Tensor [B, 3, D, H, W] or displacement field,
  "phi_total": Tensor [B, 3, D, H, W] or displacement field,
  "velocity": Tensor [B, 3, D, H, W],
  "reliability": Tensor [B, 1, D, H, W],
  "uncertainty": Tensor [B, 3 or 1, D, H, W],
  "aux": dict
}

Coordinate convention:

Use normalized grid coordinates compatible with torch.nn.functional.grid_sample, or clearly document voxel-coordinate convention. Be consistent across spatial transformer, compose_fields, and scaling-and-squaring.

## 4. Model modules

### 4.1 Spatial utilities

Implement:

- make_identity_grid(shape)
- warp_image(image, displacement_or_grid)
- compose_displacements(disp_a, disp_b)
- integrate_velocity_scaling_squaring(v, n_steps)
- jacobian_determinant(disp)
- displacement_gradient(disp)

Tests:
- zero displacement returns same image approximately;
- integrated zero velocity returns zero displacement;
- no NaNs;
- output shapes correct.

### 4.2 Skeleton branch

Purpose:
Predict low-dimensional rigid or affine-like transforms for anatomical nodes.

Input:
- moving image;
- fixed image;
- optional bone masks;
- optional tissue map.

Output:
- node parameters [B, N, P], where P can be 6 for rigid SE(3)-like parameters or 12 for affine;
- dense skeleton displacement [B, 3, D, H, W].

Implementation version 1:
- Use a small 3D CNN encoder.
- Global average pool features.
- Predict per-node 6-parameter transforms: rotation vector + translation.
- Convert small rotation vector to rotation matrix using a differentiable Rodrigues or first-order approximation.
- Apply each node transform to the identity grid.
- Blend node displacement fields using normalized bone masks.
- If bone masks are not provided, use uniform or learned soft blending as a fallback.

Advanced TODO:
- Use anatomical graph edges between skull, mandible, C1-C7, shoulders.
- Add graph regularization penalizing impossible jumps between adjacent vertebrae.

### 4.3 Residual diffeomorphic branch

Purpose:
Predict residual soft-tissue stationary velocity field after skeleton alignment.

Input:
- fixed image;
- moving image warped by skeleton deformation;
- optional tissue map;
- optional reliability prior.

Output:
- velocity field [B, 3, D, H, W];
- soft displacement after integration [B, 3, D, H, W].

Implementation:
- Use a 3D UNet-like encoder-decoder.
- Predict velocity.
- Integrate velocity using scaling and squaring.
- Compose residual soft displacement with skeleton displacement to obtain total displacement.

### 4.4 Non-correspondence branch

Purpose:
Predict reliability map R(x) in [0, 1].

Interpretation:
- R close to 1: local image correspondence is reliable.
- R close to 0: local region may correspond to tumor shrinkage, air cavity change, artifact, truncation, or missing anatomy.

Input:
- fixed image;
- moving image warped by current deformation;
- absolute difference image;
- optional tissue map.

Output:
- reliability [B, 1, D, H, W] through sigmoid.

Loss:
L_sim_R = mean(R * image_similarity_error)

Regularization:
- sparsity penalty on 1 - R;
- total variation penalty on R;
- optional anatomy prior encouraging low reliability in tumor, air, artifact regions.

Important:
Do not allow trivial all-zero reliability. Include sparsity or mean prior.

### 4.5 Uncertainty branch

Purpose:
Estimate voxel-wise deformation uncertainty.

Implementation version 1:
- Predict log variance map log_sigma2 [B, 3, D, H, W] from shared features or from fixed/moving/warped/difference inputs.
- Clamp log variance to a stable range.
- If target displacement or teacher field is available, implement Gaussian NLL:
  L_unc = mean( exp(-log_sigma2) * error^2 + log_sigma2 )

If no target field exists:
- expose uncertainty output;
- optionally regularize it weakly;
- use it for visualization and future calibration experiments.

Advanced TODO:
- ensemble inference;
- Monte Carlo dropout;
- diffusion or flow-matching deformation sampling.

## 5. Loss functions

Total loss:

L = 
  lambda_sim * L_sim_R
+ lambda_seg * L_seg
+ lambda_reg * L_reg
+ lambda_jac * L_jac
+ lambda_skel * L_skel
+ lambda_noncor * L_noncor
+ lambda_unc * L_unc
+ lambda_inv * L_inv

### 5.1 Image similarity

Implement at least:
- local normalized cross correlation loss;
- MSE fallback.

Weighted version:
L_sim_R = mean(R * sim_error) / (mean(R) + eps)

### 5.2 Segmentation loss

If segmentations are provided:
- soft Dice loss between fixed_seg and warped moving_seg.

### 5.3 Regularization

Implement:
- gradient smoothness loss for displacement or velocity;
- bending energy optional;
- tissue-aware smoothness optional.

Tissue-aware smoothness:
If tissue_map is provided, use channel-dependent weights:
bone: high rigidity
soft tissue: medium
air/tumor/artifact: lower correspondence reliability

### 5.4 Skeleton loss

Implement:
- bone rigidity loss;
- adjacent node transform smoothness placeholder;
- penalty on excessive rotations/translations.

### 5.5 Non-correspondence loss

Implement:
- sparsity on 1 - R;
- total variation on R;
- optional mean reliability target, e.g. encourage mean(R) > 0.7.

### 5.6 Jacobian loss

Implement:
- folding penalty: mean(ReLU(-detJ))
- report folding rate.

### 5.7 Inverse consistency

Optional:
If bidirectional mode is enabled:
- compute forward and backward deformation;
- penalize composition away from identity.

## 6. Training

Implement a training script that:
- reads YAML config;
- supports synthetic data mode;
- supports optional real dataset mode using npz or NIfTI paths;
- runs forward pass;
- computes total loss;
- logs component losses;
- saves checkpoints.

The first runnable version must work on synthetic data.

## 7. Evaluation

Implement metrics:
- image similarity after registration;
- Dice for propagated segmentations;
- Jacobian determinant statistics;
- folding rate;
- displacement magnitude;
- inverse consistency error placeholder;
- bone rigidity error placeholder;
- uncertainty summary.

Advanced:
- TRE if landmarks are provided;
- HD95 if scipy or surface-distance package is available;
- dose accumulation metrics if dose maps are provided.

## 8. Minimal acceptance criteria

The repository is acceptable when:

1. pytest passes on synthetic tests;
2. python -m neck_diffreg.training.train --config configs/neck_diffreg_base.yaml runs for at least 2 iterations on synthetic data;
3. the model forward pass returns all required keys;
4. all losses return finite scalars;
5. the spatial transformer identity test passes;
6. no patient-specific paths are hard-coded.
# AGENTS.md

## Project identity

This repository implements NeCK-DiffReg, a research-grade PyTorch framework for head-and-neck deformable image registration in adaptive radiotherapy.

The goal is not to build a generic registration network. The goal is to implement an anatomically informed, head-and-neck-specific registration model with:

1. a neuro-kinematic skeleton branch for skull, mandible, cervical vertebrae, and shoulder girdle motion;
2. a tissue-aware diffeomorphic residual deformation branch;
3. a non-correspondence-aware reliability map for tumor shrinkage, air cavity changes, artifacts, and missing anatomy;
4. an uncertainty head for voxel-wise deformation uncertainty and optional dose-relevant evaluation.

## Coding expectations

Use Python and PyTorch.

Prefer clear, modular, research-friendly code over overly compressed code.

Every major module must have:
- type hints where practical;
- docstrings explaining input/output tensor shapes;
- minimal unit tests using synthetic tensors;
- no hidden dependency on private datasets.

Avoid implementing everything in one giant file.

Use deterministic toy tests to verify:
- spatial transformer identity warp;
- velocity integration shape consistency;
- no NaNs in forward pass;
- loss functions return finite scalar values;
- model can perform a tiny forward/backward pass on synthetic 3D volumes.

## Repository structure target

Use this structure unless a better structure already exists:

neck_diffreg/
  models/
    neck_diffreg.py
    skeleton_branch.py
    residual_branch.py
    noncorrespondence.py
    uncertainty.py
    spatial.py
    layers.py
  losses/
    similarity.py
    regularization.py
    anatomy.py
    uncertainty.py
    total_loss.py
  data/
    dataset.py
    transforms.py
    synthetic.py
  training/
    train.py
    validate.py
    infer.py
    config.py
  evaluation/
    metrics.py
    dose.py
    visualization.py
  utils/
    io.py
    logging.py
    seed.py
configs/
  neck_diffreg_base.yaml
tests/
  test_spatial.py
  test_model_forward.py
  test_losses.py
docs/
  NECK_DIFFREG_SPEC.md
README.md

## Scientific design constraints

The main deformation should be represented as:

phi_total = phi_soft o phi_skeleton

where phi_skeleton is a low-dimensional anatomically constrained transform derived from skull, mandible, cervical vertebrae, and shoulder nodes, and phi_soft is a residual diffeomorphic deformation from a stationary velocity field.

The model should output at least:
- displacement field or deformation field;
- reliability map R(x) in [0, 1];
- optional voxel-wise uncertainty map or log-variance map.

The implementation should support 3D volumes with shape:

moving: [B, 1, D, H, W]
fixed: [B, 1, D, H, W]
moving_seg: optional [B, C, D, H, W] or [B, 1, D, H, W]
fixed_seg: optional [B, C, D, H, W] or [B, 1, D, H, W]
tissue_map: optional [B, K, D, H, W]
bone_masks: optional [B, N_bones, D, H, W]

Do not assume a fixed image size.

## Implementation priority

Build a minimal working prototype first.

Phase 1:
- spatial transformer;
- identity grid;
- displacement composition;
- scaling-and-squaring integration;
- simple CNN encoder-decoder residual registration baseline.

Phase 2:
- add skeleton branch that predicts one SE(3)-like affine transform per anatomical node;
- blend node transforms into a dense skeleton deformation using bone masks or distance-based weights.

Phase 3:
- add residual diffeomorphic branch conditioned on moving, fixed, warped moving, and optional tissue maps.

Phase 4:
- add reliability map R(x);
- use R-weighted image similarity loss;
- regularize R with sparsity and total variation.

Phase 5:
- add uncertainty output;
- implement Gaussian negative log-likelihood-style deformation uncertainty loss when pseudo labels or teacher fields are available;
- otherwise expose uncertainty as a calibrated auxiliary output.

Phase 6:
- add evaluation metrics: Dice, HD95 placeholder or optional implementation, TRE placeholder, Jacobian determinant, folding rate, inverse consistency, bone rigidity error.

## Safety and data assumptions

Do not hard-code protected patient data paths.

All examples must run with synthetic data.

Dataset classes may support NIfTI or npz loading, but tests must not require real medical images.

## When making design choices

Prefer a working, extensible prototype over a theoretically perfect but untestable system.

If the full scientific method is too large, implement the interface and a simple version, then leave clear TODOs for advanced variants.
# NeCK-DiffReg

NeCK-DiffReg is a research-oriented PyTorch prototype for head-and-neck deformable
image registration in adaptive radiotherapy. This first scaffold focuses on a
minimal, synthetic-data-only implementation that can be tested without private
medical images.

## What is implemented

- Normalized-coordinate 3D spatial transformer utilities.
- Displacement composition and scaling-and-squaring velocity integration.
- Approximate finite-difference Jacobian determinant and folding penalty.
- Minimal model with:
  - skeleton branch predicting small per-node SE(3)-like parameters;
  - residual diffeomorphic 3D UNet branch;
  - reliability map head;
  - uncertainty log-variance head.
- Synthetic 3D blob dataset with known smooth random deformation.
- Composite losses for image similarity, reliability regularization,
  deformation smoothness, folding penalty, Dice loss, and uncertainty NLL.
- Pytest smoke tests for spatial utilities, model forward/backward, and losses.

## Run tests

```bash
pytest
```

## Run synthetic smoke training

```bash
python -m neck_diffreg.training.train --config configs/neck_diffreg_base.yaml
```

The training script runs two synthetic iterations by default and saves:

```text
outputs/checkpoints/neck_diffreg_synthetic.pt
```

## Notes

All examples use synthetic tensors. No patient-specific paths or datasets are
required. Advanced anatomy graph constraints, real NIfTI/NPZ loading, dose
evaluation, and calibrated uncertainty analysis are left as explicit later-phase
extensions.

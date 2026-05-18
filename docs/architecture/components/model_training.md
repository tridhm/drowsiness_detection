# Model Training

## Purpose
Train and export model artifacts used by runtime detection and baseline pipelines.

## Responsibilities and Non-Goals
### Responsibilities
- Train deep eye/multitask models from image datasets.
- Train tabular baseline model from engineered CSV features.
- Persist artifacts with predictable filenames for runtime reuse.

### Non-Goals
- Real-time runtime orchestration.
- Dataset annotation workflows.
- Deployment packaging.

## Primary Entry Points and Public Interfaces
- [train_eye_model_torch.py](/d:/drowsiness_detection-main/train_eye_model_torch.py)
  - Eye classifier training with CLI params (`--data-dir`, `--batch-size`, `--learning-rate`, `--epochs`).
- [train_advanced_model.py](/d:/drowsiness_detection-main/train_advanced_model.py)
  - Advanced multitask training flow (supports `--config config_tools.json` + CLI overrides).
- [train_model.py](/d:/drowsiness_detection-main/train_model.py)
  - RandomForest training from `drowsiness_data.csv` (supports `--config config_tools.json`).

## Internal Logic and Data Flow
1. Resolve dataset paths.
2. Apply preprocessing/transforms and split.
3. Train model(s) and compute metrics.
4. Save artifacts for runtime consumption.

## Dependencies
- Deep learning: `torch`, `torchvision`, `Pillow`, `tqdm`
- Data handling: `numpy`, `pathlib`, `argparse`
- Classical ML: `pandas`, `scikit-learn`, `joblib`
- Canonical runtime: Python 3.12.x

## Outputs and Contracts to Runtime
- `eye_model.pth`
- `advanced_drowsiness_model.pth`
- `drowsiness_model.pkl`
- `scaler.pkl`

Contract rules:
- Runtime model class definitions must remain compatible with saved weights.
- Artifact names/paths should stay stable unless runtime config/docs are updated accordingly.

## Configuration and Environment
- Device selection is automatic (`cuda` when available).
- Hyperparameters are CLI args and can be loaded from `config_tools.json` via `--config`.
- Paths are repo-relative by default.
- Config precedence for train/data scripts: `defaults < --config JSON < CLI overrides`.

## Testing and Observability
- Console epoch metrics and progress bars.
- Explicit artifact save messages are the primary completion signal.
- Downstream validation: runtime/evaluation scripts can load generated artifacts.

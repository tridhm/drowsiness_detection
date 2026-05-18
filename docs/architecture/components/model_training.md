# Model Training

## Purpose
Train and export model artifacts used by runtime detection, baseline pipelines, and window-level research fusion experiments.

## Responsibilities and Non-Goals
### Responsibilities
- Train deep eye/multitask models from image datasets.
- Train tabular baseline model from engineered CSV features.
- Train the research Random Forest window-fusion MVP from `labeled_windows.csv`.
- Persist artifacts/reports with predictable filenames for runtime or offline analysis.

### Non-Goals
- Real-time runtime orchestration.
- Dataset annotation workflows.
- Deployment packaging.
- Claiming real KSS-aligned performance before a validated KSS/PVT-style dataset is used.

## Primary Entry Points and Public Interfaces
- [train_eye_model_torch.py](/d:/drowsiness_detection-main/train_eye_model_torch.py)
  - Eye classifier training with CLI params (`--data-dir`, `--batch-size`, `--learning-rate`, `--epochs`).
- [train_advanced_model.py](/d:/drowsiness_detection-main/train_advanced_model.py)
  - Advanced multitask training flow (supports `--config config_tools.json` + CLI overrides).
- [train_model.py](/d:/drowsiness_detection-main/train_model.py)
  - Legacy/classical RandomForest training from `drowsiness_data.csv` (supports `--config config_tools.json`).
- [tools/research/train_window_fusion_model.py](/d:/drowsiness_detection-main/tools/research/train_window_fusion_model.py)
  - Research Random Forest MVP trained on window-level features from `labeled_windows.csv`.

## Internal Logic and Data Flow
### Runtime artifact training
1. Resolve dataset paths.
2. Apply preprocessing/transforms and split.
3. Train model(s) and compute metrics.
4. Save artifacts for runtime consumption.

### Window-level research fusion
1. Read `labeled_windows.csv`.
2. Convert `kss_band` to the configured classification target, with `sleepy` treated as positive by default.
3. Select numeric/window features from eye, yawn, head-pose, and FSM groups.
4. Prefer subject-disjoint train/test split so the same subject does not appear in both train and test.
5. Train `RandomForestClassifier`.
6. Write model metrics, feature importance, permutation importance, and ablation results.

## Dependencies
- Deep learning: `torch`, `torchvision`, `Pillow`, `tqdm`
- Data handling: `numpy`, `pathlib`, `argparse`
- Legacy/classical ML: `pandas`, `scikit-learn`, `joblib`
- Research RF MVP: stdlib `csv`, `numpy`, `scikit-learn`, `joblib`; no `pandas` requirement
- Canonical runtime: Python 3.12.x

## Outputs and Contracts to Runtime / Research Reports
Runtime artifacts:
- `eye_model.pth`
- `advanced_drowsiness_model.pth`
- `drowsiness_model.pkl`
- `scaler.pkl`

Research RF reports:
- `random_forest_results.csv`
- `feature_importance.csv`
- `permutation_importance.csv`
- `ablation_results.csv`

Contract rules:
- Runtime model class definitions must remain compatible with saved weights.
- Artifact names/paths should stay stable unless runtime config/docs are updated accordingly.
- Window-level RF input must contain KSS-aligned labels; sample fixture results are validation-only.
- RF reports are generated analysis artifacts and should normally live under `reports/research/`.

## Configuration and Environment
- Device selection is automatic (`cuda` when available) for PyTorch training.
- Hyperparameters are CLI args and can be loaded from `config_tools.json` via `--config` for legacy train/data scripts.
- Paths are repo-relative by default.
- Config precedence for train/data scripts: `defaults < --config JSON < CLI overrides`.
- `train_window_fusion_model.py` exposes `--n-estimators`, `--test-subject-ratio`, `--permutation-repeats`, and `--allow-random-split`.
- Subject-disjoint split is the default safety behavior for research RF; random split is only a technical fallback.

## Testing and Observability
- Console epoch metrics and progress bars for deep-learning scripts.
- Explicit artifact/report save messages are the primary completion signal.
- Downstream validation: runtime/evaluation scripts can load generated artifacts.
- Research RF behavior is covered by unit tests that verify CSV outputs and the subject-disjoint split failure mode.

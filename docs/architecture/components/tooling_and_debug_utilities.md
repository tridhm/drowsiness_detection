# Tooling and Debug Utilities

## Purpose
Document diagnostic and support utilities used for environment checks, merge syncing, early pipeline validation, and offline research reporting.

## Responsibilities and Non-Goals
### Responsibilities
- Provide environment sanity checks.
- Provide deterministic archive import/provenance tooling.
- Preserve lightweight prototype/debug scripts.
- Provide research CLI tools that convert video/window data into reproducible CSV reports.

### Non-Goals
- Core runtime decision logic.
- Training loops for runtime production models.
- Full observability stack/telemetry backend.
- Creating KSS labels automatically from video.

## Primary Entry Points and Public Interfaces
- [debug_mp.py](/d:/drowsiness_detection-main/debug_mp.py)
  - MediaPipe import/debug inspection.
- [debug_mp_310.py](/d:/drowsiness_detection-main/debug_mp_310.py)
  - Legacy Python-version-specific MediaPipe debug check.
- [v10.py](/d:/drowsiness_detection-main/v10.py)
  - EAR-only prototype visual utility.
- [tools/merge/sync_import_archive.py](/d:/drowsiness_detection-main/tools/merge/sync_import_archive.py)
  - Controlled DA1/bundle non-heavy asset sync + manifest generation.
- [tools/research/export_video_features.py](/d:/drowsiness_detection-main/tools/research/export_video_features.py)
  - Offline video-to-CSV feature exporter for frame/window research data.
- [tools/research/align_window_labels.py](/d:/drowsiness_detection-main/tools/research/align_window_labels.py)
  - KSS-style annotation alignment for exported windows.
- [tools/research/evaluate_baselines.py](/d:/drowsiness_detection-main/tools/research/evaluate_baselines.py)
  - Rule-based baseline evaluator for labeled windows.
- [tools/research/train_window_fusion_model.py](/d:/drowsiness_detection-main/tools/research/train_window_fusion_model.py)
  - Window-level Random Forest fusion MVP with feature importance, permutation importance, and ablation reports.

## Internal Logic and Data Flow
- Debug scripts run standalone and print diagnostics.
- `sync_import_archive.py`:
  1. Enumerates approved archive patterns.
  2. Copies/verifies files into active repo.
  3. Writes checksum/status manifest (`docs/merge/import_manifest.json`).
- Research tools:
  1. Export reproducible per-frame and per-window CSV rows from a video.
  2. Align window rows with pre-existing KSS annotations by subject/session/video/time overlap.
  3. Evaluate simple baselines from `labeled_windows.csv`.
  4. Train a window-level Random Forest MVP and write metrics/explanation CSVs.

## Dependencies
- Debug: `mediapipe`, `opencv-python`, `numpy` (depending on script)
- Merge tool: `argparse`, `hashlib`, `json`, `pathlib`, `shutil`
- Research exporter: existing `runtime` modules, OpenCV video capture, stdlib `csv`
- Research evaluator/trainer: stdlib `csv`, `numpy`, `scikit-learn`, `joblib`; no `pandas` requirement

## Outputs and Contracts
- Debug scripts: console diagnostics only.
- Merge tool: machine-readable provenance contract via
  - [import_manifest.json](/d:/drowsiness_detection-main/docs/merge/import_manifest.json)
- Research exporter:
  - `frame_features.csv`: one row per frame with timestamp, raw signals, rolling features, and FSM fields.
  - `window_features.csv`: one row per time window with aggregated eye/yawn/head-pose/FSM features.
- Research alignment/evaluation/training:
  - `labeled_windows.csv`
  - `baseline_results.csv`
  - `baseline_confusion_matrix.csv`
  - `random_forest_results.csv`
  - `feature_importance.csv`
  - `permutation_importance.csv`
  - `ablation_results.csv`

## Configuration and Environment
- Canonical environment target for project workflows: Python 3.12.x.
- `sync_import_archive.py` options:
  - `--archive-root`
  - `--repo-root`
  - `--overwrite`
  - `--manifest-file`
- `v10.py` assumes webcam source by default.
- Research tools are CLI-first and write outputs to user-provided CSV paths/directories, usually under `reports/research/`.
- `align_window_labels.py` requires a KSS-style annotation CSV; it does not infer KSS automatically.

## Testing and Observability
- Merge verification snapshot:
  - [verification_report.md](/d:/drowsiness_detection-main/docs/merge/verification_report.md)
- Unit tests for runtime and research contracts are under `tests/`.
- Research pipeline fixtures live under `metadata/*.sample.csv` and validate schemas/metrics without claiming real model performance.

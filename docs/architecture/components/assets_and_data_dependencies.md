# Assets and Data Dependencies

## Purpose
Define static assets, dataset expectations, and model/evaluation artifacts required by runtime, training, and offline research workflows.

## Responsibilities and Non-Goals
### Responsibilities
- Document required local datasets/artifacts.
- Clarify which scripts consume which assets.
- Capture path/layout assumptions that affect reproducibility.
- Distinguish committed sample fixtures from generated reports and heavy datasets.

### Non-Goals
- Implementing loaders/validators.
- Remote storage management.
- CI artifact publishing.
- Bundling real public KSS/PVT datasets inside the portable package.

## Primary Entry Points and Consumers
- Runtime:
  - [advanced_drowsiness_detection.py](/d:/drowsiness_detection-main/advanced_drowsiness_detection.py)
  - [runtime/app.py](/d:/drowsiness_detection-main/runtime/app.py)
  - [drowsiness_detection_with_model.py](/d:/drowsiness_detection-main/drowsiness_detection_with_model.py)
- Training:
  - [train_advanced_model.py](/d:/drowsiness_detection-main/train_advanced_model.py)
  - [train_eye_model_torch.py](/d:/drowsiness_detection-main/train_eye_model_torch.py)
  - [train_model.py](/d:/drowsiness_detection-main/train_model.py)
- Evaluation:
  - [tools/evaluation/label_yawn_frames.py](/d:/drowsiness_detection-main/tools/evaluation/label_yawn_frames.py)
  - [tools/evaluation/evaluate_mar_videos.py](/d:/drowsiness_detection-main/tools/evaluation/evaluate_mar_videos.py)
  - [tools/evaluation/analyze_mar_thresholds.py](/d:/drowsiness_detection-main/tools/evaluation/analyze_mar_thresholds.py)
- Research pipeline:
  - [tools/research/export_video_features.py](/d:/drowsiness_detection-main/tools/research/export_video_features.py)
  - [tools/research/align_window_labels.py](/d:/drowsiness_detection-main/tools/research/align_window_labels.py)
  - [tools/research/evaluate_baselines.py](/d:/drowsiness_detection-main/tools/research/evaluate_baselines.py)
  - [tools/research/train_window_fusion_model.py](/d:/drowsiness_detection-main/tools/research/train_window_fusion_model.py)

## Dependency Sets
### Local datasets / inputs
- `CEW/` (`closed/`, `open/`)
- `dataset_eyes&yawn/`
- `mrleyedataset/` (if used in advanced training)
- `Video Database/` (external heavy media)
- `tools/evaluation/metadata/yawn_labels/*.csv`
- KSS/PVT-style public dataset annotations, if doing real research validation; not included in the repo.

### Runtime static assets
- `alert.wav`
- `config.json`

### Model artifacts
- `eye_model.pth`
- `advanced_drowsiness_model.pth`
- `drowsiness_model.pkl`
- `scaler.pkl`

### Evaluation metadata artifacts
- `tools/evaluation/metadata/reports/final_evaluation_report.csv`
- `tools/evaluation/metadata/reports/mar_evaluation_report.csv`
- `tools/evaluation/metadata/reports/mar_threshold_analysis.csv`
- `tools/evaluation/metadata/mar_result/*`

### Research sample fixtures and generated artifacts
Committed lightweight fixtures:
- `metadata/kss_annotations.sample.csv`
- `metadata/window_features.sample.csv`
- `metadata/labeled_windows.sample.csv`
- `metadata/subject_split.sample.csv`

Generated outputs, normally under `reports/research/`:
- `frame_features.csv`
- `window_features.csv`
- `labeled_windows.csv`
- `baseline_results.csv`
- `baseline_confusion_matrix.csv`
- `random_forest_results.csv`
- `feature_importance.csv`
- `permutation_importance.csv`
- `ablation_results.csv`

## Contracts to Other Modules
- Runtime contracts:
  - Engine registry names in config/CLI must match `runtime/engines/registry.py`.
  - Signal schema is `fsm.DrowsinessSignals`.
- Training contracts:
  - Saved model tensor shapes must match inference class definitions.
- Evaluation contracts:
  - Label CSV schema: `frame_index,label`
  - MAR result CSV requires `MAR` column.
- Research contracts:
  - `frame_features.csv` has one row per frame with timestamp, face/signal fields, rolling features, and FSM outputs.
  - `window_features.csv` has one row per time window with aggregated eye/yawn/head-pose/FSM features.
  - KSS annotation CSV must include `subject_id,session_id,video_id,start_time_sec,end_time_sec,kss_score,kss_band,notes`.
  - `labeled_windows.csv` must include aligned `kss_score`, `kss_band`, and `label_source` before baseline/RF evaluation.

## Configuration / Path Rules
- Runtime config precedence: `defaults < config.json < CLI`.
- Most scripts assume project root as current working directory.
- Heavy archive content is kept outside active tree in:
  - `D:\drowsiness_detection-import-archive\merge_20260418_015536`
- Heavy videos/datasets should stay outside GitHub or be ignored.
- Research smoke-test outputs should go under `reports/research/` and are generated artifacts, not source files.

## Observability / Validation
- Runtime startup and frame logs surface missing source/model issues.
- Merge sync verification and SHA256 status are tracked in:
  - [import_manifest.json](/d:/drowsiness_detection-main/docs/merge/import_manifest.json)
- Verification summary is tracked in:
  - [verification_report.md](/d:/drowsiness_detection-main/docs/merge/verification_report.md)
- Research pipeline validation is covered by unit tests plus `metadata/*.sample.csv` fixtures.
- Fixture metrics validate code paths only and must not be reported as real drowsiness-detection performance.

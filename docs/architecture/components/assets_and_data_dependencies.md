# Assets and Data Dependencies

## Purpose
Define static assets, dataset expectations, and model/evaluation artifacts required by runtime, training, and offline evaluation workflows.

## Responsibilities and Non-Goals
### Responsibilities
- Document required local datasets/artifacts.
- Clarify which scripts consume which assets.
- Capture path/layout assumptions that affect reproducibility.

### Non-Goals
- Implementing loaders/validators.
- Remote storage management.
- CI artifact publishing.

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

## Dependency Sets
### Local datasets / inputs
- `CEW/` (`closed/`, `open/`)
- `dataset_eyes&yawn/`
- `mrleyedataset/` (if used in advanced training)
- `Video Database/` (external heavy media)
- `tools/evaluation/metadata/yawn_labels/*.csv`

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

## Contracts to Other Modules
- Runtime contracts:
  - Engine registry names in config/CLI must match `runtime/engines/registry.py`.
  - Signal schema is `fsm.DrowsinessSignals`.
- Training contracts:
  - Saved model tensor shapes must match inference class definitions.
- Evaluation contracts:
  - Label CSV schema: `frame_index,label`
  - MAR result CSV requires `MAR` column.

## Configuration / Path Rules
- Runtime config precedence: `defaults < config.json < CLI`.
- Most scripts assume project root as current working directory.
- Heavy archive content is kept outside active tree in:
  - `D:\drowsiness_detection-import-archive\merge_20260418_015536`

## Observability / Validation
- Runtime startup and frame logs surface missing source/model issues.
- Merge sync verification and SHA256 status are tracked in:
  - [import_manifest.json](/d:/drowsiness_detection-main/docs/merge/import_manifest.json)
- Verification summary is tracked in:
  - [verification_report.md](/d:/drowsiness_detection-main/docs/merge/verification_report.md)

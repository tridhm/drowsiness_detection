# DA1 + Bundle Merge Inventory and Conflict Matrix

## Direct Conflicts

| File | Current Root Role | DA1/Bundle Role | Resolution |
| --- | --- | --- | --- |
| `advanced_drowsiness_detection.py` | Thin runtime entrypoint (`runtime/app.py`) | DA1: offline MAR evaluation script, Bundle: pre-FSM runtime | Keep modular orchestrator; expose bundle-style logic as opt-in `legacy` decision engine |
| `drowsiness_detection_with_model.py` | Baseline runtime | DA1 variant switches capture source | Keep root baseline and CLI source options |
| `train_eye_model_torch.py` | Eye classifier training | DA1 variant changes dataset path to `CEW` | Keep parameterized root script (`--data-dir`) |
| `advanced_drowsiness_model.pth` | Existing root advanced weights | DA1 has different weights with same architecture keys | Keep both artifacts separate and documented |
| `README_ADVANCED.md` | Legacy runtime usage docs (source archive) | DA1/bundle docs reflect older runtime model | Keep root docs; merge only relevant operational notes |

## Imported DA1 Additions (Maximum Non-Heavy)

- Code/tools:
  - `tools/evaluation/label_yawn_frames.py`
  - `tools/evaluation/evaluate_mar_videos.py`
  - `tools/evaluation/analyze_mar_thresholds.py` (from DA1 `phantichketqua.py`, parameterized)
  - `tools/merge/sync_import_archive.py`
- Models:
  - `eye_model.pth` (copied + hash-verified)
- Metadata/evaluation artifacts:
  - `tools/evaluation/metadata/yawn_labels/*.csv`
  - `tools/evaluation/metadata/reports/final_evaluation_report.csv`
  - `tools/evaluation/metadata/reports/mar_evaluation_report.csv`
  - `tools/evaluation/metadata/mar_result/mar_result.csv`
  - `tools/evaluation/metadata/mar_result/Book1.xlsx`
  - `tools/evaluation/metadata/mar_result/Code_Generated_Image.png`
  - `tools/evaluation/metadata/mar_result/unnamed.png`
- References:
  - `docs/reference/*.pdf`

## Imported Bundle Additions

- `legacy_feature_overlay.py` (telemetry helper)
- `legacy` decision engine module via runtime registry (`runtime/engines/legacy_engine.py`)
- Runtime registry selection:
  - `--decision-engine fsm|legacy`

## Excluded Assets (Intentional)

- `DA1/drowsiness_detection-main/Video Database/**` (heavy videos)
- `DA1/drowsiness_detection-main/extracted_video_frames/**` (generated frame dumps)
- `DA1/drowsiness_detection-main/.git/**`
- `DA1/drowsiness_detection-main/venv/**`

Reason: preserve maximum practical integration while keeping heavy environment-specific media external.

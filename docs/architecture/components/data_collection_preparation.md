# Data Collection and Preparation

## Purpose
Generate supervised training/evaluation inputs from raw videos and landmarks.

## Responsibilities and Non-Goals
### Responsibilities
- Extract engineered feature rows (`EAR`, `MAR`, head pose) for tabular modeling.
- Sample raw frames from videos for manual labeling workflows.
- Provide label/evaluation metadata used by offline analysis tools.

### Non-Goals
- Real-time runtime alerting.
- Final model training loops.
- Dataset governance/versioning.

## Primary Entry Points and Public Interfaces
- [train_data.py](/d:/drowsiness_detection-main/train_data.py)
  - Extracts engineered features into `drowsiness_data.csv`.
- [extract_video_frames.py](/d:/drowsiness_detection-main/extract_video_frames.py)
  - Samples frames from video sets into `extracted_video_frames/`.
- [tools/evaluation/label_yawn_frames.py](/d:/drowsiness_detection-main/tools/evaluation/label_yawn_frames.py)
  - Interactive frame-level labeling for yawn videos.

## Internal Logic and Data Flow
### `train_data.py`
1. Open labeled source video.
2. Detect landmarks and compute `ear, mar, pitch, yaw, roll`.
3. Append rows to `drowsiness_data.csv` with class label.

### `extract_video_frames.py`
1. Enumerate videos under source folder.
2. Sample frames at fixed interval.
3. Write image files for downstream manual review/labeling.

### `label_yawn_frames.py`
1. Enumerate videos from dataset path.
2. Record frame labels from keyboard input.
3. Write `*_labels.csv` per video.

## Dependencies
- Landmark/data prep: `opencv-python`, `dlib`, `imutils`, `numpy`, `scipy`
- Labeling: `opencv-python`, `csv`, `pathlib`

## Outputs and Contracts
- Produces:
  - `drowsiness_data.csv`
  - `extracted_video_frames/`
  - `tools/evaluation/metadata/yawn_labels/*.csv`
- Consumes:
  - Source videos (`Video Database` or provided dataset path)
  - Optional landmark predictor assets used by `train_data.py`
- CSV schema contract for tabular training:
  - `ear, mar, pitch, yaw, roll, label`

## Configuration and Environment
- Paths are configured via CLI args.
- `train_data.py` and `extract_video_frames.py` also support shared JSON config via `--config config_tools.json`.
- `label_yawn_frames.py` supports CLI (`--dataset-path`, `--label-path`, etc.).
- Scripts expect repository root working directory unless absolute paths are provided.
- Canonical runtime: Python 3.12.x.

## Testing and Observability
- Scripts print per-run file counts and output paths.
- Data quality is validated downstream by successful training/evaluation runs.

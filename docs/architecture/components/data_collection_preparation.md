# Data Collection and Preparation

## Purpose
Generate supervised training/evaluation inputs from raw videos, landmarks, frame labels, exported research features, and KSS-style annotations.

## Responsibilities and Non-Goals
### Responsibilities
- Extract engineered feature rows (`EAR`, `MAR`, head pose) for tabular modeling.
- Sample raw frames from videos for manual labeling workflows.
- Provide label/evaluation metadata used by offline analysis tools.
- Export frame-level and window-level research features from video files.
- Align exported windows with existing KSS-style annotation segments.

### Non-Goals
- Real-time runtime alerting.
- Final model training loops.
- Dataset governance/versioning.
- Automatically generating KSS labels from facial features.

## Primary Entry Points and Public Interfaces
- [train_data.py](/d:/drowsiness_detection-main/train_data.py)
  - Extracts engineered features into `drowsiness_data.csv`.
- [extract_video_frames.py](/d:/drowsiness_detection-main/extract_video_frames.py)
  - Samples frames from video sets into `extracted_video_frames/`.
- [tools/evaluation/label_yawn_frames.py](/d:/drowsiness_detection-main/tools/evaluation/label_yawn_frames.py)
  - Interactive frame-level labeling for yawn videos.
- [tools/research/export_video_features.py](/d:/drowsiness_detection-main/tools/research/export_video_features.py)
  - Runs the modular runtime pipeline offline and writes `frame_features.csv` plus `window_features.csv`.
- [tools/research/align_window_labels.py](/d:/drowsiness_detection-main/tools/research/align_window_labels.py)
  - Matches `window_features.csv` rows to KSS-style annotation rows and writes `labeled_windows.csv`.

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

### `export_video_features.py`
1. Open a video file through OpenCV.
2. Derive timestamps from video FPS, not wall-clock time.
3. Reuse runtime perception/feature/decision modules to compute frame signals.
4. Write one row per frame to `frame_features.csv`.
5. Aggregate rows into fixed-duration windows and write `window_features.csv`.

### `align_window_labels.py`
1. Read `window_features.csv` and a KSS-style annotation CSV.
2. Match by `subject_id`, `session_id`, `video_id`.
3. Choose the annotation segment with maximum time overlap for each window.
4. Write `labeled_windows.csv` with `kss_score`, `kss_band`, and `label_source`.

## Dependencies
- Landmark/data prep: `opencv-python`, `dlib`, `imutils`, `numpy`, `scipy`
- Labeling: `opencv-python`, `csv`, `pathlib`
- Research exporter/alignment: `opencv-python`, existing `runtime` modules, stdlib `csv`, `pathlib`

## Outputs and Contracts
- Produces:
  - `drowsiness_data.csv`
  - `extracted_video_frames/`
  - `tools/evaluation/metadata/yawn_labels/*.csv`
  - `frame_features.csv`
  - `window_features.csv`
  - `labeled_windows.csv`
- Consumes:
  - Source videos (`Video Database` or provided dataset path)
  - Optional landmark predictor assets used by `train_data.py`
  - KSS-style annotation CSVs for label alignment
- CSV schema contract for tabular training:
  - `ear, mar, pitch, yaw, roll, label`
- KSS annotation contract:
  - `subject_id,session_id,video_id,start_time_sec,end_time_sec,kss_score,kss_band,notes`
- Research window contract:
  - windows must preserve subject/session/video identifiers and start/end timestamps so labels can be aligned later.

## Configuration and Environment
- Paths are configured via CLI args.
- `train_data.py` and `extract_video_frames.py` also support shared JSON config via `--config config_tools.json`.
- `label_yawn_frames.py` supports CLI (`--dataset-path`, `--label-path`, etc.).
- `export_video_features.py` supports `--video-path`, `--frame-csv`, `--window-csv`, `--window-seconds`, `--stride-seconds`, `--max-frames`, and optional `--config`.
- `align_window_labels.py` supports `--keep-unlabeled` when the caller wants to preserve windows that have no matching annotation.
- Scripts expect repository root working directory unless absolute paths are provided.
- Canonical runtime: Python 3.12.x.

## Testing and Observability
- Scripts print per-run file counts and output paths.
- Data quality is validated downstream by successful training/evaluation runs.
- Research fixtures under `metadata/*.sample.csv` validate alignment and aggregation without requiring a real KSS dataset.

# Drowsiness Detection Architecture Guide

## High-Level Purpose
This repository provides a modular driver drowsiness detection system with three aligned tracks:
1. Real-time runtime detection (`advanced_drowsiness_detection.py` -> `runtime/app.py`)
2. Training pipelines for model artifacts (`.pth`, `.pkl`, `.csv`)
3. Offline evaluation/labeling and merge traceability tooling

The runtime is now package-based (`runtime/`) to support testability and pluggable decision engines.

## Tech Stack
- Language: Python
- Canonical interpreter target: Python 3.12.x
- Computer Vision: OpenCV, MediaPipe, SciPy
- Deep Learning: PyTorch, TorchVision
- Classical ML: scikit-learn
- Data Utilities: numpy, csv, pathlib, joblib
- Alerting/UI: OpenCV overlays + `playsound`

## System / Directory Map
- `advanced_drowsiness_detection.py`: thin runtime entrypoint
- `runtime/app.py`: orchestrator
- `runtime/config.py`: typed config loader (`defaults < config.json < CLI`)
- `runtime/transports.py`: source transport (`webcam` / `file`)
- `runtime/perception.py`: face landmark + raw signal extraction
- `runtime/features.py`: EMA/PERCLOS/temporal feature pipeline
- `runtime/engines/`: decision-engine interface and implementations
- `runtime/alerts.py`: audio alert policy controller
- `tools/evaluation/`: labeling/evaluation/threshold analysis scripts
- `tools/merge/sync_import_archive.py`: controlled DA1/bundle asset sync + provenance manifest

### Component Documents
- [runtime_detection.md](./components/runtime_detection.md)
- [model_training.md](./components/model_training.md)
- [data_collection_preparation.md](./components/data_collection_preparation.md)
- [assets_and_data_dependencies.md](./components/assets_and_data_dependencies.md)
- [tooling_and_debug_utilities.md](./components/tooling_and_debug_utilities.md)

### Visual Architecture
- [Codebase_Visual_Map.md](./Codebase_Visual_Map.md)
- [modular_runtime_map.md](./modular_runtime_map.md)

## Cross-Component Contracts and Key Data Flows
### Flow A: Data Prep -> Training -> Runtime
1. [data_collection_preparation.md](./components/data_collection_preparation.md)
   - Produces `drowsiness_data.csv` and sampled frame corpora.
2. [model_training.md](./components/model_training.md)
   - Produces `eye_model.pth`, `advanced_drowsiness_model.pth`, `drowsiness_model.pkl`, `scaler.pkl`.
3. [runtime_detection.md](./components/runtime_detection.md)
   - Consumes model artifacts + `alert.wav` for online detection.

### Flow B: Runtime Modular Decision Layer
1. `runtime/perception.py` -> raw frame signals
2. `runtime/features.py` -> normalized/temporal `DrowsinessSignals`
3. `runtime/engines/registry.py` -> selected engine (`fsm`, `legacy`)
4. `runtime/alerts.py` -> sound policy from engine decision

### Flow C: Merge and Provenance
- `tools/merge/sync_import_archive.py` imports approved non-heavy archive assets.
- `docs/merge/import_manifest.json` stores source->target + checksums + status.
- `docs/merge/da1_bundle_integration_ledger.md` stores human-readable what/how/where.

## Runtime and Deployment Notes
- Local-machine execution model (no service wrapper in repo).
- Runtime supports config file + CLI overrides:
  - `--config <path>`
  - `--decision-engine fsm|legacy`
  - `--source webcam|file`
  - `--video-path <path>`
- GPU acceleration remains opportunistic via `torch.cuda.is_available()`.
- Heavy datasets/media remain external and excluded from source tree.

## How to Start
### Setup
1. Create Python 3.12 virtual environment (`py -3.12 -m venv venv`).
2. Activate virtual environment (`.\venv\Scripts\Activate.ps1`).
3. Install dependencies from canonical [README.md](/d:/drowsiness_detection-main/README.md):
   - `pip install -r requirements.txt`
   - Optional legacy tools: `pip install -r requirements-legacy.txt`
4. Ensure required local assets are available (`alert.wav`, datasets, model files as needed).

### Run (Runtime Detection)
- Default FSM runtime:
  - `python advanced_drowsiness_detection.py`
- Use config file:
  - `python advanced_drowsiness_detection.py --config config.json`
- Switch engine:
  - `python advanced_drowsiness_detection.py --decision-engine legacy`

### Train
- Eye classifier:
  - `python train_eye_model_torch.py --data-dir dataset_eyes&yawn/train`
- Advanced multitask model:
  - `python train_advanced_model.py --config config_tools.json`
- Tabular model from engineered CSV features:
  - `python train_model.py --config config_tools.json`

### Evaluate and Analyze
- Label yawn frames:
  - `python tools/evaluation/label_yawn_frames.py`
- MAR sliding-window evaluation:
  - `python tools/evaluation/evaluate_mar_videos.py`
- MAR threshold sweep analysis:
  - `python tools/evaluation/analyze_mar_thresholds.py`

### Debug
- MediaPipe checks:
  - `python debug_mp.py`
  - `python debug_mp_310.py`
- EAR prototype sanity run:
  - `python v10.py`

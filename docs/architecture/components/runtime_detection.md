# Runtime Detection

## Purpose
Provide real-time drowsiness detection from webcam/video streams using a modular perception -> feature -> decision-engine -> alert pipeline.

## Responsibilities and Non-Goals
### Responsibilities
- Capture frames from a configured source.
- Extract raw face/eye/mouth/head/gaze signals.
- Build temporal features (`EMA`, `PERCLOS`, counters).
- Execute selected decision engine (`fsm` or `legacy`).
- Render overlays and trigger alert sound policy.

### Non-Goals
- Model training/export.
- Dataset labeling/governance.
- Packaging/deployment orchestration.

## Primary Entry Points and Public Interfaces
- [advanced_drowsiness_detection.py](/d:/drowsiness_detection-main/advanced_drowsiness_detection.py)
  - Thin entrypoint delegating to modular runtime.
- [runtime/app.py](/d:/drowsiness_detection-main/runtime/app.py)
  - Main runtime orchestrator.
- [runtime/engines/registry.py](/d:/drowsiness_detection-main/runtime/engines/registry.py)
  - Python map registry for decision engines.
- [drowsiness_detection_with_model.py](/d:/drowsiness_detection-main/drowsiness_detection_with_model.py)
  - Baseline runtime path using eye model + EAR/MAR.

## Internal Logic and Data Flow
1. Load runtime config (`defaults < config.json < CLI`).
2. Open source transport (`webcam` or `file`) via `VideoTransport`.
3. Extract frame-level perception signals via `PerceptionExtractor`.
4. Build temporal `DrowsinessSignals` via `SignalFeaturePipeline`.
5. Resolve and execute decision engine from registry map:
   - `fsm`: canonical FSM path
   - `legacy`: bundle-style rule path (opt-in)
6. Apply alert policy (`none` / `double` / `continuous`) in `AudioAlertController`.
7. Render overlay/debug info and process quit event.

## Dependencies
- CV/perception: `opencv-python`, `mediapipe`, `numpy`
- Signal processing: local `ema_filter.py`, `perclos.py`, `fsm.py`
- Runtime alerts: `playsound`, `threading`

## Outputs and Contracts to Other Modules
- Consumes model artifacts and static assets:
  - `eye_model.pth` (baseline)
  - `advanced_drowsiness_model.pth` (training artifact, optional)
  - `alert.wav`
- Consumes evaluation metadata optionally for offline workflows (not runtime-critical).
- Produces runtime UI/alerts and console diagnostics.

## Configuration and CLI
Supported runtime CLI:
- `--config <path-to-json>`
- `--source webcam|file`
- `--video-path <path-to-video>`
- `--decision-engine fsm|legacy`
- `--enable-legacy-feature-overlay` / `--disable-legacy-feature-overlay`
- `--display-window` / `--no-display`

Central config file:
- [config.json](/d:/drowsiness_detection-main/config.json)

## Testing and Observability Touchpoints
- Unit coverage for config precedence, engine registry, and engine contract in `tests/`.
- Runtime logs every N frames include state, evidence, and PERCLOS values.
- Overlay includes state label, evidence score, and key signals.

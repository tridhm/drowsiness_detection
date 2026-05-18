# Modular Runtime Map

This document describes the decoupled runtime architecture introduced for testability and engine extensibility.

## Component Overview

```mermaid
flowchart TB
    CLI[CLI + config.json\n(precedence: CLI > JSON > defaults)] --> APP[runtime/app.py\nOrchestrator]

    APP --> TRANSPORT[runtime/transports.py\nVideoTransport]
    APP --> PERCEPTION[runtime/perception.py\nPerceptionExtractor]
    APP --> FEATURES[runtime/features.py\nSignalFeaturePipeline]
    APP --> REGISTRY[runtime/engines/registry.py\nENGINE_REGISTRY]
    APP --> ALERTS[runtime/alerts.py\nAudioAlertController]

    REGISTRY --> FSME[runtime/engines/fsm_engine.py\nFSMDecisionEngine]
    REGISTRY --> LEGACYE[runtime/engines/legacy_engine.py\nLegacyRuleEngine]

    FEATURES --> SIGNALS[fsm.DrowsinessSignals]
    SIGNALS --> FSME
    SIGNALS --> LEGACYE

    APP --> ENTRY[advanced_drowsiness_detection.py\nThin wrapper]
```

## Engine Extension Point

- Registry file: `runtime/engines/registry.py`
- Add a new engine by:
  1. Implementing `DecisionEngine` in `runtime/engines/<new_engine>.py`
  2. Registering it in `ENGINE_REGISTRY` (python map)
  3. Selecting it via `--decision-engine <key>` or `config.json`

This keeps extension explicit and test-friendly while avoiding dynamic import ambiguity.

## Config Contract

- Root file: `config.json`
- Major sections:
  - `input`: source selection and video path
  - `runtime`: fps, warmup, calibration
  - `thresholds`: EAR/MAR/head/gaze thresholds
  - `windows`: PERCLOS, blink, yawn windows
  - `alerts`: alert cooldown policy
  - top-level toggles: `decision_engine`, `enable_legacy_feature_overlay`, `display_window`

## Runtime Modes

- `decision_engine=fsm` (default): canonical FSM decision path
- `decision_engine=legacy`: bundle-style rule engine isolated as its own module

Both modes share the same perception/feature pipeline and alert orchestration.

## Related Docs

- High-level guide: `docs/architecture/Architecture_Guide.md`
- Visual system map: `docs/architecture/Codebase_Visual_Map.md`
- Merge provenance: `docs/merge/da1_bundle_integration_ledger.md`

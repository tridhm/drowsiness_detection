# DA1 + Bundle Merge Notes

## What Was Merged

- Modular runtime foundation:
  - `runtime/` package (`config`, `transports`, `perception`, `features`, `engines`, `alerts`, `app`)
  - Root `advanced_drowsiness_detection.py` converted to thin entrypoint
  - Engine registry map with `fsm` and `legacy` keys
  - Runtime CLI/config integration:
    - `--config`
    - `--decision-engine fsm|legacy`
    - `--source`, `--video-path`
    - `--enable-legacy-feature-overlay`
- Offline evaluation tooling:
  - `label_yawn_frames.py`
  - `evaluate_mar_videos.py`
  - `analyze_mar_thresholds.py`
- Non-heavy DA1 artifacts synced with provenance:
  - Label CSVs, compact reports, MAR result metadata assets, `eye_model.pth`, references
- Controlled import + manifest:
  - `tools/merge/sync_import_archive.py`
  - `docs/merge/import_manifest.json`

## What Was Not Merged

- Heavy DA1 trees:
  - `Video Database/**`
  - `extracted_video_frames/**`
- Repository/environment internals from archive:
  - `.git/**`
  - `venv/**`

## Why

- Preserve canonical modular runtime behavior (FSM default) while supporting optional legacy engine.
- Achieve maximum practical non-heavy integration with deterministic provenance.
- Keep repository lightweight and reproducible.

## Archive Location

- `D:/drowsiness_detection-import-archive/merge_20260418_015536`

## Traceability Artifacts

- Human ledger: `docs/merge/da1_bundle_integration_ledger.md`
- Machine manifest: `docs/merge/import_manifest.json`
- Verification snapshot: `docs/merge/verification_report.md`

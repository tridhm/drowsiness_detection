# Tooling and Debug Utilities

## Purpose
Document diagnostic and support utilities used for environment checks, merge syncing, and early pipeline validation.

## Responsibilities and Non-Goals
### Responsibilities
- Provide environment sanity checks.
- Provide deterministic archive import/provenance tooling.
- Preserve lightweight prototype/debug scripts.

### Non-Goals
- Core runtime decision logic.
- Training loops.
- Full observability stack/telemetry backend.

## Primary Entry Points and Public Interfaces
- [debug_mp.py](/d:/drowsiness_detection-main/debug_mp.py)
  - MediaPipe import/debug inspection.
- [debug_mp_310.py](/d:/drowsiness_detection-main/debug_mp_310.py)
  - Legacy Python-version-specific MediaPipe debug check.
- [v10.py](/d:/drowsiness_detection-main/v10.py)
  - EAR-only prototype visual utility.
- [tools/merge/sync_import_archive.py](/d:/drowsiness_detection-main/tools/merge/sync_import_archive.py)
  - Controlled DA1/bundle non-heavy asset sync + manifest generation.

## Internal Logic and Data Flow
- Debug scripts run standalone and print diagnostics.
- `sync_import_archive.py`:
  1. Enumerates approved archive patterns.
  2. Copies/verifies files into active repo.
  3. Writes checksum/status manifest (`docs/merge/import_manifest.json`).

## Dependencies
- Debug: `mediapipe`, `opencv-python`, `numpy` (depending on script)
- Merge tool: `argparse`, `hashlib`, `json`, `pathlib`, `shutil`

## Outputs and Contracts
- Debug scripts: console diagnostics only.
- Merge tool: machine-readable provenance contract via
  - [import_manifest.json](/d:/drowsiness_detection-main/docs/merge/import_manifest.json)

## Configuration and Environment
- Canonical environment target for project workflows: Python 3.12.x.
- `sync_import_archive.py` options:
  - `--archive-root`
  - `--repo-root`
  - `--overwrite`
  - `--manifest-file`
- `v10.py` assumes webcam source by default.

## Testing and Observability
- Merge verification snapshot:
  - [verification_report.md](/d:/drowsiness_detection-main/docs/merge/verification_report.md)
- Unit tests for runtime contract/config are under `tests/` and are part of latest verification pass.

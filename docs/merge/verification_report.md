# Verification Report

Date: 2026-04-18

## Commands Executed

1. Compile checks

```powershell
D:\drowsiness_detection-main\venv\Scripts\python.exe -m py_compile <runtime + tools + tests>
```

Result: `PY_COMPILE_OK`

2. Unit tests

```powershell
D:\drowsiness_detection-main\venv\Scripts\python.exe -m unittest discover -s D:\drowsiness_detection-main\tests -p "test_*.py" -v
```

Result: `Ran 7 tests ... OK`

3. CLI smoke checks

```powershell
python advanced_drowsiness_detection.py --help
python tools/evaluation/analyze_mar_thresholds.py --help
python tools/merge/sync_import_archive.py --help
```

Result: All help commands returned successfully.

4. DA1 non-heavy sync

```powershell
python tools/merge/sync_import_archive.py --repo-root D:\drowsiness_detection-main --archive-root D:\drowsiness_detection-import-archive\merge_20260418_015536 --overwrite
```

Result summary:
- copied: 5
- verified: 12
- skipped: 0
- missing: 0

5. MAR threshold analyzer smoke run

```powershell
python tools/evaluation/analyze_mar_thresholds.py --input-csv tools/evaluation/metadata/mar_result/mar_result.csv --output-csv tools/evaluation/metadata/reports/mar_threshold_analysis.csv --threshold-start 0.5 --threshold-end 0.5 --threshold-step 0.1
```

Result: output CSV generated and metrics printed.

## Guard Checks

- Heavy archive assets remain excluded from sync policy:
  - `Video Database/**`
  - `extracted_video_frames/**`
  - `.git/**`
  - `venv/**`
- Manifest with per-file status/hash written to:
  - `docs/merge/import_manifest.json`

## Notes

- Full webcam/file live runtime regression parity was not executed in this automated pass.
- Decision-engine contract and config precedence coverage is included in unit tests.

## Documentation Sync (Latest Pass)

- Updated architecture/component docs for modular runtime structure and CLI/config contract.
- Updated merge inventory/notes/provenance docs to reflect maximum non-heavy DA1 integration.
- Regenerated visual map PDFs from updated Mermaid source:
  - `docs/architecture/Codebase_Visual_Map-1.pdf`
  - `docs/architecture/Codebase_Visual_Map-2.pdf`
  - `docs/architecture/Codebase_Visual_Map-3.pdf`
  - `docs/architecture/Codebase_Visual_Map.pdf`

## Documentation and Environment Sync (Current Pass)

- Standardized docs to canonical environment target: **Python 3.12.x**.
- Updated setup guidance to use `README.md` as single source of truth.
- Synced training/data component docs with current CLI behavior (`--config`, shared `config_tools.json`, schema validation).

Additional checks executed:

```powershell
.\venv\Scripts\python.exe train_advanced_model.py --config config_tools.json --help
.\venv\Scripts\python.exe train_model.py --config config_tools.json --help
.\venv\Scripts\python.exe train_data.py --config config_tools.json --help
.\venv\Scripts\python.exe extract_video_frames.py --config config_tools.json --help
.\venv\Scripts\python.exe -m py_compile cli_json_config.py train_advanced_model.py train_model.py train_data.py extract_video_frames.py
```

Negative validation smoke (expected failure):

```powershell
python train_model.py --config <invalid_config_with_test_size_2> --help
# -> error: Config key 'test_size' must be < 1
```

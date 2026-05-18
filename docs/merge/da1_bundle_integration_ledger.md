# DA1 + Bundle Integration Ledger (Modular Runtime)

This ledger tracks what was integrated, how it was integrated, and where it lives in the codebase.

## Runtime Modularization Ledger

| Source/Intent | Integration Method | Target Location | Runtime Impact |
| --- | --- | --- | --- |
| Monolithic runtime orchestration | Refactor to orchestrator module | `runtime/app.py` | Main runtime now composes modules via config + registry |
| Inline runtime config constants | Extracted to typed config module | `runtime/config.py`, `config.json` | Configurable behavior with precedence `CLI > JSON > defaults` |
| Inline capture open/read logic | Transport abstraction | `runtime/transports.py` | Source selection (`webcam` / `file`) decoupled from decision logic |
| Inline perception processing | Perception extractor module | `runtime/perception.py` | Landmark + raw signal extraction isolated for testability |
| Inline feature accumulation | Feature pipeline module | `runtime/features.py` | EMA/PERCLOS/counters separated from UI/alerts |
| Mixed FSM + legacy decision code | Engine interface + registry map | `runtime/engines/base.py`, `runtime/engines/registry.py` | Engine switching via `--decision-engine` |
| Alert calls scattered in loop | Alert controller module | `runtime/alerts.py` | Alert policy isolated and reusable |
| Root script carrying full runtime | Thin entry wrapper | `advanced_drowsiness_detection.py` | Entry now delegates to modular app |

## DA1/Bundle Asset Integration Ledger

| Archive Source | Integration Method | Target Location | Status |
| --- | --- | --- | --- |
| `DA1/.../eye_model.pth` | Model copy + hash verification | `eye_model.pth` | Integrated |
| `DA1/.../Video Database/Yawn/Labels/*.csv` | Metadata copy | `tools/evaluation/metadata/yawn_labels/` | Integrated |
| `DA1/.../final_evaluation_report.csv` | Metadata copy | `tools/evaluation/metadata/reports/final_evaluation_report.csv` | Integrated |
| `DA1/.../MAR_Result/mar_evaluation_report.csv` | Metadata copy | `tools/evaluation/metadata/reports/mar_evaluation_report.csv` | Integrated |
| `DA1/.../MAR_Result/mar_result.csv` | Metadata copy | `tools/evaluation/metadata/mar_result/mar_result.csv` | Integrated |
| `DA1/.../MAR_Result/Book1.xlsx` | Metadata copy | `tools/evaluation/metadata/mar_result/Book1.xlsx` | Integrated |
| `DA1/.../MAR_Result/Code_Generated_Image.png` | Metadata copy | `tools/evaluation/metadata/mar_result/Code_Generated_Image.png` | Integrated |
| `DA1/.../MAR_Result/unnamed.png` | Metadata copy | `tools/evaluation/metadata/mar_result/unnamed.png` | Integrated |
| `DA1/.../Reference/*.pdf` | Reference copy | `docs/reference/*.pdf` | Integrated |
| `DA1/.../MAR_Result/phantichketqua.py` | Port + CLI parameterization | `tools/evaluation/analyze_mar_thresholds.py` | Integrated |
| `drowsiness_detection_bundle/...` legacy features | Isolated into optional/engine modules | `runtime/engines/legacy_engine.py`, `legacy_feature_overlay.py` | Integrated |

## Controlled Import Tool

- Tool: `tools/merge/sync_import_archive.py`
- Purpose: deterministic non-heavy asset sync + checksum verification + manifest writing
- Manifest: `docs/merge/import_manifest.json`

## Explicit Exclusions (Not Integrated)

- `DA1/drowsiness_detection-main/Video Database/**`
- `DA1/drowsiness_detection-main/extracted_video_frames/**`
- `DA1/drowsiness_detection-main/.git/**`
- `DA1/drowsiness_detection-main/venv/**`

These remain external in `D:\drowsiness_detection-import-archive\merge_20260418_015536`.

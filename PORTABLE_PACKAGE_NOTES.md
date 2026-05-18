# Portable Package Notes

Created at: 2026-04-18 09:22:02
Last refreshed from main: 2026-05-18
Source: D:\drowsiness_detection-main

This portable package is intended for creating a new GitHub repository.

## Included
- Source code (`*.py`)
- Runtime modules (`runtime/`)
- Tooling (`tools/`)
- Research pipeline tooling (`tools/research/`)
- Documentation (`README.md`, `docs/`)
- Lightweight sample metadata (`metadata/*.sample.csv`)
- Config files (`config.json`, `config_tools.json`, `config_tools.schema.json`)
- Requirements (`requirements.txt`, `requirements-legacy.txt`)

## Excluded
- Virtual environment: `venv/`
- Caches: `__pycache__/`, `*.pyc`
- Generated reports: `reports/`
- Runtime logs: `logs/`
- Heavy datasets: `CEW/`, `dataset_eyes&yawn/`, `mrleyedataset/`, `Video Database/`, `extracted_video_frames/`
- Model artifacts: `*.pth`, `*.pkl`
- Temp files: `tempCodeRunnerFile*`

## Latest refresh notes
- Synced the offline research pipeline MVP from `D:\drowsiness_detection-main`.
- Added feature export, KSS/window alignment, baseline evaluation, and Random Forest window-fusion tooling.
- Added Vietnamese research pipeline changelog docs and slide-friendly changelog docs.
- Generated CSV reports are intentionally excluded; rerun the documented commands to regenerate them locally.

## Quick setup
1. `py -3.12 -m venv venv`
2. `./venv/Scripts/Activate.ps1`
3. `pip install -r requirements.txt`

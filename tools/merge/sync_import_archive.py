"""Sync approved DA1/bundle archive assets into the active repository with provenance."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable


DEFAULT_ARCHIVE_ROOT = Path(r"D:\drowsiness_detection-import-archive\merge_20260418_015536")

ASSET_SPECS = [
    {
        "source_pattern": "DA1/drowsiness_detection-main/Video Database/Yawn/Labels/*.csv",
        "target": "tools/evaluation/metadata/yawn_labels/",
        "type": "metadata_copy",
        "subsystem": "evaluation",
    },
    {
        "source_pattern": "DA1/drowsiness_detection-main/final_evaluation_report.csv",
        "target": "tools/evaluation/metadata/reports/final_evaluation_report.csv",
        "type": "metadata_copy",
        "subsystem": "evaluation",
    },
    {
        "source_pattern": "DA1/drowsiness_detection-main/MAR_Result/mar_evaluation_report.csv",
        "target": "tools/evaluation/metadata/reports/mar_evaluation_report.csv",
        "type": "metadata_copy",
        "subsystem": "evaluation",
    },
    {
        "source_pattern": "DA1/drowsiness_detection-main/MAR_Result/mar_result.csv",
        "target": "tools/evaluation/metadata/mar_result/mar_result.csv",
        "type": "metadata_copy",
        "subsystem": "evaluation",
    },
    {
        "source_pattern": "DA1/drowsiness_detection-main/MAR_Result/Book1.xlsx",
        "target": "tools/evaluation/metadata/mar_result/Book1.xlsx",
        "type": "metadata_copy",
        "subsystem": "evaluation",
    },
    {
        "source_pattern": "DA1/drowsiness_detection-main/MAR_Result/Code_Generated_Image.png",
        "target": "tools/evaluation/metadata/mar_result/Code_Generated_Image.png",
        "type": "metadata_copy",
        "subsystem": "evaluation",
    },
    {
        "source_pattern": "DA1/drowsiness_detection-main/MAR_Result/unnamed.png",
        "target": "tools/evaluation/metadata/mar_result/unnamed.png",
        "type": "metadata_copy",
        "subsystem": "evaluation",
    },
    {
        "source_pattern": "DA1/drowsiness_detection-main/eye_model.pth",
        "target": "eye_model.pth",
        "type": "model_copy",
        "subsystem": "models",
    },
    {
        "source_pattern": "DA1/drowsiness_detection-main/Reference/OTSU_paper.pdf",
        "target": "docs/reference/OTSU_paper.pdf",
        "type": "reference_copy",
        "subsystem": "docs",
    },
    {
        "source_pattern": "DA1/drowsiness_detection-main/Reference/real-time-driver-drowsiness-detection-system-using-facial-5eb70f1r04.pdf",
        "target": "docs/reference/real-time-driver-drowsiness-detection-system-using-facial-5eb70f1r04.pdf",
        "type": "reference_copy",
        "subsystem": "docs",
    },
]

EXCLUDED_PATTERNS = [
    "DA1/drowsiness_detection-main/Video Database/**",
    "DA1/drowsiness_detection-main/extracted_video_frames/**",
    "DA1/drowsiness_detection-main/.git/**",
    "DA1/drowsiness_detection-main/venv/**",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_sources(archive_root: Path, pattern: str) -> list[Path]:
    return sorted(archive_root.glob(pattern))


def target_path(repo_root: Path, target_spec: str, source_path: Path, is_pattern: bool) -> Path:
    raw_target = repo_root / target_spec
    if is_pattern or target_spec.endswith("/") or target_spec.endswith("\\"):
        return raw_target / source_path.name
    return raw_target


def copy_asset(src: Path, dst: Path, overwrite: bool) -> tuple[str, str | None]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_hash = sha256(src)

    if dst.exists():
        dst_hash = sha256(dst)
        if dst_hash == src_hash:
            return "verified", dst_hash
        if not overwrite:
            return "different_skipped", dst_hash

    shutil.copy2(src, dst)
    return "copied", sha256(dst)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync approved DA1/bundle assets into active repo.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT), help="Archive root path.")
    parser.add_argument("--repo-root", default=None, help="Repository root. Defaults to script parent parent.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite when source and target hashes differ.")
    parser.add_argument(
        "--manifest-file",
        default="docs/merge/import_manifest.json",
        help="Manifest output path relative to repo root.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    archive_root = Path(args.archive_root)
    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]

    if not archive_root.exists():
        raise FileNotFoundError(f"Archive root not found: {archive_root}")

    records = []
    copied_count = 0
    verified_count = 0
    skipped_count = 0
    missing_count = 0

    for spec in ASSET_SPECS:
        pattern = spec["source_pattern"]
        sources = resolve_sources(archive_root, pattern)
        is_pattern = any(ch in pattern for ch in "*?[")

        if not sources:
            records.append(
                {
                    "source": pattern,
                    "target": spec["target"],
                    "type": spec["type"],
                    "subsystem": spec["subsystem"],
                    "status": "missing_source",
                }
            )
            missing_count += 1
            continue

        for src in sources:
            dst = target_path(repo_root, spec["target"], src, is_pattern)
            status, dst_hash = copy_asset(src, dst, overwrite=args.overwrite)
            src_hash = sha256(src)
            rel_src = src.relative_to(archive_root).as_posix()
            rel_dst = dst.relative_to(repo_root).as_posix()
            records.append(
                {
                    "source": rel_src,
                    "target": rel_dst,
                    "type": spec["type"],
                    "subsystem": spec["subsystem"],
                    "status": status,
                    "sha256_source": src_hash,
                    "sha256_target": dst_hash,
                }
            )
            if status == "copied":
                copied_count += 1
            elif status == "verified":
                verified_count += 1
            else:
                skipped_count += 1

    manifest = {
        "merge_timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_folders": ["DA1/drowsiness_detection-main", "drowsiness_detection_bundle"],
        "import_scope": "maximum_non_heavy",
        "excluded_artifacts": EXCLUDED_PATTERNS,
        "summary": {
            "copied": copied_count,
            "verified": verified_count,
            "skipped": skipped_count,
            "missing": missing_count,
            "total_records": len(records),
        },
        "records": records,
        "archive_policy": {
            "status": "completed",
            "destination_root": str(archive_root.parent),
            "archive_path": str(archive_root),
        },
    }

    manifest_path = repo_root / args.manifest_file
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {manifest_path}")
    print(
        "Summary -> copied=%d verified=%d skipped=%d missing=%d"
        % (copied_count, verified_count, skipped_count, missing_count)
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

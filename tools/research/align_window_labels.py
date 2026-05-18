from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from tools.research.common import KSS_COLUMNS, read_csv_rows, write_csv_rows
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.research.common import KSS_COLUMNS, read_csv_rows, write_csv_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align window feature rows with KSS-style time annotations.")
    parser.add_argument("--window-csv", required=True, help="Input window_features.csv path.")
    parser.add_argument("--annotations-csv", required=True, help="Input KSS annotation CSV path.")
    parser.add_argument("--output-csv", required=True, help="Output labeled_windows.csv path.")
    parser.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Keep windows without matching annotations using empty KSS fields.",
    )
    return parser


def align_windows(
    windows: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    keep_unlabeled: bool = False,
) -> list[dict[str, Any]]:
    labeled_rows: list[dict[str, Any]] = []
    for window in windows:
        match, overlap_seconds = _best_annotation(window, annotations)
        if match is None:
            if keep_unlabeled:
                output = dict(window)
                output.update({"kss_score": "", "kss_band": "", "label_source": "unlabeled"})
                labeled_rows.append(output)
            continue

        output = dict(window)
        output["kss_score"] = match.get("kss_score", "")
        output["kss_band"] = match.get("kss_band", "")
        output["label_source"] = f"overlap_seconds={overlap_seconds:.3f}"
        labeled_rows.append(output)
    return labeled_rows


def _best_annotation(
    window: dict[str, Any], annotations: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    window_key = (window.get("subject_id"), window.get("session_id"), window.get("video_id"))
    window_start = _to_float(window.get("window_start_sec"))
    window_end = _to_float(window.get("window_end_sec"))

    best: dict[str, Any] | None = None
    best_overlap = 0.0
    for annotation in annotations:
        annotation_key = (
            annotation.get("subject_id"),
            annotation.get("session_id"),
            annotation.get("video_id"),
        )
        if annotation_key != window_key:
            continue
        overlap = _overlap_seconds(
            window_start,
            window_end,
            _to_float(annotation.get("start_time_sec")),
            _to_float(annotation.get("end_time_sec")),
        )
        if overlap > best_overlap:
            best = annotation
            best_overlap = overlap
    return best, best_overlap


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def run(args: argparse.Namespace) -> int:
    windows = read_csv_rows(Path(args.window_csv))
    annotations = read_csv_rows(Path(args.annotations_csv))
    if annotations:
        missing = [column for column in KSS_COLUMNS if column not in annotations[0]]
        if missing:
            raise ValueError(f"Missing annotation column(s): {missing}")
    labeled = align_windows(windows, annotations, keep_unlabeled=args.keep_unlabeled)
    if windows:
        fieldnames = list(windows[0].keys()) + ["kss_score", "kss_band", "label_source"]
    else:
        fieldnames = ["kss_score", "kss_band", "label_source"]
    write_csv_rows(Path(args.output_csv), fieldnames, labeled)
    print(f"Aligned {len(labeled)} labeled window row(s) -> {args.output_csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from tools.research.common import binary_metrics, fsm_state_is_sleepy, kss_is_sleepy, read_csv_rows, write_csv_rows
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.research.common import binary_metrics, fsm_state_is_sleepy, kss_is_sleepy, read_csv_rows, write_csv_rows


RESULT_COLUMNS = ["baseline", "accuracy", "precision", "recall", "f1", "support", "tp", "tn", "fp", "fn"]
CONFUSION_COLUMNS = ["baseline", "tp", "tn", "fp", "fn"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate simple drowsiness baselines on labeled windows.")
    parser.add_argument("--input-csv", required=True, help="Input labeled_windows.csv path.")
    parser.add_argument("--results-csv", required=True, help="Output baseline_results.csv path.")
    parser.add_argument("--confusion-csv", required=True, help="Output baseline_confusion_matrix.csv path.")
    parser.add_argument("--perclos-threshold", type=float, default=0.35, help="PERCLOS threshold for sleepy prediction.")
    return parser


def evaluate_baseline_rows(
    rows: list[dict[str, Any]],
    perclos_threshold: float = 0.35,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    if not rows:
        raise ValueError("No labeled window rows found")
    y_true = [1 if kss_is_sleepy(row) else 0 for row in rows]
    predictions = {
        "perclos_only": [1 if _float(row.get("perclos_60s")) >= perclos_threshold else 0 for row in rows],
        "yawn_only": [1 if _float(row.get("yawn_count")) >= 1.0 else 0 for row in rows],
        "head_pose_only": [1 if _float(row.get("head_drop_count")) >= 1.0 else 0 for row in rows],
        "fsm_state": [1 if fsm_state_is_sleepy(row.get("fsm_state_mode")) else 0 for row in rows],
    }

    result_rows: list[dict[str, Any]] = []
    confusion_rows: dict[str, dict[str, int]] = {}
    for name, y_pred in predictions.items():
        metrics = binary_metrics(y_true, y_pred)
        result_rows.append({"baseline": name, **metrics})
        confusion_rows[name] = {
            "tp": int(metrics["tp"]),
            "tn": int(metrics["tn"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
        }
    return result_rows, confusion_rows


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def run(args: argparse.Namespace) -> int:
    rows = read_csv_rows(Path(args.input_csv))
    result_rows, confusion = evaluate_baseline_rows(rows, perclos_threshold=args.perclos_threshold)
    write_csv_rows(Path(args.results_csv), RESULT_COLUMNS, result_rows)
    write_csv_rows(
        Path(args.confusion_csv),
        CONFUSION_COLUMNS,
        [{"baseline": name, **values} for name, values in confusion.items()],
    )
    print(f"Wrote baseline metrics -> {args.results_csv}")
    print(f"Wrote confusion matrix -> {args.confusion_csv}")
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

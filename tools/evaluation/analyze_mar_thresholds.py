"""Analyze MAR thresholds against pseudo-ground-truth clustering."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MAR threshold sweep using KMeans-derived labels.")
    parser.add_argument(
        "--input-csv",
        default="tools/evaluation/metadata/mar_result/mar_result.csv",
        help="Input CSV containing MAR column.",
    )
    parser.add_argument(
        "--output-csv",
        default="tools/evaluation/metadata/reports/mar_threshold_analysis.csv",
        help="Output CSV report path.",
    )
    parser.add_argument("--threshold-start", type=float, default=0.30, help="Threshold sweep start.")
    parser.add_argument("--threshold-end", type=float, default=0.80, help="Threshold sweep end (inclusive).")
    parser.add_argument("--threshold-step", type=float, default=0.025, help="Threshold step size.")
    return parser


def threshold_range(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold-step must be > 0")
    values: list[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def load_mar_values(input_csv: Path) -> np.ndarray:
    mar_values: list[float] = []
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "MAR" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a MAR column.")
        for row in reader:
            mar_values.append(float(row["MAR"]))
    if not mar_values:
        raise ValueError("Input CSV contains no MAR rows.")
    return np.array(mar_values, dtype=float)


def build_pseudo_labels(mar_values: np.ndarray) -> np.ndarray:
    mar_reshaped = mar_values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(mar_reshaped)

    centers = kmeans.cluster_centers_.flatten()
    open_cluster_idx = int(centers.argmax())
    return (kmeans.labels_ == open_cluster_idx).astype(int)


def write_rows(output_csv: Path, rows: list[dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Threshold",
        "TP",
        "TN",
        "FP",
        "FN",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_rows(rows: list[dict[str, object]]) -> None:
    headers = ["Threshold", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1-Score"]
    print(" ".join(h.ljust(10) for h in headers))
    for row in rows:
        print(" ".join(str(row[h]).ljust(10) for h in headers))


def run(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    mar_values = load_mar_values(input_csv)
    y_true = build_pseudo_labels(mar_values)

    rows = []
    for threshold in threshold_range(args.threshold_start, args.threshold_end, args.threshold_step):
        y_pred = (mar_values > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        rows.append(
            {
                "Threshold": f"{threshold:.3f}",
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "Accuracy": f"{acc:.2%}",
                "Precision": f"{pre:.3f}",
                "Recall": f"{rec:.3f}",
                "F1-Score": f"{f1:.3f}",
            }
        )

    write_rows(output_csv, rows)
    print_rows(rows)
    print(f"\nSaved: {output_csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

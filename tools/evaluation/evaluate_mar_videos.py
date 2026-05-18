"""Evaluate MAR-based yawn detection against frame-level labels."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
MOUTH = [61, 291, 13, 14, 17, 78, 308]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sliding-window MAR evaluation on labeled yawn videos.",
    )
    parser.add_argument(
        "--dataset-path",
        default="Video Database/Yawn",
        help="Folder containing videos to evaluate.",
    )
    parser.add_argument(
        "--label-path",
        default="Video Database/Yawn/Labels",
        help="Folder containing *_labels.csv ground truth files.",
    )
    parser.add_argument(
        "--report-file",
        default="final_evaluation_report.csv",
        help="Output CSV file to append evaluation rows.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=150,
        help="Sliding MAR window size (frames).",
    )
    parser.add_argument(
        "--threshold-factor",
        type=float,
        default=0.7,
        help="Dynamic threshold factor applied to max(MAR window).",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.15,
        help="Lower bound for dynamic threshold.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show real-time preview window during evaluation.",
    )
    return parser


def mouth_aspect_ratio(mouth_points: list[tuple[int, int]]) -> float:
    vertical = distance.euclidean(mouth_points[2], mouth_points[3])
    horizontal = distance.euclidean(mouth_points[0], mouth_points[1])
    return vertical / horizontal if horizontal != 0 else 0.0


def load_ground_truth(label_path: Path, video_name: str) -> Optional[list[int]]:
    label_file = label_path / f"{video_name}_labels.csv"
    if not label_file.exists():
        return None

    labels: list[int] = []
    with label_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
    return labels


def save_report_row(report_file: Path, row: list[object]) -> None:
    exists = report_file.exists()
    with report_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "Timestamp",
                    "Video Name",
                    "Method",
                    "Avg Threshold",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                    "Accuracy",
                    "F1-Score",
                ]
            )
        writer.writerow(row)


def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset_path)
    label_path = Path(args.label_path)
    report_file = Path(args.report_file)

    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        return 2
    if not label_path.exists():
        print(f"Label path not found: {label_path}")
        return 2

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    video_files = [
        p for p in sorted(dataset_path.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    print(f"Starting MAR evaluation for {len(video_files)} videos...")

    for video_file in video_files:
        y_true = load_ground_truth(label_path, video_file.name)
        if y_true is None:
            print(f"Skip {video_file.name}: no labels found.")
            continue

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Skip {video_file.name}: cannot open video.")
            continue

        mar_window = deque(maxlen=args.window_size)
        y_pred: list[int] = []
        threshold_history: list[float] = []

        print(f"Evaluating {video_file.name} ...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                mouth_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH]
                mar = mouth_aspect_ratio(mouth_pts)
                mar_window.append(mar)

                if len(mar_window) >= 10:
                    dynamic_threshold = max(float(np.max(mar_window)) * args.threshold_factor, args.min_threshold)
                else:
                    dynamic_threshold = 0.3

                threshold_history.append(dynamic_threshold)
                prediction = 1 if mar >= dynamic_threshold else 0
                y_pred.append(prediction)

                if args.preview:
                    color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                    cv2.putText(
                        frame,
                        f"MAR: {mar:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Dynamic Thresh: {dynamic_threshold:.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                    cv2.imshow("MAR Evaluation", frame)

            if args.preview and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

        if not y_pred:
            print(f"No usable predictions for {video_file.name}.")
            continue

        min_len = min(len(y_pred), len(y_true))
        y_pred_sync = y_pred[:min_len]
        y_true_sync = y_true[:min_len]

        tn, fp, fn, tp = confusion_matrix(y_true_sync, y_pred_sync, labels=[0, 1]).ravel()
        acc = accuracy_score(y_true_sync, y_pred_sync)
        f1 = f1_score(y_true_sync, y_pred_sync, zero_division=0)
        avg_thresh = float(np.mean(threshold_history)) if threshold_history else 0.0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_report_row(
            report_file,
            [
                timestamp,
                video_file.name,
                "Sliding Window",
                f"{avg_thresh:.4f}",
                tp,
                tn,
                fp,
                fn,
                f"{acc:.4f}",
                f"{f1:.4f}",
            ],
        )
        print(f"Done {video_file.name}: accuracy={acc:.4f}, f1={f1:.4f}")

    if args.preview:
        cv2.destroyAllWindows()

    print(f"Evaluation complete. Report saved to: {report_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

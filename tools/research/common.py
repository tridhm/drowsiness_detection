from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable


FRAME_FEATURE_COLUMNS = [
    "subject_id",
    "session_id",
    "video_id",
    "frame_index",
    "timestamp_sec",
    "face_detected",
    "ear",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "eye_closed",
    "mouth_open",
    "head_nod_detected",
    "perclos_60s",
    "perclos_5s",
    "blink_frequency",
    "yawn_frequency",
    "pitch_velocity",
    "gaze_stable",
    "fsm_state",
    "fsm_evidence",
    "fsm_reasons",
]

WINDOW_FEATURE_COLUMNS = [
    "subject_id",
    "session_id",
    "video_id",
    "window_start_sec",
    "window_end_sec",
    "frame_count",
    "valid_face_ratio",
    "mean_ear",
    "min_ear",
    "mean_mar",
    "max_mar",
    "perclos_60s",
    "perclos_5s",
    "max_eye_closed_duration_sec",
    "blink_rate_per_min",
    "yawn_count",
    "head_drop_count",
    "max_pitch_velocity",
    "mean_fsm_evidence",
    "max_fsm_evidence",
    "fsm_state_mode",
]

KSS_COLUMNS = [
    "subject_id",
    "session_id",
    "video_id",
    "start_time_sec",
    "end_time_sec",
    "kss_score",
    "kss_band",
    "notes",
]

EYE_FEATURES = [
    "mean_ear",
    "min_ear",
    "perclos_60s",
    "perclos_5s",
    "max_eye_closed_duration_sec",
    "blink_rate_per_min",
]
PERCLOS_FEATURES = ["perclos_60s", "perclos_5s"]
YAWN_FEATURES = ["mean_mar", "max_mar", "yawn_count"]
HEAD_POSE_FEATURES = ["head_drop_count", "max_pitch_velocity"]
FSM_FEATURES = ["mean_fsm_evidence", "max_fsm_evidence", "fsm_state_mode"]
FULL_FEATURES = EYE_FEATURES + YAWN_FEATURES + HEAD_POSE_FEATURES + FSM_FEATURES

FSM_STATE_ENCODING = {
    "ALERT": 0.0,
    "SUSPICIOUS": 1.0,
    "DROWSY": 2.0,
    "CRITICAL": 3.0,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in fieldnames})


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return value


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    return int(round(as_float(row, key, float(default))))


def as_bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key, "")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def kss_is_sleepy(row: dict[str, Any]) -> bool:
    band = str(row.get("kss_band", "")).strip().lower()
    if band:
        return band == "sleepy"
    return as_float(row, "kss_score", 0.0) >= 7.0


def fsm_state_is_sleepy(value: Any) -> bool:
    state = str(value).strip().upper()
    return state in {"DROWSY", "CRITICAL"}


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    total = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support": total,
    }


def aggregate_frame_rows(
    frame_rows: list[dict[str, Any]],
    window_seconds: float,
    stride_seconds: float,
) -> list[dict[str, Any]]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be > 0")
    if not frame_rows:
        return []

    rows = sorted(frame_rows, key=lambda row: as_float(row, "timestamp_sec"))
    timestamps = [as_float(row, "timestamp_sec") for row in rows]
    start = timestamps[0]
    end = timestamps[-1]
    sample_interval = _estimate_sample_interval(timestamps)
    windows: list[dict[str, Any]] = []

    window_start = start
    while window_start <= end + 1e-9:
        window_end = window_start + window_seconds
        members = [row for row in rows if window_start <= as_float(row, "timestamp_sec") < window_end]
        if len(members) >= 2:
            windows.append(_aggregate_one_window(members, window_start, window_end, sample_interval))
        window_start += stride_seconds
    return windows


def _estimate_sample_interval(timestamps: list[float]) -> float:
    diffs = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not diffs:
        return 0.0
    return float(median(diffs))


def _aggregate_one_window(
    rows: list[dict[str, Any]],
    window_start: float,
    window_end: float,
    sample_interval: float,
) -> dict[str, Any]:
    first = rows[0]
    timestamps = [as_float(row, "timestamp_sec") for row in rows]
    observed_seconds = max(timestamps[-1] - timestamps[0], sample_interval, 1e-9)

    ear_values = [as_float(row, "ear") for row in rows]
    mar_values = [as_float(row, "mar") for row in rows]
    evidence_values = [as_float(row, "fsm_evidence") for row in rows]
    pitch_velocity_values = [abs(as_float(row, "pitch_velocity")) for row in rows]
    states = [str(row.get("fsm_state", "")) for row in rows]

    blink_count = _rolling_count_delta(rows, "blink_frequency")
    yawn_count = _rolling_count_delta(rows, "yawn_frequency")

    return {
        "subject_id": first.get("subject_id", ""),
        "session_id": first.get("session_id", ""),
        "video_id": first.get("video_id", ""),
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "frame_count": len(rows),
        "valid_face_ratio": sum(1 for row in rows if as_bool(row, "face_detected")) / len(rows),
        "mean_ear": _mean(ear_values),
        "min_ear": min(ear_values) if ear_values else 0.0,
        "mean_mar": _mean(mar_values),
        "max_mar": max(mar_values) if mar_values else 0.0,
        "perclos_60s": as_float(rows[-1], "perclos_60s"),
        "perclos_5s": as_float(rows[-1], "perclos_5s"),
        "max_eye_closed_duration_sec": _max_consecutive_true_seconds(rows, "eye_closed", sample_interval),
        "blink_rate_per_min": (blink_count / observed_seconds) * 60.0,
        "yawn_count": yawn_count,
        "head_drop_count": sum(1 for row in rows if as_bool(row, "head_nod_detected")),
        "max_pitch_velocity": max(pitch_velocity_values) if pitch_velocity_values else 0.0,
        "mean_fsm_evidence": _mean(evidence_values),
        "max_fsm_evidence": max(evidence_values) if evidence_values else 0.0,
        "fsm_state_mode": _mode(states),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mode(values: list[str]) -> str:
    clean_values = [value for value in values if value]
    if not clean_values:
        return ""
    counts = Counter(clean_values)
    severity = {"ALERT": 0, "SUSPICIOUS": 1, "DROWSY": 2, "CRITICAL": 3}
    return max(counts, key=lambda value: (counts[value], severity.get(value.upper(), 0)))


def _rolling_count_delta(rows: list[dict[str, Any]], key: str) -> int:
    values = [as_int(row, key) for row in rows]
    if not values:
        return 0
    return max(0, max(values) - min(values))


def _max_consecutive_true_seconds(rows: list[dict[str, Any]], key: str, sample_interval: float) -> float:
    if sample_interval <= 0:
        sample_interval = 1.0
    longest = 0
    current = 0
    for row in rows:
        if as_bool(row, key):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest * sample_interval


def numeric_feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append([numeric_feature_value(row, feature) for feature in feature_names])
    return matrix


def numeric_feature_value(row: dict[str, Any], feature: str) -> float:
    if feature == "fsm_state_mode":
        return FSM_STATE_ENCODING.get(str(row.get(feature, "")).strip().upper(), 0.0)
    return as_float(row, feature)


def available_features(rows: list[dict[str, Any]], requested: list[str]) -> list[str]:
    if not rows:
        return []
    keys = set(rows[0].keys())
    return [feature for feature in requested if feature in keys]

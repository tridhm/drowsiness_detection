from __future__ import annotations

import argparse
from pathlib import Path

import cv2

try:
    from runtime.config import load_runtime_config
    from runtime.contracts import EngineContext
    from runtime.engines.registry import available_engines, create_engine
    from runtime.features import SignalFeaturePipeline
    from runtime.perception import PerceptionExtractor
    from tools.research.common import FRAME_FEATURE_COLUMNS, WINDOW_FEATURE_COLUMNS, aggregate_frame_rows, write_csv_rows
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from runtime.config import load_runtime_config
    from runtime.contracts import EngineContext
    from runtime.engines.registry import available_engines, create_engine
    from runtime.features import SignalFeaturePipeline
    from runtime.perception import PerceptionExtractor
    from tools.research.common import FRAME_FEATURE_COLUMNS, WINDOW_FEATURE_COLUMNS, aggregate_frame_rows, write_csv_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export per-frame and per-window drowsiness research features from a video.")
    parser.add_argument("--video-path", required=True, help="Input video file path.")
    parser.add_argument("--subject-id", required=True, help="Subject identifier for exported rows.")
    parser.add_argument("--session-id", required=True, help="Session identifier for exported rows.")
    parser.add_argument("--video-id", required=True, help="Video identifier for exported rows.")
    parser.add_argument("--frame-csv", required=True, help="Output frame_features.csv path.")
    parser.add_argument("--window-csv", required=True, help="Output window_features.csv path.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Window length in seconds.")
    parser.add_argument("--stride-seconds", type=float, default=10.0, help="Window stride in seconds.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process; 0 means all frames.")
    parser.add_argument("--config", default=None, help="Optional runtime config JSON path.")
    parser.add_argument(
        "--decision-engine",
        choices=available_engines(),
        default=None,
        help="Optional decision engine override; defaults to runtime config.",
    )
    return parser


def export_video_features(args: argparse.Namespace) -> tuple[int, int]:
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")
    if args.stride_seconds <= 0:
        raise ValueError("--stride-seconds must be > 0")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")

    config = load_runtime_config(args.config, {"source": "file", "video_path": args.video_path})
    if args.decision_engine:
        config.decision_engine = args.decision_engine

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {args.video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or config.runtime.fps or 30.0)
    if fps <= 0:
        fps = 30.0
    config.runtime.fps = fps

    perception = PerceptionExtractor()
    features = SignalFeaturePipeline(config)
    engine = create_engine(config.decision_engine, config)
    engine.initialize(EngineContext(fps=fps, metadata={"mode": "offline_export"}))

    frame_rows = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break
            timestamp_sec = frame_index / fps
            raw = perception.process(frame)
            signals, debug = features.update(raw, now=timestamp_sec)
            decision = engine.update(signals)
            reasons = ";".join(decision.reasons)

            frame_rows.append(
                {
                    "subject_id": args.subject_id,
                    "session_id": args.session_id,
                    "video_id": args.video_id,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "face_detected": int(raw.face_detected),
                    "ear": raw.ear,
                    "mar": raw.mar,
                    "pitch": raw.pitch,
                    "yaw": raw.yaw,
                    "roll": raw.roll,
                    "eye_closed": int(signals.ear_below_threshold),
                    "mouth_open": int(signals.mar_above_threshold),
                    "head_nod_detected": int(signals.head_nod_detected),
                    "perclos_60s": signals.perclos,
                    "perclos_5s": signals.perclos_short,
                    "blink_frequency": signals.blink_frequency,
                    "yawn_frequency": signals.yawn_frequency,
                    "pitch_velocity": signals.pitch_velocity,
                    "gaze_stable": int(signals.gaze_stable),
                    "fsm_state": decision.state.value,
                    "fsm_evidence": decision.evidence,
                    "fsm_reasons": reasons,
                }
            )
            frame_index += 1
    finally:
        cap.release()
        perception.close()

    window_rows = aggregate_frame_rows(frame_rows, args.window_seconds, args.stride_seconds)
    write_csv_rows(Path(args.frame_csv), FRAME_FEATURE_COLUMNS, frame_rows)
    write_csv_rows(Path(args.window_csv), WINDOW_FEATURE_COLUMNS, window_rows)
    return len(frame_rows), len(window_rows)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        frame_count, window_count = export_video_features(args)
        print(f"Wrote {frame_count} frame row(s) -> {args.frame_csv}")
        print(f"Wrote {window_count} window row(s) -> {args.window_csv}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

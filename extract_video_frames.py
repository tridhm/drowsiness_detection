"""
Extract frames from a video folder for downstream training workflows.
"""

import argparse
from pathlib import Path

import cv2

from cli_json_config import parse_args_with_json_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract sampled frames from videos into labeled output folders.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to shared JSON config file (section: extract_video_frames).",
    )
    parser.add_argument(
        "--video-dir",
        default="Video Database",
        help="Input folder containing source videos.",
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_video_frames",
        help="Output root folder for extracted frames.",
    )
    parser.add_argument(
        "--glob",
        default="*.avi",
        help="Glob pattern for input videos (example: *.avi, *.mp4).",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Recursively search videos under --video-dir.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Search only top-level files under --video-dir.",
    )
    parser.set_defaults(recursive=False)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Frames sampled per second from each video.",
    )
    parser.add_argument(
        "--label-folder",
        default="alert",
        help="Output subfolder label to store extracted frames (default: alert).",
    )
    return parser


def _iter_videos(video_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(path for path in video_dir.rglob(pattern) if path.is_file())
    return sorted(path for path in video_dir.glob(pattern) if path.is_file())


def extract_frames_from_videos(args: argparse.Namespace) -> int:
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be > 0.")

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    label_dir = output_dir / args.label_folder
    drowsy_dir = output_dir / "drowsy"

    if not video_dir.exists():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")

    label_dir.mkdir(parents=True, exist_ok=True)
    drowsy_dir.mkdir(parents=True, exist_ok=True)

    video_files = _iter_videos(video_dir, args.glob, args.recursive)
    print(f"📹 Found {len(video_files)} video files in {video_dir}")

    for video_path in video_files:
        print(f"\n🎬 Processing: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ❌ Cannot open {video_path.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 30.0
        sample_rate = max(1, int(round(fps / args.sample_fps)))

        print(f"  Total frames: {total_frames}, FPS: {fps:.2f}, sample_rate: every {sample_rate} frame(s)")

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % sample_rate == 0:
                filename = f"{video_path.stem}_frame_{frame_count:06d}.jpg"
                output_path = label_dir / filename
                cv2.imwrite(str(output_path), frame)
                saved_count += 1

        cap.release()
        print(f"  ✅ Saved {saved_count} frames to {label_dir}")

    print("\n✅ Extraction complete!")
    print(f"📁 Frames saved in: {output_dir}")
    print("⚠️  Note: Manual labeling may be needed to separate drowsy/alert frames.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_json_config(parser, argv, section="extract_video_frames")
    try:
        return extract_frames_from_videos(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

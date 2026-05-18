import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from cli_json_config import parse_args_with_json_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract EAR/MAR/head-pose features from video and append to training CSV.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to shared JSON config file (section: train_data).",
    )
    parser.add_argument(
        "--video-source",
        default="video_buon_ngu.mp4",
        help="Input video file path.",
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        default=1,
        help="Label for all extracted rows: 0=alert, 1=drowsy.",
    )
    parser.add_argument(
        "--csv-file",
        default="drowsiness_data.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--predictor-path",
        default="models/shape_predictor_68_face_landmarks.dat",
        help="Path to dlib 68-point landmark predictor file.",
    )
    parser.add_argument("--frame-width", type=int, default=450, help="Resize frame width before processing.")
    parser.add_argument(
        "--detector-upsamples",
        type=int,
        default=0,
        help="dlib face detector upsample times.",
    )
    parser.add_argument("--show-window", dest="show_window", action="store_true", help="Display preview window while extracting.")
    parser.add_argument("--no-show-window", dest="show_window", action="store_false", help="Disable preview window.")
    parser.add_argument("--quit-key", default="q", help="Quit key when --show-window is enabled.")
    parser.add_argument("--write-header", dest="write_header", action="store_true", help="Write CSV header when starting file.")
    parser.add_argument("--no-write-header", dest="write_header", action="store_false", help="Do not write CSV header.")
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite CSV file instead of appending.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Append to CSV file instead of overwriting.",
    )
    parser.set_defaults(show_window=False, write_header=False, overwrite=False)
    return parser


def eye_aspect_ratio(eye, distance_module):
    a = distance_module.euclidean(eye[1], eye[5])
    b = distance_module.euclidean(eye[2], eye[4])
    c = distance_module.euclidean(eye[0], eye[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(mouth, distance_module):
    a = distance_module.euclidean(mouth[13], mouth[19])
    b = distance_module.euclidean(mouth[14], mouth[18])
    c = distance_module.euclidean(mouth[15], mouth[17])
    d = distance_module.euclidean(mouth[12], mouth[16])
    if d == 0:
        return 0.0
    return (a + b + c) / (3.0 * d)


def get_head_pose(shape, frame_shape):
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ]
    )

    image_points = np.array(
        [
            shape[30],
            shape[8],
            shape[36],
            shape[45],
            shape[48],
            shape[54],
        ],
        dtype="double",
    )

    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _mtx_r, _mtx_q, _qx, _qy, _qz = cv2.RQDecomp3x3(rotation_matrix)
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]
    return pitch, yaw, roll


def run(args: argparse.Namespace) -> int:
    import dlib
    import imutils
    from imutils import face_utils
    from scipy.spatial import distance

    predictor_path = Path(args.predictor_path)
    if not predictor_path.exists():
        raise FileNotFoundError(f"Predictor file not found: {predictor_path}")
    if args.frame_width <= 0:
        raise ValueError("--frame-width must be > 0.")
    if args.detector_upsamples < 0:
        raise ValueError("--detector-upsamples must be >= 0.")
    if not args.quit_key:
        raise ValueError("--quit-key cannot be empty.")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))

    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    (m_start, m_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

    csv_file = Path(args.csv_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_file.exists()
    mode = "w" if args.overwrite else "a"

    cap = cv2.VideoCapture(args.video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.video_source}")

    quit_key_code = ord(args.quit_key[0].lower())
    row_count = 0

    try:
        with csv_file.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            should_write_header = args.write_header and (mode == "w" or not file_exists)
            if should_write_header:
                writer.writerow(["ear", "mar", "pitch", "yaw", "roll", "label"])

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = imutils.resize(frame, width=args.frame_width)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_shape = frame.shape

                subjects = detector(gray, args.detector_upsamples)

                for subject in subjects:
                    shape = predictor(gray, subject)
                    shape_np = face_utils.shape_to_np(shape)

                    left_eye = shape_np[l_start:l_end]
                    right_eye = shape_np[r_start:r_end]
                    left_ear = eye_aspect_ratio(left_eye, distance)
                    right_ear = eye_aspect_ratio(right_eye, distance)
                    ear = (left_ear + right_ear) / 2.0

                    mouth = shape_np[m_start:m_end]
                    mar = mouth_aspect_ratio(mouth, distance)

                    pitch, yaw, roll = get_head_pose(shape_np, frame_shape)

                    print(
                        f"EAR: {ear:.2f}, MAR: {mar:.2f}, Pitch: {pitch:.2f}, "
                        f"Yaw: {yaw:.2f}, Label: {args.label}"
                    )
                    writer.writerow([ear, mar, pitch, yaw, roll, args.label])
                    row_count += 1

                if args.show_window:
                    cv2.imshow("Feature Extraction", frame)
                    if cv2.waitKey(1) & 0xFF == quit_key_code:
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Đã trích xuất xong dữ liệu từ {args.video_source}")
    print(f"Đã ghi {row_count} dòng vào {csv_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_json_config(parser, argv, section="train_data")
    try:
        return run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

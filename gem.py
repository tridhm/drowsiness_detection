import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import deque
import glob
import os
import argparse

class DynamicEAR:
    def __init__(self, window_size=150, alpha=0.3, k_low=2.5, k_high=1.5):
        self.window = deque(maxlen=window_size)
        self.alpha = alpha
        self.k_low = k_low
        self.k_high = k_high
        self.ewma = None
        self.is_closed = False

    def update(self, current_ear):
        if self.ewma is None:
            self.ewma = current_ear
        else:
            self.ewma = self.alpha * current_ear + (1 - self.alpha) * self.ewma
            
        self.window.append(self.ewma)
        
        if len(self.window) < 30:
            self.is_closed = self.ewma < 0.20
            return self.is_closed, self.ewma, 0.20
            
        window_arr = np.array(self.window)
        median_ear = np.median(window_arr)
        mad = np.median(np.abs(window_arr - median_ear))
        mad = max(mad, 0.01) 

        T_low = median_ear - self.k_low * mad
        T_high = median_ear - self.k_high * mad

        if not self.is_closed:
            if self.ewma < T_low:
                self.is_closed = True
        else:
            if self.ewma > T_high:
                self.is_closed = False

        return self.is_closed, self.ewma, T_low

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def extract_closed_eye_frames(
    video_path,
    output_dir,
    min_closed_seconds=1.0,
    padding_seconds=0.5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """
    Quet video va xuat cac khung hinh (frames) co nguoi nham mat thanh anh tinh.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Khong the mo duoc file video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Gia tri mac dinh du phong neu video bi loi header khong doc duoc fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    mp_face_mesh = mp.solutions.face_mesh
    dynamic_ear = DynamicEAR(window_size=int(fps*3), alpha=0.3, k_low=2.5, k_high=1.5)
    
    # Tao thu muc con rieng cho tung video dua tren ten video
    video_stem = Path(video_path).stem
    out_dir_path = Path(output_dir) / video_stem
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"   -> Dang phan tich (FPS: {fps:.1f}, Total frames: {total_frames})...")
    
    # --- BUOC 1: QUET TIM CAC DOAN NHAM MAT ---
    frame_count = 0
    is_currently_closed = False
    start_closed_frame = 0
    closed_segments = []
    
    detected_face_frames = 0
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                detected_face_frames += 1
                landmarks = results.multi_face_landmarks[0].landmark
                points = np.array([[(int(l.x * width), int(l.y * height))] for l in landmarks])
                
                left_pts = points[LEFT_EYE].squeeze()
                right_pts = points[RIGHT_EYE].squeeze()
                ear_raw = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0
                
                is_closed, smoothed_ear, _ = dynamic_ear.update(ear_raw)
                
                if is_closed and not is_currently_closed:
                    is_currently_closed = True
                    start_closed_frame = frame_count
                
                elif not is_closed and is_currently_closed:
                    is_currently_closed = False
                    end_closed_frame = frame_count
                    duration = (end_closed_frame - start_closed_frame) / fps
                    
                    if duration >= min_closed_seconds:
                        closed_segments.append((start_closed_frame, end_closed_frame))

    if is_currently_closed:
        duration = (frame_count - start_closed_frame) / fps
        if duration >= min_closed_seconds:
            closed_segments.append((start_closed_frame, frame_count))

    if not closed_segments:
        detect_ratio = detected_face_frames / frame_count if frame_count > 0 else 0.0
        print(f"   -> Face-detect ratio: {detected_face_frames}/{frame_count} ({detect_ratio:.1%})")
        if detected_face_frames == 0:
            print("   [WARN] Khong phat hien khuon mat nao trong video nay (co the do goc quay/anh sang/video khong co mat).")
        print("   -> Khong tim thay doan nham mat nao du dai trong video nay.")
        cap.release()
        return {"saved_frames": 0, "closed_segments": 0, "face_frames": detected_face_frames, "total_frames": frame_count}

    print(f"   -> TIM THAY {len(closed_segments)} chuoi nham mat. Dang tien hanh cat frame...")

    # --- BUOC 2: CAT VA LUU TUNG FRAME ---
    pad_frames = int(padding_seconds * fps)
    saved_frames_count = 0
    
    for idx, (start_f, end_f) in enumerate(closed_segments):
        cut_start = max(0, start_f - pad_frames)
        cut_end = min(total_frames, end_f + pad_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, cut_start)
        
        current_f = cut_start
        while current_f <= cut_end:
            ret, frame = cap.read()
            if not ret: break
            
            output_filename = out_dir_path / f"clip_{idx+1}_frame_{current_f}.jpg"
            # Luu truc tiep anh ra o cung thay vi VideoWriter
            cv2.imwrite(str(output_filename), frame) 
            
            current_f += 1
            saved_frames_count += 1

    detect_ratio = detected_face_frames / frame_count if frame_count > 0 else 0.0
    print(f"   -> Face-detect ratio: {detected_face_frames}/{frame_count} ({detect_ratio:.1%})")
    print(f"   [OK] Xong! Da luu {saved_frames_count} frames vao: {out_dir_path.name}/")
    cap.release()
    return {
        "saved_frames": saved_frames_count,
        "closed_segments": len(closed_segments),
        "face_frames": detected_face_frames,
        "total_frames": frame_count,
    }


def discover_videos(input_folder, recursive=True):
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    videos = []
    pattern_type = "**" if recursive else ""
    for ext in video_extensions:
        lower_pattern = os.path.join(input_folder, pattern_type, ext) if pattern_type else os.path.join(input_folder, ext)
        upper_pattern = os.path.join(input_folder, pattern_type, ext.upper()) if pattern_type else os.path.join(input_folder, ext.upper())
        videos.extend(glob.glob(lower_pattern, recursive=recursive))
        videos.extend(glob.glob(upper_pattern, recursive=recursive))
    return sorted(set(videos))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trich xuat frame nham mat tu video bang MediaPipe FaceMesh + EAR dong."
    )
    parser.add_argument(
        "--input-folder",
        default=r"D:\Pj\Car\cam\Thenho2\dashcam_output\rear2",
        help="Thu muc chua video dau vao (mac dinh: D:\\Pj\\Car\\cam\\Thenho2\\dashcam_output\\rear2).",
    )
    parser.add_argument(
        "--output-directory",
        default="Extracted_Closed_Eye_Frames",
        help="Thu muc output de luu frame da cat.",
    )
    parser.add_argument(
        "--min-closed-seconds",
        type=float,
        default=1.0,
        help="Thoi luong nham mat toi thieu de giu segment (giay).",
    )
    parser.add_argument(
        "--padding-seconds",
        type=float,
        default=0.5,
        help="So giay mo rong truoc/sau segment nham mat.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Nguong phat hien mat toi thieu cho MediaPipe FaceMesh.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Nguong tracking mat toi thieu cho MediaPipe FaceMesh.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Chi quet video o thu muc goc, khong quet thu muc con.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Gioi han so video can chay (0 = khong gioi han).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    input_folder = args.input_folder
    output_directory = args.output_directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    recursive = not args.non_recursive
    videos = discover_videos(input_folder, recursive=recursive)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    print(f"[INFO] Da quet va tim thay {len(videos)} video can xu ly trong thu muc.")
    if not videos:
        print("   [WARN] Khong tim thay video nao. Kiem tra lai --input-folder va dinh dang file.")
        raise SystemExit(1)

    success_count = 0
    error_count = 0
    total_saved = 0
    total_face_frames = 0
    total_frames = 0
    
    for i, vid in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Xu ly: {Path(vid).name}")
        try:
            result = extract_closed_eye_frames(
                vid,
                output_directory,
                min_closed_seconds=args.min_closed_seconds,
                padding_seconds=args.padding_seconds,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
            )
            success_count += 1
            if result:
                total_saved += result["saved_frames"]
                total_face_frames += result["face_frames"]
                total_frames += result["total_frames"]
        except Exception as e:
            print("   [ERROR] Loi voi video nay! Bo qua va chuyen sang video tiep theo.")
            print(f"   Chi tiet loi: {e}")
            error_count += 1

    face_ratio = total_face_frames / total_frames if total_frames > 0 else 0.0
    print("\n" + "="*50)
    print("BATCH XU LY VIDEO DA HOAN TAT!")
    print(f"[OK] Thanh cong: {success_count} video")
    print(f"[ERROR] That bai/Bo qua: {error_count} video")
    print(f"[INFO] Face-detect tong: {total_face_frames}/{total_frames} ({face_ratio:.1%})")
    print(f"[INFO] Tong frame da luu: {total_saved}")
    print(f"[INFO] Frame anh tinh da duoc phan loai vao thu muc: {output_directory}")
    print("="*50)

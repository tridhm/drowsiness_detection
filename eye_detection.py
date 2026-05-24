import cv2
import numpy as np
import math
import time
from collections import deque

def eye_aspect_ratio(eye):
    """
    Tính toán chỉ số EAR hình học kinh điển cho mắt.
    Minh chứng khoa học: Soukupová & Čech (2016)
    URL: https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
    """
    A = math.hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    B = math.hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    C = math.hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


class DynamicEAR:
    def __init__(self, window_size=150, alpha=0.3, k_low=4.5, k_high=1.0):
        """
        Ước lượng nền cá nhân hóa và quản lý trạng thái mắt bằng cặp ngưỡng trễ.
        Minh chứng khoa học: Lucas & Saccucci (1990), Leys et al. (2013), Akinshin (2022).
        """
        self.window = deque(maxlen=window_size)
        self.alpha = alpha
        self.k_low = k_low
        self.k_high = k_high
        self.ewma = None
        self.is_closed = False
        self.dynamic_low = 0.25
        self.threshold_locked = False
        self.locked_T_low = None

    def lock_threshold(self):
        """Khóa ngưỡng sau giai đoạn hiệu chuẩn (Calibration Lock)"""
        if len(self.window) >= 150:
            window_arr = np.array(self.window)
            median_ear = np.median(window_arr)
            mad = np.median(np.abs(window_arr - median_ear))
            mad = max(mad, 0.02)

            self.locked_T_low = median_ear - self.k_low * mad
            self.threshold_locked = True
            print(f"[SUCCESS] Threshold LOCKED at T_low = {self.locked_T_low:.4f}")
            return True
        return False

    def update(self, current_ear):
        """Chuẩn xử lý tín hiệu smoothing + Robust Statistics tính ngưỡng động thấp/cao"""
        if self.ewma is None:
            self.ewma = current_ear
        else:
            self.ewma = self.alpha * current_ear + (1 - self.alpha) * self.ewma

        self.window.append(self.ewma)

        if len(self.window) < 30:
            self.is_closed = self.ewma < 0.25
            self.dynamic_low = 0.25
            return self.is_closed, self.ewma, self.dynamic_low, 0.25, 0.02

        window_arr = np.array(self.window)
        median_ear = np.median(window_arr)
        mad = np.median(np.abs(window_arr - median_ear))
        mad = max(mad, 0.02) # Sàn nhiễu MAD tránh chia cho 0 hoặc lỗi tịnh tiến

        if self.threshold_locked:
            T_low = self.locked_T_low
            T_high = self.locked_T_low + 0.05
        else:
            T_low = median_ear - self.k_low * mad
            T_high = median_ear - self.k_high * mad
            T_low = min(T_low, 0.25)
            T_high = min(T_high, 0.25)

        self.dynamic_low = T_low

        # Thuật toán chống nhấp nháy trạng thái Hysteresis (Schmitt Trigger)
        if not self.is_closed:
            if self.ewma < T_low:
                self.is_closed = True
        else:
            if self.ewma > T_high:
                self.is_closed = False

        return self.is_closed, self.ewma, T_low, median_ear, mad


class BlinkTracker:
    def __init__(self, earm_thresh=0.18, history_size=13, blink_cooldown_frames=6, time_window=10.0):
        """
        Theo dõi và đếm cú nháy mắt bằng Heuristic EARM 13-frame liên tiếp.
        Minh chứng khoa học: Soukupová & Čech (2016) - Mục 2.2 Classification.
        """
        self.earm_thresh = earm_thresh
        self.history = deque(maxlen=history_size)
        self.blink_timestamps = deque(maxlen=200)
        self.blink_cooldown = 0
        self.blink_cooldown_frames = blink_cooldown_frames
        self.time_window = time_window

    def update(self, ear, now=None):
        """Tính toán giá trị EARM cục bộ từ cửa sổ trượt 13-frame"""
        if now is None:
            now = time.time()

        self.history.append(ear)
        blink_event = False
        earm = None

        if len(self.history) == self.history.maxlen:
            # Lấy các giá trị frame biên theo công thức vi phân đồ thị hình thung lũng của Soukupová
            ear_t_minus_6 = self.history[0]
            ear_t_minus_5 = self.history[1]
            ear_t = self.history[6]        # Frame trung tâm (đỉnh nhắm sâu nhất)
            ear_t_plus_5 = self.history[11]
            ear_t_plus_6 = self.history[12]

            # Công thức EARM chính xác của Heuristic thực nghiệm
            earm = (ear_t_minus_6 + ear_t_minus_5 + ear_t_plus_5 + ear_t_plus_6) - (4 * ear_t)

            if earm > self.earm_thresh and self.blink_cooldown == 0:
                self.blink_timestamps.append(now)
                self.blink_cooldown = self.blink_cooldown_frames
                blink_event = True

            if self.blink_cooldown > 0:
                self.blink_cooldown -= 1

        return blink_event, earm

    def get_blink_rate(self, now=None):
        """Heuristic từ thống kê hành vi: Tính toán số lượng blink trượt trên cửa sổ thời gian"""
        if now is None:
            now = time.time()
        # Loại bỏ và dọn dẹp các mốc thời gian đã nằm ngoài cửa sổ quan sát (ví dụ: quá 10 giây)
        while self.blink_timestamps and (now - self.blink_timestamps[0] > self.time_window):
            self.blink_timestamps.popleft()
        return len(self.blink_timestamps)


# =====================================================================
# HỆ THỐNG ĐIỀU PHỐI TRUNG TÂM (ĐẠT PHẢN XẠ PHÁT HIỆN BUỒN NGỦ < 1 GIÂY)
# =====================================================================
class DriverSafetySystem:
    def __init__(self, fps=30):
        self.fps = fps
        self.dynamic_ear_analyzer = DynamicEAR()
        self.blink_tracker = BlinkTracker(time_window=10.0)
        
        # Ngưỡng tối ưu phản xạ nhanh dưới 1 giây dựa trên báo cáo FAA (Stern et al., 1994)
        # 18 frames liên tục ở camera 30fps tương đương chính xác 600ms thời gian thực
        self.consecutive_closed_frames = 0
        self.drowsiness_frame_threshold = int(0.6 * self.fps) 

    def monitor(self, raw_ear, current_time=None):
        if current_time is None:
            current_time = time.time()

        # 1. Cập nhật trạng thái đóng mở mí mắt từ bộ lọc thích ứng
        is_closed, ewma_ear, T_low, median_ear, mad = self.dynamic_ear_analyzer.update(raw_ear)

        # 2. Đẩy luồng dữ liệu vào bộ đếm sự kiện nháy mắt độc lập EARM
        blink_triggered, earm_val = self.blink_tracker.update(ewma_ear, current_time)
        blink_rate = self.blink_tracker.get_blink_rate(current_time)

        # 3. Quản lý trạng thái tích lũy thời gian đóng mắt liên tục
        if is_closed:
            self.consecutive_closed_frames += 1
        else:
            self.consecutive_closed_frames = 0

        # 4. Kích hoạt tín hiệu cảnh báo cực tốc (Đảm bảo thời gian phản ứng ~0.6 giây)
        drowsiness_alert = self.consecutive_closed_frames >= self.drowsiness_frame_threshold

        return {
            "is_closed": is_closed,
            "drowsiness_alert": drowsiness_alert,
            "blink_triggered": blink_triggered,
            "blink_rate_last_10s": blink_rate,
            "ewma_ear": ewma_ear,
            "T_low": T_low,
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "earm_value": earm_val
        }

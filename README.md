# Hệ Thống Phát Hiện Buồn Ngủ Khi Lái Xe

`README.md` là tài liệu chính thức và đầy đủ nhất của dự án.
Thư mục `./docs` chỉ là tài liệu bổ sung (kiến trúc, sơ đồ, ledger).

Mục tiêu của README:
- Cài môi trường và chạy demo nhanh.
- Biết cách đưa dataset vào đúng chỗ để train/evaluate.
- Biết nên chạy script nào theo từng nhu cầu.

## Table of contents
- [Overview](#overview)
- [Get Started](#get-started)
- [Quickstart](#quickstart)
- [Dataset Setup (quan trọng)](#dataset-setup-quan-trọng)
- [Core Concepts](#core-concepts)
- [CLI / --help Reference (đầy đủ)](#cli---help-reference-đầy-đủ)
- [Runtime Guides](#runtime-guides)
- [Training Guides](#training-guides)
- [Evaluation Guides](#evaluation-guides)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [FAQ cho thành viên mới](#faq-cho-thành-viên-mới)
- [Architecture docs](#architecture-docs)
- [Merge and provenance docs](#merge-and-provenance-docs)
- [Contributing](#contributing)

## Overview
Codebase hiện có 2 luồng runtime:
- Runtime chính (khuyên dùng): `advanced_drowsiness_detection.py`
- Runtime baseline cũ: `drowsiness_detection_with_model.py`

Runtime chính đã được tách module để dễ bảo trì và dễ mở rộng:
- Input transport
- Perception (trích xuất tín hiệu)
- Feature pipeline (EMA, PERCLOS, tần suất chớp mắt/ngáp)
- Decision engine (`fsm` hoặc `legacy`)
- Alert policy

## Get Started
### 1) Yêu cầu tối thiểu
- Windows 10/11
- Python 3.12.x (chuẩn môi trường chính thức của project)
- Webcam (nếu chạy realtime webcam)
- GPU là tùy chọn, không bắt buộc

### 2) Tạo môi trường Python
Mở PowerShell tại thư mục repo:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Nếu bị chặn khi activate:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3) Cài thư viện
```powershell
pip install -r requirements.txt
```

Nếu bạn cần chạy script legacy có `dlib` (`train_data.py`), dùng:
```powershell
pip install -r requirements-legacy.txt
```

### 4) Kiểm tra nhanh môi trường
```powershell
python -c "import sys; print(sys.version)"
python -c "import cv2, mediapipe, torch; print('OK')"
```

Bạn cần thấy phiên bản `3.12.x` ở dòng đầu.

### 5) Tránh lỗi do nhiều Python trên cùng máy
Máy Windows thường có nhiều Python cùng lúc (ví dụ Python hệ thống + Python trong `venv`).
Để chắc chắn dùng đúng môi trường, luôn chạy theo mẫu:

```powershell
.\venv\Scripts\python.exe advanced_drowsiness_detection.py --help
```

Bạn có thể kiểm tra nhanh:
```powershell
where python
python -V
.\venv\Scripts\python.exe -V
```

## Quickstart
### Chạy nhanh runtime chính bằng webcam
```powershell
python advanced_drowsiness_detection.py
```

### Chạy runtime chính bằng file video
```powershell
python advanced_drowsiness_detection.py --source file --video-path "Video Database\Sub 03.avi"
```

### Đổi decision engine
```powershell
python advanced_drowsiness_detection.py --decision-engine fsm
python advanced_drowsiness_detection.py --decision-engine legacy
```

### Dùng config + override bằng CLI
```powershell
python advanced_drowsiness_detection.py --config config.json --source webcam
```

## Dataset Setup (quan trọng)
Repo này chỉ giữ `code + metadata nhẹ`, không kèm toàn bộ dataset nặng.  
Bạn cần tự đặt dataset vào đúng cấu trúc bên dưới để train/evaluate hoạt động ổn định.

### A) Dataset cho `train_eye_model_torch.py`
Script này yêu cầu thư mục:

```text
dataset_eyes&yawn/
└── train/
    ├── Closed/
    └── Open/
```

Lưu ý quan trọng:
- Tên thư mục bắt buộc đúng chữ hoa/chữ thường: `Closed`, `Open`.
- Ảnh nên là `.jpg` hoặc `.png`.
- Nếu sai tên thư mục, script sẽ báo lỗi class không tồn tại.

Tạo nhanh khung thư mục:
```powershell
New-Item -ItemType Directory -Force -Path ".\dataset_eyes&yawn\train\Closed"
New-Item -ItemType Directory -Force -Path ".\dataset_eyes&yawn\train\Open"
```

### B) Dataset cho `train_advanced_model.py`
Script này đọc đồng thời 3 nguồn dữ liệu (nếu có):

```text
CEW/
├── closed/
└── open/

mrleyedataset/
├── Close-Eyes/
└── Open-Eyes/

dataset_eyes&yawn/
└── train/
    ├── Closed/
    ├── Open/
    ├── yawn/
    └── no_yawn/
```

Lưu ý:
- `CEW/closed` và `CEW/open` đang có sẵn trong repo.
- `mrleyedataset` và `dataset_eyes&yawn` thường phải tự thêm ngoài.
- Nếu thiếu một nguồn, script vẫn có thể chạy nhưng ít dữ liệu hơn.

### C) Dataset video cho runtime file mode và evaluation

```text
Video Database/
├── Sub 03.avi
└── Yawn/
    ├── 1.mp4
    ├── 2.mp4
    └── ...
```

Nhãn yawn (`*_labels.csv`) có thể để ở 1 trong 2 nơi:
- Cách 1 (theo cấu trúc DA1 cũ): `Video Database/Yawn/Labels`
- Cách 2 (khuyên dùng trong codebase hiện tại): `tools/evaluation/metadata/yawn_labels`

### D) Nếu dataset nằm ngoài repo (khuyên dùng)
Bạn có thể để dataset ở ổ khác rồi truyền path qua CLI cho script nào hỗ trợ path:
- Runtime: `--video-path`
- Evaluation: `--dataset-path`, `--label-path`
- Train eye model: `--data-dir`

Nếu bạn muốn giữ dataset ngoài repo nhưng không phải truyền CLI mỗi lần, có thể tạo junction:
```powershell
New-Item -ItemType Junction -Path ".\mrleyedataset" -Target "D:\datasets\mrleyedataset"
New-Item -ItemType Junction -Path ".\dataset_eyes&yawn" -Target "D:\datasets\dataset_eyes&yawn"
New-Item -ItemType Junction -Path ".\Video Database" -Target "D:\datasets\Video Database"
```

### E) Kiểm tra dataset đã đặt đúng chưa
Chạy nhanh trong PowerShell:
```powershell
Test-Path ".\CEW\closed"
Test-Path ".\CEW\open"
Test-Path ".\dataset_eyes&yawn\train\Closed"
Test-Path ".\dataset_eyes&yawn\train\Open"
Test-Path ".\dataset_eyes&yawn\train\yawn"
Test-Path ".\dataset_eyes&yawn\train\no_yawn"
Test-Path ".\mrleyedataset\Close-Eyes"
Test-Path ".\mrleyedataset\Open-Eyes"
Test-Path ".\Video Database\Yawn"
```

Nếu script trả về `True` cho thư mục bạn cần, bạn có thể chuyển sang bước train/evaluate.

## Core Concepts
### 1) Pipeline runtime chính
Luồng xử lý:
1. Nạp cấu hình (`defaults < config.json < CLI`)
2. Đọc frame từ webcam hoặc file
3. Trích xuất tín hiệu thô (EAR, MAR, pitch/yaw/roll, gaze)
4. Tạo đặc trưng theo thời gian (EMA, PERCLOS, bộ đếm sự kiện)
5. Chọn decision engine từ registry map
6. Trả về trạng thái + chính sách cảnh báo âm thanh/overlay

### 2) Decision engines
- `fsm`: engine chuẩn mặc định của hệ thống (khuyên dùng).
- `legacy`: engine rule-based legacy để so sánh.

Engine map nằm ở `runtime/engines/registry.py`.

### 3) Artifact quan trọng
Các file thường dùng:
- `eye_model.pth`
- `advanced_drowsiness_model.pth`
- `drowsiness_model.pkl`
- `scaler.pkl`
- `alert.wav`

## CLI / --help Reference (đầy đủ)
Phần này là “cheat sheet” đầy đủ để thành viên mới không cần nhớ tham số.

### 1) Nhóm runtime
#### `advanced_drowsiness_detection.py`
Lệnh help:
```powershell
.\venv\Scripts\python.exe advanced_drowsiness_detection.py --help
```

| Option | Giá trị | Mặc định hiệu lực | Mô tả |
| --- | --- | --- | --- |
| `--config` | `<path>` | Không dùng file ngoài nếu không truyền | Đường dẫn file JSON config. |
| `--source` | `webcam` hoặc `file` | Theo `config.json` (mặc định `webcam`) | Chọn nguồn video. |
| `--video-path` | `<path>` | Theo `config.json` (`Video Database\Sub 03.avi`) | Video path khi `--source file`. |
| `--decision-engine` | `fsm` hoặc `legacy` | Theo `config.json` (mặc định `fsm`) | Chọn engine quyết định trạng thái. |
| `--enable-legacy-feature-overlay` | Cờ bật | Theo `config.json` (mặc định `false`) | Bật overlay telemetry kiểu legacy. |
| `--disable-legacy-feature-overlay` | Cờ tắt | Theo `config.json` | Tắt overlay telemetry legacy. |
| `--display-window` | Cờ bật | Theo `config.json` (mặc định `true`) | Bật cửa sổ hiển thị OpenCV. |
| `--no-display` | Cờ tắt | Theo `config.json` | Chạy không mở cửa sổ (headless). |

Ví dụ nhanh:
```powershell
.\venv\Scripts\python.exe advanced_drowsiness_detection.py --source webcam --decision-engine fsm
.\venv\Scripts\python.exe advanced_drowsiness_detection.py --source file --video-path "D:\videos\demo.mp4" --decision-engine legacy
.\venv\Scripts\python.exe advanced_drowsiness_detection.py --config config.json --no-display
```

#### `drowsiness_detection_with_model.py`
Lệnh help:
```powershell
.\venv\Scripts\python.exe drowsiness_detection_with_model.py --help
```

| Option | Giá trị | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `--source` | `webcam` hoặc `file` | `file` | Chọn nguồn video. |
| `--video-path` | `<path>` | `deokinhthieusang.mp4` | Video path khi chạy mode file. |
| `--model-path` | `<path>` | `eye_model.pth` | Đường dẫn model mắt. |

### 2) Nhóm training
#### `train_eye_model_torch.py`
Lệnh help:
```powershell
.\venv\Scripts\python.exe train_eye_model_torch.py --help
```

| Option | Giá trị | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `--data-dir` | `<path>` | `dataset_eyes&yawn/train` | Thư mục dataset có class `Closed`, `Open`. |
| `--batch-size` | `int` | `32` | Batch size. |
| `--learning-rate` | `float` | `0.001` | Learning rate cho Adam. |
| `--epochs` | `int` | `5` | Số epoch. |

#### `train_advanced_model.py`
- Lệnh help:
```powershell
.\venv\Scripts\python.exe train_advanced_model.py --help
```

| Option | Giá trị | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `--config` | `<path>` | `None` | File JSON cấu hình (section `train_advanced_model`). |
| `--cew-dir` | `<path>` | `CEW` | Root CEW (cần `closed/open`). |
| `--mrl-dir` | `<path>` | `mrleyedataset` | Root MRL (cần `Close-Eyes/Open-Eyes`). |
| `--eyes-yawn-dir` | `<path>` | `dataset_eyes&yawn/train` | Root dữ liệu mắt/ngáp (`Closed/Open/yawn/no_yawn`). |
| `--batch-size` | `int` | `32` | Batch size. |
| `--learning-rate` | `float` | `0.001` | Learning rate Adam. |
| `--epochs` | `int` | `20` | Số epoch train. |
| `--val-split` | `float` | `0.2` | Tỷ lệ validation split. |
| `--num-workers` | `int` | `2` | Số worker DataLoader. |
| `--output-model` | `<path>` | `advanced_drowsiness_model.pth` | File model đầu ra. |
| `--seed` | `int` | `42` | Random seed cho split. |

#### `train_model.py`
- Lệnh help:
```powershell
.\venv\Scripts\python.exe train_model.py --help
```

| Option | Giá trị | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `--config` | `<path>` | `None` | File JSON cấu hình (section `train_model`). |
| `--input-csv` | `<path>` | `drowsiness_data.csv` | File CSV đầu vào. |
| `--has-header` | Cờ | `False` | Bật nếu CSV đã có header. |
| `--no-header` | Cờ | `False` | Ép đọc CSV không có header. |
| `--test-size` | `float` | `0.2` | Tỷ lệ test split. |
| `--random-state` | `int` | `42` | Seed cho split/model. |
| `--n-estimators` | `int` | `100` | Số cây RandomForest. |
| `--n-jobs` | `int` | `-1` | Số luồng huấn luyện RF. |
| `--output-model` | `<path>` | `drowsiness_model.pkl` | File model đầu ra. |
| `--output-scaler` | `<path>` | `scaler.pkl` | File scaler đầu ra. |

#### `train_data.py`
- Lệnh help:
```powershell
.\venv\Scripts\python.exe train_data.py --help
```

| Option | Giá trị | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `--config` | `<path>` | `None` | File JSON cấu hình (section `train_data`). |
| `--video-source` | `<path>` | `video_buon_ngu.mp4` | Video nguồn để trích đặc trưng. |
| `--label` | `0` hoặc `1` | `1` | Nhãn cho toàn bộ dòng xuất ra. |
| `--csv-file` | `<path>` | `drowsiness_data.csv` | File CSV output. |
| `--predictor-path` | `<path>` | `models/shape_predictor_68_face_landmarks.dat` | File landmark predictor của dlib. |
| `--frame-width` | `int` | `450` | Kích thước resize chiều rộng frame. |
| `--detector-upsamples` | `int` | `0` | Số lần upsample detector dlib. |
| `--show-window` | Cờ | `False` | Bật preview khi trích dữ liệu. |
| `--no-show-window` | Cờ | `False` | Tắt preview (hữu ích khi config bật sẵn). |
| `--quit-key` | `char` | `q` | Phím thoát khi bật preview. |
| `--write-header` | Cờ | `False` | Ghi header CSV khi bắt đầu file. |
| `--no-write-header` | Cờ | `False` | Không ghi header CSV. |
| `--overwrite` | Cờ | `False` | Ghi đè file CSV thay vì append. |
| `--no-overwrite` | Cờ | `False` | Append vào file CSV. |

#### `extract_video_frames.py`
- Lệnh help:
```powershell
.\venv\Scripts\python.exe extract_video_frames.py --help
```

| Option | Giá trị | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `--config` | `<path>` | `None` | File JSON cấu hình (section `extract_video_frames`). |
| `--video-dir` | `<path>` | `Video Database` | Thư mục video nguồn. |
| `--output-dir` | `<path>` | `extracted_video_frames` | Thư mục output frame. |
| `--glob` | `pattern` | `*.avi` | Mẫu file video cần quét. |
| `--recursive` | Cờ | `False` | Quét đệ quy thư mục con. |
| `--no-recursive` | Cờ | `False` | Chỉ quét thư mục cấp cao nhất. |
| `--sample-fps` | `float` | `1.0` | Số frame lấy mỗi giây. |
| `--label-folder` | `string` | `alert` | Tên subfolder output để lưu frame. |

### 3) Nhóm evaluation
#### `tools/evaluation/label_yawn_frames.py`
Lệnh help:
```powershell
.\venv\Scripts\python.exe tools\evaluation\label_yawn_frames.py --help
```

| Option | Giá trị | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `--dataset-path` | `<path>` | `Video Database/Yawn` | Thư mục video cần label. |
| `--label-path` | `<path>` | `Video Database/Yawn/Labels` | Thư mục xuất file `*_labels.csv`. |
| `--delay-ms` | `int` | `30` | Delay mỗi frame (ms) để bắt phím. |
| `--label-key` | `char` | `l` | Giữ phím này để gán frame là ngáp (`1`). |
| `--skip-key` | `char` | `s` | Bỏ qua video hiện tại. |
| `--quit-key` | `char` | `q` | Lưu phần đã label và thoát. |
| `--overwrite` | Cờ | `False` | Ghi đè file nhãn đã có. |

#### `tools/evaluation/evaluate_mar_videos.py`
Lệnh help:
```powershell
.\venv\Scripts\python.exe tools\evaluation\evaluate_mar_videos.py --help
```

| Option | Giá trị | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `--dataset-path` | `<path>` | `Video Database/Yawn` | Thư mục video để đánh giá. |
| `--label-path` | `<path>` | `Video Database/Yawn/Labels` | Thư mục nhãn ground truth. |
| `--report-file` | `<path>` | `final_evaluation_report.csv` | File CSV tổng hợp kết quả. |
| `--window-size` | `int` | `150` | Kích thước sliding window MAR (frame). |
| `--threshold-factor` | `float` | `0.7` | Hệ số ngưỡng động theo max(MAR window). |
| `--min-threshold` | `float` | `0.15` | Ngưỡng sàn tối thiểu. |
| `--preview` | Cờ | `False` | Hiển thị preview khi đánh giá. |

#### `tools/evaluation/analyze_mar_thresholds.py`
Lệnh help:
```powershell
.\venv\Scripts\python.exe tools\evaluation\analyze_mar_thresholds.py --help
```

| Option | Giá trị | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `--input-csv` | `<path>` | `tools/evaluation/metadata/mar_result/mar_result.csv` | File có cột `MAR`. |
| `--output-csv` | `<path>` | `tools/evaluation/metadata/reports/mar_threshold_analysis.csv` | File kết quả phân tích ngưỡng. |
| `--threshold-start` | `float` | `0.30` | Giá trị bắt đầu quét ngưỡng. |
| `--threshold-end` | `float` | `0.80` | Giá trị kết thúc quét ngưỡng. |
| `--threshold-step` | `float` | `0.025` | Bước nhảy mỗi lần quét. |

### 4) One-shot kiểm tra tất cả lệnh `--help`
Bạn có thể chạy một lượt để kiểm tra môi trường:

```powershell
.\venv\Scripts\python.exe advanced_drowsiness_detection.py --help
.\venv\Scripts\python.exe drowsiness_detection_with_model.py --help
.\venv\Scripts\python.exe train_eye_model_torch.py --help
.\venv\Scripts\python.exe train_advanced_model.py --help
.\venv\Scripts\python.exe train_model.py --help
.\venv\Scripts\python.exe train_data.py --help
.\venv\Scripts\python.exe extract_video_frames.py --help
.\venv\Scripts\python.exe tools\evaluation\label_yawn_frames.py --help
.\venv\Scripts\python.exe tools\evaluation\evaluate_mar_videos.py --help
.\venv\Scripts\python.exe tools\evaluation\analyze_mar_thresholds.py --help
```

## Runtime Guides
### 1) Runtime chính (modular)
Xem help:
```powershell
python advanced_drowsiness_detection.py --help
```

Các tham số chính:
- `--config <path>`
- `--source webcam|file`
- `--video-path <path>`
- `--decision-engine fsm|legacy`
- `--enable-legacy-feature-overlay`
- `--disable-legacy-feature-overlay`
- `--display-window`
- `--no-display`

Ví dụ:
```powershell
python advanced_drowsiness_detection.py --source webcam --decision-engine fsm
python advanced_drowsiness_detection.py --source file --video-path "D:\videos\test.mp4" --decision-engine legacy
```

### 2) Runtime baseline cũ
```powershell
python drowsiness_detection_with_model.py --help
python drowsiness_detection_with_model.py --source file --video-path "deokinhthieusang.mp4" --model-path eye_model.pth
```

## Training Guides
### 1) Train eye classifier (PyTorch)
```powershell
python train_eye_model_torch.py --data-dir "dataset_eyes&yawn/train" --batch-size 32 --learning-rate 0.001 --epochs 5
```

Output: `eye_model.pth`

### 2) Train advanced multitask model
```powershell
python train_advanced_model.py --config config_tools.json
# Ví dụ override từ CLI:
python train_advanced_model.py --config config_tools.json --epochs 25 --output-model advanced_drowsiness_model_v2.pth
```

Output: `advanced_drowsiness_model.pth`

### 3) Train model tabular (classical ML)
```powershell
python train_model.py --config config_tools.json
# Ví dụ override:
python train_model.py --config config_tools.json --input-csv custom_data.csv --has-header
```

Yêu cầu đầu vào: `drowsiness_data.csv` (cột: `ear, mar, pitch, yaw, roll, label`).

Output:
- `drowsiness_model.pkl`
- `scaler.pkl`

### 4) Tạo dữ liệu tabular / frame
```powershell
python train_data.py --config config_tools.json
python extract_video_frames.py --config config_tools.json
# Ví dụ override:
python train_data.py --config config_tools.json --video-source "video_tinh_tao.mp4" --label 0 --show-window
```

Lưu ý cho `train_data.py`:
- Cần file `models/shape_predictor_68_face_landmarks.dat`.
- Nếu muốn xem preview khi trích dữ liệu: thêm `--show-window`.

## Evaluation Guides
### 1) Gán nhãn frame cho video yawn
```powershell
python tools/evaluation/label_yawn_frames.py --dataset-path "Video Database/Yawn" --label-path "tools/evaluation/metadata/yawn_labels"
```

Mặc định thao tác:
- Giữ phím `l`: frame đang ngáp (`label=1`)
- Không giữ `l`: bình thường (`label=0`)
- Phím `s`: bỏ qua video hiện tại
- Phím `q`: lưu nhãn hiện có và thoát

### 2) Đánh giá MAR theo sliding window
```powershell
python tools/evaluation/evaluate_mar_videos.py --dataset-path "Video Database/Yawn" --label-path "tools/evaluation/metadata/yawn_labels" --report-file "tools/evaluation/metadata/reports/final_evaluation_report.csv"
```

### 3) Quét ngưỡng MAR từ file CSV có sẵn
```powershell
python tools/evaluation/analyze_mar_thresholds.py --input-csv "tools/evaluation/metadata/mar_result/mar_result.csv" --output-csv "tools/evaluation/metadata/reports/mar_threshold_analysis.csv"
```

## Configuration
File cấu hình mặc định: `config.json`.

Thứ tự ưu tiên cấu hình:
1. Giá trị mặc định trong code
2. `config.json`
3. Tham số CLI (ưu tiên cao nhất)

Ví dụ rút gọn:
```json
{
  "input": {
    "source": "webcam",
    "video_path": "Video Database\\Sub 03.avi",
    "loop_file": true
  },
  "runtime": {
    "fps": 30.0,
    "warmup_seconds": 3.0,
    "calibration_frames": 60
  },
  "decision_engine": "fsm",
  "enable_legacy_feature_overlay": false
}
```

### Shared JSON config cho 4 script train/data
File mẫu dùng chung: `config_tools.json`.
Schema validate: `config_tools.schema.json`.

Cách hoạt động:
1. Giá trị mặc định trong script
2. Giá trị trong `--config`
3. Tham số CLI ghi đè lần cuối

Validation hoạt động trước khi parse CLI:
- Sai kiểu dữ liệu (ví dụ string thay vì number) -> báo lỗi rõ key.
- Sai range (ví dụ `test_size=2`) -> báo lỗi rõ ràng.
- Key không hợp lệ trong section -> báo lỗi ngay (không bỏ qua âm thầm).

Section mapping trong `config_tools.json`:
- `train_advanced_model`
- `train_model`
- `train_data`
- `extract_video_frames`

Ví dụ:
```powershell
python train_advanced_model.py --config config_tools.json --epochs 30
python train_model.py --config config_tools.json --no-header
python train_data.py --config config_tools.json --video-source "video01.mp4" --show-window
python extract_video_frames.py --config config_tools.json --glob "*.mp4" --recursive
```

## Project Structure
```text
.
├── advanced_drowsiness_detection.py        # entrypoint runtime chính
├── drowsiness_detection_with_model.py      # runtime baseline
├── cli_json_config.py                      # helper nạp --config JSON cho script CLI
├── config_tools.schema.json                # schema validate config cho train/data scripts
├── runtime/
│   ├── app.py                              # orchestrator
│   ├── config.py                           # cấu hình + precedence
│   ├── transports.py                       # webcam/file transport
│   ├── perception.py                       # trích xuất tín hiệu thô
│   ├── features.py                         # EMA/PERCLOS/temporal features
│   ├── alerts.py                           # điều khiển cảnh báo âm thanh
│   └── engines/                            # engine interface + fsm/legacy
├── tools/
│   ├── evaluation/                         # labeling/evaluation scripts
│   └── merge/                              # sync archive + provenance
├── docs/
│   ├── architecture/                       # tài liệu kiến trúc
│   ├── merge/                              # ledger/manifest/verification
│   └── reference/                          # tài liệu tham khảo
├── requirements.txt                        # dependencies chính
├── requirements-legacy.txt                 # dependencies cho script legacy (dlib)
├── config_tools.json                       # config mẫu chung cho train/data scripts
└── config.json                             # config mặc định
```

## Troubleshooting
### `Cannot open video file ...`
- Kiểm tra path trong `--video-path`.
- Thử đổi video sang `.mp4` chuẩn H.264 nếu codec lạ.
- Đảm bảo file không bị app khác giữ lock.

### `Cannot open webcam device 0`
- Đóng các app đang dùng camera (Zoom, Teams, Chrome).
- Kiểm tra camera hoạt động trong app Camera của Windows.

### Không nghe được `alert.wav`
- Đảm bảo `alert.wav` có ở root repo.
- Kiểm tra loa/hệ thống âm thanh.
- Kiểm tra thư viện `playsound` đã cài.

### `Class Closed/Open not found in dataset`
- Kiểm tra tên thư mục phải đúng: `Closed`, `Open`.
- Tránh viết thường toàn bộ (`closed`, `open`) cho script `train_eye_model_torch.py`.

### Lỗi thiếu MediaPipe/OpenCV/PyTorch
- Kích hoạt đúng virtual environment.
- Cài lại thư viện bằng `pip install -r requirements.txt`.
- Kiểm tra đúng interpreter:
  - `where python`
  - `python -V`
  - `.\venv\Scripts\python.exe -V`
  - `.\venv\Scripts\python.exe -c "import cv2, torch, sklearn; print('OK')"`

### Sai phiên bản Python (không phải 3.12)
- Xóa môi trường cũ nếu tạo nhầm:
  - `Remove-Item -Recurse -Force .\venv`
- Tạo lại đúng chuẩn:
  - `py -3.12 -m venv venv`
  - `.\venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`

### Lỗi thiếu `shape_predictor_68_face_landmarks.dat`
- Chỉ ảnh hưởng script `train_data.py`.
- Runtime chính không cần file này.

## FAQ cho thành viên mới
### Tôi chỉ muốn chạy demo nhanh nhất, làm gì?
1. Cài môi trường theo mục Get Started.
2. Chạy `python advanced_drowsiness_detection.py` (webcam).
3. Nếu không có webcam, chạy mode file với `--source file --video-path ...`.

### Tôi chưa có dataset đầy đủ, có làm được gì?
- Có thể chạy runtime webcam ngay.
- Có thể chạy evaluation threshold với file metadata sẵn trong `tools/evaluation/metadata`.
- Train đầy đủ cần bổ sung dataset như mục Dataset Setup.

### Tôi muốn thử engine mới (ví dụ LSTM) thì bắt đầu ở đâu?
1. Tạo engine mới trong `runtime/engines/`.
2. Implement interface `DecisionEngine`.
3. Đăng ký vào `runtime/engines/registry.py`.
4. Chạy bằng `--decision-engine <engine_moi>`.

## Architecture docs
- [Architecture Guide](docs/architecture/Architecture_Guide.md)
- [Modular Runtime Map](docs/architecture/modular_runtime_map.md)
- [Codebase Visual Map](docs/architecture/Codebase_Visual_Map.md)

## Merge and provenance docs
- [Integration Ledger](docs/merge/da1_bundle_integration_ledger.md)
- [Import Manifest](docs/merge/import_manifest.json)
- [Model Provenance](docs/merge/model_provenance.md)
- [Verification Report](docs/merge/verification_report.md)

## Contributing
Khi thêm thay đổi lớn, đặc biệt là decision engine:
1. Tách phần logic thành module rõ ràng (không trộn vào `runtime/app.py`).
2. Thêm test tương ứng vào `tests/`.
3. Cập nhật `config.json` nếu có option mới.
4. Cập nhật README để thành viên khác chạy lại được ngay.

---
Nếu bạn chưa biết bắt đầu từ đâu, chạy:
```powershell
python advanced_drowsiness_detection.py --help
```
và đọc thêm [Architecture Guide](docs/architecture/Architecture_Guide.md).

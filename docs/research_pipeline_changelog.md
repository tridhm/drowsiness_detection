# Changelog Chi Tiết: MVP Pipeline Nghiên Cứu Offline

## 1. Tóm Tắt Ngắn Gọn

Cập nhật này bổ sung một pipeline nghiên cứu offline cho hệ thống phát hiện buồn ngủ. Trước đây project chủ yếu phục vụ demo realtime: mở webcam/video, xử lý từng frame, hiển thị cảnh báo. Sau cập nhật này, hệ thống có thêm một hướng dùng cho nghiên cứu: chạy video offline, xuất đặc trưng ra CSV, căn chỉnh với nhãn KSS, đánh giá baseline, và train thử Random Forest trên đặc trưng cấp window.

Pipeline tổng quát:

```text
Video có sẵn
  -> export frame_features.csv và window_features.csv
  -> căn chỉnh window với annotation kiểu KSS
  -> tạo labeled_windows.csv
  -> đánh giá baseline rule-based
  -> train Random Forest window-fusion MVP
  -> xuất metrics, feature importance, permutation importance, ablation results
```

Điểm quan trọng: cập nhật này **xây nền tảng kỹ thuật để làm thí nghiệm**, chưa phải là kết quả chứng minh hệ thống phát hiện buồn ngủ tốt. Trong folder hiện tại chưa có dataset KSS/PVT thật, nên các CSV sample chỉ dùng để kiểm thử pipeline.

## 2. Vì Sao Thay Đổi Này Meaningful

Nếu chỉ chạy realtime demo, sau khi video kết thúc mình chỉ thấy cảnh báo trên màn hình. Cách đó khó dùng cho báo cáo nghiên cứu vì không có dữ liệu sạch để tính metric, vẽ biểu đồ, hoặc so sánh phương pháp.

Pipeline mới giải quyết vấn đề này bằng cách biến video thành bảng dữ liệu:

```text
Video -> bảng đặc trưng -> bảng có nhãn -> metrics/report
```

Nhờ đó project có thể trả lời các câu hỏi nghiên cứu như:

- PERCLOS-only hoạt động thế nào so với FSM hiện tại?
- Các feature về mắt, ngáp, head pose đóng góp khác nhau ra sao?
- Random Forest fusion có cải thiện so với baseline rule-based không?
- Khi bỏ PERCLOS hoặc bỏ yawn/head-pose features, F1-score thay đổi thế nào?

Tức là thay đổi này chuyển project từ hướng chỉ demo realtime sang hướng có thể đánh giá lặp lại và có artifact nghiên cứu.

## 3. Những Gì Đã Thêm

### 3.1 Shared research helpers

File chính:

```text
tools/research/common.py
```

Vai trò:

- Đọc/ghi CSV bằng standard library `csv`, không phụ thuộc `pandas`.
- Định nghĩa schema cho frame-level và window-level features.
- Gom nhiều frame thành một window thời gian.
- Tính metric nhị phân như accuracy, precision, recall, F1, TP/TN/FP/FN.
- Định nghĩa nhóm feature cho Random Forest và ablation.

Lý do không dùng `pandas`: môi trường `venv` hiện tại có `numpy`, `sklearn`, `joblib`, nhưng thiếu `pandas`. Vì vậy tool được viết bằng `csv` + `numpy/sklearn` để chạy được ngay trong môi trường hiện có.

### 3.2 Offline feature exporter

File:

```text
tools/research/export_video_features.py
```

Tool này đọc một video và xuất ra hai loại CSV:

1. `frame_features.csv`: mỗi row là một frame.
2. `window_features.csv`: mỗi row là một window thời gian, ví dụ 10 giây hoặc 60 giây.

Tool tái sử dụng pipeline hiện có trong runtime:

- `runtime.perception.PerceptionExtractor`: trích EAR, MAR, head pose, gaze bằng MediaPipe.
- `runtime.features.SignalFeaturePipeline`: tính feature theo thời gian như PERCLOS, blink/yawn frequency, head nod.
- `runtime.engines`: chạy decision engine như FSM.

Điểm quan trọng: exporter dùng timestamp từ FPS của video:

```text
timestamp_sec = frame_index / fps
```

Không dùng `time.time()` vì offline research cần reproducible timestamp. Cùng một video chạy lại phải ra mốc thời gian giống nhau.

Lệnh smoke test:

```powershell
.\venv\Scripts\python.exe tools\research\export_video_features.py `
  --video-path "Video Database\0392.mp4" `
  --subject-id S0392 `
  --session-id current `
  --video-id 0392 `
  --max-frames 150 `
  --window-seconds 10 `
  --stride-seconds 5 `
  --frame-csv reports\research\smoke_frame_features.csv `
  --window-csv reports\research\smoke_window_features.csv
```

Ý nghĩa của smoke test:

- Xác nhận tool mở được video có sẵn.
- Xác nhận MediaPipe/runtime pipeline chạy được offline.
- Xác nhận CSV được sinh ra đúng schema.
- Không dùng kết quả này để claim performance vì video chưa có nhãn KSS thật.

### 3.3 KSS/window label alignment

File:

```text
tools/research/align_window_labels.py
```

Tool này ghép `window_features.csv` với file annotation kiểu KSS.

Input annotation kỳ vọng:

```csv
subject_id,session_id,video_id,start_time_sec,end_time_sec,kss_score,kss_band,notes
S01,session_01,video_01,0,60,3,alert,baseline awake
S01,session_01,video_01,60,120,8,sleepy,clear drowsiness
```

Cách match:

- Match cùng `subject_id`.
- Match cùng `session_id`.
- Match cùng `video_id`.
- Chọn annotation có thời gian overlap lớn nhất với window.

Ví dụ:

```text
Window: 10s -> 70s
Annotation A: 0s -> 30s, overlap 20s
Annotation B: 30s -> 90s, overlap 40s
=> chọn Annotation B
```

Output là `labeled_windows.csv`, tức là window feature đã có thêm:

```text
kss_score
kss_band
label_source
```

Lệnh ví dụ dùng fixture:

```powershell
.\venv\Scripts\python.exe tools\research\align_window_labels.py `
  --window-csv metadata\window_features.sample.csv `
  --annotations-csv metadata\kss_annotations.sample.csv `
  --output-csv reports\research\labeled_windows.csv
```

### 3.4 Baseline evaluator

File:

```text
tools/research/evaluate_baselines.py
```

Tool này đánh giá các baseline đơn giản trước khi dùng Random Forest.

Các baseline hiện có:

| Baseline | Ý nghĩa |
|---|---|
| `perclos_only` | Chỉ dùng ngưỡng PERCLOS để đoán sleepy/non-sleepy |
| `yawn_only` | Nếu có yawn event thì đoán sleepy |
| `head_pose_only` | Nếu có head drop thì đoán sleepy |
| `fsm_state` | Nếu FSM state là `DROWSY` hoặc `CRITICAL` thì đoán sleepy |

Target mặc định:

```text
kss_band == sleepy -> positive
kss_band != sleepy -> negative
```

Output:

```text
baseline_results.csv
baseline_confusion_matrix.csv
```

Các metric được xuất:

- Accuracy
- Precision
- Recall
- F1
- TP, TN, FP, FN
- Support

Lệnh ví dụ:

```powershell
.\venv\Scripts\python.exe tools\research\evaluate_baselines.py `
  --input-csv metadata\labeled_windows.sample.csv `
  --results-csv reports\research\baseline_results.csv `
  --confusion-csv reports\research\baseline_confusion_matrix.csv
```

Nếu bị hỏi vì sao phải có baseline trước Random Forest, câu trả lời là: baseline giúp biết model phức tạp có thật sự cải thiện hay không. Nếu Random Forest không tốt hơn PERCLOS-only/FSM, thì dùng RF chưa có nhiều giá trị nghiên cứu.

### 3.5 Random Forest window-fusion MVP

File:

```text
tools/research/train_window_fusion_model.py
```

Tool này train `RandomForestClassifier` trên feature cấp window, không train trên frame đơn lẻ.

Sai lầm cần tránh:

```text
random frame -> label 0/1 -> random train/test split
```

Cách đúng hơn mà tool này hướng tới:

```text
window-level features -> KSS band -> subject-disjoint train/test split
```

Feature groups hiện dùng:

| Nhóm | Feature |
|---|---|
| Eye | `mean_ear`, `min_ear`, `perclos_60s`, `perclos_5s`, `max_eye_closed_duration_sec`, `blink_rate_per_min` |
| Yawn | `mean_mar`, `max_mar`, `yawn_count` |
| Head pose | `head_drop_count`, `max_pitch_velocity` |
| FSM | `mean_fsm_evidence`, `max_fsm_evidence`, `fsm_state_mode` |

Output:

```text
random_forest_results.csv
feature_importance.csv
permutation_importance.csv
ablation_results.csv
```

Ý nghĩa từng output:

- `random_forest_results.csv`: metric tổng quát của RF full model.
- `feature_importance.csv`: built-in importance của Random Forest.
- `permutation_importance.csv`: đo mức giảm score khi shuffle từng feature, đáng tin hơn built-in importance trong nhiều trường hợp.
- `ablation_results.csv`: train lại model với các nhóm feature bị bỏ bớt để xem nhóm nào thật sự quan trọng.

Các ablation variant:

| Variant | Ý nghĩa |
|---|---|
| `full` | Dùng tất cả feature groups |
| `eye_only` | Chỉ dùng feature mắt |
| `perclos_only` | Chỉ dùng PERCLOS |
| `no_perclos` | Bỏ PERCLOS để xem model tụt bao nhiêu |
| `no_yawn` | Bỏ nhóm yawn |
| `no_head_pose` | Bỏ nhóm head pose |
| `no_fsm` | Bỏ nhóm evidence/state từ FSM |

Lệnh ví dụ:

```powershell
.\venv\Scripts\python.exe tools\research\train_window_fusion_model.py `
  --input-csv metadata\labeled_windows.sample.csv `
  --output-dir reports\research\rf_sample `
  --n-estimators 25 `
  --permutation-repeats 3
```

## 4. Schema Dữ Liệu

### 4.1 Frame-level features

`frame_features.csv` có schema:

```text
subject_id, session_id, video_id, frame_index, timestamp_sec,
face_detected, ear, mar, pitch, yaw, roll,
eye_closed, mouth_open, head_nod_detected,
perclos_60s, perclos_5s,
blink_frequency, yawn_frequency,
pitch_velocity, gaze_stable,
fsm_state, fsm_evidence, fsm_reasons
```

Dùng để debug tín hiệu theo từng frame.

Ví dụ một vài row đầu của `frame_features.csv` sẽ có dạng như sau:

```csv
subject_id,session_id,video_id,frame_index,timestamp_sec,face_detected,ear,mar,pitch,yaw,roll,eye_closed,mouth_open,head_nod_detected,perclos_60s,perclos_5s,blink_frequency,yawn_frequency,pitch_velocity,gaze_stable,fsm_state,fsm_evidence,fsm_reasons
S0392,current,0392,0,0.000000,1,0.284321,0.032115,2.512300,-1.220100,0.531200,0,0,0,0.000000,0.000000,0,0,0.000000,0,ALERT,0.000000,
S0392,current,0392,1,0.040000,1,0.279812,0.033441,2.601900,-1.180400,0.520900,0,0,0,0.000000,0.000000,0,0,0.089600,0,ALERT,0.000000,
S0392,current,0392,2,0.080000,1,0.182000,0.040210,3.010000,-0.950000,0.600000,1,0,0,0.333333,0.333333,0,0,0.408100,0,SUSPICIOUS,0.250000,EAR_BELOW_THRESHOLD
```

Ý nghĩa một row frame:

- `subject_id`, `session_id`, `video_id`: định danh subject/session/video để sau này ghép với nhãn KSS.
- `frame_index`, `timestamp_sec`: frame thứ mấy và thời điểm trong video; `timestamp_sec = frame_index / fps`.
- `face_detected`: `1` nếu MediaPipe thấy mặt, `0` nếu không thấy mặt.
- `ear`, `mar`: Eye Aspect Ratio và Mouth Aspect Ratio của frame đó.
- `pitch`, `yaw`, `roll`: góc head pose ước lượng ở frame đó.
- `eye_closed`, `mouth_open`, `head_nod_detected`: các cờ sự kiện nhị phân do pipeline suy ra.
- `perclos_60s`, `perclos_5s`: tỷ lệ eye-closed trong cửa sổ dài/ngắn tại thời điểm frame đó.
- `blink_frequency`, `yawn_frequency`: số blink/yawn được đếm trong rolling window hiện tại của runtime.
- `pitch_velocity`: độ thay đổi pitch giữa các frame, dùng để bắt head-drop/head-nod dynamics.
- `gaze_stable`: cờ gaze ổn định theo logic runtime hiện tại.
- `fsm_state`, `fsm_evidence`, `fsm_reasons`: kết quả decision engine tại frame đó.

Nói đơn giản: `frame_features.csv` là log chi tiết từng frame, dùng để debug vì sao tại thời điểm đó hệ thống xem mắt đang nhắm, miệng đang mở, evidence FSM đang tăng, hoặc state chuyển từ `ALERT` sang `SUSPICIOUS/DROWSY`.

### 4.2 Window-level features

`window_features.csv` có schema:

```text
subject_id, session_id, video_id,
window_start_sec, window_end_sec,
frame_count, valid_face_ratio,
mean_ear, min_ear,
mean_mar, max_mar,
perclos_60s, perclos_5s,
max_eye_closed_duration_sec,
blink_rate_per_min,
yawn_count,
head_drop_count,
max_pitch_velocity,
mean_fsm_evidence, max_fsm_evidence,
fsm_state_mode
```

Dùng cho đánh giá baseline, alignment với KSS, và train Random Forest.

Ví dụ một vài row của `window_features.csv` sẽ có dạng như sau:

```csv
subject_id,session_id,video_id,window_start_sec,window_end_sec,frame_count,valid_face_ratio,mean_ear,min_ear,mean_mar,max_mar,perclos_60s,perclos_5s,max_eye_closed_duration_sec,blink_rate_per_min,yawn_count,head_drop_count,max_pitch_velocity,mean_fsm_evidence,max_fsm_evidence,fsm_state_mode
S0392,current,0392,0.000000,10.000000,150,0.760000,0.233262,0.000000,0.020456,0.156174,0.220000,0.224000,0.200000,70.469799,0,0,102.156232,0.152227,0.500000,ALERT
S0392,current,0392,5.000000,15.000000,25,1.000000,0.321563,0.186165,0.034229,0.101015,0.220000,0.224000,0.000000,0.000000,0,0,81.605743,0.081156,0.101587,ALERT
```

Ý nghĩa một row window:

- `window_start_sec`, `window_end_sec`: khoảng thời gian của window, ví dụ từ giây 0 đến giây 10.
- `frame_count`: số frame nằm trong window đó.
- `valid_face_ratio`: tỷ lệ frame trong window có detect được mặt; giá trị thấp nghĩa là dữ liệu window kém tin cậy.
- `mean_ear`, `min_ear`: EAR trung bình và EAR thấp nhất trong window.
- `mean_mar`, `max_mar`: MAR trung bình và MAR cao nhất trong window.
- `perclos_60s`, `perclos_5s`: PERCLOS tại cuối window, lấy từ rolling calculation của pipeline.
- `max_eye_closed_duration_sec`: chuỗi nhắm mắt liên tục dài nhất trong window, tính bằng giây.
- `blink_rate_per_min`: blink rate quy đổi theo phút từ blink count trong window.
- `yawn_count`: số yawn event xuất hiện trong window.
- `head_drop_count`: số frame/event có head-nod/head-drop flag.
- `max_pitch_velocity`: pitch velocity lớn nhất trong window.
- `mean_fsm_evidence`, `max_fsm_evidence`: evidence trung bình/lớn nhất của FSM trong window.
- `fsm_state_mode`: state xuất hiện nhiều nhất trong window; nếu hòa thì ưu tiên state rủi ro cao hơn.

Nói đơn giản: `window_features.csv` nén nhiều frame thành một dòng đặc trưng theo thời gian. Đây là file quan trọng hơn cho nghiên cứu, vì KSS cũng là trạng thái theo khoảng thời gian chứ không phải nhãn cho từng frame đơn lẻ.

### 4.3 KSS annotation

`kss_annotations.csv` kỳ vọng schema:

```text
subject_id, session_id, video_id,
start_time_sec, end_time_sec,
kss_score, kss_band, notes
```

Mapping khuyến nghị:

```text
KSS 1-4 -> alert
KSS 5-6 -> caution
KSS 7-9 -> sleepy
```

Khi train/evaluate binary classifier, tool hiện coi:

```text
sleepy -> positive
alert/caution -> negative
```

## 5. Fixture Và Dữ Liệu Sample

Các file sample đã thêm:

```text
metadata/kss_annotations.sample.csv
metadata/subject_split.sample.csv
metadata/window_features.sample.csv
metadata/labeled_windows.sample.csv
```

Mục đích của các file này:

- Cho phép test label alignment mà không cần dataset thật.
- Cho phép test baseline evaluator.
- Cho phép test Random Forest output reports.
- Giúp pipeline chạy end-to-end trong môi trường hiện tại.

Không nên dùng các file này làm evidence cho thesis vì chúng là synthetic fixtures, không phải dữ liệu thực nghiệm.

## 6. Output Report Sinh Ra

Các report thường được sinh trong `reports/research/`:

```text
reports/research/smoke_frame_features.csv
reports/research/smoke_window_features.csv
reports/research/labeled_windows.csv
reports/research/baseline_results.csv
reports/research/baseline_confusion_matrix.csv
reports/research/rf_sample/random_forest_results.csv
reports/research/rf_sample/feature_importance.csv
reports/research/rf_sample/permutation_importance.csv
reports/research/rf_sample/ablation_results.csv
```

Folder `reports/` đã được thêm vào `.gitignore`, vì đây là generated artifact. Nếu sau này có report chính thức cần nộp, nên review rồi mới add thủ công.

## 7. Validation Đã Chạy

Các lệnh validation chính:

```powershell
.\venv\Scripts\python.exe -m unittest discover tests
.\venv\Scripts\python.exe -m py_compile tools\research\common.py tools\research\export_video_features.py tools\research\align_window_labels.py tools\research\evaluate_baselines.py tools\research\train_window_fusion_model.py
```

Smoke test exporter:

```powershell
.\venv\Scripts\python.exe tools\research\export_video_features.py --video-path "Video Database\0392.mp4" --subject-id S0392 --session-id current --video-id 0392 --max-frames 150 --window-seconds 10 --stride-seconds 5 --frame-csv reports\research\smoke_frame_features.csv --window-csv reports\research\smoke_window_features.csv
```

Kết quả smoke test đã sinh:

```text
150 frame rows
2 window rows
```

Lưu ý: MediaPipe có thể in warning TensorFlow Lite/protobuf khi chạy. Đây là warning từ thư viện, không làm command fail.

## 8. Nếu Bị Hỏi Thì Trả Lời Sao?

### Hỏi: Pipeline này khác gì demo realtime?

Trả lời:

> Demo realtime dùng để hiển thị cảnh báo trực tiếp. Pipeline offline dùng để xuất dữ liệu nghiên cứu ra CSV, từ đó có thể tính metric, vẽ biểu đồ, so sánh baseline, train Random Forest, và chạy lại cùng một video để có kết quả tái lập.

### Hỏi: Vì sao cần window-level features?

Trả lời:

> Buồn ngủ không nên kết luận từ một frame đơn lẻ. Các dấu hiệu như PERCLOS, nhắm mắt kéo dài, yawn count, head drop, blink rate đều là tín hiệu theo thời gian. Vì vậy mình gom frame thành window, ví dụ 30s hoặc 60s, rồi đánh giá trạng thái trong window đó.

### Hỏi: Random Forest đã chứng minh hệ thống tốt chưa?

Trả lời:

> Chưa. Hiện tại Random Forest MVP mới chứng minh pipeline train/evaluate chạy được trên dữ liệu window-level. Để chứng minh performance thật, cần dataset có nhãn KSS/PVT thật và subject-disjoint evaluation.

### Hỏi: Feature importance dùng để làm gì?

Trả lời:

> Feature importance giúp giải thích model đang dựa nhiều vào tín hiệu nào, ví dụ PERCLOS, eye closure duration, yawn count, hay head-pose features. Nhưng built-in importance của Random Forest có thể bias, nên pipeline cũng có permutation importance và ablation để kiểm tra đáng tin hơn.

### Hỏi: Ablation là gì?

Trả lời:

> Ablation là train/evaluate lại model khi bỏ một nhóm feature. Ví dụ so sánh full model với no-PERCLOS. Nếu bỏ PERCLOS mà F1 giảm mạnh, có thể nói PERCLOS là nhóm feature quan trọng. Đây là bằng chứng dễ giải thích trong báo cáo.

### Hỏi: Vì sao subject-disjoint split quan trọng?

Trả lời:

> Nếu cùng một người xuất hiện ở cả train và test, model có thể học đặc điểm khuôn mặt, ánh sáng hoặc background của người đó thay vì học dấu hiệu buồn ngủ. Subject-disjoint split giúp đánh giá khả năng generalize sang người mới tốt hơn.

### Hỏi: Hiện tại thiếu gì để ra kết quả nghiên cứu thật?

Trả lời:

> Thiếu dữ liệu có nhãn KSS/PVT thật. Code pipeline đã sẵn sàng để nhận `window_features.csv` và `kss_annotations.csv`, nhưng performance thật chỉ nên report sau khi có dữ liệu nhãn đáng tin cậy.

### Hỏi: Có dùng dataset public chưa?

Trả lời:

> Chưa. Cập nhật này không giả định dataset bên ngoài. Nó dùng video có sẵn trong folder để smoke test exporter và synthetic fixtures để test alignment/evaluation/RF. Bước tiếp theo là import hoặc thu thập KSS annotations thật.

## 9. Cách Báo Cáo An Toàn

Có thể viết trong báo cáo tuần này:

> Em đã triển khai pipeline nghiên cứu offline cho hệ thống phát hiện buồn ngủ, bao gồm xuất đặc trưng từ video, căn chỉnh nhãn theo window kiểu KSS, đánh giá các baseline đơn giản, và MVP Random Forest fusion ở cấp window. Các tool đã được kiểm thử bằng synthetic fixtures và smoke test trên video có sẵn trong project. Kết quả performance thật sẽ được đánh giá sau khi có dữ liệu KSS/PVT-labeled phù hợp.

Không nên viết:

> Random Forest đã chứng minh hệ thống phát hiện buồn ngủ chính xác.

Lý do: hiện tại chưa có dataset KSS/PVT thật trong folder, nên các metrics từ fixture chỉ chứng minh pipeline chạy đúng, không chứng minh chất lượng phát hiện buồn ngủ ngoài thực tế.

## 10. Giới Hạn Hiện Tại

- Folder hiện tại chưa có dataset KSS/PVT-labeled thật.
- Các video trong `Video Database/*.mp4` chỉ dùng để smoke test exporter.
- Metrics từ `metadata/labeled_windows.sample.csv` chỉ kiểm thử code path, không phải kết quả nghiên cứu.
- Exporter hiện vẫn tái sử dụng một số temporal internals dựa trên frame-count của runtime hiện tại; cải tiến timestamp-based đầy đủ cho runtime nên để phase sau.
- Random Forest MVP là experimental baseline, không phải live detector mới trong runtime.

## 11. Bước Tiếp Theo

1. Thu thập hoặc import annotation KSS thật.
2. Chạy exporter trên từng video được chọn.
3. Căn chỉnh nhãn thật để tạo `labeled_windows.csv`.
4. Chạy baseline và Random Forest fusion bằng subject-disjoint split.
5. Dùng metrics, confusion matrix, feature importance, permutation importance, và ablation table thật trong báo cáo/thesis.

Nếu có nhiều thời gian hơn, nên làm thêm:

- Adapter cho dataset public có KSS/PVT nếu được phép dùng.
- Script batch export nhiều video từ manifest.
- Biểu đồ timeline PERCLOS/FSM/KSS.
- Báo cáo lỗi/error analysis cho những window model dự đoán sai.

# Lộ Trình Nghiên Cứu Đồ Án 1: Hệ Thống Giám Sát Buồn Ngủ Tài Xế Bằng Thị Giác Máy Tính

## 1. Mục Đích

Tài liệu này là phiên bản tiếng Việt của lộ trình nghiên cứu mới cho codebase hiện tại. Nó thay thế vai trò định hướng của các tài liệu thesis/audit AI-generated cũ trong `docs/old/`, và được viết theo mục tiêu **Đồ Án 1 / tiền luận văn tốt nghiệp**.

Định hướng nên dùng cho đề tài:

> Một nguyên mẫu hệ thống giám sát buồn ngủ tài xế bằng thị giác máy tính, trích xuất các đặc trưng theo thời gian từ mắt, miệng/ngáp, ánh nhìn và tư thế đầu, sau đó đánh giá khả năng phân loại mức độ buồn ngủ theo nhãn KSS đồng bộ theo cửa sổ thời gian.

Định hướng này tốt hơn việc chỉ nói “hệ thống phát hiện mệt mỏi”, vì buồn ngủ là một trạng thái sinh lý ẩn. Một đề tài có chất lượng nghiên cứu cần chứng minh mối liên hệ giữa tín hiệu quan sát được trên camera và một nhãn/chân trị có cơ sở, ví dụ KSS, PVT hoặc một chỉ báo hiệu năng/tỉnh táo tương đương.

### Cách Đọc Tài Liệu Này Nếu Chưa Quen Thuật Ngữ

Một số từ tiếng Anh được giữ lại vì chúng thường xuất hiện trong tài liệu kỹ thuật, paper và code. Khi gặp các từ này, có thể hiểu nhanh như sau:

| Thuật ngữ | Hiểu đơn giản |
|---|---|
| Claim | Điều mình tuyên bố hệ thống làm được. Ví dụ: “hệ thống hỗ trợ giám sát buồn ngủ” là claim an toàn hơn “hệ thống chẩn đoán mệt mỏi lâm sàng”. |
| Ground truth | Nhãn/chân trị dùng để so sánh với kết quả hệ thống. Trong roadmap này, ground truth chính nên là KSS hoặc nhãn tương đương. |
| KSS | Karolinska Sleepiness Scale, thang điểm tự đánh giá mức buồn ngủ từ 1 đến 9. |
| PVT | Psychomotor Vigilance Task, bài kiểm tra phản ứng/tỉnh táo, thường dùng trong nghiên cứu buồn ngủ. |
| PERCLOS | Tỷ lệ thời gian mắt nhắm trong một khoảng thời gian. Ví dụ PERCLOS 60 giây = trong 60 giây vừa qua mắt nhắm bao nhiêu phần trăm thời gian. |
| Feature | Đặc trưng/số đo đưa vào đánh giá hoặc model. Ví dụ: `perclos_60s`, `mean_ear`, `yawn_count`. |
| Window | Cửa sổ thời gian để gom dữ liệu. Ví dụ lấy từng đoạn 60 giây để tính đặc trưng. |
| Stride | Bước trượt giữa hai window liên tiếp. Ví dụ window dài 60 giây, stride 10 giây nghĩa là cứ mỗi 10 giây tạo một window mới. |
| Baseline | Phương pháp đơn giản dùng làm mốc so sánh. Ví dụ PERCLOS-only là baseline. |
| Fusion | Gộp nhiều tín hiệu/đặc trưng lại để quyết định. Ví dụ gộp mắt + ngáp + đầu. |
| Random Forest | Một mô hình machine learning dạng nhiều cây quyết định, dễ dùng làm baseline có tính giải thích. |
| Feature importance | Cách xem đặc trưng nào đóng góp nhiều hơn cho model. Không nên xem đây là bằng chứng duy nhất. |
| Ablation study | Thử bỏ từng nhóm đặc trưng để xem kết quả giảm bao nhiêu. Đây là cách chứng minh đóng góp của từng phần. |
| Subject-disjoint split | Cách chia train/test sao cho người trong tập train không xuất hiện trong tập test. |
| False alert | Cảnh báo sai, tức hệ thống báo buồn ngủ khi người lái vẫn tỉnh táo. |
| Detection latency | Độ trễ phát hiện, tức từ lúc dấu hiệu buồn ngủ xuất hiện đến lúc hệ thống cảnh báo mất bao lâu. |
| Runtime | Phần chương trình chạy thật với webcam/video để demo hoặc sử dụng. |
| Pipeline | Chuỗi xử lý từ input đến output. Ví dụ: video -> trích đặc trưng -> FSM/model -> cảnh báo. |

## 2. Tình Trạng Hiện Tại Của Dự Án

Codebase hiện tại đã có nền tảng kỹ thuật khá tốt:

- Entrypoint runtime dạng module: `advanced_drowsiness_detection.py`
- Điều phối runtime: `runtime/app.py`
- Trích xuất tín hiệu nhận thức: `runtime/perception.py`
- Pipeline đặc trưng theo thời gian: `runtime/features.py`
- Tiện ích tính PERCLOS: `perclos.py`
- Logic quyết định FSM: `fsm.py` và `runtime/engines/fsm_engine.py`
- Runtime legacy dùng eye model: `drowsiness_detection_with_model.py`
- Script huấn luyện: `train_eye_model_torch.py`, `train_advanced_model.py`, `train_model.py`
- Test nhẹ cho config và contract của decision engine trong `tests/`

Điểm yếu chính hiện tại không phải là thiếu code. Điểm yếu chính là thiếu lớp dữ liệu và đánh giá đạt chất lượng nghiên cứu.

Các hạn chế hiện tại:

- Một số logic thu thập dữ liệu cũ vẫn hỗ trợ kiểu gán một nhãn nhị phân cho cả video hoặc cả lần trích xuất.
- Công cụ đánh giá hiện tại mạnh hơn ở bài toán yawn/MAR, chưa đủ mạnh cho trạng thái buồn ngủ tổng thể.
- Huấn luyện model hiện tại chưa chứng minh được phân loại buồn ngủ theo cách tách tập không trùng chủ thể.
- `advanced_drowsiness_model.pth` hiện chủ yếu là artifact huấn luyện, trong khi runtime module chính đang thiên về đặc trưng + FSM.
- Ngưỡng và trọng số trong FSM có vẻ hợp lý, nhưng chưa được tune từ dữ liệu validation có nhãn KSS.

## 3. Câu Hỏi Nghiên Cứu Chính

Câu hỏi nghiên cứu nên dùng:

> Các đặc trưng thị giác theo thời gian như PERCLOS, thời lượng nhắm mắt, episode ngáp, hành vi chớp mắt, độ ổn định ánh nhìn và động học tư thế đầu có thể phân loại mức độ buồn ngủ của tài xế hay không, khi được đánh giá bằng nhãn KSS đồng bộ theo cửa sổ thời gian?

Các câu hỏi phụ nên có:

1. PERCLOS đơn lẻ có phải baseline hữu ích cho phân loại buồn ngủ không?
2. FSM đa tín hiệu có cải thiện so với baseline đơn tín hiệu không?
3. Random Forest fusion trên đặc trưng theo cửa sổ có cải thiện so với rule-based baseline không?
4. Nhóm đặc trưng nào đóng góp nhiều nhất: mắt, ngáp, tư thế đầu hay gaze?
5. Hệ thống tạo bao nhiêu cảnh báo sai mỗi phút trong điều kiện tỉnh táo?

## 4. Khái Niệm Cốt Lõi

Nên tách dự án thành ba lớp:

| Lớp | Ý nghĩa | Ví dụ |
|---|---|---|
| Sự kiện camera | Quan sát trực tiếp từ hình ảnh | mắt nhắm, miệng mở giống ngáp, đầu gục |
| Trạng thái buồn ngủ | Nhãn/chân trị hoặc proxy | KSS alert/caution/sleepy |
| Quyết định hệ thống | Cảnh báo runtime sinh ra | ALERT, CAUTION, DROWSY, CRITICAL |

Mục tiêu nghiên cứu không chỉ là phát hiện sự kiện camera. Mục tiêu là đánh giá xem các sự kiện camera và đặc trưng theo thời gian có tương ứng với trạng thái buồn ngủ hay không.

Mapping KSS đề xuất:

| Điểm KSS | Nhóm | Ý nghĩa |
|---|---|---|
| 1-4 | `alert` | Tỉnh táo / gần như tỉnh táo |
| 5-6 | `caution` | Vùng chuyển tiếp, bắt đầu buồn ngủ |
| 7-9 | `sleepy` | Buồn ngủ rõ, rủi ro cao |

Nếu dữ liệu nhỏ và cần bài toán nhị phân:

```text
KSS 1-6 -> non_sleepy
KSS 7-9 -> sleepy
```

## 5. Tóm Tắt Roadmap

| Phase | Tên | Output chính |
|---|---|---|
| 0 | Reset phạm vi | Claim trung thực và framing cho Đồ Án 1 |
| 1 | Protocol nghiên cứu | Schema nhãn và luật đánh giá |
| 2 | Offline feature exporter | CSV đặc trưng theo frame và theo window |
| 3 | Chiến lược dataset | Kế hoạch dataset có KSS/PVT |
| 4 | Baseline experiments | Metrics cho PERCLOS/FSM/tín hiệu đơn |
| 5 | Random Forest fusion | Baseline ML có tính giải thích và feature importance |
| 6 | Cải thiện runtime | FSM/config được tune bằng evidence |
| 7 | Đóng gói thesis | Hình, bảng, báo cáo và demo |

## 6. Phase 0: Reset Phạm Vi

### Mục Tiêu

Làm cho claim của dự án trung thực và bảo vệ được trước hội đồng.

### Nên Dùng Claim Này

Ở đây, **claim** nghĩa là câu tuyên bố chính thức về năng lực của đề tài. Claim nên vừa đủ mạnh để có giá trị, nhưng không được quá đà so với dữ liệu và thí nghiệm hiện có.

> Đây là một nguyên mẫu hệ thống giám sát buồn ngủ tài xế bằng thị giác máy tính, có trích xuất đặc trưng theo thời gian và đánh giá nghiên cứu dựa trên nhãn buồn ngủ KSS đồng bộ theo thời gian.

### Chưa Nên Dùng Claim Này

> Đây là một hệ thống phát hiện mệt mỏi đã được xác thực lâm sàng.

### Việc Cần Làm

- Xem các báo cáo cũ trong `docs/old/` là tài liệu audit lịch sử, không phải roadmap hiện tại.
- Giữ runtime module hiện tại làm nền tảng kỹ thuật chính.
- Tạm dừng ưu tiên train model lớn hơn cho đến khi có lớp dữ liệu/đánh giá tốt hơn.
- Tách rõ claim demo kỹ thuật và claim nghiên cứu.

### Deliverables

- `docs/DA1_RESEARCH_ROADMAP.md`
- `docs/DA1_RESEARCH_ROADMAP_VI.md`
- Có thể thêm sau: `docs/research_protocol.md`
- Có thể thêm sau: `docs/label_schema.md`

## 7. Phase 1: Protocol Nghiên Cứu Và Schema Nhãn

### Mục Tiêu

Định nghĩa rõ hệ thống đo cái gì, nhãn được tạo như thế nào, và đánh giá ra sao.

### Đơn Vị Nhãn Đề Xuất

Dùng cửa sổ thời gian, không dùng random frame.

Mặc định đề xuất:

- Độ dài window: 60 giây
- Stride: 10 giây. Stride là bước trượt: cứ mỗi 10 giây tạo một window mới.
- Nguồn nhãn: điểm KSS gần nhất, hoặc điểm KSS được thu trong khoảng thời gian đó

Phương án cho experiment nhỏ:

- Độ dài window: 30 giây
- Stride: 5 giây

Ví dụ dễ hiểu:

```text
Video dài 180 giây
Window = 60 giây
Stride = 10 giây

Window 1: 0-60 giây
Window 2: 10-70 giây
Window 3: 20-80 giây
...
```

Cách này giúp hệ thống đánh giá theo trạng thái trong một đoạn thời gian, thay vì phán đoán buồn ngủ chỉ từ một frame đơn lẻ.

### Schema Nhãn Đề Xuất

```csv
subject_id,session_id,video_id,start_time_sec,end_time_sec,kss_score,kss_band,notes
S01,session_01,video_01,0,60,3,alert,baseline awake
S01,session_01,video_01,60,120,5,caution,early signs
S01,session_01,video_01,120,180,8,sleepy,clear drowsiness
```

### Quy Tắc Chia Train/Test Theo Chủ Thể

Phải dùng subject-disjoint evaluation.

**Subject** nghĩa là người tham gia/thực nghiệm viên. **Subject-disjoint** nghĩa là người đã dùng để train model thì không được xuất hiện trong test. Đây là điểm rất quan trọng trong đề tài có dữ liệu khuôn mặt.

Không tốt:

```text
Randomly split frames from all subjects into train/test.
```

Tốt:

```text
Train subjects: S01-S07
Validation subjects: S08
Test subjects: S09-S10
```

Lý do: nếu cùng một người xuất hiện ở cả train và test, model có thể học đặc điểm khuôn mặt, ánh sáng, góc camera hoặc background thay vì học buồn ngủ.

### Deliverables

- `docs/research_protocol.md`
- `docs/label_schema.md`
- `metadata/kss_annotations.csv`
- `metadata/subject_split.csv`

## 8. Phase 2: Offline Feature Exporter

### Mục Tiêu

Biến video thành dữ liệu nghiên cứu có thể phân tích được.

UI OpenCV realtime hữu ích cho demo, nhưng nghiên cứu cần file đặc trưng và prediction được export ra CSV/Parquet.

### Script Mới Đề Xuất

```text
tools/research/export_video_features.py
```

### Input Arguments Đề Xuất

```text
--video-path <path>
--subject-id <id>
--session-id <id>
--output-csv <path>
--window-seconds 60
--stride-seconds 10
```

### Cột Output Theo Frame

```text
subject_id
session_id
video_id
frame_index
timestamp_sec
ear
mar
pitch
yaw
roll
eye_closed
mouth_open
head_nod_candidate
gaze_x
gaze_y
fsm_state
fsm_evidence
```

### Cột Output Theo Window

```text
subject_id
session_id
video_id
window_start_sec
window_end_sec
mean_ear
min_ear
mean_mar
max_mar
perclos_30s
perclos_60s
max_eye_closed_duration_sec
ocular_lapse_count
blink_rate_per_min
yawn_count
mean_yawn_duration_sec
head_drop_count
max_pitch_velocity_deg_s
gaze_dispersion
fsm_state_mode
fsm_max_evidence
```

### Ghi Chú Implementation

- Reuse `runtime/perception.py`, không nên copy lại logic EAR/MAR/head pose.
- Reuse hoặc adapt `runtime/features.py` cho đặc trưng theo thời gian.
- Nếu có thể, export cả raw signals và derived features.
- Lưu timestamp theo FPS của video hoặc timing thực tế khi capture.

### Deliverables

- `tools/research/export_video_features.py`
- `reports/features/frame_features.csv`
- `reports/features/window_features.csv`

## 9. Phase 3: Chiến Lược Dataset

### Mục Tiêu

Dùng nhãn đủ tốt để bảo vệ claim nghiên cứu.

### Chiến Lược Đề Xuất

Dùng hai track dataset:

1. **Dataset buồn ngủ public** để validation có tính khoa học.
2. **Dataset webcam nhỏ tự thu** để demo local và domain adaptation.

### Track A: Dataset Public

DROZY hữu ích vì có KSS, PVT reaction time, tín hiệu PSG và dữ liệu video/liên quan video. Nó có cơ sở nghiên cứu tốt hơn việc tự gán cả video là “drowsy”.

Dùng DROZY để trả lời:

```text
Pipeline đặc trưng hiện tại có phân loại được các window buồn ngủ có nhãn KSS trên dataset được công nhận không?
```

### Track B: Dataset Pilot Tự Thu

Để demo Đồ Án 1 ở HUST, nên có một dataset nhỏ tự thu.

Protocol gợi ý:

```text
Subjects: 5-10
Sessions per subject: 2-3
Session duration: 10-20 minutes
KSS prompt interval: every 1-2 minutes
Camera: normal webcam
Conditions: normal light, low light, glasses/no glasses if possible
```

Quy tắc an toàn:

Không yêu cầu người tham gia thiếu ngủ nguy hiểm. Dataset pilot có thể gồm session mệt tự nhiên, ngáp giả lập cho event testing, và session tỉnh táo để đo false alert.

### Dataset Manifest

```csv
video_id,subject_id,session_id,path,fps,duration_sec,condition,has_kss,has_pvt,notes
video_01,S01,session_01,data/videos/S01_session_01.mp4,30,900,normal_light,true,false,custom pilot
```

### Deliverables

- `metadata/data_manifest.csv`
- `metadata/kss_annotations.csv`
- `metadata/subject_split.csv`
- Có thể thêm: `metadata/pvt_results.csv`

## 10. Phase 4: Baseline Experiments

### Mục Tiêu

Thiết lập các baseline đơn giản và bảo vệ được trước khi dùng ML fusion.

**Baseline** là mốc so sánh đơn giản. Nếu model phức tạp không tốt hơn baseline, thì model phức tạp chưa có giá trị rõ ràng.

### Baseline Bắt Buộc

| Baseline | Mô tả |
|---|---|
| Eye-closure rule | Chỉ dùng nhắm mắt kéo dài |
| PERCLOS-only | Chỉ dùng ngưỡng PERCLOS |
| Yawn-only | Chỉ dùng số lượng/thời lượng episode ngáp |
| Head-pose-only | Chỉ dùng head-drop hoặc pitch dynamics |
| Current FSM | Dùng decision engine đa tín hiệu hiện tại |

### Metrics

Cần report:

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
False alerts per minute
Detection latency
```

Với thesis, nên ưu tiên:

- Recall cho window sleepy
- False alerts per minute trong window alert
- F1-score để so sánh cân bằng
- Confusion matrix để dễ giải thích

### Deliverables

- `tools/research/evaluate_baselines.py`
- `reports/baseline_results.csv`
- `reports/confusion_matrices/`
- `reports/figures/perclos_vs_kss.png`
- `reports/figures/fsm_state_timeline.png`

## 11. Phase 5: Random Forest Fusion Và Feature Importance

### Mục Tiêu

Dùng Random Forest như một baseline fusion theo window có tính giải thích.

Không nên xem Random Forest là “magic detector”. Nó nên được dùng để kiểm tra xem learned feature fusion có cải thiện so với simple rules hay không.

Nói đơn giản:

- **Random Forest**: model học từ nhiều đặc trưng theo window để dự đoán `alert`, `caution`, hoặc `sleepy`.
- **Fusion**: gộp nhiều tín hiệu lại, ví dụ mắt + ngáp + đầu + gaze.
- **Feature importance**: xem feature nào có vẻ quan trọng hơn trong quyết định của model.

### Cách Huấn Luyện Đúng

Dùng cấu trúc này:

```text
window-level features -> KSS band -> subject-disjoint train/test split
```

Không dùng cấu trúc này:

```text
random frames -> whole-video binary labels -> random train/test split
```

### Feature Đề Xuất

```text
perclos_60s
perclos_30s
max_eye_closed_duration_sec
ocular_lapse_count
blink_rate_per_min
mean_ear
min_ear
yawn_count
mean_yawn_duration_sec
max_mar
head_drop_count
max_pitch_velocity_deg_s
gaze_dispersion
```

### Model Cần So Sánh

Tối thiểu:

- PERCLOS-only rule
- Current FSM
- Random Forest

Có thể thêm:

- Logistic Regression
- Gradient Boosting
- XGBoost hoặc LightGBM nếu phù hợp constraint của đồ án

### Feature Importance

Nên dùng ba cách:

1. Built-in Random Forest importance để có overview nhanh.
2. Permutation importance để giải thích đáng tin hơn.
3. Ablation study để có evidence mạnh nhất cho thesis.

Giải thích thêm:

- Built-in importance dễ lấy nhưng có thể bị lệch, nên chỉ dùng để tham khảo ban đầu.
- Permutation importance đáng tin hơn vì nó đo kết quả giảm thế nào khi một feature bị xáo trộn.
- Ablation study dễ giải thích nhất trong báo cáo: bỏ nhóm feature đó đi, nếu F1-score giảm mạnh thì nhóm đó thật sự có ích.

Ví dụ ablation plan:

| Variant | Mục đích |
|---|---|
| Full features | Model đầy đủ tốt nhất |
| Eye-only | Đóng góp của nhóm ocular |
| PERCLOS-only | Baseline đơn giản mạnh |
| No PERCLOS | Vai trò của closure-time accumulation |
| No yawn features | Vai trò của mouth/yawn cues |
| No head-pose features | Vai trò của head dynamics |
| No gaze features | Vai trò của gaze stability |

### Deliverables

- `tools/research/train_window_fusion_model.py`
- `reports/random_forest_results.csv`
- `reports/feature_importance.csv`
- `reports/permutation_importance.csv`
- `reports/ablation_results.csv`

## 12. Phase 6: Cải Thiện Runtime Dựa Trên Evidence

### Mục Tiêu

Cập nhật demo realtime sau khi đã có kết quả đánh giá, không tune bằng cảm giác.

### Cải Tiến Nên Ưu Tiên

1. Chuyển PERCLOS sang timestamp-based thay vì chỉ dựa trên config FPS.
2. Chuyển yawn detection từ frame-count threshold sang episode duration theo giây.
3. Chuyển head nod detection từ static pitch threshold sang pitch drop + rebound dynamics.
4. Tune threshold FSM bằng validation results.
5. Thêm config nghiên cứu/evaluation riêng với config demo mặc định.

### Tách Config Runtime

```text
config.json              # config demo mặc định
config_research.json     # threshold được chọn từ validation experiments
```

### Deliverables

- Cập nhật `perclos.py`
- Cập nhật `runtime/features.py`
- Cập nhật `fsm.py` hoặc thêm research engine mới
- `config_research.json`
- `reports/runtime_comparison.csv`

## 13. Phase 7: Cấu Trúc Báo Cáo Đồ Án 1

### Chương 1: Giới Thiệu

- Bài toán buồn ngủ khi lái xe
- Vì sao camera-based monitoring hữu ích
- Động lực đề tài
- Phạm vi và giới hạn
- Câu hỏi nghiên cứu

### Chương 2: Cơ Sở Lý Thuyết

- Buồn ngủ là trạng thái ẩn
- KSS và nhãn buồn ngủ
- PERCLOS
- EAR và MAR
- Head pose và gaze features
- FSM và temporal decision logic
- Random Forest và interpretable feature fusion

### Chương 3: Thiết Kế Hệ Thống

- Kiến trúc runtime
- Perception pipeline
- Temporal feature pipeline
- FSM decision engine
- Offline evaluation pipeline
- Data flow diagram

### Chương 4: Dataset Và Gán Nhãn

- Nguồn dataset
- Mapping nhãn KSS
- Cách tạo window
- Subject-disjoint split
- Ghi chú chất lượng dữ liệu

### Chương 5: Thí Nghiệm Và Kết Quả

- Baseline methods
- Random Forest fusion
- Feature importance
- Ablation study
- Error analysis
- Kết quả runtime demo

### Chương 6: Kết Luận

- Hệ thống làm được gì
- Giới hạn còn lại
- Hướng phát triển cho Đồ Án Tốt Nghiệp / luận văn tốt nghiệp

## 14. Timeline Đề Xuất

Cho kế hoạch 8-10 tuần:

| Tuần | Trọng tâm |
|---|---|
| 1 | Reset phạm vi, research protocol, label schema |
| 2 | Offline feature exporter |
| 3 | Dataset adapter và format KSS annotation |
| 4 | Window feature generation và KSS alignment |
| 5 | Baseline evaluation |
| 6 | Random Forest fusion và feature importance |
| 7 | Ablation study và error analysis |
| 8 | Cải thiện runtime FSM |
| 9 | Viết báo cáo, hình, bảng |
| 10 | Demo cuối, cleanup, slides bảo vệ |

## 15. Nên Làm Gì Đầu Tiên

Các bước thực tế tiếp theo:

1. Tạo `docs/research_protocol.md`.
2. Tạo `docs/label_schema.md`.
3. Tạo `tools/research/export_video_features.py`.
4. Tạo ví dụ nhỏ `metadata/kss_annotations.csv`.
5. Chạy một video qua exporter và kiểm tra feature sinh ra.

Không nên bắt đầu bằng việc train model lớn hơn. Hãy bắt đầu bằng việc làm cho project đo lường được.

## 16. Những Điều Cần Tránh

Tránh các lỗi sau:

- Không dùng random frame-level train/test split cho claim về trạng thái buồn ngủ.
- Không dùng whole-video binary labels làm ground truth nghiên cứu.
- Không claim hệ thống là clinical fatigue detection.
- Không chỉ dựa vào accuracy.
- Không dùng Gini feature importance đơn lẻ làm bằng chứng.
- Không tune threshold FSM chỉ bằng cảm giác nhìn demo.
- Không để thesis AI-generated cũ định nghĩa roadmap hiện tại.

## 17. Contribution Đồ Án 1 Đề Xuất

Contribution mạnh nhất cho Đồ Án 1:

> Một pipeline giám sát buồn ngủ bằng thị giác máy tính có thể đo lường và giải thích được, so sánh các đặc trưng temporal rule-based và Random Forest feature fusion với nhãn buồn ngủ KSS-aligned, sử dụng subject-disjoint evaluation.

Contribution này bảo vệ được vì kết hợp:

- Thị giác máy tính
- Xử lý tín hiệu theo thời gian
- Gán nhãn ground-truth
- Machine learning có tính giải thích
- Metrics đánh giá đúng
- Khả năng demo realtime

## 18. Tài Liệu Tham Khảo Cho Hướng Nghiên Cứu

- Kaida et al., “Validation of the Karolinska Sleepiness Scale against performance and EEG variables,” Clinical Neurophysiology, 2006: https://pubmed.ncbi.nlm.nih.gov/16679057/
- Abe, “PERCLOS-based technologies for detecting drowsiness: current evidence and future directions,” Sleep Advances, 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10108649/
- DROZY: The ULg Multimodality Drowsiness Database: https://www.drozy.ulg.ac.be/
- Euro NCAP Safe Driving Driver Engagement Protocol, Version 1.1, October 2025: https://cdn.euroncap.com/cars/assets/euro_ncap_protocol_safe_driving_driver_engagement_v11_a30e874152.pdf

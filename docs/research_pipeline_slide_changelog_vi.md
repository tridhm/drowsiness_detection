# Changelog Ngắn Gọn Cho Slide: Research Pipeline MVP

## Slide 1 — Vấn Đề Trước Đây

- Hệ thống chủ yếu chạy theo kiểu demo realtime: mở video/webcam và xem cảnh báo trên UI.
- Cách này tốt để trình diễn, nhưng khó dùng để đánh giá nghiên cứu.
- Sau khi video chạy xong, không có bảng dữ liệu rõ ràng để tính accuracy, recall, F1, false alarm.
- Random Forest cũ train theo từng frame với label 0/1 khá thô, chưa phù hợp để chứng minh drowsiness theo thời gian.

## Slide 2 — Thay Đổi Chính

Em bổ sung một pipeline nghiên cứu offline:

```text
Video có sẵn
-> xuất đặc trưng theo frame/window
-> ghép với nhãn KSS nếu có
-> tạo bảng labeled_windows.csv
-> đánh giá baseline rule-based
-> train thử Random Forest window-level
-> xuất metrics và bảng phân tích feature
```

Ý nghĩa: chuyển từ “nhìn demo UI để nhận xét” sang “xuất dữ liệu để đánh giá hàng loạt bằng metric”.

## Slide 3 — Offline Feature Exporter

File mới:

```text
tools/research/export_video_features.py
```

Tool này đọc video và xuất 2 file:

- `frame_features.csv`: mỗi dòng là một frame.
- `window_features.csv`: mỗi dòng là một đoạn thời gian, ví dụ 10s hoặc 60s.

Ví dụ `frame_features.csv` ghi:

```text
frame_index, timestamp_sec, EAR, MAR, head pose, eye_closed, mouth_open, FSM state
```

Ví dụ `window_features.csv` ghi:

```text
0-60s: mean_EAR, min_EAR, PERCLOS, yawn_count, head_drop_count, FSM_state_mode
```

## Slide 4 — Vì Sao Cần Window Features?

- Một frame đơn lẻ không đủ kết luận buồn ngủ.
- Mắt nhắm trong một frame có thể chỉ là chớp mắt.
- Drowsiness là trạng thái theo thời gian, nên cần gom nhiều frame thành window.
- Các feature theo window như PERCLOS, eye-closed duration, yawn count, head-drop count phù hợp hơn cho nghiên cứu.

Nói ngắn:

```text
Frame = dữ liệu chi tiết để debug.
Window = dữ liệu chính để đánh giá/train.
```

## Slide 5 — KSS Label Alignment

File mới:

```text
tools/research/align_window_labels.py
```

Tool này không tự đoán KSS từ video.

Nó cần một file annotation KSS có sẵn, ví dụ:

```text
0-60s: KSS 3 -> alert
60-120s: KSS 8 -> sleepy
```

Sau đó tool tự động ghép mỗi window với đoạn KSS phù hợp nhất dựa trên thời gian overlap.

Output:

```text
labeled_windows.csv = window_features.csv + kss_score + kss_band
```

## Slide 6 — Baseline Evaluator

File mới:

```text
tools/research/evaluate_baselines.py
```

Baseline là mốc so sánh đơn giản trước khi dùng model phức tạp.

Các baseline hiện có:

- `PERCLOS-only`: chỉ dùng tỷ lệ mắt nhắm.
- `Yawn-only`: chỉ dùng số lần ngáp.
- `Head-pose-only`: chỉ dùng tín hiệu cúi/gật đầu.
- `FSM-state`: dùng decision engine hiện tại của hệ thống.

Mục đích: kiểm tra Random Forest có thật sự tốt hơn luật đơn giản không.

## Slide 7 — Random Forest Window-Fusion MVP

File mới:

```text
tools/research/train_window_fusion_model.py
```

Random Forest mới không train theo từng frame như trước.

Cách mới:

```text
window-level features -> KSS label -> sleepy/non-sleepy
```

Feature dùng nhiều nhóm tín hiệu:

- Mắt: EAR, PERCLOS, eye-closed duration.
- Miệng/ngáp: MAR, yawn_count.
- Đầu: head_drop_count, pitch_velocity.
- FSM: evidence và state của decision engine.

Mục tiêu: kiểm tra việc kết hợp nhiều tín hiệu theo thời gian có tốt hơn baseline không.

## Slide 8 — Output Báo Cáo

Pipeline xuất các file report:

```text
baseline_results.csv
baseline_confusion_matrix.csv
random_forest_results.csv
feature_importance.csv
permutation_importance.csv
ablation_results.csv
```

Ý nghĩa:

- `baseline_results.csv`: so sánh các luật đơn giản.
- `random_forest_results.csv`: metric của Random Forest.
- `feature_importance.csv`: feature nào model dùng nhiều.
- `permutation_importance.csv`: feature nào làm model giảm mạnh nếu bị xáo trộn.
- `ablation_results.csv`: bỏ từng nhóm feature để xem nhóm nào quan trọng.

## Slide 9 — Kết Quả Kỹ Thuật Đã Kiểm Tra

Đã chạy validation:

```text
unittest discover tests -> 12 tests OK
py_compile tools/research/*.py -> OK
```

Đã smoke test exporter trên video có sẵn:

```text
Video Database/0392.mp4
max_frames = 150
output = 150 frame rows + 2 window rows
```

Đã kiểm tra baseline/RF bằng synthetic fixture để đảm bảo tool chạy end-to-end.

## Slide 10 — Giới Hạn Và Cách Nói An Toàn

Hiện tại chưa có dataset KSS/PVT thật trong folder.

Vì vậy chưa được claim:

```text
Random Forest chứng minh hệ thống phát hiện buồn ngủ chính xác.
```

Cách nói đúng hơn:

```text
Em đã triển khai pipeline nghiên cứu offline để xuất feature, ghép nhãn KSS, đánh giá baseline và train thử Random Forest window-level. Pipeline đã chạy được bằng fixture và video có sẵn. Kết quả performance thật sẽ được đánh giá sau khi có dữ liệu KSS/PVT-labeled phù hợp.
```

## Slide 11 — Nếu Bị Hỏi: Vì Sao Meaningful?

Trả lời ngắn:

- Trước đây chủ yếu đánh giá bằng demo UI.
- Bây giờ có thể xuất CSV để đánh giá hàng loạt.
- Có thể so sánh PERCLOS-only, FSM và Random Forest bằng cùng một bảng dữ liệu.
- Có thể tính metric rõ ràng như precision, recall, F1, confusion matrix.
- Có nền tảng để làm thesis/research evaluation khi có KSS labels thật.

## Slide 12 — Next Steps

- Thu thập hoặc import annotation KSS thật.
- Chạy exporter trên nhiều video.
- Tạo `labeled_windows.csv` thật.
- Chạy baseline và Random Forest bằng subject-disjoint split.
- Dùng metrics, feature importance và ablation để viết báo cáo/thesis.

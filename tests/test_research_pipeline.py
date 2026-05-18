import csv
import tempfile
import unittest
from pathlib import Path


class ResearchWindowAggregationTests(unittest.TestCase):
    def test_window_aggregation_computes_temporal_features(self):
        from tools.research.common import aggregate_frame_rows

        frame_rows = [
            {
                "subject_id": "S01",
                "session_id": "A",
                "video_id": "V1",
                "timestamp_sec": "0.0",
                "face_detected": "1",
                "ear": "0.20",
                "mar": "0.30",
                "eye_closed": "1",
                "head_nod_detected": "0",
                "perclos_60s": "0.25",
                "perclos_5s": "0.20",
                "blink_frequency": "1",
                "yawn_frequency": "0",
                "pitch_velocity": "0.10",
                "fsm_state": "ALERT",
                "fsm_evidence": "0.10",
            },
            {
                "subject_id": "S01",
                "session_id": "A",
                "video_id": "V1",
                "timestamp_sec": "1.0",
                "face_detected": "1",
                "ear": "0.10",
                "mar": "0.70",
                "eye_closed": "1",
                "head_nod_detected": "1",
                "perclos_60s": "0.50",
                "perclos_5s": "0.60",
                "blink_frequency": "2",
                "yawn_frequency": "1",
                "pitch_velocity": "1.25",
                "fsm_state": "DROWSY",
                "fsm_evidence": "0.90",
            },
            {
                "subject_id": "S01",
                "session_id": "A",
                "video_id": "V1",
                "timestamp_sec": "2.0",
                "face_detected": "0",
                "ear": "0.30",
                "mar": "0.20",
                "eye_closed": "0",
                "head_nod_detected": "0",
                "perclos_60s": "0.40",
                "perclos_5s": "0.30",
                "blink_frequency": "2",
                "yawn_frequency": "1",
                "pitch_velocity": "0.05",
                "fsm_state": "DROWSY",
                "fsm_evidence": "0.30",
            },
        ]

        windows = aggregate_frame_rows(frame_rows, window_seconds=2.0, stride_seconds=2.0)

        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual(window["subject_id"], "S01")
        self.assertEqual(window["frame_count"], 2)
        self.assertAlmostEqual(window["valid_face_ratio"], 1.0)
        self.assertAlmostEqual(window["mean_ear"], 0.15)
        self.assertAlmostEqual(window["min_ear"], 0.10)
        self.assertAlmostEqual(window["mean_mar"], 0.50)
        self.assertAlmostEqual(window["max_mar"], 0.70)
        self.assertAlmostEqual(window["perclos_60s"], 0.50)
        self.assertAlmostEqual(window["max_eye_closed_duration_sec"], 2.0)
        self.assertAlmostEqual(window["blink_rate_per_min"], 60.0)
        self.assertEqual(window["yawn_count"], 1)
        self.assertEqual(window["head_drop_count"], 1)
        self.assertAlmostEqual(window["max_pitch_velocity"], 1.25)
        self.assertEqual(window["fsm_state_mode"], "DROWSY")


class LabelAlignmentTests(unittest.TestCase):
    def test_alignment_uses_maximum_time_overlap_and_skips_unlabeled(self):
        from tools.research.align_window_labels import align_windows

        windows = [
            {
                "subject_id": "S01",
                "session_id": "A",
                "video_id": "V1",
                "window_start_sec": "0",
                "window_end_sec": "10",
            },
            {
                "subject_id": "S01",
                "session_id": "A",
                "video_id": "V1",
                "window_start_sec": "10",
                "window_end_sec": "20",
            },
            {
                "subject_id": "S02",
                "session_id": "B",
                "video_id": "V2",
                "window_start_sec": "0",
                "window_end_sec": "10",
            },
        ]
        annotations = [
            {
                "subject_id": "S01",
                "session_id": "A",
                "video_id": "V1",
                "start_time_sec": "0",
                "end_time_sec": "8",
                "kss_score": "3",
                "kss_band": "alert",
                "notes": "awake",
            },
            {
                "subject_id": "S01",
                "session_id": "A",
                "video_id": "V1",
                "start_time_sec": "8",
                "end_time_sec": "20",
                "kss_score": "8",
                "kss_band": "sleepy",
                "notes": "sleepy",
            },
        ]

        labeled = align_windows(windows, annotations, keep_unlabeled=False)

        self.assertEqual(len(labeled), 2)
        self.assertEqual(labeled[0]["kss_score"], "3")
        self.assertEqual(labeled[0]["kss_band"], "alert")
        self.assertEqual(labeled[1]["kss_score"], "8")
        self.assertEqual(labeled[1]["kss_band"], "sleepy")
        self.assertEqual(labeled[1]["label_source"], "overlap_seconds=10.000")


class BaselineEvaluatorTests(unittest.TestCase):
    def test_baseline_evaluator_produces_deterministic_metrics(self):
        from tools.research.evaluate_baselines import evaluate_baseline_rows

        rows = [
            {"kss_band": "alert", "perclos_60s": "0.10", "yawn_count": "0", "head_drop_count": "0", "fsm_state_mode": "ALERT"},
            {"kss_band": "sleepy", "perclos_60s": "0.45", "yawn_count": "0", "head_drop_count": "0", "fsm_state_mode": "ALERT"},
            {"kss_band": "sleepy", "perclos_60s": "0.50", "yawn_count": "1", "head_drop_count": "1", "fsm_state_mode": "DROWSY"},
            {"kss_band": "alert", "perclos_60s": "0.20", "yawn_count": "1", "head_drop_count": "0", "fsm_state_mode": "SUSPICIOUS"},
        ]

        results, matrices = evaluate_baseline_rows(rows, perclos_threshold=0.35)

        by_name = {row["baseline"]: row for row in results}
        self.assertAlmostEqual(float(by_name["perclos_only"]["f1"]), 1.0)
        self.assertAlmostEqual(float(by_name["yawn_only"]["precision"]), 0.5)
        self.assertEqual(matrices["fsm_state"]["tp"], 1)
        self.assertEqual(matrices["fsm_state"]["fn"], 1)


class RandomForestFusionTests(unittest.TestCase):
    def _write_labeled_windows(self, path: Path, subjects: list[str]) -> None:
        fieldnames = [
            "subject_id",
            "session_id",
            "video_id",
            "kss_band",
            "mean_ear",
            "min_ear",
            "perclos_60s",
            "perclos_5s",
            "max_eye_closed_duration_sec",
            "blink_rate_per_min",
            "mean_mar",
            "max_mar",
            "yawn_count",
            "head_drop_count",
            "max_pitch_velocity",
            "mean_fsm_evidence",
            "max_fsm_evidence",
            "fsm_state_mode",
        ]
        rows = []
        for subject in subjects:
            rows.extend(
                [
                    [subject, "A", f"{subject}_1", "alert", 0.30, 0.25, 0.10, 0.10, 0.0, 12, 0.20, 0.30, 0, 0, 0.10, 0.10, 0.20, "ALERT"],
                    [subject, "A", f"{subject}_2", "sleepy", 0.16, 0.10, 0.55, 0.70, 3.0, 3, 0.70, 0.90, 2, 1, 2.00, 0.80, 0.95, "DROWSY"],
                ]
            )
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            writer.writerows(rows)

    def test_random_forest_outputs_reports_with_subject_disjoint_data(self):
        from tools.research.train_window_fusion_model import train_window_fusion_model

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "labeled_windows.csv"
            self._write_labeled_windows(input_csv, ["S01", "S02", "S03", "S04"])

            outputs = train_window_fusion_model(
                input_csv=input_csv,
                output_dir=tmp_path / "reports",
                random_state=7,
                n_estimators=10,
                allow_random_split=False,
                test_subject_ratio=0.5,
                permutation_repeats=2,
            )

            for path in outputs.values():
                self.assertTrue(path.exists(), path)

            with outputs["feature_importance"].open("r", encoding="utf-8") as f:
                text = f.read()
            self.assertIn("perclos_60s", text)
            self.assertIn("importance", text)

    def test_random_forest_refuses_single_subject_without_fallback(self):
        from tools.research.train_window_fusion_model import train_window_fusion_model

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "labeled_windows.csv"
            self._write_labeled_windows(input_csv, ["S01"])

            with self.assertRaisesRegex(ValueError, "subject-disjoint"):
                train_window_fusion_model(
                    input_csv=input_csv,
                    output_dir=tmp_path / "reports",
                    random_state=7,
                    n_estimators=10,
                    allow_random_split=False,
                    test_subject_ratio=0.5,
                    permutation_repeats=2,
                )


if __name__ == "__main__":
    unittest.main()

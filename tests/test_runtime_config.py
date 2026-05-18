import json
import tempfile
import unittest
from pathlib import Path

from runtime.config import load_runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_cli_overrides_json_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "input": {"source": "file", "video_path": "sample.mp4"},
                        "decision_engine": "legacy",
                        "enable_legacy_feature_overlay": True,
                    }
                ),
                encoding="utf-8",
            )

            cfg = load_runtime_config(
                str(cfg_path),
                {
                    "source": "webcam",
                    "decision_engine": "fsm",
                    "enable_legacy_feature_overlay": False,
                },
            )

            self.assertEqual(cfg.input.source, "webcam")
            self.assertEqual(cfg.input.video_path, "sample.mp4")
            self.assertEqual(cfg.decision_engine, "fsm")
            self.assertFalse(cfg.enable_legacy_feature_overlay)

    def test_missing_config_uses_defaults(self):
        cfg = load_runtime_config(None, {})
        self.assertEqual(cfg.input.source, "webcam")
        self.assertEqual(cfg.decision_engine, "fsm")


if __name__ == "__main__":
    unittest.main()

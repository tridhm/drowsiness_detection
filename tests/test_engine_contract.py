import unittest

from fsm import DrowsinessSignals, DrowsinessState
from runtime.config import default_runtime_config
from runtime.contracts import EngineContext
from runtime.engines.registry import create_engine


class EngineContractTests(unittest.TestCase):
    def test_fsm_engine_contract(self):
        config = default_runtime_config()
        engine = create_engine("fsm", config)
        engine.initialize(EngineContext(fps=config.runtime.fps))

        result = engine.update(DrowsinessSignals())
        self.assertIsInstance(result.state, DrowsinessState)
        self.assertIsInstance(result.evidence, float)
        self.assertTrue(hasattr(result, "alert_sound"))

    def test_legacy_engine_contract(self):
        config = default_runtime_config()
        engine = create_engine("legacy", config)
        engine.initialize(EngineContext(fps=config.runtime.fps))

        signals = DrowsinessSignals(
            ear=0.1,
            mar=0.8,
            perclos=0.5,
            perclos_short=0.7,
            yawn_frequency=4,
            blink_frequency=15,
            head_nod_detected=True,
            eyes_closed_consecutive=25,
            ear_below_threshold=True,
            mar_above_threshold=True,
            pitch_above_threshold=True,
        )
        result = engine.update(signals)
        self.assertIn(result.state, list(DrowsinessState))
        self.assertGreaterEqual(result.evidence, 0.0)


if __name__ == "__main__":
    unittest.main()

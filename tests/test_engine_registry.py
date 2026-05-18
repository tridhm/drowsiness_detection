import unittest

from runtime.config import default_runtime_config
from runtime.engines.registry import available_engines, create_engine


class EngineRegistryTests(unittest.TestCase):
    def test_registry_contains_expected_engines(self):
        names = available_engines()
        self.assertIn("fsm", names)
        self.assertIn("legacy", names)

    def test_create_engine_success(self):
        config = default_runtime_config()
        engine = create_engine("fsm", config)
        self.assertTrue(hasattr(engine, "initialize"))
        self.assertTrue(hasattr(engine, "update"))
        self.assertTrue(hasattr(engine, "reset"))

    def test_create_engine_invalid(self):
        config = default_runtime_config()
        with self.assertRaises(ValueError):
            create_engine("unknown", config)


if __name__ == "__main__":
    unittest.main()

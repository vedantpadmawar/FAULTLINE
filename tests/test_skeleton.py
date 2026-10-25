"""Skeleton and repository integrity tests."""

import sys
from pathlib import Path

# Ensure src is in python path
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import yaml
import faultline


import unittest


class TestSkeleton(unittest.TestCase):
    def test_package_metadata(self):
        """Verify package import and version."""
        self.assertEqual(faultline.__version__, "0.1.0")

    def test_metrics_yaml_exists_and_valid(self):
        """Verify that config/metrics.yaml exists and parses with all 23 defined metrics."""
        metrics_path = Path(__file__).resolve().parent.parent / "config" / "metrics.yaml"
        self.assertTrue(metrics_path.exists(), f"Missing config file at {metrics_path}")

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertIn("metrics", data, "metrics key missing from metrics.yaml")
        metrics = data["metrics"]
        self.assertEqual(len(metrics), 23, f"Expected 23 metrics in registry, found {len(metrics)}")

        # Check that required fields exist on every metric
        for m in metrics:
            self.assertIn("id", m, "Metric missing 'id'")
            self.assertIn("what", m, f"Metric {m.get('id')} missing 'what'")
            self.assertIn("unit", m, f"Metric {m.get('id')} missing 'unit'")
            self.assertIn("direction", m, f"Metric {m.get('id')} missing 'direction'")
            self.assertIn("measure", m, f"Metric {m.get('id')} missing 'measure'")

        # Verify key gate metrics are flagged as gate: true
        gate_metric_ids = {m["id"] for m in metrics if m.get("gate") is True}
        expected_gates = {"M03_unsupported_claims", "M04_llm_authored_numbers", "M14_determinism"}
        self.assertTrue(expected_gates.issubset(gate_metric_ids), f"Missing gate flags: {expected_gates - gate_metric_ids}")


if __name__ == "__main__":
    unittest.main()

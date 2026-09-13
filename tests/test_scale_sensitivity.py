import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "35_scale_sensitivity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("scale_sensitivity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example_mainshocks():
    x = np.arange(8.0)
    return pd.DataFrame(
        {
            "mainshock_id": [f"g{i}" for i in range(8)],
            "p": 0.4 + 0.08 * x,
            "mainshock_mag": 6.0 + 0.1 * (x % 3),
            "n_events": 30 + 5 * x,
            "pop_sum_10km": 10 ** (4.0 + 0.08 * x),
            "pop_sum_25km": 10 ** (4.5 + 0.06 * x),
            "pop_sum_50km": 10 ** (5.0 + 0.04 * x),
        }
    )


class ScaleSensitivityTests(unittest.TestCase):
    def test_scale_analysis_returns_requested_radii(self):
        module = load_module()
        out = module.analyze_scales(example_mainshocks(), radii=(10, 25, 50))
        self.assertEqual({row["radius_km"] for row in out}, {10, 25, 50})
        for row in out:
            self.assertIn("loo_pearson_min", row)
            self.assertIn("ridge_mae", row)
            self.assertEqual(row["n"], 8)

    def test_exact_permutation_count_for_eight_groups(self):
        module = load_module()
        out = module.exact_slope_permutation(np.arange(8.0), np.arange(8.0))
        self.assertEqual(out["n_permutations"], 40320)
        self.assertAlmostEqual(out["observed_slope"], 1.0)


if __name__ == "__main__":
    unittest.main()

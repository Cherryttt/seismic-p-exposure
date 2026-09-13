import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, alias):
    spec = importlib.util.spec_from_file_location(alias, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmpiricalPropagationTests(unittest.TestCase):
    def test_record_id_distinguishes_mainshock_and_subgroups(self):
        fit = load_script("03_fit_omori.py", "fit_omori_script")
        main = fit.make_record_id("g1", "mainshock", "all")
        depth = fit.make_record_id("g1", "depth", "0-10km")
        ring = fit.make_record_id("g1", "ring", "0-10km")
        self.assertEqual(len({main, depth, ring}), 3)

    def test_empirical_matrix_contains_only_saved_draws(self):
        propagation = load_script("32_uncertainty_propagation.py", "propagation_script")
        summary = pd.DataFrame(
            {
                "record_id": ["a", "b"],
                "mainshock_id": ["g1", "g2"],
                "p": [0.2, 0.8],
            }
        )
        draws = pd.DataFrame(
            {
                "record_id": ["a", "a", "b", "b"],
                "p_boot": [0.1, 0.3, 0.7, 0.9],
            }
        )
        matrix = propagation.empirical_draw_matrix(summary, draws, n_draws=20, seed=3)
        self.assertEqual(matrix.shape, (2, 20))
        self.assertTrue(set(np.unique(matrix[0])).issubset({0.1, 0.3}))
        self.assertTrue(set(np.unique(matrix[1])).issubset({0.7, 0.9}))

    def test_ml_module_loads_for_full_propagation(self):
        propagation = load_script("32_uncertainty_propagation.py", "propagation_loader")
        ml = propagation._load_ml_module()
        self.assertTrue(callable(ml.run_logo_comparison))


if __name__ == "__main__":
    unittest.main()

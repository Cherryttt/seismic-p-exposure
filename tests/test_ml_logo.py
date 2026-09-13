from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from analysis_core import regression_metrics
from tests.test_analysis_core import example_subgroups


ROOT = Path(__file__).resolve().parents[1]


def load_ml_module():
    spec = importlib.util.spec_from_file_location("ml40", ROOT / "40_ml_groupkfold.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MlLogoTests(unittest.TestCase):
    def test_test_targets_do_not_change_training_thresholds(self) -> None:
        module = load_ml_module()
        first = module.training_quantile_bins(
            np.array([0.2, 0.4, 0.8, 1.0]), np.array([0.3, 0.7])
        )
        second = module.training_quantile_bins(
            np.array([0.2, 0.4, 0.8, 1.0]), np.array([3.0, 7.0])
        )

        self.assertEqual(first.thresholds, second.thresholds)

    def test_fold_r2_is_none_when_test_target_has_no_variance(self) -> None:
        result = regression_metrics([0.5, 0.5], [0.4, 0.6])

        self.assertIsNone(result["r2"])

    def test_group_level_permutation_preserves_within_group_constancy(self) -> None:
        module = load_ml_module()
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

        permuted = module.permute_group_level_feature(values, groups, seed=7)

        for group in np.unique(groups):
            self.assertEqual(np.unique(permuted[groups == group]).size, 1)
        self.assertEqual(sorted(np.unique(permuted)), [1.0, 2.0, 3.0])

    def test_logo_comparison_contains_all_baselines_and_macro_metrics(self) -> None:
        module = load_ml_module()

        result = module.run_logo_comparison(
            example_subgroups(), alpha_grid=(0.1, 1.0), importance_repeats=2, seed=7
        )

        self.assertEqual(set(result["models"]), {"mean", "ols", "ridge"})
        self.assertIn("macro", result["models"]["ridge"])
        self.assertIn("pooled", result["models"]["ridge"])
        self.assertTrue(all("alpha" in fold for fold in result["folds"]))
        self.assertTrue(all("class_thresholds" in fold for fold in result["folds"]))


if __name__ == "__main__":
    unittest.main()

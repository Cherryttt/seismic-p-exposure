from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from analysis_core import ANALYSIS_SPEC
from tests.test_analysis_core import example_subgroups


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RobustnessPipelineTests(unittest.TestCase):
    def test_fe_and_permutation_use_identical_analysis_spec(self) -> None:
        robustness = load_script("robustness31", "31_robustness.py")
        permutation = load_script("permutation34", "34_within_mainshock_permutation.py")
        frame = example_subgroups()

        fe = robustness.run_subgroup_fe(frame, n_boot=30, seed=7)
        perm = permutation.run_permutation(frame, n_perm=20, seed=7)

        self.assertEqual(fe["analysis_spec"], ANALYSIS_SPEC)
        self.assertEqual(fe["analysis_spec"], perm["analysis_spec"])
        self.assertEqual(fe["n_groups"], perm["n_groups"])
        self.assertNotIn("mainshock_mag", fe["coef"])
        self.assertIn("cluster_bootstrap_ci95", fe)
        self.assertEqual(perm["n_perm"], 20)


if __name__ == "__main__":
    unittest.main()

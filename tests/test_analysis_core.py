from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from analysis_core import (
    build_within_design,
    filter_analysis_sample,
    fit_within_fe,
    permute_exposure_within_groups,
    regression_metrics,
)


def example_subgroups() -> pd.DataFrame:
    rows = []
    for group_idx, group in enumerate(("m1", "m2", "m3")):
        magnitude = 6.0 + 0.3 * group_idx
        for row_idx, (kind, label) in enumerate(
            (
                ("depth", "0-10km"),
                ("depth", "10-20km"),
                ("region", "East_of_ms"),
                ("region", "West_of_ms"),
                ("ring", "0-50km"),
                ("ring", "50-400km"),
                ("depth", "all_depth"),
                ("depth", "all_relaxed"),
                ("ring", "0-400km"),
            )
        ):
            rows.append(
                {
                    "mainshock_id": group,
                    "subgroup_type": kind,
                    "subgroup": label,
                    "mainshock_mag": magnitude,
                    "n_events": 25 + row_idx,
                    "pop_sum_subgroup": 10 ** (5.0 + 0.1 * row_idx + 0.05 * group_idx),
                    "p": 0.45 + 0.04 * group_idx - 0.015 * row_idx,
                }
            )
    return pd.DataFrame(rows)


class AnalysisCoreTests(unittest.TestCase):
    def test_filter_analysis_sample_removes_aggregate_fallbacks(self) -> None:
        out = filter_analysis_sample(example_subgroups())

        self.assertFalse(out["subgroup"].isin(["all_depth", "all_relaxed", "0-400km"]).any())
        self.assertEqual(len(out), 18)
        self.assertTrue(out["analysis_included"].all())


    def test_fixed_effect_design_excludes_group_constant_magnitude_and_is_full_rank(self) -> None:
        sample = filter_analysis_sample(example_subgroups())
        design = build_within_design(sample)
        result = fit_within_fe(sample)

        self.assertNotIn("mainshock_mag", design.feature_names)
        self.assertNotIn("mainshock_mag", result["coef"])
        self.assertEqual(design.rank, design.matrix.shape[1])
        self.assertEqual(result["rank"], result["n_parameters"])


    def test_group_permutation_preserves_each_mainshock_exposure_multiset(self) -> None:
        sample = filter_analysis_sample(example_subgroups())
        permuted = permute_exposure_within_groups(sample, seed=7)

        for group, original in sample.groupby("mainshock_id"):
            got = permuted.loc[permuted["mainshock_id"] == group, "pop_sum_subgroup"]
            self.assertEqual(sorted(got.tolist()), sorted(original["pop_sum_subgroup"].tolist()))


    def test_regression_metrics_omits_r2_for_zero_variance_target(self) -> None:
        result = regression_metrics(np.array([0.5, 0.5]), np.array([0.4, 0.6]))

        self.assertIsNone(result["r2"])
        self.assertAlmostEqual(result["mae"], 0.1)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "37_mainshock_influence.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mainshock_influence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example_subgroups():
    rows = []
    for group_index in range(4):
        for row_index, subgroup_type in enumerate(("depth", "region", "ring")):
            rows.append(
                {
                    "mainshock_id": f"g{group_index}",
                    "subgroup_type": subgroup_type,
                    "subgroup": f"{row_index + 1}-{row_index + 2}km" if subgroup_type != "region" else "East_of_ms",
                    "n_events": 25 + row_index,
                    "p": 0.5 + 0.05 * group_index - 0.03 * row_index,
                    "pop_sum_subgroup": 1000 + 200 * group_index + 50 * row_index,
                    "mainshock_mag": 6.0 + 0.1 * group_index,
                }
            )
    return pd.DataFrame(rows)


class MainshockInfluenceTests(unittest.TestCase):
    def test_each_mainshock_is_omitted_once(self):
        module = load_module()
        rows = module.subgroup_influence(
            example_subgroups(), n_perm=5, include_ml=False, seed=2
        )
        self.assertEqual({row["omitted_mainshock"] for row in rows}, {"g0", "g1", "g2", "g3"})
        self.assertTrue(all(row["n_groups"] == 3 for row in rows))
        self.assertTrue(all("fe_exposure_slope" in row for row in rows))


if __name__ == "__main__":
    unittest.main()

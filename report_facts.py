from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def _read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def load_report_facts() -> dict:
    fe = _read_json("regression_controlled_subgroup_fe.json")
    permutation = _read_json("within_mainshock_permutation.json")
    uncertainty = _read_json("uncertainty_propagation.json")
    ml = _read_json("ml_logo_model_comparison.json")
    scale = _read_json("scale_sensitivity.json")
    influence = _read_json("mainshock_influence.json")
    analysis = pd.read_csv(OUT / "subgroup_analysis_sample.csv")
    scale_by_radius = {int(row["radius_km"]): row for row in scale["radii"]}
    return {
        "analysis_n": int(len(analysis)),
        "analysis_groups": int(analysis["mainshock_id"].nunique()),
        "analysis_type_counts": analysis["subgroup_type"].value_counts().to_dict(),
        "fe_slope": fe["coef"]["log10_pop_sum_subgroup"],
        "fe_ci": fe["cluster_bootstrap_ci95"]["log10_pop_sum_subgroup"],
        "fe_negative_probability": fe["cluster_bootstrap_probability_negative"]["log10_pop_sum_subgroup"],
        "fe_r2": fe["r2"],
        "fe_adj_r2": fe["adj_r2"],
        "permutation_observed": permutation["observed_slope"],
        "permutation_p": permutation["p_value_two_sided"],
        "permutation_n": permutation["n_perm"],
        "propagation_draws": uncertainty["n_draws"],
        "propagation_metrics": uncertainty["metrics"],
        "propagation_negative_probability": uncertainty["fe_slope_negative_probability"],
        "ml": ml,
        "scale_by_radius": scale_by_radius,
        "influence_summary": influence["summary"],
        "influence_subgroup_omissions": len(influence["subgroup_rows"]),
        "influence_scale_omissions": len({row["omitted_mainshock"] for row in influence["scale_rows"]}),
    }

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import EXPOSURE_TERM, filter_analysis_sample, fit_within_fe, json_ready, within_group_permutation


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def _load_script(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def subgroup_influence(
    frame: pd.DataFrame,
    *,
    n_perm: int = 1000,
    include_ml: bool = True,
    seed: int = 7,
) -> list[dict]:
    sample = filter_analysis_sample(frame)
    ml = _load_script("40_ml_groupkfold.py", "ml_logo_influence") if include_ml else None
    rows = []
    for index, omitted in enumerate(sorted(sample["mainshock_id"].astype(str).unique())):
        reduced = sample.loc[sample["mainshock_id"].astype(str) != omitted].copy()
        fit = fit_within_fe(reduced)
        permutation = within_group_permutation(
            reduced, n_perm=n_perm, seed=seed + index * (n_perm + 1)
        )
        row = {
            "scope": "subgroup_fe_ml",
            "omitted_mainshock": omitted,
            "n": int(len(reduced)),
            "n_groups": int(reduced["mainshock_id"].nunique()),
            "fe_exposure_slope": float(fit["coef"][EXPOSURE_TERM]),
            "permutation_p": float(permutation["p_value_two_sided"]),
        }
        if include_ml:
            comparison = ml.run_logo_comparison(reduced, importance_repeats=1, seed=seed + index)
            for model in ("mean", "ols", "ridge"):
                metrics = comparison["models"][model]["pooled"]
                row[f"{model}_mae"] = metrics["mae"]
                row[f"{model}_rmse"] = metrics["rmse"]
                row[f"{model}_r2"] = metrics["r2"]
        rows.append(row)
    return rows


def scale_influence(frame: pd.DataFrame, radii=(10, 25, 50)) -> list[dict]:
    scale = _load_script("35_scale_sensitivity.py", "scale_influence_module")
    rows = []
    for omitted in sorted(frame["mainshock_id"].astype(str).unique()):
        reduced = frame.loc[frame["mainshock_id"].astype(str) != omitted].copy()
        for result in scale.analyze_scales(reduced, radii=radii):
            rows.append(
                {
                    "scope": "mainshock_scale",
                    "omitted_mainshock": omitted,
                    "radius_km": result["radius_km"],
                    "n": result["n"],
                    "n_groups": result["n"],
                    "pearson_r": result["pearson_r"],
                    "spearman_rho": result["spearman_rho"],
                    "exact_permutation_p": result["exact_permutation_p"],
                    "ridge_mae": result["ridge_mae"],
                    "ridge_rmse": result["ridge_rmse"],
                }
            )
    return rows


def _summaries(subgroup_rows: list[dict], scale_rows: list[dict]) -> dict:
    subgroup = pd.DataFrame(subgroup_rows)
    scale = pd.DataFrame(scale_rows)
    return {
        "subgroup_fe": {
            "slope_min": float(subgroup["fe_exposure_slope"].min()),
            "slope_max": float(subgroup["fe_exposure_slope"].max()),
            "negative_fraction": float((subgroup["fe_exposure_slope"] < 0).mean()),
            "permutation_p_min": float(subgroup["permutation_p"].min()),
            "permutation_p_max": float(subgroup["permutation_p"].max()),
            "ridge_mae_min": float(subgroup["ridge_mae"].min()),
            "ridge_mae_max": float(subgroup["ridge_mae"].max()),
        },
        "scale_pearson_ranges": {
            str(int(radius)): {
                "min": float(group["pearson_r"].min()),
                "max": float(group["pearson_r"].max()),
                "positive_fraction": float((group["pearson_r"] > 0).mean()),
            }
            for radius, group in scale.groupby("radius_km")
        },
    }


def _plot(subgroup_rows: list[dict], scale_rows: list[dict], path: Path) -> None:
    subgroup = pd.DataFrame(subgroup_rows).sort_values("fe_exposure_slope")
    scale = pd.DataFrame(scale_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    axes[0].barh(subgroup["omitted_mainshock"], subgroup["fe_exposure_slope"], color="#2878b5")
    axes[0].axvline(0, color="0.3", linewidth=1)
    axes[0].set_xlabel("FE exposure coefficient after omission")
    axes[0].set_title("Subgroup model influence")

    for radius, group in scale.groupby("radius_km"):
        group = group.sort_values("omitted_mainshock")
        axes[1].plot(
            group["omitted_mainshock"], group["pearson_r"], marker="o", label=f"{int(radius)} km"
        )
    axes[1].axhline(0, color="0.3", linewidth=1)
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_ylabel("Pearson r after omission")
    axes[1].set_title("Buffer-scale influence")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    subgroup_rows = subgroup_influence(
        pd.read_csv(OUT / "subgroup_summary.csv"), n_perm=1000, include_ml=True, seed=7
    )
    scale_rows = scale_influence(pd.read_csv(OUT / "mainshock_summary.csv"))
    rows = [*subgroup_rows, *scale_rows]
    pd.DataFrame(rows).to_csv(OUT / "mainshock_influence.csv", index=False)
    payload = {
        "method": "complete leave-one-mainshock-out re-estimation",
        "subgroup_permutations_per_omission": 1000,
        "subgroup_rows": subgroup_rows,
        "scale_rows": scale_rows,
        "summary": _summaries(subgroup_rows, scale_rows),
    }
    (OUT / "mainshock_influence.json").write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _plot(subgroup_rows, scale_rows, OUT / "fig_mainshock_influence.png")
    print(json.dumps(json_ready(payload["summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

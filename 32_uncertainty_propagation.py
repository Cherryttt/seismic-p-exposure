from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis_core import EXPOSURE_TERM, filter_analysis_sample, fit_within_fe, json_ready, regression_metrics


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"


def empirical_draw_matrix(
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """Sample each fitted record only from its saved empirical bootstrap values."""
    if "record_id" not in summary or "record_id" not in draws or "p_boot" not in draws:
        raise ValueError("Both inputs require record_id; draws also require p_boot.")
    rng = np.random.default_rng(seed)
    by_record = {
        str(record_id): group["p_boot"].dropna().to_numpy(float)
        for record_id, group in draws.groupby("record_id", sort=False)
    }
    matrix = np.empty((len(summary), n_draws), dtype=float)
    missing = []
    for row_index, record_id in enumerate(summary["record_id"].astype(str)):
        values = by_record.get(record_id, np.empty(0))
        values = values[np.isfinite(values)]
        if values.size == 0:
            missing.append(record_id)
            continue
        matrix[row_index] = rng.choice(values, size=n_draws, replace=True)
    if missing:
        raise ValueError(f"No empirical bootstrap draws for record_id: {missing[:5]}")
    return matrix


def _load_ml_module():
    script = HERE / "40_ml_groupkfold.py"
    spec = importlib.util.spec_from_file_location("ml_logo", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summarize(values: np.ndarray) -> Dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "median": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "lo": float(np.quantile(finite, 0.025)),
        "hi": float(np.quantile(finite, 0.975)),
    }


def propagate_empirical_uncertainty(
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    n_draws: int = 1000,
    seed: int = 7,
) -> tuple[dict, pd.DataFrame]:
    sample = filter_analysis_sample(summary)
    ml = _load_ml_module()
    matrix, _, point_y, groups, ordered = ml.make_features(sample)
    p_draws = empirical_draw_matrix(ordered, draws, n_draws=n_draws, seed=seed)
    splits = ml.leave_one_group_out(groups)
    fixed_alphas = {
        group: ml._select_alpha(matrix[train], point_y[train], groups[train], ml.ALPHA_GRID)
        for group, train, _ in splits
    }

    output_rows = []
    x_exposure = np.log10(ordered["pop_sum_subgroup"].to_numpy(float))
    for draw_index in range(n_draws):
        y = p_draws[:, draw_index]
        draw_frame = ordered.copy()
        draw_frame["p"] = y
        fe_slope = float(fit_within_fe(draw_frame)["coef"][EXPOSURE_TERM])
        row = {
            "draw": draw_index,
            "pearson_r": float(pearsonr(x_exposure, y).statistic),
            "spearman_rho": float(spearmanr(x_exposure, y).statistic),
            "fe_exposure_slope": fe_slope,
        }
        predictions = {name: np.full(len(y), np.nan) for name in ("mean", "ols", "ridge")}
        for group, train, test in splits:
            predictions["mean"][test] = y[train].mean()
            ols_beta = ml.ols_train(matrix[train], y[train])
            predictions["ols"][test] = ml.ols_predict(ols_beta, matrix[test])
            ridge_beta, mean, scale = ml.ridge_train(
                matrix[train], y[train], alpha=fixed_alphas[group]
            )
            predictions["ridge"][test] = ml.ridge_predict(
                ridge_beta, mean, scale, matrix[test]
            )
        for model, predicted in predictions.items():
            metrics = regression_metrics(y, predicted)
            row[f"{model}_mae"] = metrics["mae"]
            row[f"{model}_rmse"] = metrics["rmse"]
            row[f"{model}_r2"] = metrics["r2"]
        output_rows.append(row)

    result_frame = pd.DataFrame(output_rows)
    metrics = {
        column: _summarize(result_frame[column].to_numpy(float))
        for column in result_frame.columns
        if column != "draw"
    }
    slopes = result_frame["fe_exposure_slope"].to_numpy(float)
    payload = {
        "method": "marginal empirical Omori bootstrap propagation",
        "analysis_spec": "Each record is independently resampled from saved bootstrap p draws; FE and fixed-split LOGO models are refit.",
        "n_rows": int(len(ordered)),
        "n_groups": int(len(np.unique(groups))),
        "n_draws": int(n_draws),
        "ridge_alpha_by_test_group": fixed_alphas,
        "metrics": metrics,
        "fe_slope_negative_probability": float(np.mean(slopes < 0)),
    }
    return payload, result_frame


def _plot(draws: pd.DataFrame, path: Path) -> None:
    slope = draws["fe_exposure_slope"].to_numpy(float)
    lo, median, hi = np.quantile(slope, [0.025, 0.5, 0.975])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    axes[0].hist(slope, bins=34, color="#2878b5", alpha=0.85)
    axes[0].axvline(0, color="0.3", linewidth=1)
    axes[0].axvline(median, color="#c43c39", linewidth=2, label=f"median={median:.3f}")
    axes[0].axvspan(lo, hi, color="#c43c39", alpha=0.12, label=f"95% [{lo:.3f}, {hi:.3f}]")
    axes[0].set_title("FE exposure coefficient")
    axes[0].set_xlabel("Coefficient")
    axes[0].set_ylabel("Empirical propagation draws")
    axes[0].legend()

    models = ("mean", "ols", "ridge")
    axes[1].boxplot(
        [draws[f"{model}_mae"] for model in models],
        tick_labels=[model.upper() for model in models],
    )
    axes[1].set_title("Pooled LOGO prediction error")
    axes[1].set_ylabel("MAE")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Propagation of empirical Omori-p uncertainty")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    summary_path = OUT_DIR / "subgroup_summary.csv"
    draw_path = OUT_DIR / "omori_bootstrap_draws.csv"
    if not draw_path.exists():
        raise SystemExit("Missing empirical draws. Run 03_fit_omori.py and 30_join_and_plot.py first.")
    payload, result_frame = propagate_empirical_uncertainty(
        pd.read_csv(summary_path), pd.read_csv(draw_path), n_draws=1000, seed=7
    )
    (OUT_DIR / "uncertainty_propagation.json").write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    result_frame.to_csv(OUT_DIR / "uncertainty_propagation_draws.csv", index=False)
    _plot(result_frame, OUT_DIR / "fig_uncertainty_propagation.png")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

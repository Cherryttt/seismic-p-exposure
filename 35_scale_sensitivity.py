from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis_core import json_ready, regression_metrics


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    denom = float(xc @ xc)
    return float((xc @ (y - y.mean())) / denom) if denom > 0 else float("nan")


def exact_slope_permutation(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        raise ValueError("x and y must have the same length of at least three")
    observed = _slope(x, y)
    permuted = np.fromiter(
        (_slope(x, y[list(order)]) for order in itertools.permutations(range(y.size))),
        dtype=float,
        count=math.factorial(y.size),
    )
    p_value = float(np.mean(np.abs(permuted) >= abs(observed) - 1e-12))
    return {
        "observed_slope": observed,
        "p_value": p_value,
        "n_permutations": int(permuted.size),
        "null_q025": float(np.quantile(permuted, 0.025)),
        "null_q975": float(np.quantile(permuted, 0.975)),
    }


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(x.shape[0]), x])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)


def _predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x]) @ beta


def _select_alpha(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 4:
        return 1.0
    scores = []
    for alpha in ALPHAS:
        predictions = np.empty_like(y)
        for i in range(len(y)):
            train = np.arange(len(y)) != i
            mean, scale = _standardize(x[train])
            beta = _ridge_fit((x[train] - mean) / scale, y[train], alpha)
            predictions[i] = _predict(beta, (x[[i]] - mean) / scale)[0]
        scores.append(float(np.mean((y - predictions) ** 2)))
    return float(ALPHAS[int(np.argmin(scores))])


def _logo_predictions(frame: pd.DataFrame, radius: int) -> dict[str, np.ndarray]:
    y = frame["p"].to_numpy(float)
    x = np.column_stack(
        [
            np.log10(frame[f"pop_sum_{radius}km"].to_numpy(float)),
            frame["mainshock_mag"].to_numpy(float),
            np.log(frame["n_events"].to_numpy(float)),
        ]
    )
    pred_mean = np.empty_like(y)
    pred_ols = np.empty_like(y)
    pred_ridge = np.empty_like(y)
    selected_alphas = []
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        mean, scale = _standardize(x[train])
        x_train = (x[train] - mean) / scale
        x_test = (x[[i]] - mean) / scale
        pred_mean[i] = y[train].mean()
        pred_ols[i] = _predict(np.linalg.lstsq(
            np.column_stack([np.ones(train.sum()), x_train]), y[train], rcond=None
        )[0], x_test)[0]
        alpha = _select_alpha(x_train, y[train])
        selected_alphas.append(alpha)
        pred_ridge[i] = _predict(_ridge_fit(x_train, y[train], alpha), x_test)[0]
    return {
        "mean": pred_mean,
        "ols": pred_ols,
        "ridge": pred_ridge,
        "alphas": np.asarray(selected_alphas),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def analyze_scales(
    frame: pd.DataFrame, radii: Iterable[int] = (10, 25, 50)
) -> list[dict]:
    rows = []
    for radius in radii:
        columns = [
            "mainshock_id", "p", "mainshock_mag", "n_events", f"pop_sum_{radius}km"
        ]
        data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
        data = data[(data["p"] > 0) & (data["n_events"] > 0) & (data[f"pop_sum_{radius}km"] > 0)]
        x = np.log10(data[f"pop_sum_{radius}km"].to_numpy(float))
        y = data["p"].to_numpy(float)
        pearson, pearson_p = _safe_corr(x, y, "pearson")
        spearman, spearman_p = _safe_corr(x, y, "spearman")
        permutation = exact_slope_permutation(x, y)
        loo_pearson = []
        loo_spearman = []
        for i in range(len(data)):
            keep = np.arange(len(data)) != i
            loo_pearson.append(_safe_corr(x[keep], y[keep], "pearson")[0])
            loo_spearman.append(_safe_corr(x[keep], y[keep], "spearman")[0])
        predictions = _logo_predictions(data, radius)
        metrics = {
            model: regression_metrics(y, predictions[model])
            for model in ("mean", "ols", "ridge")
        }
        rows.append(
            {
                "radius_km": int(radius),
                "n": int(len(data)),
                "pearson_r": pearson,
                "pearson_p": pearson_p,
                "spearman_rho": spearman,
                "spearman_p": spearman_p,
                "slope": permutation["observed_slope"],
                "exact_permutation_p": permutation["p_value"],
                "n_permutations": permutation["n_permutations"],
                "loo_pearson_min": float(np.min(loo_pearson)),
                "loo_pearson_max": float(np.max(loo_pearson)),
                "loo_spearman_min": float(np.min(loo_spearman)),
                "loo_spearman_max": float(np.max(loo_spearman)),
                "mean_mae": metrics["mean"]["mae"],
                "mean_rmse": metrics["mean"]["rmse"],
                "mean_r2": metrics["mean"]["r2"],
                "ols_mae": metrics["ols"]["mae"],
                "ols_rmse": metrics["ols"]["rmse"],
                "ols_r2": metrics["ols"]["r2"],
                "ridge_mae": metrics["ridge"]["mae"],
                "ridge_rmse": metrics["ridge"]["rmse"],
                "ridge_r2": metrics["ridge"]["r2"],
                "ridge_alpha_median": float(np.median(predictions["alphas"])),
            }
        )
    return rows


def _plot_correlations(rows: list[dict], path: Path) -> None:
    radii = [row["radius_km"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for key, label, marker in (
        ("pearson_r", "Pearson r", "o"),
        ("spearman_rho", "Spearman rho", "s"),
    ):
        ax.plot(radii, [row[key] for row in rows], marker=marker, linewidth=2, label=label)
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_xticks(radii)
    ax.set_xlabel("Population exposure radius (km)")
    ax.set_ylabel("Correlation with Omori p")
    ax.set_title("Mainshock-level association across exposure scales (N=8)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_metrics(rows: list[dict], path: Path) -> None:
    radii = np.asarray([row["radius_km"] for row in rows], dtype=float)
    width = 3.2
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for offset, model in zip((-width, 0.0, width), ("mean", "ols", "ridge")):
        ax.bar(radii + offset, [row[f"{model}_mae"] for row in rows], width=width, label=model.upper())
    ax.set_xticks(radii)
    ax.set_xlabel("Population exposure radius (km)")
    ax.set_ylabel("Pooled LOGO MAE")
    ax.set_title("Cross-mainshock prediction sensitivity to exposure scale")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    source = OUT / "mainshock_summary.csv"
    frame = pd.read_csv(source)
    rows = analyze_scales(frame)
    pd.DataFrame(rows).to_csv(OUT / "scale_sensitivity.csv", index=False)
    payload = {
        "scope": "mainshock_level_exploratory",
        "note": "Fold-level R2 is not computed because each LOGO test fold contains one mainshock.",
        "radii": rows,
    }
    (OUT / "scale_sensitivity.json").write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _plot_correlations(rows, OUT / "fig_scale_sensitivity_corr.png")
    _plot_metrics(rows, OUT / "fig_scale_sensitivity_mae.png")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

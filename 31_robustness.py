from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from analysis_core import (
    EXPOSURE_TERM,
    cluster_bootstrap_fe,
    filter_analysis_sample,
    fit_within_fe,
    json_ready,
)
from config import SEED


def load_summary(out_dir: Path) -> pd.DataFrame:
    p = out_dir / "mainshock_summary.csv"
    if not p.exists():
        raise SystemExit("Missing out/mainshock_summary.csv. Run 30_join_and_plot.py first.")
    df = pd.read_csv(p)
    return df


def load_subgroup_summary(out_dir: Path) -> pd.DataFrame:
    p = out_dir / "subgroup_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _corrs(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    from scipy.stats import pearsonr, spearmanr

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return float(pr), float(pp), float(sr), float(sp)


def leave_one_out_correlations(df: pd.DataFrame, x_col: str, y_col: str, *, group_col: str = "mainshock_id") -> pd.DataFrame:
    d = df[[x_col, y_col, group_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 4:
        raise SystemExit("Too few points for leave-one-out analysis.")

    x = np.log10(np.maximum(d[x_col].to_numpy(float), 1.0))
    y = d[y_col].to_numpy(float)

    rows = []
    # Full-sample correlation
    pr, pp, sr, sp = _corrs(x, y)
    rows.append({"left_out": "(none)", "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp, "n": len(d)})

    # Leave-one-group-out (more meaningful when df contains multiple rows per mainshock)
    groups = d[group_col].astype(str).to_numpy()
    uniq = list(dict.fromkeys(groups))  # stable unique
    for g in uniq:
        m = groups != g
        pr, pp, sr, sp = _corrs(x[m], y[m])
        rows.append(
            {
                "left_out": str(g),
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_rho": sr,
                "spearman_p": sp,
                "n": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_leave_one_out(fig_path: Path, loo: pd.DataFrame, title: str) -> None:
    import matplotlib.pyplot as plt

    d = loo[loo["left_out"] != "(none)"].copy()
    d = d.sort_values("pearson_r")
    labels = d["left_out"].to_list()
    x = np.arange(len(labels))

    plt.figure(figsize=(10.5, 5.0))
    plt.barh(labels, d["pearson_r"].to_numpy(float), alpha=0.8, label="Pearson r (leave-one-out)")
    # Spearman as markers
    plt.plot(d["spearman_rho"].to_numpy(float), labels, "o", ms=5, label="Spearman ρ (leave-one-out)")
    plt.axvline(0.0, color="0.3", lw=1)

    # Add full-sample vertical line
    full = loo[loo["left_out"] == "(none)"].iloc[0]
    plt.axvline(
        float(full["pearson_r"]),
        color="#1f77b4",
        lw=2,
        ls="--",
        alpha=0.9,
        label=f"Full-sample Pearson r={full['pearson_r']:.2f} (p={full['pearson_p']:.3g})",
    )
    # Also show full-sample Spearman summary as text (more visible than another line)
    plt.text(
        0.02,
        0.02,
        f"Full Spearman ρ={full['spearman_rho']:.2f} (p={full['spearman_p']:.3g})\nN={int(full['n'])}",
        transform=plt.gca().transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )

    plt.xlabel("Correlation")
    plt.title(title)
    plt.grid(True, axis="x", ls=":", alpha=0.35)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

def bootstrap_clustered(
    df: pd.DataFrame,
    *,
    group_col: str,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Cluster bootstrap: resample groups with replacement, keeping all rows per group."""
    groups = df[group_col].astype(str).unique().tolist()
    if len(groups) < 2:
        raise SystemExit("Not enough groups for clustered bootstrap.")
    out = []
    for _ in range(n_boot):
        picked = rng.choice(groups, size=len(groups), replace=True)
        samp = pd.concat([df[df[group_col].astype(str) == g] for g in picked], ignore_index=True)
        out.append(samp)
    return out


def bootstrap_corr(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: str,
    n_boot: int = 500,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = df[[x_col, y_col, group_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 6:
        raise SystemExit("Too few samples for bootstrap correlation.")

    reps = []
    # cluster bootstrap by mainshock_id to respect dependence
    for samp in bootstrap_clustered(d, group_col=group_col, n_boot=n_boot, rng=rng):
        x = np.log10(np.maximum(samp[x_col].to_numpy(float), 1.0))
        y = samp[y_col].to_numpy(float)
        pr, pp, sr, sp = _corrs(x, y)
        reps.append({"pearson_r": pr, "spearman_rho": sr})
    return pd.DataFrame(reps)


def plot_bootstrap_corr(fig_path: Path, reps: pd.DataFrame, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9.5, 4.2))
    for i, (col, label, color) in enumerate(
        [
            ("pearson_r", "Pearson r", "#1f77b4"),
            ("spearman_rho", "Spearman ρ", "#ff7f0e"),
        ]
    ):
        vals = reps[col].to_numpy(float)
        plt.hist(vals, bins=25, alpha=0.55, color=color, label=label, density=True)
        q = np.quantile(vals, [0.025, 0.5, 0.975])
        plt.axvline(q[1], color=color, lw=2)
        plt.axvline(q[0], color=color, lw=1, ls="--")
        plt.axvline(q[2], color=color, lw=1, ls="--")
    plt.xlabel("Correlation")
    plt.title(title)
    plt.grid(True, axis="y", ls=":", alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def fit_controlled_regression(df: pd.DataFrame, *, add_mainshock_fe: bool = False) -> Dict:
    """
    Fit OLS: p ~ log_pop50 + mainshock_mag + log(n_events)
    Return coefficients + 95% CI.
    """
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    # Prefer subgroup-specific exposure if present; otherwise fall back to mainshock-centered 50km.
    x_exposure = "pop_sum_subgroup" if "pop_sum_subgroup" in d.columns else "pop_sum_50km"
    needed = ["p", x_exposure, "mainshock_mag", "n_events"]
    for c in needed:
        if c not in d.columns:
            raise SystemExit(f"Missing column {c} in summary.")
    extra = ["mainshock_id"] if "mainshock_id" in d.columns else []
    d = d[needed + extra].dropna()
    if len(d) < 6:
        raise SystemExit("Too few samples for regression (need at least ~6).")

    y = d["p"].to_numpy(float)
    base_cols = [
        np.ones(len(d), dtype=float),
        np.log10(np.maximum(d[x_exposure].to_numpy(float), 1.0)),
        d["mainshock_mag"].to_numpy(float),
        np.log(np.maximum(d["n_events"].to_numpy(float), 1.0)),
    ]
    names = ["const", f"log10_{x_exposure}", "mainshock_mag", "log_n_events"]

    if add_mainshock_fe and "mainshock_id" in d.columns:
        # One-hot encode mainshock_id, drop first to avoid multicollinearity with intercept.
        cats = d["mainshock_id"].astype(str)
        uniq = sorted(cats.unique().tolist())
        if len(uniq) >= 2:
            for u in uniq[1:]:
                base_cols.append((cats == u).to_numpy(float))
                names.append(f"fe_{u}")

    X = np.column_stack(base_cols)

    # OLS via numpy + t-based CI/p-values (avoids statsmodels dependency).
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n = len(y)
    k = X.shape[1]
    dof = max(n - k, 1)
    s2 = float((resid @ resid) / dof)
    # Use pseudo-inverse for stability (small N / collinearity can make XtX singular).
    XtX_inv = np.linalg.pinv(X.T @ X)
    var = np.diag(XtX_inv) * s2
    se = np.sqrt(np.maximum(var, 0.0))

    # t distribution for CI/p-values
    from scipy.stats import t as t_dist

    tval = t_dist.ppf(0.975, dof)
    ci_lo = beta - tval * se
    ci_hi = beta + tval * se

    tstats = beta / np.maximum(se, 1e-12)
    pvals = 2.0 * (1.0 - t_dist.cdf(np.abs(tstats), dof))

    # R^2
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(resid @ resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float("nan")

    out = {
        "n": int(n),
        "dof": int(dof),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "add_mainshock_fe": bool(add_mainshock_fe),
        "coef": {names[i]: float(beta[i]) for i in range(k)},
        "ci95": {names[i]: [float(ci_lo[i]), float(ci_hi[i])] for i in range(k)},
        "pvalues": {names[i]: float(pvals[i]) for i in range(k)},
    }
    return out


def run_subgroup_fe(
    df: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = SEED,
) -> Dict[str, object]:
    """Fit the corrected subgroup FE model with a mainshock-clustered interval."""
    sample = filter_analysis_sample(df)
    fitted = fit_within_fe(sample)
    bootstrap = cluster_bootstrap_fe(sample, n_boot=n_boot, seed=seed)
    fitted["add_mainshock_fe"] = True
    fitted["ci_method_primary"] = "mainshock_cluster_bootstrap_percentile"
    fitted["cluster_bootstrap_ci95"] = {
        EXPOSURE_TERM: [bootstrap["lo"], bootstrap["hi"]]
    }
    fitted["cluster_bootstrap_median"] = {
        EXPOSURE_TERM: bootstrap["median"]
    }
    fitted["cluster_bootstrap_probability_negative"] = {
        EXPOSURE_TERM: bootstrap["probability_negative"]
    }
    fitted["cluster_bootstrap_n_valid"] = bootstrap["n_boot_valid"]
    fitted["ci95"] = dict(fitted["naive_iid_ci95"])
    fitted["ci95"][EXPOSURE_TERM] = [bootstrap["lo"], bootstrap["hi"]]
    fitted["pvalues"] = dict(fitted["naive_iid_pvalues"])
    fitted["pvalues"][EXPOSURE_TERM] = None
    return fitted


def plot_regression_coeffs(fig_path: Path, reg: Dict) -> None:
    import matplotlib.pyplot as plt

    # Plot only the main predictors (keep FE coefficients out of the chart to reduce clutter).
    keys = [k for k in reg.get("coef", {}).keys() if k != "const" and not k.startswith("fe_")]
    names = []
    for k in keys:
        if k.startswith("log10_"):
            names.append(k)
    for k in ("mainshock_mag", "log_n_events"):
        if k in keys:
            names.append(k)
    for k in keys:
        if k not in names:
            names.append(k)

    coefs = [reg["coef"][n] for n in names]
    cis = [reg["ci95"][n] for n in names]
    lo = np.array([c[0] for c in cis], dtype=float)
    hi = np.array([c[1] for c in cis], dtype=float)
    mid = np.array(coefs, dtype=float)
    y = np.arange(len(names))

    plt.figure(figsize=(8.0, 4.2))
    plt.errorbar(mid, y, xerr=np.vstack([mid - lo, hi - mid]), fmt="o", capsize=4)
    plt.axvline(0.0, color="0.3", lw=1)
    plt.yticks(y, names)
    plt.xlabel("Coefficient (OLS) with 95% CI")
    plt.title(
        f"Controlled regression (OLS, small N; interpret cautiously):\n"
        f"p ~ ({' + '.join(names)}){' + FE(mainshock)' if reg.get('add_mainshock_fe') else ''}  "
        f"(R2={reg['r2']:.2f}, N={reg['n']}, dof={reg.get('dof','?')})"
    )
    plt.grid(True, axis="x", ls=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    import json

    # Run robustness on both mainshock-level and subgroup-level (if available).
    datasets = [("mainshock", load_summary(out_dir))]
    sub = load_subgroup_summary(out_dir)
    if not sub.empty:
        filtered_sub = filter_analysis_sample(sub)
        filtered_sub.to_csv(out_dir / "subgroup_analysis_sample.csv", index=False)
        datasets.append(("subgroup", filtered_sub))

    for mode, df in datasets:
        # Leave-one-out robustness for exposure vs p (prefer subgroup-specific exposure if present)
        x_col = "pop_sum_subgroup" if "pop_sum_subgroup" in df.columns else "pop_sum_50km"
        if x_col in df.columns and "p" in df.columns and "mainshock_id" in df.columns:
            loo = leave_one_out_correlations(df, x_col, "p", group_col="mainshock_id")
            loo.to_csv(out_dir / f"robustness_leave_one_out_{mode}.csv", index=False)
            plot_leave_one_out(
                out_dir / f"fig_leave_one_out_corr_{mode}.png",
                loo,
                title=f"Leave-one-out sensitivity ({mode}): corr(log10({x_col}), p)",
            )

        # Controlled regression
        reg = fit_controlled_regression(df, add_mainshock_fe=False)
        with open(out_dir / f"regression_controlled_{mode}.json", "w", encoding="utf-8") as f:
            json.dump(json_ready(reg), f, ensure_ascii=False, indent=2, allow_nan=False)
        plot_regression_coeffs(out_dir / f"fig_regression_coeffs_{mode}.png", reg)

        # A fixed-effect model is identified only for subgroup data with within-mainshock variation.
        if mode == "subgroup" and df["mainshock_id"].nunique() >= 2:
            reg_fe = run_subgroup_fe(df, n_boot=2000, seed=SEED)
            with open(out_dir / f"regression_controlled_{mode}_fe.json", "w", encoding="utf-8") as f:
                json.dump(json_ready(reg_fe), f, ensure_ascii=False, indent=2, allow_nan=False)
            plot_regression_coeffs(out_dir / f"fig_regression_coeffs_{mode}_fe.png", reg_fe)

        # Cluster bootstrap for correlation (more defensible than single p-value)
        if x_col in df.columns and "p" in df.columns and "mainshock_id" in df.columns and df["mainshock_id"].nunique() >= 2:
            reps = bootstrap_corr(df, x_col=x_col, y_col="p", group_col="mainshock_id", n_boot=400, seed=SEED)
            reps.to_csv(out_dir / f"bootstrap_corr_{mode}.csv", index=False)
            plot_bootstrap_corr(out_dir / f"fig_bootstrap_corr_{mode}.png", reps, title=f"Cluster bootstrap corr ({mode}): log10({x_col}) vs p")

    print("Saved robustness outputs to:", out_dir)


if __name__ == "__main__":
    main()

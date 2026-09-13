from __future__ import annotations

import json
from pathlib import Path

import math
import numpy as np
import pandas as pd


def load_results(out_dir: Path) -> pd.DataFrame:
    p = out_dir / "results_batch.json"
    if not p.exists():
        raise SystemExit("Missing out/results_batch.json. Run 03_fit_omori.py first.")
    data = json.load(open(p, "r", encoding="utf-8"))
    rows = data.get("by_mainshock") or []
    if not rows:
        raise SystemExit("No by_mainshock rows in results_batch.json.")
    df = pd.DataFrame(rows)
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    return df


def load_subgroup_results(out_dir: Path) -> pd.DataFrame:
    p = out_dir / "results_batch.json"
    data = json.load(open(p, "r", encoding="utf-8"))
    rows = data.get("by_subgroup") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def load_exposure(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "exposure_mainshocks.csv"
    if not p.exists():
        raise SystemExit("Missing data/exposure_mainshocks.csv. Run 20_exposure_population.py first.")
    df = pd.read_csv(p)
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    return df


def load_subgroup_exposure(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "exposure_subgroups.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ("mainshock_id", "subgroup_type", "subgroup"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def scatter_with_fit(
    fig_path: Path,
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    *,
    annotate: bool = True,
    label_col: str = "mainshock_id",
) -> None:
    import matplotlib.pyplot as plt

    cols = [c for c in (x, y, "mainshock_id", "p_lo", "p_hi", "n_events", "mainshock_mag") if c in df.columns]
    d = df[cols].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        print("Skip plot (too few points):", fig_path.name)
        return

    xs = d[x].to_numpy(float)
    ys = d[y].to_numpy(float)

    # Plot exposure on log scale to reduce leverage of large-population outliers.
    xs_plot = np.log10(np.maximum(xs, 1.0))

    plt.figure(figsize=(7.4, 5.3))
    sizes = 40
    if "n_events" in d.columns:
        sizes = 30 + 8 * np.sqrt(np.maximum(d["n_events"].to_numpy(float), 1.0))
    colors = None
    if "mainshock_mag" in d.columns:
        colors = d["mainshock_mag"].to_numpy(float)

    # Add p CI if available
    if "p_lo" in d.columns and "p_hi" in d.columns:
        lo = d["p_lo"].to_numpy(float)
        hi = d["p_hi"].to_numpy(float)
        yerr = np.vstack([ys - lo, hi - ys])
        plt.errorbar(xs_plot, ys, yerr=yerr, fmt="none", ecolor="0.5", elinewidth=1, alpha=0.6, capsize=3)

    sc = plt.scatter(xs_plot, ys, s=sizes, c=colors, cmap="viridis" if colors is not None else None, alpha=0.85, edgecolors="k", linewidths=0.3)
    if colors is not None:
        cb = plt.colorbar(sc)
        cb.set_label("Mainshock magnitude")

    # Simple linear fit (for course report visual guidance)
    try:
        m, b = np.polyfit(xs_plot, ys, deg=1)
        xx = np.linspace(xs_plot.min(), xs_plot.max(), 100)
        plt.plot(xx, m * xx + b, "-", lw=2, alpha=0.8)
    except Exception:
        pass

    # Correlations
    try:
        from scipy.stats import spearmanr, pearsonr

        pr, pp = pearsonr(xs_plot, ys)
        sr, sp = spearmanr(xs_plot, ys)
        plt.text(
            0.02,
            0.98,
            f"Pearson r={pr:.2f} (p={pp:.3g})\nSpearman ρ={sr:.2f} (p={sp:.3g})\nN={len(d)}",
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
        )
    except Exception:
        pass

    if annotate and label_col in d.columns:
        for _, r in d.iterrows():
            plt.annotate(
                str(r[label_col]),
                (math.log10(max(float(r[x]), 1.0)), float(r[y])),
                fontsize=8,
                alpha=0.85,
                xytext=(3, 3),
                textcoords="offset points",
            )

    plt.xlabel(xlabel + " (log10)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, ls=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = load_results(out_dir)
    sub = load_subgroup_results(out_dir)
    exp = load_exposure(data_dir)
    subexp = load_subgroup_exposure(data_dir)

    # Join on mainshock_id (USGS id)
    merged = res.merge(exp, how="left", left_on="mainshock_id", right_on="usgs_id", suffixes=("", "_exp"))

    # Keep a compact summary for reporting
    keep_cols = [
        "record_id",
        "mainshock_id",
        "time_utc",
        "mainshock_mag",
        "mainshock_lat",
        "mainshock_lon",
        "n_events",
        "Mc",
        "tmin_days",
        "p",
        "p_lo",
        "p_hi",
    ]
    exposure_cols = [c for c in merged.columns if c.startswith("pop_sum_") or c.startswith("pop_mean_") or c.startswith("pop_max_")]
    summary = merged[keep_cols + exposure_cols].copy()

    summary_path = out_dir / "mainshock_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    # Subgroup summary (more samples for correlation analysis)
    if not sub.empty:
        sub2 = sub.merge(exp, how="left", left_on="mainshock_id", right_on="usgs_id")
        if not subexp.empty:
            sub2 = sub2.merge(subexp, how="left", on=["mainshock_id", "subgroup_type", "subgroup"])
        sub_keep = ["record_id", "mainshock_id", "subgroup_type", "subgroup", "n_events", "p", "p_lo", "p_hi", "mainshock_mag"]
        exposure_cols = [c for c in sub2.columns if c.startswith("pop_sum_") or c.startswith("pop_mean_") or c.startswith("pop_max_")]
        sub_summary = sub2[sub_keep + exposure_cols].copy()
        sub_path = out_dir / "subgroup_summary.csv"
        sub_summary.to_csv(sub_path, index=False)
        print("Saved:", sub_path)

    # Plots (use population sum as the simplest exposure proxy)
    if "pop_sum_50km" in summary.columns:
        scatter_with_fit(
            out_dir / "fig_p_vs_pop_sum_50km.png",
            summary,
            x="pop_sum_50km",
            y="p",
            xlabel="Population exposure (sum within 50km)",
            ylabel="Omori p (MLE)",
            title="Aftershock decay (p) vs population exposure (50km)",
        )

    if "pop_sum_25km" in summary.columns:
        scatter_with_fit(
            out_dir / "fig_p_vs_pop_sum_25km.png",
            summary,
            x="pop_sum_25km",
            y="p",
            xlabel="Population exposure (sum within 25km)",
            ylabel="Omori p (MLE)",
            title="Aftershock decay (p) vs population exposure (25km)",
        )

    # Subgroup-level plots (more samples; disable annotations to reduce clutter).
    if not sub.empty:
        # Prefer subgroup-specific exposure when available (avoids pseudo-replication from repeating
        # the same mainshock-centered exposure across multiple subgroups).
        x_sub = "pop_sum_subgroup" if "pop_sum_subgroup" in sub_summary.columns else "pop_sum_50km"
        title_suffix = " (subgroup mask)" if x_sub == "pop_sum_subgroup" else ""
        if x_sub in sub_summary.columns:
            scatter_with_fit(
                out_dir / "fig_p_vs_pop_sum_50km_subgroup.png",
                sub_summary,
                x=x_sub,
                y="p",
                xlabel=f"Population exposure (sum){title_suffix}",
                ylabel="Omori p (MLE)",
                title=f"Subgroups: p vs population exposure{title_suffix}",
                annotate=False,
            )
        # Keep the old 25km plot only if we don't have subgroup exposure, to avoid confusing repeats.
        if x_sub != "pop_sum_subgroup" and "pop_sum_25km" in sub_summary.columns:
            scatter_with_fit(
                out_dir / "fig_p_vs_pop_sum_25km_subgroup.png",
                sub_summary,
                x="pop_sum_25km",
                y="p",
                xlabel="Population exposure (sum within 25km)",
                ylabel="Omori p (MLE)",
                title="Subgroups: p vs population exposure (25km)",
                annotate=False,
            )

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()

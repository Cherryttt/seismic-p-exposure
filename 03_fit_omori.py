from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    BOOTSTRAP_N,
    DEPTH_BINS_KM,
    MC_GRID,
    REGIONS,
    SEED,
    TMIN_GRID_DAYS,
    DAYS_AFTER,
    DISTANCE_RINGS_KM,
)
from omori_fit import binned_rate, bootstrap_p, fit_omori_mle

MIN_SUBGROUP_EVENTS = 25  # per-subgroup minimum for a stable p fit


def make_record_id(mainshock_id: str, subgroup_type: str, subgroup: str) -> str:
    """Stable key joining fitted rows to their empirical bootstrap draws."""
    return "|".join((str(mainshock_id), str(subgroup_type), str(subgroup)))


def _append_bootstrap_draws(
    collector: List[Dict] | None,
    *,
    record_id: str,
    mainshock_id: str,
    subgroup_type: str,
    subgroup: str,
    values: np.ndarray,
) -> None:
    if collector is None:
        return
    collector.extend(
        {
            "record_id": record_id,
            "mainshock_id": str(mainshock_id),
            "subgroup_type": str(subgroup_type),
            "subgroup": str(subgroup),
            "draw": int(index),
            "p_boot": float(value),
        }
        for index, value in enumerate(np.asarray(values, dtype=float))
        if np.isfinite(value)
    )
MIN_SUBGROUP_ROWS_PER_MAINSHOCK = 6  # try to keep each mainshock contributing >= this many subgroup rows
# Depth segmentation is only meaningful when the mainshock has enough aftershocks.
# With < 2*MIN_SUBGROUP_EVENTS, it's impossible to form 2 non-overlapping depth bins
# each meeting the minimum count requirement.
DEPTH_TARGET_SEGMENTS = 3
DEPTH_MIN_SEGMENTS = 2
DEPTH_ENFORCE_MIN_EVENTS = True  # if True, only output depth bins when each has >= MIN_SUBGROUP_EVENTS

def load_catalog(data_dir: Path) -> pd.DataFrame:
    # Prefer batch catalog if present; else fall back to single-event catalog.
    for name in ("catalog_all.parquet", "catalog_all.csv", "catalog.parquet", "catalog.csv"):
        p = data_dir / name
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            df = pd.read_csv(p)
            # Parse date columns if present (avoid passing unsupported kwargs to read_csv)
            for col in ("time_utc", "mainshock_time_utc"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            return df
    raise FileNotFoundError(f"No catalog found in {data_dir} (expected catalog_all.* or catalog.*).")


def load_mainshock(data_dir: Path) -> Dict:
    with open(data_dir / "mainshock.json", "r", encoding="utf-8") as f:
        return json.load(f)


def depth_group(depth: float, bins: List[float]) -> str:
    if np.isnan(depth):
        return "unknown"
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= depth < hi:
            return f"{lo:g}-{hi:g}km"
    return f">={bins[-1]:g}km"


def dynamic_subregions(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    # For batch catalogs: define simple per-event masks based on each mainshock's lat/lon.
    # Returns dict of name -> boolean mask aligned to df rows.
    lon0 = df["mainshock_lon"].to_numpy(float)
    lat0 = df["mainshock_lat"].to_numpy(float)
    lon = df["lon"].to_numpy(float)
    lat = df["lat"].to_numpy(float)
    return {
        "West_of_ms": lon < lon0,
        "East_of_ms": lon >= lon0,
        "South_of_ms": lat < lat0,
        "North_of_ms": lat >= lat0,
    }


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    # Vectorized great-circle distance.
    r = 6371.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

def _fit_row(
    *,
    tt: np.ndarray,
    ms_id: str,
    subgroup_type: str,
    subgroup: str,
    mc0: float,
    tmin0: float,
    tmax: float,
    rng: np.random.Generator,
    mainshock_mag: float,
    bootstrap_draws: List[Dict] | None = None,
) -> Dict | None:
    """Fit Omori p for one subgroup; return a results row or None if insufficient data."""
    if tt.size < MIN_SUBGROUP_EVENTS:
        return None
    try:
        fitg = fit_omori_mle(tt, tmin=float(tmin0), tmax=float(tmax))
    except Exception:
        return None
    psg = bootstrap_p(tt, tmin=float(tmin0), tmax=float(tmax), n=BOOTSTRAP_N, rng=rng)
    if psg.size:
        lo_g, hi_g = float(np.quantile(psg, 0.025)), float(np.quantile(psg, 0.975))
    else:
        lo_g, hi_g = float("nan"), float("nan")
    record_id = make_record_id(ms_id, subgroup_type, subgroup)
    _append_bootstrap_draws(
        bootstrap_draws,
        record_id=record_id,
        mainshock_id=ms_id,
        subgroup_type=subgroup_type,
        subgroup=subgroup,
        values=psg,
    )
    return {
        "record_id": record_id,
        "mainshock_id": ms_id,
        "subgroup_type": subgroup_type,
        "subgroup": subgroup,
        "n_events": int(fitg.n_events),
        "Mc": float(mc0),
        "tmin_days": float(tmin0),
        **fitg.to_dict(),
        "p_lo": lo_g,
        "p_hi": hi_g,
        "mainshock_mag": float(mainshock_mag),
    }


def _merged_rings(dist_km: np.ndarray, rings: List[float], *, min_events: int) -> List[Tuple[float, float]]:
    """
    Merge adjacent distance rings until each ring has >= min_events,
    to avoid mainshocks contributing only 1 ring sample (LOGO n_test=1).
    """
    edges = sorted(set(float(x) for x in rings))
    if len(edges) < 2:
        return []
    segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def count(seg):
        lo, hi = seg
        return int(((dist_km >= lo) & (dist_km < hi)).sum())

    segs = segs[:]
    while True:
        if len(segs) <= 1:
            break
        counts = [count(s) for s in segs]
        m = min(counts)
        if m >= min_events:
            break
        i = int(np.argmin(counts))
        # merge seg i with a neighbor (pick the neighbor with smaller count)
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        lo = min(segs[i][0], segs[j][0])
        hi = max(segs[i][1], segs[j][1])
        new = (lo, hi)
        # remove i and j, insert merged
        keep = [s for k, s in enumerate(segs) if k not in (i, j)]
        keep.append(new)
        segs = sorted(keep, key=lambda x: x[0])
    return segs


def _merged_depth_segments(depth_km: np.ndarray, bins: List[float], *, min_events: int, max_segments: int = 3) -> List[Tuple[float, float]]:
    """
    Build depth segments from DEPTH_BINS_KM and merge adjacent segments until:
    - each segment has >= min_events (when possible)
    - number of segments <= max_segments (merge the smallest segments)

    Returns list of (lo_km, hi_km) where hi_km may be inf for the last bin.
    """
    d = np.asarray(depth_km, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return []
    edges = [float(x) for x in bins]
    if len(edges) < 2:
        return []

    segs: List[Tuple[float, float]] = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    segs.append((edges[-1], float("inf")))  # >= last edge

    def count(seg: Tuple[float, float]) -> int:
        lo, hi = seg
        if math.isinf(hi):
            return int((d >= lo).sum())
        return int(((d >= lo) & (d < hi)).sum())

    segs = segs[:]

    # Merge sparse segments.
    while True:
        if len(segs) <= 1:
            break
        counts = [count(s) for s in segs]
        m = min(counts)
        if m >= min_events:
            break
        i = int(np.argmin(counts))
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        lo = min(segs[i][0], segs[j][0])
        hi = max(segs[i][1], segs[j][1])
        new = (lo, hi)
        keep = [s for k, s in enumerate(segs) if k not in (i, j)]
        keep.append(new)
        segs = sorted(keep, key=lambda x: x[0])

    # Limit the number of segments for comparability (aim for 2-3 segments per mainshock).
    while len(segs) > max_segments:
        counts = [count(s) for s in segs]
        i = int(np.argmin(counts))
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        lo = min(segs[i][0], segs[j][0])
        hi = max(segs[i][1], segs[j][1])
        new = (lo, hi)
        keep = [s for k, s in enumerate(segs) if k not in (i, j)]
        keep.append(new)
        segs = sorted(keep, key=lambda x: x[0])

    segs = [s for s in segs if count(s) > 0]

    # If bins+merging collapse to <=1 segment but we have enough events, fall back to
    # quantile-based splits to get 2-3 segments with adequate counts.
    if len(segs) <= 1:
        if d.size >= 3 * min_events and max_segments >= 3:
            q1, q2 = np.quantile(d, [1 / 3, 2 / 3]).tolist()
            cuts = sorted(set([float(np.min(d)), float(q1), float(q2), float(np.max(d) + 1e-6)]))
            out = [(cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], float("inf"))]
            # Ensure each has enough events; if not, reduce to 2 splits.
            if min((count(s) for s in out)) >= min_events:
                return out
        if d.size >= 2 * min_events and max_segments >= 2:
            q = float(np.quantile(d, 0.5))
            out = [(float(np.min(d)), q), (q, float("inf"))]
            if min((count(s) for s in out)) >= min_events:
                return out

    # If we only end up with a single segment covering essentially all depths,
    # drop it (we already include an all_depth backstop subgroup).
    if len(segs) == 1:
        lo, hi = segs[0]
        if math.isinf(hi) and lo <= min(edges) + 1e-9:
            return []
    return segs


def _fixed_depth_segments(
    depth_km: np.ndarray,
    *,
    min_events: int,
    target_segments: int,
    min_segments: int,
) -> List[Tuple[float, float]]:
    """
    Produce 2-3 depth segments with >= min_events each (when possible).

    Strategy:
    - Use quantile cuts (more adaptive than fixed bins for small samples).
    - Merge adjacent segments until each has enough events.
    - Prefer `target_segments` (usually 3), fall back to 2.
    - Return [] if it's mathematically impossible (not enough events).
    """
    d = np.asarray(depth_km, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < min_segments * min_events:
        return []

    def seg_counts(edges: List[float]) -> List[int]:
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            out.append(int(((d >= lo) & (d < hi)).sum()))
        out.append(int((d >= edges[-1]).sum()))
        return out

    def build(k: int) -> List[Tuple[float, float]]:
        # k segments -> k-1 quantile cutpoints
        if k == 2:
            cuts = [float(np.quantile(d, 0.5))]
        elif k == 3:
            cuts = [float(np.quantile(d, 1 / 3)), float(np.quantile(d, 2 / 3))]
        else:
            qs = [(i + 1) / k for i in range(k - 1)]
            cuts = [float(np.quantile(d, q)) for q in qs]
        lo_min = float(np.min(d))
        # Drop degenerate cutpoints that don't actually split the support.
        cuts = [c for c in cuts if c > lo_min + 1e-9]
        cuts = sorted(set(cuts))
        if len(cuts) < k - 1:
            # Quantiles can collapse when many depths are identical (common for USGS where
            # depth is often reported as exactly 10km). Fall back to searching cutpoints
            # over unique depth values to find a split with enough events per segment.
            u = np.unique(d)
            if u.size <= 1:
                return []
            u = np.sort(u)

            def ok_edges(edges: List[float]) -> bool:
                segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
                segs.append((edges[-1], float("inf")))
                # Use <= on the first segment to absorb mass at an exact cutpoint.
                def cnt(seg, first: bool) -> int:
                    lo, hi = seg
                    if math.isinf(hi):
                        return int((d >= lo).sum()) if first else int((d > lo).sum())
                    if first:
                        return int(((d >= lo) & (d <= hi)).sum())
                    return int(((d > lo) & (d <= hi)).sum())
                counts = [cnt(segs[0], True)] + [cnt(s, False) for s in segs[1:]]
                return (len(counts) >= min_segments) and (min(counts) >= min_events)

            # k=2: try single cutpoint.
            if k == 2:
                for cut in u[1:-1]:
                    if float(cut) <= lo_min + 1e-9:
                        continue
                    # edges are [min, cut]
                    if ok_edges([float(np.min(d)), float(cut)]):
                        cuts = [float(cut)]
                        break
                else:
                    return []
            # k=3: try pairs of cutpoints (small u size here; brute force OK).
            elif k == 3:
                found = None
                for i in range(1, len(u) - 2):
                    for j in range(i + 1, len(u) - 1):
                        if float(u[i]) <= lo_min + 1e-9:
                            continue
                        edges = [float(np.min(d)), float(u[i]), float(u[j])]
                        if ok_edges(edges):
                            found = [float(u[i]), float(u[j])]
                            break
                    if found:
                        break
                if not found:
                    return []
                cuts = found
            else:
                return []
        lo0 = float(np.min(d))
        edges = [lo0] + cuts
        segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        segs.append((edges[-1], float("inf")))
        # Guard against zero-width segments (e.g., min depth equals cutpoint).
        if any((not math.isinf(hi)) and (hi <= lo + 1e-12) for lo, hi in segs):
            return []

        def count(seg: Tuple[float, float], *, first: bool) -> int:
            # Disjoint bins (lo, hi] (first bin is [min, hi]) to match the selection logic below.
            lo, hi = seg
            if math.isinf(hi):
                return int((d >= lo).sum()) if first else int((d > lo).sum())
            if first:
                return int(((d >= lo) & (d <= hi)).sum())
            return int(((d > lo) & (d <= hi)).sum())

        # Merge sparse segments.
        segs2 = segs[:]
        while True:
            if len(segs2) <= 1:
                break
            counts = [count(segs2[0], first=True)] + [count(s, first=False) for s in segs2[1:]]
            if min(counts) >= min_events:
                break
            i = int(np.argmin(counts))
            if i == 0:
                j = 1
            elif i == len(segs2) - 1:
                j = i - 1
            else:
                j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
            lo = min(segs2[i][0], segs2[j][0])
            hi = max(segs2[i][1], segs2[j][1])
            keep = [s for idx, s in enumerate(segs2) if idx not in (i, j)]
            keep.append((lo, hi))
            segs2 = sorted(keep, key=lambda x: x[0])

        segs2 = [s for s in segs2 if (count(segs2[0], first=True) if s == segs2[0] else count(s, first=False)) > 0]
        if len(segs2) >= min_segments:
            counts2 = [count(segs2[0], first=True)] + [count(s, first=False) for s in segs2[1:]]
            if min(counts2) < min_events:
                return []
            # If too many segments remain (rare after merging), merge the smallest.
            while len(segs2) > k:
                counts = [count(segs2[0], first=True)] + [count(s, first=False) for s in segs2[1:]]
                i = int(np.argmin(counts))
                j = 1 if i == 0 else i - 1
                lo = min(segs2[i][0], segs2[j][0])
                hi = max(segs2[i][1], segs2[j][1])
                keep = [s for idx, s in enumerate(segs2) if idx not in (i, j)]
                keep.append((lo, hi))
                segs2 = sorted(keep, key=lambda x: x[0])
            counts3 = [count(segs2[0], first=True)] + [count(s, first=False) for s in segs2[1:]]
            if len(segs2) >= min_segments and min(counts3) >= min_events:
                return segs2
        return []

    # Prefer 3 segments, else 2.
    if target_segments >= 3:
        out = build(3)
        if out:
            return out
    return build(2)


def _depth_label(lo: float, hi: float) -> str:
    if math.isinf(hi):
        return f">={lo:g}km"
    return f"{lo:g}-{hi:g}km"


def region_labels(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, pred in REGIONS:
        out[name] = np.array([bool(pred(lat, lon)) for lat, lon in zip(df["lat"].values, df["lon"].values)])
    return out


def ensure_dirs(here: Path) -> Tuple[Path, Path, Path]:
    data_dir = here / "data"
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir
    return data_dir, out_dir, fig_dir


def plot_rate_fit(fig_path: Path, t: np.ndarray, fit, title: str) -> None:
    import matplotlib.pyplot as plt

    centers, rate = binned_rate(t, fit.tmin, fit.tmax, n_bins=35)
    tt = np.logspace(np.log10(fit.tmin), np.log10(fit.tmax), 300)
    lam = fit.K / (tt + fit.c) ** fit.p

    plt.figure(figsize=(7.5, 5.2))
    plt.loglog(centers, np.maximum(rate, 1e-12), "o", ms=4, alpha=0.7, label="binned rate (events/day)")
    plt.loglog(tt, lam, "-", lw=2, label=f"fit: p={fit.p:.2f}, c={fit.c:.4f}d")
    plt.xlabel("t since mainshock (days)")
    plt.ylabel("rate (events/day)")
    plt.title(title)
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_p_by_group(fig_path: Path, rows: List[Dict], group_key: str, title: str) -> None:
    import matplotlib.pyplot as plt

    # rows: dict with keys group_key, p, p_lo, p_hi, n_events
    items = [r for r in rows if r.get(group_key)]
    items = sorted(items, key=lambda r: r[group_key])
    labels = [r[group_key] for r in items]
    p = np.array([r["p"] for r in items], dtype=float)
    lo = np.array([r["p_lo"] for r in items], dtype=float)
    hi = np.array([r["p_hi"] for r in items], dtype=float)
    yerr = np.vstack([p - lo, hi - p])

    plt.figure(figsize=(9.0, 4.8))
    x = np.arange(len(labels))
    plt.errorbar(x, p, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Omori p")
    plt.title(title)
    plt.grid(True, axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_sensitivity(fig_path: Path, sens: Dict[Tuple[float, float], Dict]) -> None:
    import matplotlib.pyplot as plt

    mcs = sorted({k[0] for k in sens.keys()})
    tmins = sorted({k[1] for k in sens.keys()})
    grid = np.full((len(mcs), len(tmins)), np.nan, dtype=float)
    for i, mc in enumerate(mcs):
        for j, tmin in enumerate(tmins):
            v = sens.get((mc, tmin))
            if v:
                grid[i, j] = v["p"]

    plt.figure(figsize=(8.0, 4.8))
    im = plt.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="p")
    plt.xticks(np.arange(len(tmins)), [str(x) for x in tmins])
    plt.yticks(np.arange(len(mcs)), [str(x) for x in mcs])
    plt.xlabel("t_min (days)")
    plt.ylabel("Mc (min magnitude)")
    plt.title("Sensitivity of p to Mc and t_min")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_p_by_mainshock(fig_path: Path, rows: List[Dict]) -> None:
    import matplotlib.pyplot as plt

    # Sort by sample size (descending) to emphasize reliability.
    items = sorted(rows, key=lambda r: int(r.get("n_events") or 0), reverse=True)
    labels = [f"{r['mainshock_id']}\n(n={r.get('n_events','')})" for r in items]
    p = np.array([r["p"] for r in items], dtype=float)
    lo = np.array([r["p_lo"] for r in items], dtype=float)
    hi = np.array([r["p_hi"] for r in items], dtype=float)
    yerr = np.vstack([p - lo, hi - p])
    n = np.array([int(r.get("n_events") or 0) for r in items], dtype=int)

    plt.figure(figsize=(10.5, 5.2))
    x = np.arange(len(labels))
    # Highlight low-count mainshocks (n<50) as hollow gray markers.
    is_low = n < 50
    # error bars
    plt.errorbar(x, p, yerr=yerr, fmt="none", ecolor="0.5", elinewidth=1.2, alpha=0.75, capsize=3)
    # points
    plt.scatter(x[~is_low], p[~is_low], s=70, c="#1f77b4", edgecolors="k", linewidths=0.3, label="n ≥ 50")
    plt.scatter(x[is_low], p[is_low], s=70, facecolors="none", edgecolors="0.45", linewidths=1.2, label="n < 50 (low count)")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Omori p (MLE)")
    plt.title("Omori p by mainshock (baseline Mc/tmin)")
    plt.grid(True, axis="y", ls=":", alpha=0.4)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def batch_main(df: pd.DataFrame, out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)

    # Baseline config
    # Use the most inclusive baseline so smaller mainshocks still have enough events
    # to support subgroup (ring/depth) splits.
    mc0 = MC_GRID[0]
    tmin0 = TMIN_GRID_DAYS[0]

    rows: List[Dict] = []
    rows_sub: List[Dict] = []
    bootstrap_draw_rows: List[Dict] = []
    for ms_id, g in df.groupby("mainshock_id"):
        g2 = g[g["mag"] >= mc0]
        if g2.empty:
            continue
        t = g2["t_days"].to_numpy(dtype=float)
        # Allow more mainshocks into the plot; we'll carry n_events and can down-weight
        # low-count mainshocks visually in downstream plots.
        if t.size < 30:
            continue
        # Use the configured window as tmax so fits are comparable across mainshocks.
        tmax = float(DAYS_AFTER)
        try:
            fit = fit_omori_mle(t, tmin=float(tmin0), tmax=tmax)
        except Exception:
            continue
        ps = bootstrap_p(t, tmin=float(tmin0), tmax=tmax, n=BOOTSTRAP_N, rng=rng)
        if ps.size:
            lo, hi = float(np.quantile(ps, 0.025)), float(np.quantile(ps, 0.975))
        else:
            lo, hi = float("nan"), float("nan")

        r0 = g2.iloc[0]
        ms_mag = float(r0.get("mainshock_mag", np.nan))
        mainshock_record_id = make_record_id(ms_id, "mainshock", "all")
        _append_bootstrap_draws(
            bootstrap_draw_rows,
            record_id=mainshock_record_id,
            mainshock_id=str(ms_id),
            subgroup_type="mainshock",
            subgroup="all",
            values=ps,
        )
        rows.append(
            {
                "record_id": mainshock_record_id,
                "mainshock_id": ms_id,
                "time_utc": str(r0.get("mainshock_time_utc", "")),
                "mainshock_mag": float(r0.get("mainshock_mag", np.nan)),
                "mainshock_lat": float(r0.get("mainshock_lat", np.nan)),
                "mainshock_lon": float(r0.get("mainshock_lon", np.nan)),
                "n_events": int(fit.n_events),
                "Mc": float(mc0),
                "tmin_days": float(tmin0),
                **fit.to_dict(),
                "p_lo": lo,
                "p_hi": hi,
            }
        )

        # ---- Subgroup expansion (more samples): depth groups + simple spatial splits ----
        # We aim to keep each mainshock contributing a comparable number of subgroup rows. To do so:
        # - use a minimum events threshold (MIN_SUBGROUP_EVENTS)
        # - adaptively merge sparse distance rings
        # - always include an "all_depth" subgroup as a backstop
        rows_sub_ms: List[Dict] = []

        # Backstop: all aftershocks (acts like a depth-aggregated subgroup)
        rr = _fit_row(
            tt=g2["t_days"].to_numpy(dtype=float),
            ms_id=str(ms_id),
            subgroup_type="depth",
            subgroup="all_depth",
            mc0=float(mc0),
            tmin0=float(tmin0),
            tmax=float(tmax),
            rng=rng,
            mainshock_mag=ms_mag,
            bootstrap_draws=bootstrap_draw_rows,
        )
        if rr:
            rows_sub_ms.append(rr)

        # Depth segments: try to build 2-3 segments with >= MIN_SUBGROUP_EVENTS each.
        # If it's impossible (not enough events), keep only all_depth and don't force a split.
        if "depth_km" in g2.columns:
            dep = g2["depth_km"].to_numpy(dtype=float)
            segs = _fixed_depth_segments(
                dep,
                min_events=MIN_SUBGROUP_EVENTS,
                target_segments=DEPTH_TARGET_SEGMENTS,
                min_segments=DEPTH_MIN_SEGMENTS,
            )
            # Optional fallback to older bin+merge logic when strict fixed segmentation isn't achievable.
            if not segs and not DEPTH_ENFORCE_MIN_EVENTS:
                segs = _merged_depth_segments(dep, list(DEPTH_BINS_KM), min_events=MIN_SUBGROUP_EVENTS, max_segments=DEPTH_TARGET_SEGMENTS)
            # Use disjoint (lo, hi] bins (first bin is [min, hi]) to handle depth quantization
            # at exactly 10km without duplicating events across bins.
            segs = sorted(segs, key=lambda x: (x[0], x[1]))
            for i_seg, (lo_d, hi_d) in enumerate(segs):
                if i_seg == 0:
                    if math.isinf(hi_d):
                        gg = g2[dep >= lo_d]
                    else:
                        gg = g2[(dep >= lo_d) & (dep <= hi_d)]
                else:
                    if math.isinf(hi_d):
                        gg = g2[dep > lo_d]
                    else:
                        gg = g2[(dep > lo_d) & (dep <= hi_d)]
                rr = _fit_row(
                    tt=gg["t_days"].to_numpy(dtype=float),
                    ms_id=str(ms_id),
                    subgroup_type="depth",
                    subgroup=_depth_label(lo_d, hi_d),
                    mc0=float(mc0),
                    tmin0=float(tmin0),
                    tmax=float(tmax),
                    rng=rng,
                    mainshock_mag=ms_mag,
                    bootstrap_draws=bootstrap_draw_rows,
                )
                if rr:
                    rows_sub_ms.append(rr)

        # Spatial splits relative to mainshock location (per-event)
        if {"mainshock_lat", "mainshock_lon", "lat", "lon"}.issubset(set(g2.columns)):
            masks = dynamic_subregions(g2)
            for name, mask in masks.items():
                gg = g2[mask]
                tt = gg["t_days"].to_numpy(dtype=float)
                rr = _fit_row(
                    tt=tt,
                    ms_id=str(ms_id),
                    subgroup_type="region",
                    subgroup=str(name),
                    mc0=float(mc0),
                    tmin0=float(tmin0),
                    tmax=float(tmax),
                    rng=rng,
                    mainshock_mag=ms_mag,
                    bootstrap_draws=bootstrap_draw_rows,
                )
                if rr:
                    rows_sub_ms.append(rr)

            # Distance rings relative to mainshock (per-event)
            lat0 = g2["mainshock_lat"].to_numpy(float)
            lon0 = g2["mainshock_lon"].to_numpy(float)
            lat = g2["lat"].to_numpy(float)
            lon = g2["lon"].to_numpy(float)
            dist = haversine_km(lat0, lon0, lat, lon)
            rings_merged = _merged_rings(dist, list(DISTANCE_RINGS_KM), min_events=MIN_SUBGROUP_EVENTS)
            for lo_r, hi_r in rings_merged:
                gg = g2[(dist >= lo_r) & (dist < hi_r)]
                rr = _fit_row(
                    tt=gg["t_days"].to_numpy(dtype=float),
                    ms_id=str(ms_id),
                    subgroup_type="ring",
                    subgroup=f"{lo_r:g}-{hi_r:g}km",
                    mc0=float(mc0),
                    tmin0=float(tmin0),
                    tmax=float(tmax),
                    rng=rng,
                    mainshock_mag=ms_mag,
                    bootstrap_draws=bootstrap_draw_rows,
                )
                if rr:
                    rows_sub_ms.append(rr)

        # If we still have too few subgroup rows for this mainshock, relax the per-subgroup minimum.
        # This is better than having LOGO folds with n_test=1.
        if len(rows_sub_ms) < MIN_SUBGROUP_ROWS_PER_MAINSHOCK:
            relaxed_min = 10
            if g2["t_days"].to_numpy(dtype=float).size >= relaxed_min and not any(r["subgroup"] == "all_relaxed" for r in rows_sub_ms):
                try:
                    fitg = fit_omori_mle(g2["t_days"].to_numpy(dtype=float), tmin=float(tmin0), tmax=float(tmax))
                    rows_sub_ms.append(
                        {
                            "record_id": make_record_id(ms_id, "depth", "all_relaxed"),
                            "mainshock_id": ms_id,
                            "subgroup_type": "depth",
                            "subgroup": "all_relaxed",
                            "n_events": int(fitg.n_events),
                            "Mc": float(mc0),
                            "tmin_days": float(tmin0),
                            **fitg.to_dict(),
                            "p_lo": float("nan"),
                            "p_hi": float("nan"),
                            "mainshock_mag": ms_mag,
                        }
                    )
                except Exception:
                    pass

        rows_sub.extend(rows_sub_ms)

    results = {
        "mode": "batch",
        "baseline": {"Mc": mc0, "tmin_days": tmin0, "tmax_days": float(DAYS_AFTER)},
        "by_mainshock": rows,
        "by_subgroup": rows_sub,
    }
    with open(out_dir / "results_batch.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(bootstrap_draw_rows).to_csv(out_dir / "omori_bootstrap_draws.csv", index=False)

    plot_p_by_mainshock(out_dir / "fig_p_by_mainshock.png", rows)
    print("Saved batch results to:", out_dir)


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir, out_dir, _ = ensure_dirs(here)

    df = load_catalog(data_dir)
    if "mainshock_id" in df.columns:
        batch_main(df, out_dir)
        return

    ms = load_mainshock(data_dir)
    rng = np.random.default_rng(SEED)

    # Assign depth groups
    df["depth_group"] = [depth_group(d, DEPTH_BINS_KM) for d in df["depth_km"].astype(float).values]
    regions = region_labels(df)

    # Pick a baseline configuration for "headline" results
    mc0 = MC_GRID[1] if len(MC_GRID) > 1 else MC_GRID[0]
    tmin0 = TMIN_GRID_DAYS[1] if len(TMIN_GRID_DAYS) > 1 else TMIN_GRID_DAYS[0]
    tmax = float(df["t_days"].max())

    base = df[df["mag"] >= mc0].copy()
    t = base["t_days"].to_numpy(dtype=float)
    fit_all = fit_omori_mle(t, tmin=tmin0, tmax=tmax)
    ps = bootstrap_p(t, tmin=tmin0, tmax=tmax, n=BOOTSTRAP_N, rng=rng)
    p_lo, p_hi = (float(np.quantile(ps, 0.025)), float(np.quantile(ps, 0.975))) if ps.size else (float("nan"), float("nan"))

    results: Dict = {
        "mainshock": ms,
        "baseline": {"Mc": mc0, "tmin_days": tmin0, "tmax_days": tmax},
        "all": {**fit_all.to_dict(), "p_lo": p_lo, "p_hi": p_hi},
        "by_depth": [],
        "by_region": [],
        "sensitivity": {},
    }

    # Plot rate + fit
    plot_rate_fit(out_dir / "fig_rate_fit.png", t, fit_all, title="Lushan aftershocks: rate decay + Omori fit (all)")

    # Depth groups
    for g, gdf in base.groupby("depth_group"):
        tt = gdf["t_days"].to_numpy(dtype=float)
        if tt.size < 50:
            continue
        fit = fit_omori_mle(tt, tmin=tmin0, tmax=tmax)
        ps = bootstrap_p(tt, tmin=tmin0, tmax=tmax, n=BOOTSTRAP_N, rng=rng)
        if ps.size:
            lo, hi = float(np.quantile(ps, 0.025)), float(np.quantile(ps, 0.975))
        else:
            lo, hi = float("nan"), float("nan")
        results["by_depth"].append({"depth_group": g, **fit.to_dict(), "p_lo": lo, "p_hi": hi})

    plot_p_by_group(out_dir / "fig_p_by_depth.png", results["by_depth"], "depth_group", "Omori p by depth group (baseline Mc/tmin)")

    # Regions
    for name, mask in regions.items():
        sub = base[mask].copy()
        tt = sub["t_days"].to_numpy(dtype=float)
        if tt.size < 80:
            continue
        fit = fit_omori_mle(tt, tmin=tmin0, tmax=tmax)
        ps = bootstrap_p(tt, tmin=tmin0, tmax=tmax, n=BOOTSTRAP_N, rng=rng)
        if ps.size:
            lo, hi = float(np.quantile(ps, 0.025)), float(np.quantile(ps, 0.975))
        else:
            lo, hi = float("nan"), float("nan")
        results["by_region"].append({"region": name, **fit.to_dict(), "p_lo": lo, "p_hi": hi, "n_events": int(tt.size)})

    plot_p_by_group(out_dir / "fig_p_by_region.png", results["by_region"], "region", "Omori p by subregion (baseline Mc/tmin)")

    # Sensitivity: Mc x tmin (all events only; fast + interpretable)
    sens: Dict[Tuple[float, float], Dict] = {}
    for mc in MC_GRID:
        dff = df[df["mag"] >= mc]
        tt_all = dff["t_days"].to_numpy(dtype=float)
        for tmin in TMIN_GRID_DAYS:
            if tt_all.size < 200:
                continue
            try:
                fit = fit_omori_mle(tt_all, tmin=float(tmin), tmax=tmax)
                sens[(float(mc), float(tmin))] = {"p": float(fit.p), "c": float(fit.c), "K": float(fit.K), "n": int(fit.n_events)}
            except Exception:
                continue

    results["sensitivity"] = {f"Mc={mc},tmin={tmin}": v for (mc, tmin), v in sens.items()}
    plot_sensitivity(out_dir / "fig_sensitivity.png", sens)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved figures and results to:", out_dir)


if __name__ == "__main__":
    main()

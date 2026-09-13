from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

from config import DISTANCE_RINGS_KM, EXPOSURE_BUFFER_KM, POP_RASTER_PATH


def load_mainshocks() -> pd.DataFrame:
    here = Path(__file__).resolve().parent
    ms_path = here / "data" / "mainshocks.json"
    if not ms_path.exists():
        raise SystemExit("Missing data/mainshocks.json. Run 10_batch_mainshocks.py first.")
    df = pd.read_json(ms_path)
    # Ensure canonical types
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    return df


def load_subgroups() -> pd.DataFrame:
    """Load subgroup definitions from out/results_batch.json (created by 03_fit_omori.py)."""
    here = Path(__file__).resolve().parent
    p = here / "out" / "results_batch.json"
    if not p.exists():
        raise SystemExit("Missing out/results_batch.json. Run 03_fit_omori.py first.")
    import json

    data = json.load(open(p, "r", encoding="utf-8"))
    rows = data.get("by_subgroup") or []
    if not rows:
        return pd.DataFrame(columns=["mainshock_id", "subgroup_type", "subgroup"])
    df = pd.DataFrame(rows)[["mainshock_id", "subgroup_type", "subgroup"]].drop_duplicates()
    df["mainshock_id"] = df["mainshock_id"].astype(str)
    df["subgroup_type"] = df["subgroup_type"].astype(str)
    df["subgroup"] = df["subgroup"].astype(str)
    return df


def _circle_bounds_wgs84(lat: float, lon: float, r_km: float) -> Tuple[float, float, float, float]:
    # Approximate degrees for small buffers (good enough for 10-50km).
    dlat = r_km / 111.0
    dlon = r_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _read_window_and_latlon(
    *,
    ds: rasterio.io.DatasetReader,
    lat: float,
    lon: float,
    r_km: float,
):
    """Read a bounding window and build lon/lat arrays for fast masking (CRS-unit agnostic)."""
    bounds_wgs84 = _circle_bounds_wgs84(lat, lon, r_km)

    tf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    (minx, miny) = tf.transform(bounds_wgs84[0], bounds_wgs84[1])
    (maxx, maxy) = tf.transform(bounds_wgs84[2], bounds_wgs84[3])

    win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
    win = win.round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return None, None, None, None

    arr = ds.read(1, window=win, masked=True)
    if arr.size == 0:
        return None, None, None, None

    rows, cols = np.indices(arr.shape)
    wt = ds.window_transform(win)
    # rasterio.transform.xy flattens when given arrays; reshape back to 2D.
    xs, ys = rasterio.transform.xy(wt, rows.ravel(), cols.ravel(), offset="center")
    xs = np.asarray(xs, dtype=float).reshape(arr.shape)
    ys = np.asarray(ys, dtype=float).reshape(arr.shape)

    # Convert pixel centers back to lon/lat for distance computations.
    tf_back = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
    lons, lats = tf_back.transform(xs, ys)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    return arr, lats, lons


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Vectorized haversine distance (km)."""
    r = 6371.0088
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = math.radians(lat2)
    lon2r = math.radians(lon2)
    dlat = lat1r - lat2r
    dlon = lon1r - lon2r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * math.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _stats_from_mask(arr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(arr.filled(np.nan), dtype=float)
    vals[~mask] = np.nan
    if np.all(np.isnan(vals)):
        return {"pop_sum": float("nan"), "pop_mean": float("nan"), "pop_max": float("nan")}
    return {
        "pop_sum": float(np.nansum(vals)),
        "pop_mean": float(np.nanmean(vals)),
        "pop_max": float(np.nanmax(vals)),
    }


def _parse_ring_km(s: str) -> Tuple[float, float] | None:
    try:
        lo, hi = s.replace("km", "").split("-")
        return float(lo), float(hi)
    except Exception:
        return None


def zonal_stats_population(
    *,
    ds: rasterio.io.DatasetReader,
    lat: float,
    lon: float,
    r_km: float,
) -> Dict[str, float]:
    # Compute approximate circular-buffer stats by masking a bounding window then applying a circle mask.
    # This is lightweight and avoids heavy GIS dependencies.
    arr, lats, lons = _read_window_and_latlon(ds=ds, lat=lat, lon=lon, r_km=r_km)
    if arr is None:
        return {"pop_sum": float("nan"), "pop_mean": float("nan"), "pop_max": float("nan")}
    dist_km = _haversine_km(lats, lons, lat, lon)
    mask = dist_km <= float(r_km)
    return _stats_from_mask(arr, mask)


def zonal_stats_population_subgroup(
    *,
    ds: rasterio.io.DatasetReader,
    lat: float,
    lon: float,
    subgroup_type: str,
    subgroup: str,
    r_max_km: float,
) -> Dict[str, float]:
    """
    Subgroup-specific exposure:
    - ring: annulus [lo,hi] km
    - region: half-plane (E/W/N/S of mainshock) within r_max_km
    - depth: whole circle within r_max_km (depth isn't a spatial split)
    """
    arr, lats, lons = _read_window_and_latlon(ds=ds, lat=lat, lon=lon, r_km=r_max_km)
    if arr is None:
        return {"pop_sum": float("nan"), "pop_mean": float("nan"), "pop_max": float("nan")}
    dist_km = _haversine_km(lats, lons, lat, lon)
    within = dist_km <= float(r_max_km)

    st = str(subgroup_type)
    sg = str(subgroup)

    if st == "ring":
        rr = _parse_ring_km(sg)
        if rr is None:
            return {"pop_sum": float("nan"), "pop_mean": float("nan"), "pop_max": float("nan")}
        lo, hi = rr
        m = within & (dist_km > float(lo)) & (dist_km <= float(hi))
        return _stats_from_mask(arr, m)

    if st == "region":
        if "East_of" in sg:
            half = lons >= float(lon)
        elif "West_of" in sg:
            half = lons < float(lon)
        elif "North_of" in sg:
            half = lats >= float(lat)
        elif "South_of" in sg:
            half = lats < float(lat)
        else:
            half = np.ones_like(within, dtype=bool)
        return _stats_from_mask(arr, within & half)

    return _stats_from_mask(arr, within)


def main() -> None:
    if not POP_RASTER_PATH:
        raise SystemExit("Set POP_RASTER_PATH in config.py to your local population raster .tif file.")

    ms = load_mainshocks()
    sub = load_subgroups()
    out_rows: List[Dict] = []
    out_sub_rows: List[Dict] = []

    r_max_km = float(max(DISTANCE_RINGS_KM) if DISTANCE_RINGS_KM else 400.0)
    ms_map = ms.set_index("usgs_id")[["lat", "lon"]].to_dict(orient="index")

    with rasterio.open(POP_RASTER_PATH) as ds:
        for _, row in ms.iterrows():
            base = {
                "usgs_id": row["usgs_id"],
                "time_utc": str(row["time_utc"]),
                "mag": float(row["mag"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "depth_km": float(row["depth_km"]) if pd.notna(row.get("depth_km")) else float("nan"),
            }
            for r in EXPOSURE_BUFFER_KM:
                stats = zonal_stats_population(ds=ds, lat=base["lat"], lon=base["lon"], r_km=float(r))
                for k, v in stats.items():
                    base[f"{k}_{int(r)}km"] = v
            out_rows.append(base)
            print("Exposure done:", base["usgs_id"])

        if not sub.empty:
            for _, r in sub.iterrows():
                msid = str(r["mainshock_id"])
                if msid not in ms_map:
                    continue
                stats = zonal_stats_population_subgroup(
                    ds=ds,
                    lat=float(ms_map[msid]["lat"]),
                    lon=float(ms_map[msid]["lon"]),
                    subgroup_type=str(r["subgroup_type"]),
                    subgroup=str(r["subgroup"]),
                    r_max_km=r_max_km,
                )
                out_sub_rows.append(
                    {
                        "mainshock_id": msid,
                        "subgroup_type": str(r["subgroup_type"]),
                        "subgroup": str(r["subgroup"]),
                        "pop_sum_subgroup": stats["pop_sum"],
                        "pop_mean_subgroup": stats["pop_mean"],
                        "pop_max_subgroup": stats["pop_max"],
                        "r_max_km": r_max_km,
                    }
                )
            print("Subgroup exposure done:", len(out_sub_rows))

    out = pd.DataFrame(out_rows)
    here = Path(__file__).resolve().parent
    out_path = here / "data" / "exposure_mainshocks.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    if out_sub_rows:
        out2 = pd.DataFrame(out_sub_rows)
        out2_path = here / "data" / "exposure_subgroups.csv"
        out2.to_csv(out2_path, index=False)
        print("Saved:", out2_path)


if __name__ == "__main__":
    main()

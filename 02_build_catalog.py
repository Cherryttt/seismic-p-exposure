from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_geojson(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mainshock(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_catalog(events: Dict[str, Any], mainshock: Dict[str, Any]) -> pd.DataFrame:
    feats = events.get("features") or []
    rows: List[Dict[str, Any]] = []
    for f in feats:
        props = f.get("properties") or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None, None]
        lon, lat, depth = (coords + [None, None, None])[:3]
        t_ms = props.get("time")
        mag = props.get("mag")
        if t_ms is None or mag is None or lat is None or lon is None:
            continue
        rows.append(
            {
                "usgs_id": f.get("id"),
                "time_ms": int(t_ms),
                "time_utc": pd.to_datetime(int(t_ms), unit="ms", utc=True),
                "lat": float(lat),
                "lon": float(lon),
                "depth_km": float(depth) if depth is not None else np.nan,
                "mag": float(mag),
                "place": props.get("place"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Remove the mainshock itself (match by usgs_id when possible; else by time).
    ms_id = mainshock.get("usgs_id")
    ms_time = pd.to_datetime(mainshock["time_utc"], utc=True)
    if ms_id:
        df = df[df["usgs_id"] != ms_id]
    df = df[df["time_utc"] > ms_time]

    # Add time since mainshock in days.
    df["t_days"] = (df["time_utc"] - ms_time) / pd.Timedelta(days=1)

    # Basic sanity filters.
    df = df[df["t_days"] > 0]
    df = df[df["mag"].between(-1, 10)]
    df = df[df["depth_km"].between(-5, 800) | df["depth_km"].isna()]

    df = df.sort_values("time_utc").reset_index(drop=True)
    return df


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_geojson(data_dir / "events.geojson")
    mainshock = load_mainshock(data_dir / "mainshock.json")

    df = to_catalog(events, mainshock)
    if df.empty:
        raise SystemExit("Catalog is empty. Check download step.")

    # Save parquet if available, else CSV.
    out_parquet = data_dir / "catalog.parquet"
    out_csv = data_dir / "catalog.csv"
    try:
        df.to_parquet(out_parquet, index=False)
        print("Saved:", out_parquet)
    except Exception:
        df.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

    print("Rows:", len(df))
    print("Mag range:", (df["mag"].min(), df["mag"].max()))
    print("Depth range:", (df["depth_km"].min(), df["depth_km"].max()))
    print("Time range days:", (df["t_days"].min(), df["t_days"].max()))


if __name__ == "__main__":
    main()

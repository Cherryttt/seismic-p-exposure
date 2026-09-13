from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_geojson(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_catalog(events: Dict[str, Any], mainshock: Dict[str, Any]) -> pd.DataFrame:
    feats = events.get("features") or []
    rows: List[Dict[str, Any]] = []
    ms_time = pd.to_datetime(mainshock["time_utc"], utc=True)
    ms_id = mainshock.get("usgs_id")
    for f in feats:
        props = f.get("properties") or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None, None]
        lon, lat, depth = (coords + [None, None, None])[:3]
        t_ms = props.get("time")
        mag = props.get("mag")
        if t_ms is None or mag is None or lat is None or lon is None:
            continue
        usgs_id = f.get("id")
        t_utc = pd.to_datetime(int(t_ms), unit="ms", utc=True)
        if ms_id and usgs_id == ms_id:
            continue
        if t_utc <= ms_time:
            continue
        rows.append(
            {
                "mainshock_id": ms_id or "",
                "mainshock_time_utc": ms_time,
                "mainshock_mag": float(mainshock.get("mag", np.nan)),
                "mainshock_lat": float(mainshock.get("lat", np.nan)),
                "mainshock_lon": float(mainshock.get("lon", np.nan)),
                "usgs_id": usgs_id,
                "time_ms": int(t_ms),
                "time_utc": t_utc,
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
    df["t_days"] = (df["time_utc"] - ms_time) / pd.Timedelta(days=1)
    df = df[df["t_days"] > 0]
    df = df.sort_values(["mainshock_id", "time_utc"]).reset_index(drop=True)
    return df


def main() -> None:
    here = Path(__file__).resolve().parent
    runs_dir = here / "data" / "runs"
    out_dir = here / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_parts: List[pd.DataFrame] = []
    for run in sorted(runs_dir.glob("*")):
        if not run.is_dir():
            continue
        ms_path = run / "mainshock.json"
        ev_path = run / "events.geojson"
        if not (ms_path.exists() and ev_path.exists()):
            continue
        ms = json.load(open(ms_path, "r", encoding="utf-8"))
        ev = json.load(open(ev_path, "r", encoding="utf-8"))
        df = to_catalog(ev, ms)
        if not df.empty:
            all_parts.append(df)
            print("Built catalog for", run.name, "rows", len(df))

    if not all_parts:
        raise SystemExit("No run catalogs found. Run 11_batch_download.py first.")

    full = pd.concat(all_parts, ignore_index=True)

    out_pqt = out_dir / "catalog_all.parquet"
    out_csv = out_dir / "catalog_all.csv"
    try:
        full.to_parquet(out_pqt, index=False)
        print("Saved:", out_pqt)
    except Exception:
        full.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

    print("Total rows:", len(full))


if __name__ == "__main__":
    main()


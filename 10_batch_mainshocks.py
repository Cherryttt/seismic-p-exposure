from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config import (
    DECLUSTER_DAYS,
    MAINSHOCK_BATCH_END_UTC,
    MAINSHOCK_BATCH_MINMAG,
    MAINSHOCK_BATCH_START_UTC,
    MAX_MAINSHOCKS,
    REGION_MAX_LAT,
    REGION_MAX_LON,
    REGION_MIN_LAT,
    REGION_MIN_LON,
)
from usgs_api import search_events_geojson, save_json


def _dt_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _parse_features(events: Dict[str, Any]) -> pd.DataFrame:
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
                "depth_km": float(depth) if depth is not None else None,
                "mag": float(mag),
                "place": props.get("place"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("time_utc").reset_index(drop=True)
    return df


def decluster_pick(df: pd.DataFrame, max_n: int, window_days: int) -> pd.DataFrame:
    # Simple declustering: iterate by time asc, keep event if not within +/-window_days
    # of any already selected event. This avoids selecting aftershocks as "mainshocks".
    selected: List[int] = []
    times = df["time_utc"].to_list()
    for i, t in enumerate(times):
        keep = True
        for j in selected:
            dt_days = abs((t - times[j]).total_seconds()) / 86400.0
            if dt_days < window_days:
                keep = False
                break
        if keep:
            selected.append(i)
        if len(selected) >= max_n:
            break
    return df.iloc[selected].reset_index(drop=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    events = search_events_geojson(
        start_utc=MAINSHOCK_BATCH_START_UTC,
        end_utc=MAINSHOCK_BATCH_END_UTC,
        minlat=REGION_MIN_LAT,
        maxlat=REGION_MAX_LAT,
        minlon=REGION_MIN_LON,
        maxlon=REGION_MAX_LON,
        minmag=MAINSHOCK_BATCH_MINMAG,
        orderby="time-asc",
        limit=20000,
    )

    df = _parse_features(events)
    if df.empty:
        raise SystemExit("No events found for region/time range.")

    picked = decluster_pick(df, max_n=MAX_MAINSHOCKS, window_days=DECLUSTER_DAYS)

    # Ensure JSON-serializable output
    picked2 = picked.copy()
    picked2["time_utc"] = picked2["time_utc"].astype(str)
    out = picked2.to_dict(orient="records")
    save_json(str(data_dir / "mainshocks.json"), out)
    print("Candidates:", len(df))
    print("Picked mainshocks:", len(picked))
    print("Saved:", data_dir / "mainshocks.json")

    # Print a quick table for user review
    print(picked[["time_utc", "mag", "lat", "lon", "depth_km", "place", "usgs_id"]].to_string(index=False))


if __name__ == "__main__":
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    main()

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config import DAYS_AFTER, MINMAG_DOWNLOAD, RADIUS_KM
from usgs_api import dt_to_usgs, download_aftershocks_geojson, save_json


def load_mainshocks(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    runs_dir = data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    mainshocks = load_mainshocks(data_dir / "mainshocks.json")
    if not mainshocks:
        raise SystemExit("mainshocks.json is empty. Run 10_batch_mainshocks.py first.")

    index_rows: List[Dict[str, Any]] = []
    for ms in mainshocks:
        usgs_id = ms["usgs_id"]
        out_dir = runs_dir / usgs_id
        out_dir.mkdir(parents=True, exist_ok=True)

        ms_time = pd.to_datetime(ms["time_utc"], utc=True).to_pydatetime()
        ms_time_utc = dt_to_usgs(ms_time)

        save_json(str(out_dir / "mainshock.json"), ms)

        events = download_aftershocks_geojson(
            mainshock_time_utc=ms_time_utc,
            lat=float(ms["lat"]),
            lon=float(ms["lon"]),
            radius_km=RADIUS_KM,
            days_after=DAYS_AFTER,
            minmag=MINMAG_DOWNLOAD,
        )
        save_json(str(out_dir / "events.geojson"), events)
        n = len(events.get("features") or [])
        index_rows.append(
            {
                "usgs_id": usgs_id,
                "time_utc": ms["time_utc"],
                "mag": ms["mag"],
                "lat": ms["lat"],
                "lon": ms["lon"],
                "depth_km": ms.get("depth_km"),
                "place": ms.get("place"),
                "aftershocks_n": n,
                "radius_km": RADIUS_KM,
                "days_after": DAYS_AFTER,
                "minmag_download": MINMAG_DOWNLOAD,
            }
        )
        print(f"Downloaded {n} aftershocks for {usgs_id} (M{ms['mag']})")

    df = pd.DataFrame(index_rows)
    df.to_csv(data_dir / "runs_index.csv", index=False)
    print("Saved:", data_dir / "runs_index.csv")


if __name__ == "__main__":
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    main()


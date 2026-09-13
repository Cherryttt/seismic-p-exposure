from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import os
from pathlib import Path

from config import (
    DAYS_AFTER,
    MAINSHOCK_LAT,
    MAINSHOCK_LON,
    MAINSHOCK_MINMAG,
    MAINSHOCK_SEARCH_END_UTC,
    MAINSHOCK_SEARCH_RADIUS_KM,
    MAINSHOCK_SEARCH_START_UTC,
    MINMAG_DOWNLOAD,
    RADIUS_KM,
)
from usgs_api import download_aftershocks_geojson, find_mainshock, save_json


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    mainshock = find_mainshock(
        start_utc=MAINSHOCK_SEARCH_START_UTC,
        end_utc=MAINSHOCK_SEARCH_END_UTC,
        lat=MAINSHOCK_LAT,
        lon=MAINSHOCK_LON,
        radius_km=MAINSHOCK_SEARCH_RADIUS_KM,
        minmag=MAINSHOCK_MINMAG,
    )
    save_json(str(data_dir / "mainshock.json"), mainshock.__dict__)

    events = download_aftershocks_geojson(
        mainshock_time_utc=mainshock.time_utc,
        lat=mainshock.lat,
        lon=mainshock.lon,
        radius_km=RADIUS_KM,
        days_after=DAYS_AFTER,
        minmag=MINMAG_DOWNLOAD,
    )
    save_json(str(data_dir / "events.geojson"), events)

    print("Mainshock:", mainshock)
    print("Saved:", data_dir / "mainshock.json")
    print("Saved:", data_dir / "events.geojson")
    n = len(events.get("features") or [])
    print("Aftershock events downloaded:", n)


if __name__ == "__main__":
    # Avoid proxy surprises if user has stale env vars.
    for k in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(k, None)

    main()

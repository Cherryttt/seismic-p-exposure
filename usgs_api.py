from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

import requests


USGS_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def _parse_utc(s: str) -> datetime:
    # Accept "...Z" or ISO strings.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def dt_to_usgs(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Mainshock:
    time_utc: str
    lat: float
    lon: float
    depth_km: float
    mag: float
    place: str
    usgs_id: str


def find_mainshock(
    *,
    start_utc: str,
    end_utc: str,
    lat: float,
    lon: float,
    radius_km: float,
    minmag: float,
    timeout_s: int = 30,
) -> Mainshock:
    params = {
        "format": "geojson",
        "starttime": start_utc,
        "endtime": end_utc,
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": minmag,
        "orderby": "magnitude",
        "limit": 10,
    }
    r = requests.get(USGS_QUERY, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    feats = data.get("features") or []
    if not feats:
        raise RuntimeError(f"No mainshock candidates returned by USGS for params={params}")

    # Pick the largest magnitude (already ordered), break ties by earliest time.
    def key(f: Dict[str, Any]) -> Tuple[float, int]:
        mag = f["properties"].get("mag") or -999.0
        t_ms = f["properties"].get("time") or 0
        return (mag, -t_ms)

    f = sorted(feats, key=key, reverse=True)[0]
    props = f["properties"]
    coords = f["geometry"]["coordinates"]
    t_ms = props["time"]
    dt = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
    return Mainshock(
        time_utc=dt_to_usgs(dt),
        lat=float(coords[1]),
        lon=float(coords[0]),
        depth_km=float(coords[2]) if len(coords) > 2 else float("nan"),
        mag=float(props["mag"]),
        place=str(props.get("place") or ""),
        usgs_id=str(f.get("id") or ""),
    )


def download_aftershocks_geojson(
    *,
    mainshock_time_utc: str,
    lat: float,
    lon: float,
    radius_km: float,
    days_after: int,
    minmag: float,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    t0 = _parse_utc(mainshock_time_utc)
    t1 = t0 + timedelta(days=days_after)
    params = {
        "format": "geojson",
        "starttime": dt_to_usgs(t0),
        "endtime": dt_to_usgs(t1),
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": minmag,
        "orderby": "time-asc",
    }
    return _paged_geojson(params=params, timeout_s=timeout_s)


def search_events_geojson(
    *,
    start_utc: str,
    end_utc: str,
    minlat: float,
    maxlat: float,
    minlon: float,
    maxlon: float,
    minmag: float,
    orderby: str = "time-asc",
    limit: int = 20000,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    params = {
        "format": "geojson",
        "starttime": start_utc,
        "endtime": end_utc,
        "minlatitude": minlat,
        "maxlatitude": maxlat,
        "minlongitude": minlon,
        "maxlongitude": maxlon,
        "minmagnitude": minmag,
        "orderby": orderby,
    }
    return _paged_geojson(params=params, timeout_s=timeout_s, page_limit=limit)


def _paged_geojson(*, params: Dict[str, Any], timeout_s: int, page_limit: int = 20000) -> Dict[str, Any]:
    """
    USGS GeoJSON has a hard-ish page size limit (commonly 20k). Use `offset` pagination
    and aggregate features to avoid silent truncation when building 100k+ catalogs.
    """
    # USGS uses 1-based offset in some docs; in practice offset=1 returns first event.
    # We'll use offset=1 and increase by page size.
    page_size = min(int(page_limit), 20000)
    offset = 1
    all_features = []
    meta = None

    while True:
        q = dict(params)
        q["limit"] = page_size
        q["offset"] = offset
        r = requests.get(USGS_QUERY, params=q, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        if meta is None:
            meta = data.get("metadata") or {}
        feats = data.get("features") or []
        all_features.extend(feats)

        # Stop conditions: fewer than a full page, or we've reached reported count.
        if len(feats) < page_size:
            break
        total = meta.get("count")
        if isinstance(total, int) and (offset - 1 + len(feats)) >= total:
            break
        offset += page_size

    out = {
        "type": "FeatureCollection",
        "metadata": meta or {},
        "features": all_features,
    }
    return out


def save_json(path: str, obj: Any) -> None:
    # Write atomically and tolerate common non-JSON types (e.g., pandas Timestamp)
    # by stringifying them. This keeps batch pipelines robust.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    # Atomic-ish on Windows when target doesn't exist; for overwrite this is still safe enough
    # for our use (prevents leaving a half-written JSON on crash).
    import os

    os.replace(tmp_path, path)

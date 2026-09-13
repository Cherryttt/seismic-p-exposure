from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"


def _safe(v) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return ""
    except Exception:
        pass
    return str(v)


def main() -> None:
    import folium
    from folium.features import GeoJsonTooltip

    ms = pd.read_csv(OUT_DIR / "mainshock_summary.csv")
    if not {"mainshock_lat", "mainshock_lon"}.issubset(ms.columns):
        raise RuntimeError("mainshock_summary.csv missing lat/lon columns.")

    # Center map
    lat0 = float(ms["mainshock_lat"].mean())
    lon0 = float(ms["mainshock_lon"].mean())

    m = folium.Map(location=[lat0, lon0], zoom_start=6, tiles="OpenStreetMap", control_scale=True)

    # Color by p
    pmin = float(np.nanmin(ms["p"].to_numpy(float)))
    pmax = float(np.nanmax(ms["p"].to_numpy(float)))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        pmin, pmax = 0.2, 1.5

    def color(p: float) -> str:
        # simple blue->red
        t = (p - pmin) / (pmax - pmin + 1e-9)
        t = float(np.clip(t, 0, 1))
        r = int(255 * t)
        b = int(255 * (1 - t))
        return f"#{r:02x}00{b:02x}"

    for _, r in ms.iterrows():
        lat = float(r["mainshock_lat"])
        lon = float(r["mainshock_lon"])
        pid = str(r["mainshock_id"])
        mag = float(r.get("mainshock_mag", np.nan))
        p = float(r.get("p", np.nan))
        pop50 = float(r.get("pop_sum_50km", np.nan))

        html = (
            f"<b>{pid}</b><br/>"
            f"M={mag:.1f} &nbsp; p={p:.3f}<br/>"
            f"pop_sum_50km={pop50:,.0f}<br/>"
            f"n_events={int(r.get('n_events', 0))}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=color(p),
            fill=True,
            fill_color=color(p),
            fill_opacity=0.85,
            tooltip=pid,
            popup=folium.Popup(html, max_width=360),
        ).add_to(m)

    # Add legend as a simple HTML overlay
    legend = f"""
    <div style="
        position: fixed;
        bottom: 24px; left: 24px; z-index: 9999;
        background: rgba(255,255,255,0.92);
        padding: 10px 12px; border: 1px solid #ddd; border-radius: 10px;
        font-family: Arial, sans-serif; font-size: 12px; color: #111;
    ">
      <div style="font-weight: 700; margin-bottom: 6px;">Mainshock p (color)</div>
      <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:140px; height:10px; background: linear-gradient(90deg, #0000ff, #ff0000); border:1px solid #ccc;"></div>
        <div>{pmin:.2f} → {pmax:.2f}</div>
      </div>
      <div style="margin-top:6px; color:#555;">Click a point to view details.</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    out = OUT_DIR / "map_mainshocks.html"
    m.save(out)
    print("Saved map:", out)


if __name__ == "__main__":
    main()


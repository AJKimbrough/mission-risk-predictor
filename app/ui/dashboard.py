import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (…/mission_go_no_go)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any

import requests
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# Reuse our local DB helper to read TFRs/features directly
from app.db import fetch_df

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# --- Aetheris branding ---
st.set_page_config(page_title="Aetheris", page_icon="🛡️", layout="wide")
st.title("Aetheris")
st.caption("Go/No-Go decisions from FAA & NOAA data — fast, explainable, map-first.")

# Keep route overlays across reruns: {route_index: [pydeck.Layer, ...]}
if "route_overlays" not in st.session_state:
    st.session_state["route_overlays"] = {}

# ---------------- Inputs ----------------
col1, col2, col3 = st.columns(3)
lat = col1.number_input("Latitude", value=32.7767, format="%.6f")
lon = col2.number_input("Longitude", value=-96.7970, format="%.6f")
runway = col3.number_input("Runway Heading (deg)", value=170)

now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
default_end = now_utc + timedelta(hours=24)

# Date/time inputs (Streamlit 3.8-safe)
s_date = col1.date_input("Start date (UTC)", value=now_utc.date())
s_time = col2.time_input("Start time (UTC)", value=now_utc.time())
e_date = col1.date_input("End date (UTC)", value=default_end.date(), key="end_date")
e_time = col2.time_input("End time (UTC)", value=default_end.time(), key="end_time")

def combine_utc(d, t):
    dt = datetime.combine(d, t)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

start = combine_utc(s_date, s_time)
end = combine_utc(e_date, e_time)

st.markdown("#### Corridor (optional)")
st.caption("Enter waypoints as `lat,lon; lat,lon; ...` to render a corridor polyline.")
corridor_text = st.text_input("Waypoints", value="")

# Map filters
st.markdown("#### Map Options")
mcol1, mcol2, mcol3 = st.columns(3)
radius_km = mcol1.slider("Map radius (km)", 10, 300, 80)
show_heat = mcol2.checkbox("Show Risk Heatmap", value=True)
show_tfr = mcol3.checkbox("Show No-Fly (TFR) Polygons", value=True)

# ---------------- Helpers ----------------
def parse_corridor(s: str) -> List[Tuple[float, float]]:
    pts = []
    if not s.strip():
        return pts
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            a, b = seg.split(",")
            pts.append((float(a.strip()), float(b.strip())))
        except Exception:
            pass
    return pts

def km_to_deg(km: float) -> float:
    return km / 111.0  # rough conversion

def load_tfr_geojson(lat: float, lon: float, start_iso: str, end_iso: str, radius_km: float) -> List[dict]:
    # (Time-filter only; bbox filtering happens client-side if desired)
    sql = """
    SELECT geometry_geojson, name, tfr_type, eff_start, eff_end
    FROM tfr
    WHERE eff_start <= :end AND eff_end >= :start
      AND geometry_geojson IS NOT NULL
    """
    df = fetch_df(sql, start=start_iso, end=end_iso)
    features = []
    for _, r in df.iterrows():
        try:
            gj = json.loads(r["geometry_geojson"])
        except Exception:
            continue
        props = {
            "name": r.get("name"),
            "tfr_type": r.get("tfr_type"),
            "eff_start": r.get("eff_start"),
            "eff_end": r.get("eff_end"),
        }
        if gj.get("type") == "FeatureCollection":
            for f in gj.get("features", []):
                f.setdefault("properties", {}).update(props)
                features.append(f)
        elif gj.get("type") == "Feature":
            gj.setdefault("properties", {}).update(props)
            features.append(gj)
        else:
            features.append({"type": "Feature", "geometry": gj, "properties": props})
    return features

def load_features_points(lat: float, lon: float, start_iso: str, end_iso: str, radius_km: float) -> pd.DataFrame:
    sql = """
    SELECT lat, lon, valid_time, wind_kts, gust_kts, pop_pct,
           convective_flag, icing_flag, tfr_active_flag, vis_sm, ceiling_ft
    FROM features
    WHERE valid_time BETWEEN :start AND :end
    """
    df = fetch_df(sql, start=start_iso, end=end_iso)
    if df.empty:
        return df
    deg = km_to_deg(radius_km)
    df = df[(df["lat"].astype(float).between(lat-deg, lat+deg)) &
            (df["lon"].astype(float).between(lon-deg, lon+deg))].copy()
    if df.empty:
        return df

    def risk_row(r):
        risk = 0.0
        if pd.notna(r.get("pop_pct")):
            risk += min(float(r["pop_pct"]) / 100.0, 1.0) * 0.3
        if pd.notna(r.get("wind_kts")):
            risk += min(float(r["wind_kts"]) / 35.0, 1.0) * 0.3
        if pd.notna(r.get("gust_kts")):
            risk += min(float(r["gust_kts"]) / 45.0, 1.0) * 0.1
        if pd.notna(r.get("vis_sm")) and float(r["vis_sm"]) < 3.0:
            risk += 0.3
        if pd.notna(r.get("ceiling_ft")) and float(r["ceiling_ft"]) < 1000:
            risk += 0.3
        if bool(r.get("convective_flag")):
            risk += 0.2
        if bool(r.get("tfr_active_flag")):
            risk += 0.3
        return min(risk, 1.5)

    df["risk_weight"] = df.apply(risk_row, axis=1)
    return df

def make_map_layers(center_lat: float, center_lon: float, start_iso: str, end_iso: str,
                    radius_km: float, show_heat: bool, show_tfr: bool,
                    corridor_pts: List[Tuple[float, float]]):
    layers = []
    # Heatmap
    if show_heat:
        fdf = load_features_points(center_lat, center_lon, start_iso, end_iso, radius_km)
        if not fdf.empty:
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=fdf.rename(columns={"lat": "latitude", "lon": "longitude"}),
                    get_position='[longitude, latitude]',
                    get_weight="risk_weight",
                    radiusPixels=60,
                    aggregation='"SUM"',
                    opacity=0.7,
                )
            )
    # TFR polygons
    if show_tfr:
        feats = load_tfr_geojson(center_lat, center_lon, start_iso, end_iso, radius_km)
        rows = []
        def poly_coords(geom) -> Optional[List[List[float]]]:
            t = geom.get("type")
            if t == "Polygon":
                return geom.get("coordinates", [])[0]
            if t == "MultiPolygon":
                polys = geom.get("coordinates", [])
                return polys[0][0] if polys and polys[0] else None
            return None
        for f in feats:
            geom = f.get("geometry", {})
            coords = poly_coords(geom)
            if not coords:
                continue
            rows.append({
                "polygon": coords,
                "name": f.get("properties", {}).get("name", "TFR"),
                "tfr_type": f.get("properties", {}).get("tfr_type", "TFR"),
            })
        if rows:
            layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=rows,
                    get_polygon="polygon",
                    stroked=True,
                    filled=True,
                    wireframe=False,
                    get_line_width=2,
                    lineWidthMinPixels=2,
                    get_line_color=[255, 0, 0, 255],
                    get_fill_color=[255, 0, 0, 80],
                )
            )
    # Corridor
    if corridor_pts:
        path = [[lon, lat] for lat, lon in corridor_pts]
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=pd.DataFrame({"path": [path]}),
                get_path="path",
                get_width=4,
                widthMinPixels=4,
                get_color=[0, 120, 240, 220],
            )
        )
    # Center marker
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({"latitude": [center_lat], "longitude": [center_lon]}),
            get_position='[longitude, latitude]',
            get_radius=150,
            radiusMinPixels=4,
            radiusMaxPixels=10,
            get_fill_color=[0, 200, 80, 255],
        )
    )
    return layers

def _segment_color(label: str):
    return {"GREEN": [0, 180, 80, 220], "AMBER": [255, 170, 0, 220], "RED": [220, 40, 40, 240]}.get(
        label, [128, 128, 128, 200]
    )

def _layers_from_segments(segments, route_idx: int, show_segment_labels: bool = False):
    """Build a PathLayer for segments; optionally add per-segment TextLayer."""
    rows = []
    for s in segments:
        a = s["a"]          # [lat, lon]
        b = s["b"]
        cen = s["center"]
        rows.append({
            "path": [[a[1], a[0]], [b[1], b[0]]],   # [lon, lat]
            "label": s["label"],
            "hazards": ", ".join(s["hazards"]) if s["hazards"] else "",
            "risk": float(s["risk"]),
            "lon": float(cen[1]),
            "lat": float(cen[0]),
            "route": f"Route {route_idx+1}",
            "color": _segment_color(s["label"]),
        })

    if not rows:
        return []

    seg_df = pd.DataFrame(rows)

    seg_layer = pdk.Layer(
        "PathLayer",
        data=seg_df,
        get_path="path",
        get_color="color",
        get_width=6,
        widthMinPixels=4,
        pickable=True,
    )

    layers = [seg_layer]

    # ONLY if you really want per-segment labels later:
    if show_segment_labels:
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=seg_df,
                get_position='[lon, lat]',
                get_text='route + " — " + hazards',
                get_size=12,
                get_color=[0, 0, 0, 255],
                background=True,
                billboard=True,
            )
        )

    return layers

def _route_name_layer(route_points, route_idx: int):
    """Single 'Route N' label at the route centroid."""
    if not route_points:
        return None
    lat = sum(p[0] for p in route_points) / len(route_points)
    lon = sum(p[1] for p in route_points) / len(route_points)
    return pdk.Layer(
        "TextLayer",
        data=pd.DataFrame([{"lon": lon, "lat": lat, "name": f"Route {route_idx+1}"}]),
        get_position='[lon, lat]',
        get_text='name',
        get_size=14,
        get_color=[0, 0, 0, 255],
        background=True,
        billboard=True,
    )


# ---------------- Decide + Map ----------------
left, right = st.columns([1, 1])

with left:
    if st.button("Decide"):
        body = {
            "lat": float(lat),
            "lon": float(lon),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "runway_heading": int(runway),
        }
        try:
            r = requests.post(f"{API_BASE}/v1/decide", json=body, timeout=30)
            if r.ok:
                data = r.json()
                st.subheader(f"Decision: **{data.get('label','?')}**")
                hourly = data.get("hourly", [])
                ml = data.get("ml_score")
                if ml is not None:
                    st.metric(label="ML risk (No-Go prob)", value=f"{ml:.2f}")
                if hourly:
                    st.dataframe(pd.DataFrame(hourly))
                with st.expander("Raw response"):
                    st.json(data)
            else:
                st.error(f"API error: {r.status_code} {r.text}")
        except Exception as ex:
            st.exception(ex)

with right:
    corridor_pts = parse_corridor(corridor_text)
    layers = make_map_layers(
        center_lat=float(lat),
        center_lon=float(lon),
        start_iso=start.isoformat(),
        end_iso=end.isoformat(),
        radius_km=float(radius_km),
        show_heat=show_heat,
        show_tfr=show_tfr,
        corridor_pts=corridor_pts,
    )

    # Add any route overlays from session state (all evaluated routes)
    for i in sorted(st.session_state.get("route_overlays", {}).keys()):
        for lyr in st.session_state["route_overlays"][i]:
            layers.append(lyr)

    view_state = pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=8)
    if radius_km <= 20:
        view_state.zoom = 11
    elif radius_km <= 50:
        view_state.zoom = 10
    elif radius_km <= 100:
        view_state.zoom = 9
    else:
        view_state.zoom = 8

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "{name} {tfr_type}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

# ---------------- Routes UI ----------------
st.markdown("### Routes")
st.caption("Enter one or more routes: each line is `lat,lon; lat,lon; ...`")

routes_text = st.text_area(
    "Route options",
    height=100,
    value=(
        "32.7767,-96.7970; 33.0,-97.0; 33.3,-97.2\n"
        "32.7767,-96.7970; 32.9,-96.5; 33.2,-96.8; 33.4,-97.1"
    ),
)

def parse_route(line: str):
    pts = []
    for seg in line.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            a, b = seg.split(",")
            pts.append([float(a.strip()), float(b.strip())])
        except Exception:
            continue
    return pts

route_lines = [ln.strip() for ln in routes_text.splitlines() if ln.strip()]
ROUTES = [r for r in (parse_route(ln) for ln in route_lines) if len(r) >= 2]
if not ROUTES:
    st.info("Add at least one valid route (two or more waypoints).")

opt_idx = st.selectbox(
    "Choose route",
    options=list(range(len(ROUTES))) if ROUTES else [0],
    format_func=lambda i: f"Option {i+1}",
    disabled=not ROUTES,
)

dep_offset = st.slider("Departure offset from chosen Start (UTC)", -2, 24, 0, 1)
dep_time = start + timedelta(hours=dep_offset)

gs = st.slider("Ground speed (kts)", 40, 180, 80)
buf_km = st.slider("Route hazard buffer (km)", 1, 10, 3)

eval_body_base = {
    "departure_time": dep_time.isoformat(),
    "ground_speed_kts": float(gs),
    "buffer_km": float(buf_km),
    "horizon_pad_min": 20,
}

cols = st.columns(2)

# --- Evaluate the selected route (adds to overlays) ---
# --- Evaluate the selected route (clears others, adds only chosen) ---
if cols[0].button("Evaluate Selected", disabled=not ROUTES):
    try:
        i = opt_idx
        body = {**eval_body_base, "path": ROUTES[i]}
        r = requests.post(f"{API_BASE}/v1/route_evaluate", json=body, timeout=40)
        r.raise_for_status()
        out = r.json()

        st.subheader(f"Route {i+1} Summary")
        st.write(out.get("summary", {}))

        segs = out.get("segments", [])
        if segs:
            st.dataframe(pd.DataFrame(segs)[
                ["idx_from","idx_to","start_time","end_time","label","hazards","risk"]
            ])
            # clear all overlays, keep only current
            st.session_state["route_overlays"] = {}
            layers = _layers_from_segments(segs, i, show_segment_labels=False)
            name_layer = _route_name_layer(ROUTES[i], i)
            if name_layer:
                layers.append(name_layer)
            st.session_state["route_overlays"][i] = layers
        else:
            st.info("No segments returned from evaluator.")

    except Exception as ex:
        st.error(f"Route evaluation failed: {ex}")


# --- Evaluate ALL routes (replaces overlays with all) ---
if cols[1].button("Evaluate All Routes", disabled=not ROUTES):
    try:
        st.session_state["route_overlays"] = {}  # clear before adding all
        added = 0
        for i, route in enumerate(ROUTES):
            body = {**eval_body_base, "path": route}
            rr = requests.post(f"{API_BASE}/v1/route_evaluate", json=body, timeout=40)
            if not rr.ok:
                continue
            out = rr.json()
            segs = out.get("segments", [])
            if segs:
                layers = _layers_from_segments(segs, i, show_segment_labels=False)
                name_layer = _route_name_layer(route, i)
                if name_layer:
                    layers.append(name_layer)
                st.session_state["route_overlays"][i] = layers
                added += 1

        if added == 0:
            st.info("No overlays created — API returned no segments.")
        else:
            st.success(f"Added overlays for {added} route(s).")
    except Exception as ex:
        st.error(f"Ranking/evaluation failed: {ex}")

with st.expander("Legend & How to Read"):
    st.markdown("""
**Layers**
- Heatmap = areas with higher risk signals (from recent METAR/TAF-derived features).
- Red polygons = TFR no-fly zones (time-filtered).
- Blue line = optional corridor you typed above.
- Colored route segments:
  - 🟩 **Green** safe
  - 🟧 **Amber** elevated risk
  - 🟥 **Red** no-go

**Tables**
- Segment table lists the time window each colored segment would be flown and its hazards.
- Route Summary shows `total_km`, and distance split across `red_km`, `amber_km`, `green_km`.

**Tips**
- Use *Departure offset* to time-shift risk along the route.
- Use *Evaluate All Routes* to compare options on the same map.
""")

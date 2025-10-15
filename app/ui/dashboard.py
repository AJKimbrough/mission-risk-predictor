import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (…/mission_go_no_go)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# --- Reuse our local DB helper to read TFRs/features directly ---
from app.db import fetch_df

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

# --- Local Decision Logic (replaces API call) ---
def local_decide(lat: float, lon: float, start: datetime, end: datetime, runway: int) -> Dict[str, Any]:
    df = load_features_points(lat, lon, start.isoformat(), end.isoformat(), radius_km=80)
    if df.empty:
        return {"label": "UNKNOWN", "hourly": [], "ml_score": None}
    mean_risk = df["risk_weight"].mean()
    label = "GREEN"
    if mean_risk > 0.9:
        label = "RED"
    elif mean_risk > 0.6:
        label = "AMBER"
    return {
        "label": label,
        "ml_score": float(mean_risk),
        "hourly": df[["valid_time", "risk_weight", "wind_kts", "gust_kts", "vis_sm"]].to_dict(orient="records")
    }

# ---------------- Decide + Map ----------------
left, right = st.columns([1, 1])
with left:
    if st.button("Decide"):
        try:
            data = local_decide(lat, lon, start, end, runway)
            st.subheader(f"Decision: **{data.get('label','?')}**")
            ml = data.get("ml_score")
            if ml is not None:
                st.metric(label="ML risk (No-Go prob)", value=f"{ml:.2f}")
            hourly = data.get("hourly", [])
            if hourly:
                st.dataframe(pd.DataFrame(hourly))
            with st.expander("Raw response"):
                st.json(data)
        except Exception as ex:
            st.exception(ex)

# --- Map rendering (unchanged) ---
with right:
    corridor_pts = parse_corridor(corridor_text)
    layers = make_map_layers(lat, lon, start.isoformat(), end.isoformat(),
                             radius_km, show_heat, show_tfr, corridor_pts)
    for i in sorted(st.session_state.get("route_overlays", {}).keys()):
        for lyr in st.session_state["route_overlays"][i]:
            layers.append(lyr)
    view_state = pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=8)
    deck = pdk.Deck(map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=view_state, layers=layers,
                    tooltip={"text": "{name} {tfr_type}"})
    st.pydeck_chart(deck, use_container_width=True)

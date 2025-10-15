import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

from app.db import fetch_df

# --- Branding & Layout ---
st.set_page_config(page_title="Aetheris", page_icon="🛡️", layout="wide")
st.title("Aetheris")
st.caption("Go/No-Go decisions from FAA & NOAA data — fast, explainable, map-first.")

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

# Map options
st.markdown("#### Map Options")
mcol1, mcol2, mcol3 = st.columns(3)
radius_km = mcol1.slider("Map radius (km)", 10, 300, 80)
show_heat = mcol2.checkbox("Show Risk Heatmap", value=True)
show_tfr = mcol3.checkbox("Show No-Fly (TFR) Polygons", value=True)

# ---------------- Helpers ----------------
def parse_corridor(s: str) -> List[Tuple[float, float]]:
    pts = []
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
    return km / 111.0

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

# ---------------- Local Models ----------------
def local_decide(lat, lon, start, end, runway):
    df = load_features_points(lat, lon, start.isoformat(), end.isoformat(), radius_km=80)
    if df.empty:
        return {"label": "UNKNOWN", "hourly": [], "ml_score": None}
    mean_risk = df["risk_weight"].mean()
    label = "GREEN" if mean_risk < 0.6 else "AMBER" if mean_risk < 0.9 else "RED"
    return {
        "label": label,
        "ml_score": float(mean_risk),
        "hourly": df[["valid_time", "risk_weight", "wind_kts", "gust_kts", "vis_sm"]].to_dict(orient="records")
    }

def local_route_evaluate(route_points, dep_time, ground_speed_kts, buffer_km, horizon_pad_min):
    segs = []
    total_km = 0.0
    for i in range(len(route_points)-1):
        a = route_points[i]
        b = route_points[i+1]
        dist_km = np.linalg.norm(np.array(a)-np.array(b))*111
        total_km += dist_km
        start_time = dep_time + timedelta(hours=total_km / ground_speed_kts)
        end_time = start_time + timedelta(hours=dist_km / ground_speed_kts)
        df = load_features_points(a[0], a[1], start_time.isoformat(), end_time.isoformat(), radius_km=buffer_km)
        mean_risk = df["risk_weight"].mean() if not df.empty else 0.0
        label = "GREEN" if mean_risk < 0.6 else "AMBER" if mean_risk < 0.9 else "RED"
        segs.append({
            "idx_from": i,
            "idx_to": i+1,
            "a": a,
            "b": b,
            "center": [(a[0]+b[0])/2, (a[1]+b[1])/2],
            "label": label,
            "hazards": ["low vis"] if mean_risk > 0.8 else [],
            "risk": mean_risk,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        })
    red_km = sum(np.linalg.norm(np.array(s["a"])-np.array(s["b"]))*111 for s in segs if s["label"]=="RED")
    amber_km = sum(np.linalg.norm(np.array(s["a"])-np.array(s["b"]))*111 for s in segs if s["label"]=="AMBER")
    green_km = total_km - red_km - amber_km
    return {
        "summary": {"total_km": total_km, "red_km": red_km, "amber_km": amber_km, "green_km": green_km},
        "segments": segs
    }

# ---------------- Decision UI ----------------
left, right = st.columns([1, 1])

with left:
    if st.button("Decide"):
        try:
            data = local_decide(lat, lon, start, end, runway)
            st.subheader(f"Decision: **{data.get('label','?')}**")
            if data.get("ml_score") is not None:
                st.metric(label="ML risk (No-Go prob)", value=f"{data['ml_score']:.2f}")
            if data.get("hourly"):
                st.dataframe(pd.DataFrame(data["hourly"]))
        except Exception as ex:
            st.exception(ex)

with right:
    corridor_pts = parse_corridor(corridor_text)

    # Always include base marker
    base_layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({"latitude": [lat], "longitude": [lon]}),
            get_position='[longitude, latitude]',
            get_radius=150,
            get_fill_color=[0, 200, 80, 255],
        )
    ]

    # Corridor visualization
    if corridor_pts:
        path = [[lon, lat] for lat, lon in corridor_pts]
        base_layers.append(
            pdk.Layer(
                "PathLayer",
                data=pd.DataFrame({"path": [path]}),
                get_path="path",
                get_width=4,
                widthMinPixels=4,
                get_color=[0, 120, 240, 220],
                pickable=True,
            )
        )

    # Optional heatmap layer
    if show_heat:
        fdf = load_features_points(lat, lon, start.isoformat(), end.isoformat(), radius_km)
        if not fdf.empty:
            base_layers.append(
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

    # Add stored overlays (route segments)
    overlays = []
    if "route_overlays" in st.session_state:
        for route_layers in st.session_state["route_overlays"].values():
            overlays.extend(route_layers)

    all_layers = base_layers + overlays
    if not all_layers:
        all_layers = base_layers

    view_state = pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=8)
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=all_layers,
            tooltip={"text": "{label} {tfr_type}"},
        ),
        use_container_width=True,
    )

# ---------------- Routes UI ----------------
st.markdown("### Routes")
st.caption("Enter routes: each line is `lat,lon; lat,lon; ...`")
routes_text = st.text_area("Route options", height=100, value="32.7767,-96.7970; 33.0,-97.0; 33.3,-97.2")

def parse_route(line):
    pts = []
    for seg in line.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            a,b = seg.split(",")
            pts.append([float(a.strip()), float(b.strip())])
        except Exception:
            continue
    return pts

route_lines = [ln.strip() for ln in routes_text.splitlines() if ln.strip()]
ROUTES = [r for r in (parse_route(ln) for ln in route_lines) if len(r) >= 2]

opt_idx = st.selectbox("Choose route", options=list(range(len(ROUTES))) if ROUTES else [0],
                       format_func=lambda i: f"Option {i+1}", disabled=not ROUTES)

dep_offset = st.slider("Departure offset (hours)", -2, 24, 0)
dep_time = start + timedelta(hours=dep_offset)
gs = st.slider("Ground speed (kts)", 40, 180, 80)
buf_km = st.slider("Buffer (km)", 1, 10, 3)
cols = st.columns(2)

# --- Evaluate Selected Route ---
if cols[0].button("Evaluate Selected", disabled=not ROUTES):
    try:
        route = ROUTES[opt_idx]
        out = local_route_evaluate(route, dep_time, gs, buf_km, 20)
        st.subheader(f"Route {opt_idx+1} Summary")
        st.write(out["summary"])
        segs = out["segments"]

        if segs:
            st.dataframe(pd.DataFrame(segs)[["idx_from", "idx_to", "label", "risk", "start_time", "end_time"]])
            path_rows = []
            for s in segs:
                color = {"GREEN": [0, 180, 80, 220],
                         "AMBER": [255, 170, 0, 220],
                         "RED": [220, 40, 40, 240]}[s["label"]]
                path = [[s["a"][1], s["a"][0]], [s["b"][1], s["b"][0]]]
                path_rows.append({
                    "path": path,
                    "color": color,
                    "risk": round(float(s["risk"]), 2),
                    "label": s["label"],
                    "hazards": ", ".join(s["hazards"]) if s["hazards"] else "None",
                    "start_time": s["start_time"][11:16],
                    "end_time": s["end_time"][11:16]
                })

            path_df = pd.DataFrame(path_rows)
            seg_layer = pdk.Layer(
                "PathLayer",
                data=path_df,
                get_path="path",
                get_color="color",
                get_width=6,
                pickable=True,
            )

            tooltip = {
                "html": "<b>{label}</b><br/>Risk: {risk}<br/>Hazards: {hazards}<br/>Time: {start_time}–{end_time}",
                "style": {"backgroundColor": "white", "color": "black"},
            }

            lat_c = np.mean([p[0] for p in route])
            lon_c = np.mean([p[1] for p in route])
            label_layer = pdk.Layer(
                "TextLayer",
                data=pd.DataFrame([{"lon": lon_c, "lat": lat_c, "text": f"Route {opt_idx+1}"}]),
                get_position='[lon, lat]',
                get_text='text',
                get_size=16,
                get_color=[0, 0, 0, 255],
                background=True,
                billboard=True,
            )

            st.session_state["route_overlays"] = {opt_idx: [seg_layer, label_layer]}

            all_layers = [lyr for group in st.session_state["route_overlays"].values() for lyr in group]
            view_state = pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=8)
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=view_state,
                    layers=all_layers,
                    tooltip=tooltip,
                ),
                use_container_width=True,
            )

    except Exception as ex:
        st.error(f"Evaluation failed: {ex}")

# --- Evaluate ALL Routes ---
if cols[1].button("Evaluate All Routes", disabled=not ROUTES):
    try:
        st.session_state["route_overlays"] = {}
        for i, route in enumerate(ROUTES):
            out = local_route_evaluate(route, dep_time, gs, buf_km, 20)
            segs = out["segments"]
            if segs:
                path_rows = []
                for s in segs:
                    color = {"GREEN": [0, 180, 80, 220],
                             "AMBER": [255, 170, 0, 220],
                             "RED": [220, 40, 40, 240]}[s["label"]]
                    path = [[s["a"][1], s["a"][0]], [s["b"][1], s["b"][0]]]
                    path_rows.append({
                        "path": path,
                        "color": color,
                        "risk": round(float(s["risk"]), 2),
                        "label": s["label"],
                        "hazards": ", ".join(s["hazards"]) if s["hazards"] else "None",
                        "start_time": s["start_time"][11:16],
                        "end_time": s["end_time"][11:16]
                    })
                path_df = pd.DataFrame(path_rows)
                seg_layer = pdk.Layer(
                    "PathLayer",
                    data=path_df,
                    get_path="path",
                    get_color="color",
                    get_width=6,
                    pickable=True,
                )
                lat_c = np.mean([p[0] for p in route])
                lon_c = np.mean([p[1] for p in route])
                label_layer = pdk.Layer(
                    "TextLayer",
                    data=pd.DataFrame([{"lon": lon_c, "lat": lat_c, "text": f"Route {i+1}"}]),
                    get_position='[lon, lat]',
                    get_text='text',
                    get_size=16,
                    get_color=[0, 0, 0, 255],
                    background=True,
                    billboard=True,
                )
                st.session_state["route_overlays"][i] = [seg_layer, label_layer]

        all_layers = [lyr for group in st.session_state["route_overlays"].values() for lyr in group]
        view_state = pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=8)
        tooltip = {
            "html": "<b>{label}</b><br/>Risk: {risk}<br/>Hazards: {hazards}<br/>Time: {start_time}–{end_time}",
            "style": {"backgroundColor": "white", "color": "black"},
        }
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=all_layers,
                tooltip=tooltip,
            ),
            use_container_width=True,
        )

        st.success("All routes evaluated and visualized on map.")
    except Exception as ex:
        st.error(f"Evaluation failed: {ex}")

# app/api/route.py
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import math, json
import pandas as pd
from sqlalchemy import text
from shapely.geometry import shape, Point

from app.db import engine

router = APIRouter(prefix="/v1", tags=["route"])

class RouteEvalRequest(BaseModel):
    path: List[List[float]] = Field(..., description="[[lat,lon], ...]")
    departure_time: datetime
    ground_speed_kts: float = 80.0
    buffer_km: float = 3.0
    horizon_pad_min: int = 20  # +/- minutes around sample time when querying features

class SegmentResult(BaseModel):
    idx_from: int
    idx_to: int
    start_time: str
    end_time: str
    center: List[float]       # [lat, lon]
    a: List[float]            # [lat, lon]  <-- NEW
    b: List[float]            # [lat, lon]  <-- NEW
    risk: float
    label: str
    hazards: List[str]

class RouteEvalResponse(BaseModel):
    summary: Dict[str, Any]
    segments: List[SegmentResult]

def _km_to_deg(km: float) -> float:
    return km/111.0

def _haversine_km(a, b):
    # a,b = (lat,lon)
    R=6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def _densify(path, max_step_km=5.0):
    out=[path[0]]
    for i in range(1,len(path)):
        a, b = path[i-1], path[i]
        d = _haversine_km(a,b)
        if d<=max_step_km:
            out.append(b); continue
        n = max(1,int(math.ceil(d/max_step_km)))
        for k in range(1,n+1):
            t = k/n
            lat = a[0] + t*(b[0]-a[0])
            lon = a[1] + t*(b[1]-a[1])
            out.append([lat,lon])
    return out

def _load_active_tfrs(t_iso: str):
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT geometry_geojson, name, tfr_type FROM tfr WHERE eff_start <= :t AND eff_end >= :t"
        ), {"t": t_iso}).fetchall()
    geoms=[]
    for gjson,name,typ in rows:
        try:
            gj = json.loads(gjson)
            # normalize: either Feature or FeatureCollection
            if gj.get("type")=="Feature":
                geoms.append((shape(gj["geometry"]), name, typ))
            elif gj.get("type")=="FeatureCollection":
                for f in gj.get("features", []):
                    geoms.append((shape(f["geometry"]), name, typ))
            else:
                geoms.append((shape(gj), name, typ))
        except Exception:
            continue
    return geoms

def _query_features_timebox(lat: float, lon: float, t: datetime, pad_min: int, radius_km: float) -> pd.DataFrame:
    start = (t - timedelta(minutes=pad_min)).isoformat()
    end   = (t + timedelta(minutes=pad_min)).isoformat()
    with engine.begin() as conn:
        df = pd.read_sql(text(
            "SELECT lat,lon,valid_time,vis_sm,ceiling_ft,wind_kts,gust_kts,xwind_kts,pop_pct,"
            "convective_flag,icing_flag,tfr_active_flag,daylight_flag "
            "FROM features WHERE valid_time BETWEEN :s AND :e"
        ), conn, params={"s": start, "e": end})
    if df.empty: 
        return df
    deg = _km_to_deg(radius_km)
    return df[(df["lat"].astype(float).between(lat-deg,lat+deg)) &
              (df["lon"].astype(float).between(lon-deg,lon+deg))].copy()

def _rule_label(row) -> (int, list):
    reasons=[]
    if pd.notna(row.get("vis_sm")) and float(row["vis_sm"])<3.0: reasons.append("Low visibility")
    if pd.notna(row.get("ceiling_ft")) and float(row["ceiling_ft"])<1000: reasons.append("Low ceiling")
    if pd.notna(row.get("wind_kts")) and float(row["wind_kts"])>25: reasons.append("High wind")
    if pd.notna(row.get("gust_kts")) and float(row["gust_kts"])>35: reasons.append("High gusts")
    if int(row.get("convective_flag") or 0)==1: reasons.append("Convective")
    if int(row.get("icing_flag") or 0)==1: reasons.append("Icing")
    if int(row.get("tfr_active_flag") or 0)==1: reasons.append("TFR active (point)")
    return (1 if reasons else 0, reasons)

@router.post("/route_evaluate", response_model=RouteEvalResponse)
def route_evaluate(req: RouteEvalRequest):
    path = req.path
    if not path or len(path)<2:
        return {"summary":{"error":"need at least 2 points"}, "segments":[]}
    path_d = _densify(path, max_step_km=max(req.buffer_km*1.5, 3.0))
    # time along path
    times=[]
    t0 = req.departure_time if req.departure_time.tzinfo else req.departure_time.replace(tzinfo=timezone.utc)
    total=0.0
    times.append(t0.isoformat())
    for i in range(1,len(path_d)):
        dkm = _haversine_km(path_d[i-1], path_d[i])
        hours = dkm / (req.ground_speed_kts*1.852)  # 1 kt = 1.852 km/h
        total += hours
        times.append((t0 + timedelta(hours=total)).isoformat())

    # Active TFR shapes at each sample time (approx: use end time)
    # (Cheaper: use the mid-route time to fetch once; better: check per segment)
    segs=[]
    red_tot=0.0; amber_tot=0.0; dist_tot=0.0
    for i in range(1,len(path_d)):
        a, b = path_d[i-1], path_d[i]
        ta = datetime.fromisoformat(times[i-1].replace("Z","+00:00"))
        tb = datetime.fromisoformat(times[i].replace("Z","+00:00"))
        midt = ta + (tb-ta)/2
        dkm = _haversine_km(a,b)
        dist_tot += dkm
        # features near midpoint
        fdf = _query_features_timebox((a[0]+b[0])/2, (a[1]+b[1])/2, midt, req.horizon_pad_min, req.buffer_km)
        # aggregate hazards (take worst in window)
        label=0; hazards=[]; risk=0.0
        if not fdf.empty:
            # mark if any row violates
            labs = fdf.apply(_rule_label, axis=1).tolist()
            any1 = any(l for l,_ in labs)
            all_reasons=[]
            for l, rs in labs:
                if l: all_reasons.extend(rs)
            label = 1 if any1 else 0
            hazards = sorted(set(all_reasons))
            # simple risk score
            def rrow(r):
                s=0.0
                if pd.notna(r.get("pop_pct")): s += min(float(r["pop_pct"])/100.0,1.0)*0.2
                if pd.notna(r.get("wind_kts")): s += min(float(r["wind_kts"])/35.0,1.0)*0.3
                if pd.notna(r.get("gust_kts")): s += min(float(r["gust_kts"])/45.0,1.0)*0.2
                if pd.notna(r.get("vis_sm")) and float(r["vis_sm"])<3: s += 0.5
                if pd.notna(r.get("ceiling_ft")) and float(r["ceiling_ft"])<1000: s += 0.5
                if int(r.get("convective_flag") or 0)==1: s+=0.3
                if int(r.get("tfr_active_flag") or 0)==1: s+=0.4
                return s
            risk = float(min(fdf.apply(rrow,axis=1).max(), 2.0))
        # TFR polygon check (spatial)
        tfr_geoms = _load_active_tfrs(midt.isoformat())
        if tfr_geoms:
            ptA, ptB = Point(a[1],a[0]), Point(b[1],b[0])
            # cheap check: midpoint + endpoints
            for geom,name,typ in tfr_geoms:
                if geom.intersects(ptA) or geom.intersects(ptB) or geom.intersects(Point((a[1]+b[1])/2,(a[0]+b[0])/2)):
                    label=1
                    hazards = sorted(set(hazards + [f"TFR: {name}"]))
                    risk = max(risk, 1.0)

        color_label = "RED" if label else ("AMBER" if risk>=0.6 else "GREEN")
        if color_label=="RED": red_tot += dkm
        elif color_label=="AMBER": amber_tot += dkm

        segs.append(SegmentResult(
            idx_from=i-1, idx_to=i,
            start_time=times[i-1], end_time=times[i],
            center=[ (a[0]+b[0])/2, (a[1]+b[1])/2 ],
            a=a,                         # <-- NEW
            b=b,                         # <-- NEW
            risk=risk, label=color_label, hazards=hazards
        ))

    summary = {
        "total_km": dist_tot,
        "red_km": red_tot,
        "amber_km": amber_tot,
        "green_km": max(0.0, dist_tot - red_tot - amber_tot),
    }
    return RouteEvalResponse(summary=summary, segments=segs)

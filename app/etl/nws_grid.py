import json, requests, os
from shapely.geometry import shape, Point
from sqlalchemy import text
from app.db import engine

URL = os.getenv("FAA_TFR_JSON", "https://tfr.faa.gov/tfr2/list.json")  

def fetch_tfr_list():
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    return r.json()  

def upsert_tfrs():
    data = fetch_tfr_list()
    count = 0
    with engine.begin() as conn:
        for t in data.get("tfrs", []):
            gj = t.get("geometry") or t.get("geojson") 
            if not gj: continue
            payload = json.dumps(gj)
            conn.execute(text(
                "INSERT OR REPLACE INTO tfr(tfr_id,name,tfr_type,eff_start,eff_end,geometry_geojson)"
                " VALUES (:id,:name,:typ,:s,:e,:geo)"
            ), {
                "id": t.get("id") or t.get("number"),
                "name": t.get("title") or "TFR",
                "typ":  t.get("type") or "TFR",
                "s":    t.get("start") or t.get("effective"),
                "e":    t.get("end")   or t.get("expires"),
                "geo":  payload
            })
            count += 1
    return count

def point_in_active_tfr(lat, lon, iso_when) -> bool:
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT geometry_geojson, eff_start, eff_end FROM tfr "
            "WHERE eff_start <= :t AND eff_end >= :t"
        ), {"t": iso_when}).fetchall()
    pt = Point(lon, lat)
    for row in rows:
        try:
            geom = shape(json.loads(row[0])["features"][0]["geometry"])
            if geom.contains(pt): return True
        except Exception:
            continue
    return False

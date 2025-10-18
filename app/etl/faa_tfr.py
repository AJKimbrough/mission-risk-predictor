import os, json, requests
from typing import Tuple, List
from sqlalchemy import text
from app.db import engine

#FAA Graphic TFRs JSON 
FAA_TFR_JSON = os.getenv("FAA_TFR_JSON", "https://tfr.faa.gov/tfr2/list.json")

S = requests.Session()
S.headers.update({
    "User-Agent": os.getenv("NWS_USER_AGENT", "SkySafe (email@example.com)"),
    "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8"
})

def fetch_tfr_list() -> dict:
    r = S.get(FAA_TFR_JSON, timeout=30)
    r.raise_for_status()
    ctype = r.headers.get("content-type","").lower()
    if "json" not in ctype:
        preview = r.text[:200].replace("\n"," ")
        raise ValueError(f"Non-JSON TFR list (Content-Type={ctype}). First 200 chars: {preview!r}")
    return r.json()

def _extract_geojson(t: dict) -> dict:
    if "geometry" in t:
        return t["geometry"]
    if "geojson" in t:
        return t["geojson"]
    # Skip link entries 
    return None

def upsert_tfrs() -> int:
    payload = fetch_tfr_list()
    items = payload.get("tfrs") or payload.get("items") or []
    count = 0
    with engine.begin() as conn:
        for t in items:
            gj = _extract_geojson(t)
            if not gj:
                continue
            tfr_id = t.get("id") or t.get("number") or t.get("tfr_number") or t.get("title") or "tfr"
            name = t.get("title") or t.get("name") or "TFR"
            typ  = t.get("type") or "TFR"
            eff_start = t.get("start") or t.get("effective") or t.get("eff_start")
            eff_end   = t.get("end")   or t.get("expires")   or t.get("eff_end")
            conn.execute(text(
                "INSERT OR REPLACE INTO tfr(tfr_id,name,tfr_type,eff_start,eff_end,geometry_geojson)"
                " VALUES (:id,:name,:typ,:s,:e,:geo)"
            ), {
                "id": str(tfr_id),
                "name": name,
                "typ": typ,
                "s": eff_start,
                "e": eff_end,
                "geo": json.dumps(gj)
            })
            count += 1
    return count

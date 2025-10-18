import os, re, requests, pandas as pd
from sqlalchemy import text
from app.db import engine

UA = os.getenv("NWS_USER_AGENT", "SkySafe (you@example.com)")

S = requests.Session()
S.headers.update({
    "User-Agent": UA,
    "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
    "Referer": "https://aviationweather.gov/",
})

#Parse lowest BKN/OVC from METAR (i.e., "BKN020 OVC080")
_CLOUD_RE = re.compile(r"\b(BKN|OVC)(\d{3})\b")

def _parse_ceiling_ft_from_raw(raw: str):
    if not raw:
        return None
    bases = []
    for cov, base3 in _CLOUD_RE.findall(raw.upper()):
        try:
            bases.append(int(base3) * 100)  # hundreds -> feet
        except Exception:
            pass
    return min(bases) if bases else None

def _parse_visib_to_sm(val):
    """AWC JSON 'visib' can be '10+' or a number; normalize to float statute miles."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    
    s = str(val).strip()
    if not s:
        return None
    if s.endswith("+"):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return None

def _awc_metar_json(ids, hours=6):
    ids_str = ",".join(ids)
    url = f"https://aviationweather.gov/api/data/metar?format=json&ids={ids_str}&hours={hours}"
    r = S.get(url, timeout=25)
    r.raise_for_status()
    txt = (r.text or "").strip()
    if not (txt.startswith("{") or txt.startswith("[")):
        return []
    return r.json() or []

def upsert_from_metars(lat, lon, km=50):
    ids_env = os.getenv("SKYSAFE_METAR_IDS")
    if ids_env:
        ids = [s.strip() for s in ids_env.split(",") if s.strip()]
    else:
        ids = ["KDAL","KDFW","KADS","KAFW","KFTW","KGKY","KRBD","KTKI","KDTO","KGPM"]

    rows = []
    data = _awc_metar_json(ids, hours=6)
    for m in data:
        try:
            #lat/lon
            la = float(m.get("lat"))
            lo = float(m.get("lon"))
            #Time
            ts = m.get("reportTime") or m.get("time")  
            #Visibility
            vis_sm = _parse_visib_to_sm(m.get("visib"))
            #Ceiling via METAR decode (lowest BKN/OVC layer)
            ceiling_ft = _parse_ceiling_ft_from_raw(m.get("rawOb"))
            rows.append({"lat": la, "lon": lo, "valid_time": ts, "vis_sm": vis_sm, "ceiling_ft": ceiling_ft})
        except Exception:
            continue

    if not rows:
        return 0

    with engine.begin() as conn:
        for r in rows:
            conn.execute(text(
                "INSERT OR REPLACE INTO features("
                "lat,lon,valid_time,vis_sm,ceiling_ft,wind_kts,xwind_kts,gust_kts,pop_pct,"
                "convective_flag,icing_flag,tfr_active_flag,daylight_flag) "
                "VALUES (:lat,:lon,:valid_time,:vis_sm,:ceiling_ft,NULL,0,NULL,NULL,0,0,0,1)"
            ), r)
    return len(rows)

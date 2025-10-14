import os, requests, pandas as pd
from datetime import timezone
from sqlalchemy import text
from app.db import engine

UA = os.getenv("NWS_USER_AGENT", "SkySafe (email@example.com)")
S = requests.Session(); S.headers.update({"User-Agent": UA, "Accept": "application/ld+json"})

def fetch_grid(lat: float, lon: float) -> pd.DataFrame:
    p = S.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", timeout=20)
    p.raise_for_status()
    grid_url = p.json()["properties"]["forecastGridData"]
    g = S.get(grid_url, timeout=30); g.raise_for_status()
    J = g.json()["properties"]
    # Pull the few series we care about
    def parse_series(key):
        if key not in J: return pd.DataFrame()
        rows = []
        for v in J[key]["values"]:
            rows.append({"valid_time": v["validTime"].split("/")[0], key: v["value"]})
        return pd.DataFrame(rows)
    df = parse_series("windSpeed")
    for k in ["windGust","probabilityOfPrecipitation","skyCover"]:
        df = df.merge(parse_series(k), on="valid_time", how="outer")
    # convert units (m/s or km/h -> kts depends on office; safest: numeric value is in km/h)
    # NWS grid uses "unitCode". When kts not available, roughly convert mph->kts if needed.
    # Keep it simple: treat numbers as mph and convert to kts if unitCode contains "unit:km_h-1" or "unit:si"
    for k in ["windSpeed","windGust"]:
        if k in J and "unitCode" in J[k]:
            u = J[k]["unitCode"]
            if "unit:km_h-1" in u: df[k] = df[k].astype(float) * 0.539957
            elif "unit:si" in u or "unit:m_s-1" in u: df[k] = df[k].astype(float) * 1.94384
    df.rename(columns={"windSpeed":"wind_kts","windGust":"gust_kts","probabilityOfPrecipitation":"pop_pct"}, inplace=True)
    df["lat"], df["lon"] = lat, lon
    return df

def upsert_features_from_grid(lat: float, lon: float):
    df = fetch_grid(lat, lon)
    if df.empty: return 0
    with engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(text(
                "INSERT INTO nws_hourly(lat,lon,valid_time,wind_kts,gust_kts,pop_pct,wx_phrase) "
                "VALUES (:lat,:lon,:valid_time,:wind_kts,:gust_kts,:pop_pct,'Forecast')"
            ), {
                "lat": lat, "lon": lon, "valid_time": r["valid_time"],
                "wind_kts": float(r.get("wind_kts") or 0),
                "gust_kts": float(r.get("gust_kts") or 0),
                "pop_pct": float(r.get("pop_pct") or 0),
            })
            conn.execute(text(
                "INSERT OR REPLACE INTO features(lat,lon,valid_time,vis_sm,ceiling_ft,wind_kts,xwind_kts,gust_kts,pop_pct,"
                "convective_flag,icing_flag,tfr_active_flag,daylight_flag) "
                "VALUES (:lat,:lon,:valid_time,NULL,NULL,:wind_kts,0,:gust_kts,:pop_pct,0,0,0,1)"
            ), {
                "lat": lat, "lon": lon, "valid_time": r["valid_time"],
                "wind_kts": float(r.get("wind_kts") or 0),
                "gust_kts": float(r.get("gust_kts") or 0),
                "pop_pct": float(r.get("pop_pct") or 0),
            })
    return len(df)

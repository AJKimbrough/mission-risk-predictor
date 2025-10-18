from datetime import datetime
import pandas as pd
from typing import Optional
from .db import fetch_df




def load_features_for_window(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    sql = (
        "SELECT * FROM features WHERE lat=:lat AND lon=:lon "
        "AND valid_time BETWEEN :start AND :end ORDER BY valid_time"
    )
    try:
        df = fetch_df(sql, lat=lat, lon=lon, start=start, end=end)
    except Exception:
        df = pd.DataFrame()
    return df
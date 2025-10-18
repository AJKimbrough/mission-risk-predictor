from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any
from .model import load_model
import numpy as np

THRESHOLD = float(os.getenv("SKYSAFE_THRESHOLD", "0.50")) if 'os' in globals() else 0.50

def _rule_flags(row) -> List[str]:
    reasons = []
    if pd.notna(row.get("vis_sm")) and row["vis_sm"] < 3:
        reasons.append("Visibility < 3sm")
    if pd.notna(row.get("ceiling_ft")) and row["ceiling_ft"] < 1000:
        reasons.append("Ceiling < 1000 ft")
    if pd.notna(row.get("wind_kts")) and row["wind_kts"] > 25:
        reasons.append("Wind > 25 kts")
    if pd.notna(row.get("gust_kts")) and row["gust_kts"] > 35:
        reasons.append("Gust > 35 kts")
    if pd.notna(row.get("tfr_active_flag")) and bool(row["tfr_active_flag"]):
        reasons.append("Active TFR")
    return reasons


def _ml_score(feats: pd.DataFrame) -> float:
    try:
        model, feature_cols = load_model()
    except Exception:
        return None
    if feats is None or feats.empty:
        return None

    df = feats.copy()
    t = pd.to_datetime(df["valid_time"], utc=True, errors="coerce")
    df["hour"] = t.dt.hour
    df["month"] = t.dt.month
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    #NA flags, fills
    for col in ["vis_sm","ceiling_ft","wind_kts","gust_kts","xwind_kts","pop_pct"]:
        df[f"{col}_na"] = df[col].isna().astype(int)
    df["vis_sm"] = df["vis_sm"].fillna(10.0).clip(0, 20)
    df["ceiling_ft"] = df["ceiling_ft"].fillna(20000).clip(0, 30000)
    for col in ["wind_kts","gust_kts","xwind_kts","pop_pct"]:
        df[col] = df[col].fillna(0).clip(lower=0)
    for c in ["convective_flag","icing_flag","tfr_active_flag","daylight_flag"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    #Align columns
    X = df.reindex(columns=feature_cols, fill_value=0).to_numpy()
    probs = model.predict_proba(X)[:,1]
    #Aggregate across hours/take max risk within the window
    return float(np.max(probs)) if len(probs) else None


def rule_decide(feats: pd.DataFrame, req) -> Dict[str, Any]:
    hourly = []
    rule_no_go = False
    if feats is not None and not feats.empty:
        for _, r in feats.iterrows():
            rs = _rule_flags(r)
            hourly.append({"time": r.get("valid_time"), "reasons": rs})
        if rs:
            rule_no_go = True
    label = "NO-GO" if rule_no_go else "GO"
    return {
        "label": label,
        "rule_score": 1 if rule_no_go else 0,
        "ml_score": None,
        "reasons": list({s for h in hourly for s in h["reasons"]}),
        "hourly": hourly,
    }
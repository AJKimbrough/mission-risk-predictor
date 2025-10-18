import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import text
from app.db import engine
from joblib import dump

SQL = """
SELECT f.id, f.lat, f.lon, f.valid_time,
       f.vis_sm, f.ceiling_ft, f.wind_kts, f.gust_kts, f.xwind_kts, f.pop_pct,
       f.convective_flag, f.icing_flag, f.tfr_active_flag, f.daylight_flag,
       CASE
           WHEN (vis_sm IS NOT NULL AND vis_sm < 3)
             OR (ceiling_ft IS NOT NULL AND ceiling_ft < 1000)
             OR (wind_kts IS NOT NULL AND wind_kts > 25)
             OR (gust_kts IS NOT NULL AND gust_kts > 35)
             OR (tfr_active_flag = 1)
           THEN 1 ELSE 0
       END AS label
FROM features f
WHERE valid_time IS NOT NULL
ORDER BY valid_time;
"""

with engine.begin() as conn:
    df = pd.read_sql(text(SQL), conn)

if df.empty:
    raise SystemExit("No feature rows found — seed or ingest first.")

#Features: Time
def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    #Normalize times 
    s = df["valid_time"].astype(str).str.strip().str.replace("Z", "+00:00", regex=False)
    try:
        t = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    except Exception:
        t = pd.to_datetime(s, utc=True, errors="coerce")
    df = df.copy()
    df["_ts"] = t
    df = df.dropna(subset=["_ts"])
    if df.empty:
        return df
    #Cyclic features
    df["hour"]  = df["_ts"].dt.hour
    df["month"] = df["_ts"].dt.month
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

df = add_time_feats(df)
if df.empty:
    raise SystemExit("No rows with parseable valid_time. Check your ingestion timestamps.")
df = df.sort_values("_ts").reset_index(drop=True)

#NA handling, caps, NA flags
for col in ["vis_sm","ceiling_ft","wind_kts","gust_kts","xwind_kts","pop_pct"]:
    df[f"{col}_na"] = df[col].isna().astype(int)

df["vis_sm"]     = df["vis_sm"].fillna(10.0).clip(0, 20)
df["ceiling_ft"] = df["ceiling_ft"].fillna(20000).clip(0, 30000)
for col in ["wind_kts","gust_kts","xwind_kts","pop_pct"]:
    df[col] = df[col].fillna(0).clip(lower=0)

for c in ["convective_flag","icing_flag","tfr_active_flag","daylight_flag"]:
    if c in df.columns:
        df[c] = df[c].fillna(0).astype(int)

#Feature Matrix 
REQUESTED_FEATURES = [
    "vis_sm","ceiling_ft","wind_kts","gust_kts","xwind_kts","pop_pct",
    "convective_flag","icing_flag","tfr_active_flag","daylight_flag",
    "vis_sm_na","ceiling_ft_na","wind_kts_na","gust_kts_na","xwind_kts_na","pop_pct_na",
    "hour_sin","hour_cos","month_sin","month_cos",
]
FEATURES = [c for c in REQUESTED_FEATURES if c in df.columns]
if not FEATURES:
    raise SystemExit("No usable feature columns after preprocessing.")

X = df[FEATURES].to_numpy()
y = df["label"].astype(int).to_numpy()

#Time-based split: oldest 80% train newest 20% test
n = len(df)
if n < 50:
    print(f"Warning: small dataset (n={n}); metrics may be noisy.")
split = max(int(n * 0.8), 1)
X_tr, y_tr = X[:split], y[:split]
X_te, y_te = X[split:], y[split:]

#Checks
u_train = np.unique(y_tr)
if len(u_train) < 2:
    raise SystemExit(f"Training needs ≥2 classes; found only {u_train}. Ingest hazard/NO-GO rows and retry.")
if len(y_te):
    u_test = np.unique(y_te)
    if len(u_test) < 2:
        print("Note: test set has a single class; metrics like ROC-AUC will be degenerate.")

#Models
models = {
    "logreg": CalibratedClassifierCV(
        base_estimator=LogisticRegression(max_iter=500, class_weight="balanced"),
        method="sigmoid", cv=2),
    "rf": CalibratedClassifierCV(
        base_estimator=RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1
        ),
        method="sigmoid", cv=2),
}

#Train, evaluate, persist
os.makedirs("models", exist_ok=True)
report = {}
for name, clf in models.items():
    clf.fit(X_tr, y_tr)
    if len(y_te):
        te_prob = clf.predict_proba(X_te)[:, 1]
        report[f"{name}_rocauc"] = float(roc_auc_score(y_te, te_prob)) if len(np.unique(y_te)) > 1 else None
        report[f"{name}_prauc"]  = float(average_precision_score(y_te, te_prob))
        report[f"{name}_brier"]  = float(brier_score_loss(y_te, te_prob))
    dump({"model": clf, "features": FEATURES}, f"models/{name}.joblib")

print("Metrics:", report)

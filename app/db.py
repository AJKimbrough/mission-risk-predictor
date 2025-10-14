import os
import pandas as pd
from sqlalchemy import create_engine, text

# Ensure directory exists for SQLite
os.makedirs("data", exist_ok=True)

DB_URL = os.getenv("DB_URL", "sqlite:///data/gonogo.sqlite")
engine = create_engine(DB_URL, future=True)

def fetch_df(sql: str, **params):
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)

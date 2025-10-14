import os
from sqlalchemy import create_engine, text

# Determine the database URL
DB_URL = os.getenv("DB_URL")

if not DB_URL:
    # Create a guaranteed writable directory for Streamlit Cloud
    DB_DIR = os.path.join(os.getcwd(), "tmp", "data")
    os.makedirs(DB_DIR, exist_ok=True)
    DB_PATH = os.path.join(DB_DIR, "gonogo.sqlite")
    DB_URL = f"sqlite:///{DB_PATH}"

# Create the engine
engine = create_engine(DB_URL, connect_args={"check_same_thread": False}, future=True)


def fetch_df(sql: str, **params):
    import pandas as pd
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)

import os
from sqlalchemy import create_engine, text


DB_URL = os.getenv("DB_URL", "sqlite:///data/gonogo.sqlite")
engine = create_engine(DB_URL, future=True)


def fetch_df(sql: str, **params):
    import pandas as pd
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)
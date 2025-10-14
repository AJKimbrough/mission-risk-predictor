CREATE EXTENSION IF NOT EXISTS postgis;


CREATE TABLE IF NOT EXISTS metar (
    id BIGSERIAL PRIMARY KEY,
    station_id TEXT,
    obs_time TIMESTAMPTZ,
    vis_sm REAL,
    ceiling_ft INT,
    wind_dir INT,
    wind_kts REAL,
    gust_kts REAL,
    temp_c REAL,
    wx_codes TEXT
);


CREATE TABLE IF NOT EXISTS nws_hourly (
    id BIGSERIAL PRIMARY KEY,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    valid_time TIMESTAMPTZ,
    temp_c REAL,
    rh_pct REAL,
    wind_kts REAL,
    gust_kts REAL,
    pop_pct REAL,
    wx_phrase TEXT
);


CREATE TABLE IF NOT EXISTS tfr (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    geometry GEOMETRY,
    eff_start TIMESTAMPTZ,
    eff_end TIMESTAMPTZ,
    tfr_type TEXT
);


CREATE TABLE IF NOT EXISTS features (
    id BIGSERIAL PRIMARY KEY,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    valid_time TIMESTAMPTZ,
    vis_sm REAL,
    ceiling_ft INT,
    wind_kts REAL,
    xwind_kts REAL,
    gust_kts REAL,
    pop_pct REAL,
    convective_flag BOOLEAN,
    icing_flag BOOLEAN,
    tfr_active_flag BOOLEAN,
    daylight_flag BOOLEAN
);


CREATE INDEX IF NOT EXISTS idx_features_time ON features(valid_time);
CREATE INDEX IF NOT EXISTS idx_features_latlon ON features(lat, lon);
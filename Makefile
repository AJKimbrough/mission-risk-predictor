User-Agent: SkySafe (akimbrough14@gmail.com)
Accept: application/ld+json


.PHONY: setup run-api run-ui makedb seed test fmt lint

PY=python
PIP=pip

setup:
	$(PIP) install -r requirements.txt
	mkdir -p data
	$(PY) -c "import os,sqlite3; os.makedirs('data', exist_ok=True); sqlite3.connect('data/gonogo.sqlite').close()"
	$(PY) -c "import sqlite3,sys; sql=open('sql/schema.sqlite.sql').read(); con=sqlite3.connect('data/gonogo.sqlite'); con.executescript(sql); con.close(); print('DB initialized')"

seed:
	$(PY) -m scripts.seed_examples

run-api:
	uvicorn app.api.main:app --reload --port 8000

run-ui:
	streamlit run app/ui/dashboard.py

makedb:
	$(PY) -c "import sqlite3; sql=open('sql/schema.sqlite.sql').read(); con=sqlite3.connect('data/gonogo.sqlite'); con.executescript(sql); con.close(); print('migrated')"

seed:
	$(PY) scripts/seed_examples.py

test:
	$(PY) -m pytest -q

fmt:
	$(PY) -m pip install ruff==0.6.9
	ruff check . --fix


lint:
	ruff check .

train:
	$(PY) -m app.train.train

ingest-nws:
	$(PY) -c "from app.etl.nws_nwp import upsert_features_from_grid as run; print('NWS rows:', run(32.7767,-96.7970))"

ingest-metar:
	$(PY) -c "from app.etl.awc_metar_taf import upsert_from_metars as run; print('METAR rows:', run(32.7767,-96.7970,50))"

ingest-tfr:
	$(PY) -c 'from app.etl.faa_tfr import upsert_tfrs as run; print("TFR rows:", run())'

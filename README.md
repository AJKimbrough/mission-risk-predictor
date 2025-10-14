<<<<<<< HEAD
# mission-risk-predictor
Predictive decision-support tool for aviation Go/No-Go analysis using airspace and weather dat
=======
# Go/No-Go Predictor (MVP Skeleton)


Minimal scaffold for a mission decision tool using weather + airspace signals.


## Quickstart (Local, SQLite)
```bash
python -m venv .venv && source .venv/bin/activate
cp .env.example .env
make setup # installs deps + creates data/gonogo.sqlite + runs schema
make seed # inserts 24h dummy weather/features for Dallas
make run-api # http://localhost:8000/v1/healthz
make run-ui # http://localhost:8501
>>>>>>> 56f87f8 (Initial commit)

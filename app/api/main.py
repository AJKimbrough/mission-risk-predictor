# app/api/main.py
from fastapi import FastAPI
from app.schemas import DecisionRequest, DecisionResponse
from app.rules import rule_decide
from app.features import load_features_for_window

app = FastAPI(title="SkySafe API", version="0.1.0")

@app.get("/v1/healthz")
def healthz():
    return {"ok": True}

@app.post("/v1/decide", response_model=DecisionResponse)
def decide(req: DecisionRequest):
    feats = load_features_for_window(req.lat, req.lon, req.start_time, req.end_time)
    decision = rule_decide(feats, req)
    return decision

try:
    from app.api.route import router as route_router
    app.include_router(route_router)  
except Exception as e:
    print("WARN: failed to include route router:", e)

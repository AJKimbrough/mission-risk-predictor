import pandas as pd
from app.rules import rule_decide
from types import SimpleNamespace




def test_rule_nogo_on_visibility():
    df = pd.DataFrame([
        {"valid_time": "2025-01-01T00:00:00Z", "vis_sm": 2.5, "ceiling_ft": 1500, "wind_kts": 5, "gust_kts": 10}
    ])
    req = SimpleNamespace(lat=0, lon=0, start_time="", end_time="", runway_heading=None)
    out = rule_decide(df, req)
    assert out["label"] == "NO-GO"
    assert any("Visibility" in r for r in out["reasons"])
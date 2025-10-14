from typing import Optional, List
from pydantic import BaseModel, Field

class DecisionRequest(BaseModel):
    lat: float
    lon: float
    start_time: str
    end_time: str
    runway_heading: Optional[int] = Field(default=None)

class HourlyReason(BaseModel):
    time: Optional[str]
    reasons: List[str]

class DecisionResponse(BaseModel):
    label: str
    rule_score: int
    ml_score: Optional[float]
    reasons: List[str]
    hourly: List[HourlyReason]

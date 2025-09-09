# Mission Go/No-Go Predictor — Scenario Library

This document defines baseline scenarios used to validate the predictor.  
Each scenario specifies **inputs, expected conditions, expected outcome, rationale, and acceptance criteria**.  
These scenarios should be logged and tracked as GitHub Issues, but are kept here for reference.

---

## Scenario 1: Foggy visibility under 3 miles

### Inputs
- Latitude: 32.8998
- Longitude: -97.0403
- Time (UTC): 2025-09-10 12:00
- Forecast horizon: 6h

### Expected Conditions
- Visibility: 2.5 mi  
- Ceiling: 1500 ft  
- Wind gusts: 10 kt  
- Precipitation: No  
- TFR Active: No  

### Expected Outcome
- Predicted label: **No-Go**  
- Risk score range: **≥ 80**

### Rationale
Visibility < 3 miles should trigger a hard No-Go regardless of other conditions.

### Acceptance Criteria
- [ ] Scenario runs without error  
- [ ] Signals logged in `queries.csv`  
- [ ] Output = No-Go  

---

## Scenario 2: TFR active with clear weather

### Inputs
- Latitude: 34.2000
- Longitude: -118.3500
- Time (UTC): 2025-09-10 18:00
- Forecast horizon: 12h

### Expected Conditions
- Visibility: 10 mi  
- Ceiling: 5000 ft  
- Wind gusts: 8 kt  
- Precipitation: No  
- TFR Active: Yes  

### Expected Outcome
- Predicted label: **No-Go**  
- Risk score range: **≥ 80**

### Rationale
Even with good weather, an active TFR must trigger No-Go for compliance/safety.

### Acceptance Criteria
- [ ] Scenario runs without error  
- [ ] Signals logged in `queries.csv`  
- [ ] Output = No-Go  

---

## Scenario 3: Clear day with no restrictions

### Inputs
- Latitude: 36.0800
- Longitude: -115.1522
- Time (UTC): 2025-09-11 15:00
- Forecast horizon: 4h

### Expected Conditions
- Visibility: 10 mi  
- Ceiling: 8000 ft  
- Wind gusts: 6 kt  
- Precipitation: No  
- TFR Active: No  

### Expected Outcome
- Predicted label: **Go**  
- Risk score range: **≤ 30**

### Rationale
All conditions nominal → predictor should return Go with low risk.

### Acceptance Criteria
- [ ] Scenario runs without error  
- [ ] Signals logged in `queries.csv`  
- [ ] Output = Go  

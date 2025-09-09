---
name: Mission Scenario
about: Log and validate a mission risk scenario (weather/TFR conditions)
title: "[Scenario] "
labels: scenario
assignees: ""
---

## Scenario Name
_Example: Foggy visibility under 3 miles_

## Inputs
- **Latitude:** 
- **Longitude:** 
- **Time (UTC):** 
- **Forecast horizon (hours ahead):** 

## Expected Conditions
- Visibility: 
- Ceiling: 
- Wind gusts: 
- Precipitation: 
- TFR Active: Yes / No  

## Expected Outcome
- Predicted label: Go / Watch / No-Go  
- Risk score range: (e.g., 70–90)  

## Rationale
_Why should the predictor give that outcome?_  

## Acceptance Criteria
- [ ] Scenario runs in the app without error  
- [ ] Signals are correctly logged in `queries.csv`  
- [ ] Predictor output matches expected outcome  
- [ ] Issue closed after verification  

---

# 📖 Example Baseline Scenarios

### Scenario 1: Foggy visibility under 3 miles
- **Inputs:** Lat 32.8998, Lon -97.0403, +6h  
- **Expected Outcome:** **No-Go**, risk ≥ 80  
- **Rationale:** Visibility < 3 mi is a hard stop.  

### Scenario 2: TFR active with clear weather
- **Inputs:** Lat 34.2000, Lon -118.3500, +12h  
- **Expected Outcome:** **No-Go**, risk ≥ 80  
- **Rationale:** Active TFR overrides good weather.  

### Scenario 3: Clear day with no restrictions
- **Inputs:** Lat 36.0800, Lon -115.1522, +4h  
- **Expected Outcome:** **Go**, risk ≤ 30  
- **Rationale:** All conditions nominal.  

### Scenario 4: Marginal ceiling and gusty winds (Watch case)
- **Inputs:** Lat 39.7392, Lon -104.9903, +8h  
- **Expected Conditions:** Visibility 5 mi, Ceiling 1800 ft, Wind gusts 22 kt, Precipitation No, TFR Active No  
- **Expected Outcome:** **Watch**, risk ≈ 40–60  
- **Rationale:** Conditions are flyable but marginal—low ceiling + gusty winds push the risk score into the Watch band.  

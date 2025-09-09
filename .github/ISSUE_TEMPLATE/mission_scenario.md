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

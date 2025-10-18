# Aetheris – Mission Go/No-Go Risk Intelligence Platform
## Real-Time Aviation Decision Support Using FAA & NOAA Feeds

Aetheris is a geospatial risk-analysis platform that assists pilots, mission planners, and UAV operators with go/no-go decisions using live meteorological and regulatory data from FAA TFR feeds, AWC METAR/TAF, and NOAA forecast models.

## This application provides:
 - Real-time route hazard visualization
 - Automated risk scoring (Green / Amber / Red)
 - Explainable hazard detection (visibility, icing, turbulence, weather, TFR conflicts)
 - Map-first UI powered by Streamlit & PyDeck
 - Data-driven insights for aviation safety & mission-critical operations

## Key Features
 - FAA + NOAA Integrated Data
    - Pulls METAR, TAF, and NextGen Forecast Model data
    - Incorporates FAA Temporary Flight Restrictions (TFRs)
    - Local SQLite caching w/ rolling updates

 - Route Hazard Evaluation
    - Users input custom flight routes
    - System computes time-aware risk per segment
    - Identifies weather hazards, TFR violations, visibility issues, icing risk, etc
    - Provides segment-by-segment breakdown

 - Mission Readiness Decision Engine
    - Computes Go / No-Go recommendation based on weighted risk factors
    - Generates explainable outputs for pilots and mission control

 - Interactive Map UI
    - Streamlit + Pydeck for web visualization
    - Heatmap overlays show environmental risk gradients
    - Routes are color-coded: 
        - Green (Safe)
        - Amber (Caution)
        - Red (No-Go)

## Example Use Cases
Use Case	            Description
Mission Planning	    Compare multiple route options and timing scenarios
UAV Logistics	        Assess low-altitude risk when flying drones beyond visual line-of-sight
Emergency Services	    Determine safest corridor for air ambulance or wildfire surveillance
Pilot Decision-Making	Get data-backed confidence before takeoff

## Tech Stack
Layer	                    Technology
- Frontend	                Streamlit, PyDeck, Mapbox
- Backend (Optional API)	FastAPI (can be local or external)
- Data	                    SQLite/PostgreSQL, pandas, SQLAlchemy
- ML & Risk Engine	        Scikit-learn, joblib models (logistic regression, random forest)
- Geospatial	            Shapely, PyProj, GeoJSON parsing
- Deployment	            Streamlit Cloud, Docker-ready

## Installation & Setup
 - git clone https://github.com/AJKimbrough/mission-risk-predictor.git
 - cd mission-risk-predictor
 - python3 -m venv .venv
 - source .venv/bin/activate
 - pip install -r requirements.txt
 - streamlit run app/ui/dashboard.py

Ensure your database is located at: data/gonogo.sqlite
You can seed sample data using:
python scripts/seed_examples.py

## Go/No-Go Decision Logic
This platform uses a weighted scoring model:
 - Feature	Condition	Risk Weight
 - Visibility	< 3 statute miles	+0.5
 - Ceiling	< 1000 ft AGL	+0.5
 - Wind	> 25 knots	+0.3
 - Gusts	> 35 knots	+0.2
 - Icing Flag	True	+0.3
 - Convective Weather	True	+0.4
 - TFR Polygon Intersect	True	Instant No-Go

## User Interface
 - Inputs
    - Choose departure time & runway heading
    - Enter custom route as "lat,lon; lat,lon; ..."
    - Adjust ground speed and hazard buffer

 - Outputs
    - Risk heatmap
    - Color-coded flight path
    - Hazard tooltip (click any segment)
    - Route summary table

Example Route Input
 - 32.7767,-96.7970; 33.00,-97.10; 33.35,-97.25

## Roadmap - (Upcoming Enhancements)
 - Integration with live FAA API keys
 - ML prediction model (Random Forest & Gradient Boosting)
 - 3D flight corridor visualization
 - Cloud deployment with PostgreSQL + Supabase
 - SMS/Email alert integration for TFR events

## License
MIT License © 2025 — Amber K.

## Author
AJ Kimbrough
Software Engineer | Quantitative Systems | Aviation Intelligence
✈️ Passionate about building real-time safety systems powered by data.

## Live Demo & GitHub
 - Demo: https://aetheris.streamlit.app
 - Repo: https://github.com/AJKimbrough/mission-risk-predictor

Final Note
 - This project showcases full-stack engineering, geospatial analytics, real-time data processing, and applied machine    
   learning in aviation safety. A high-impact system for real world mission-critical operations.
# Antarctic Mission Control — Team Handoff

## 1. What this project is

**Project:** AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System

**Goal:** Build a decision-support system for Antarctic expedition logistics that combines environmental forecasts, iceberg movement prediction, uncertainty, navigation constraints, POLARIS/RIO-based ice risk, time-dependent A* routing, and a lightweight operational dashboard.

### Core product story

The final demo should feel like an operational mission-control workflow:

**Observe → Forecast → Plan → Respond / Replan**

The forecast animation is the product centerpiece. The routing system then uses the forecast at the vessel's actual arrival time. Finally, the user can inject a new iceberg hazard and demonstrate a route replan.

---

## 2. Current project status

### Backend / science / routing

The backend and core routing pipeline are substantially implemented and tested.

The current architecture is:

```text
Environmental data
      ↓
Cleaning + harmonization
      ↓
Common Antarctic grid (EPSG:3976)
      ↓
Sea-ice forecast
      ↓
Iceberg forecast + uncertainty
      ↓
Dynamic environmental / iceberg risk
      ↓
POLARIS / RIO interpretation
      ↓
Time-dependent A* routing
      ↓
Route profiles + ETA / route metrics
      ↓
FastAPI API
      ↓
React + OpenLayers frontend
```

### Sea-ice work completed

- Bremen AMSR2 Antarctic SIC processed on the project grid.
- Full 2025 sequence is available locally, with the unavailable 2025-06-08 date preserved as a real gap rather than fabricated.
- Persistence benchmark implemented and evaluated.
- HGB sea-ice model implemented and evaluated.
- Independent OSI-430-a cross-consistency comparison implemented.
- Model outputs and validation metrics are treated as validation evidence, not as proof of perfect forecast accuracy.

Important benchmark figures already obtained:

- Persistence pooled: 369 days, 366 forecast pairs, MAE 7.83 percentage points, RMSE 14.84, R² 0.480.
- HGB test: MAE 11.073 percentage points, RMSE 17.818, R² 0.5694.
- MIZ errors are materially larger than open-water / easier regimes, which is important for honest uncertainty communication.

### Ocean currents

- Copernicus GLORYS12V1 daily 2025 currents processed to the project CRS/grid.
- Current rasters store east/north components in the project grid basis.
- Current data are used downstream for iceberg movement and navigation costs.

### Wind / ERA5

- CDS/ERA5 monthly 2025 files downloaded and verified.
- Wind matching is implemented with antimeridian handling.
- Wind vectors stay in native east/north form until the physics module combines them with current vectors.

### Iceberg data + prediction

Official BYU/SCP consolidated iceberg data are used.

Implemented:

- Dataset standardization across sensor-specific columns.
- Observed vs interpolated observation quality tracking.
- One-step movement construction without crossing sensor tracks.
- Physics baseline using current + windage.
- ML residual model infrastructure.
- Calibrated uncertainty cone / radius.
- Great-circle distance for final iceberg exposure calculations.

Important scientific result:

- The residual ML model did **not** improve moved-test displacement error in the current configuration, so it is not treated as a claimed production improvement.
- A windage sweep suggested a small validation improvement around 0.002, but the effect was operationally weak and the shipped configuration remains `windage_coefficient = 0.0`.
- Current uncertainty calibration is 24-hour based; 48-hour uncertainty is explicitly extrapolated using a sqrt-time assumption and must be labelled as such.

### Iceberg uncertainty

Current configuration:

- Quantile: q = 0.90.
- 24-hour calibrated radius ≈ 18.10 km.
- 48-hour radius ≈ 25.60 km using explicit extrapolation.
- Test coverage for calibrated 24-hour radii: q50 ≈ 65.8%, q80 ≈ 93.0%, q90 ≈ 97.0%.

The UI must distinguish:

- **Observed** position
- **Predicted +6h**
- **Predicted +12h**
- **Predicted +24h**
- **Predicted +48h**

The animation between forecast horizons is only a visual interpolation. It must not be presented as extra model forecast points.

### Navigation / routing

Implemented and tested:

- Navigation grid and domain handling.
- Explicit land / bathymetry constraints.
- Static 8-connected A*.
- Time-dependent A*.
- Environmental navigation cost.
- POLARIS / RIO integration.
- Iceberg exposure term.
- Three route profiles.

The **time-dependent A*** implementation is frozen and tested against an exhaustive optimum in its bounded temporal discretization fixture.

The important operational concept is:

> A cell's forecast-dependent cost is evaluated using the forecast bucket corresponding to the vessel's predicted arrival time at that cell.

### Hard navigation constraints

Current design uses:

- Land exclusion from BedMachine Antarctica V4.1.
- Bathymetry check using required depth = vessel draft + safety margin.
- Missing / unknown navigation information is **not** treated as permissive navigation space.

The project deliberately does **not** use GEBCO as the safety-navigation bathymetry source. The GEBCO package may exist locally but is excluded from the Git repository.

### POLARIS / RIO

Implemented:

- POLARIS RIV tables 1.3 and 1.4 in configuration files.
- RIO calculation.
- SIGRID-3 to POLARIS interpretation policy.
- Special-consideration handling.
- Separate project-specific risk penalties.

Important wording rule:

**POLARIS is not a Go / No-Go methodology.**

"Special consideration" must not be represented as an automatic hard block merely because the class is special consideration.

Project-specific risk penalties are clearly labelled as project parameters, not POLARIS outputs.

### Iceberg routing risk

The iceberg navigation cost uses a configurable exposure term.

Current shipped weight:

```text
iceberg_exposure_weight = 5.0
```

This value is a project parameter. It is **not** a collision probability, physical law, or POLARIS value.

The current iceberg exposure calculation uses the maximum exposure over active icebergs rather than summing every iceberg's exposure.

### Historical / real-data validation

A real historical route case has been demonstrated using:

- 2025-01-08 mission date.
- PC6 / POLARIS Table 1.3 configuration.
- BedMachine land exclusion.
- Historical chart override where explicitly configured.
- Vessel speed around 5.0 m/s.

A historical USNIC Antarctic chart from 2020-12-24 is used for the ice-regime / POLARIS side of the example. It is **not** used as a future iceberg trajectory predictor.

---

## 3. Frontend status

### Existing stack

- React 18
- TypeScript
- Vite
- OpenLayers
- proj4
- Vitest
- FastAPI backend

### Existing frontend capabilities

The frontend already has:

- Antarctic map using EPSG:3976.
- SIC / land / coverage presentation layers.
- Forecast timeline.
- Predicted iceberg states.
- Vessel playback / transit controls.
- Mission configuration.
- Route planning and route profiles.
- Route explanation panels.
- Simulation control for temporary iceberg injection.
- API integration.
- Frontend tests and production build configuration.

### Current UX problem

The frontend works, but the current experience is still more like a dashboard than a clear operational story.

The biggest remaining UX goal is:

> **Make the forecast animation the main experience instead of showing too many panels at once.**

The user should first understand what is happening to the environment, then watch the forecast evolve, then request a route, then trigger a replan.

Avoid putting every card / panel on screen at the same time.

---

## 4. Required demo flow

A strong demo should follow this sequence.

### Stage 1 — Observe

Show:

- Antarctic operating area.
- Vessel starting position.
- Current sea-ice / navigation context.
- Active iceberg observations.

Keep this visually simple.

### Stage 2 — Forecast

This is the main "wow" moment.

Show the forecast timeline:

```text
Observed → +6h → +12h → +24h → +48h
```

Animate iceberg positions and uncertainty cones.

The viewer should immediately understand:

1. where the iceberg was observed,
2. where the model predicts it will move,
3. that uncertainty grows with forecast horizon.

Do not hide the fact that uncertainty grows.

### Stage 3 — Plan

Once the forecast is understood, open the mission planning interface.

Calculate three route objectives using the same routing engine:

- fastest / travel-time oriented,
- risk-oriented / full configured risk cost,
- shortest-distance.

Do not describe shortest-distance as fuel.

The UI should explain route differences rather than merely display three lines.

### Stage 4 — Respond / Replan

Inject a temporary iceberg directly onto or near the currently selected route.

Then:

```text
Hazard appears → route becomes impacted → planner replans → new route displayed
```

Show why the route changed, including useful evidence such as affected cells and closest approach / time where available.

This should be the climax of the demo.

---

## 5. Highest-priority work remaining

### P0 — Finish the frontend story

Refactor the frontend into a clearer staged experience:

```text
OBSERVE → FORECAST → PLAN → RESPOND
```

The map should remain the hero element.

Reduce simultaneous UI panels and move secondary information into drawers / overlays / contextual panels.

### P0 — Deterministic iceberg injection demo

The current simulation infrastructure already supports temporary in-memory iceberg injection and rerouting.

The next UX feature should make the demo deterministic:

- inject an iceberg at a known point on the currently displayed route,
- force a visible impact,
- recalculate the route,
- animate or highlight the route change.

The demo should not depend on randomly selecting a location that may or may not alter the route.

### P0 — Make forecast animation trustworthy

Do not fake extra scientific forecast data.

Use backend forecast states at the supported horizons and visually interpolate only for smooth animation.

Clearly label the difference between observed and predicted positions.

### P1 — Mission / discharge-site workflow

The product should support the India / Antarctic expedition logistics framing.

The design target is to support multiple possible discharge / logistics destinations rather than hard-coding a single endpoint.

Also consider departure-time comparisons such as:

```text
Depart now
vs
Wait for better forecast conditions
```

The system should be able to represent a situation where no feasible route exists under the configured hard constraints.

### P1 — Explain rejected alternatives

For a route or site that was not selected, expose useful evidence:

- route objective value,
- risk components,
- hard-constraint rejection if applicable,
- closest-approach distance/time for iceberg-related rejection where available,
- relevant forecast horizon.

### P1 — Historical voyage backtest

Build a proper historical backtest that shows whether route decisions made from forecast-era information reproduce sensible historical decisions.

Do not mix future information into historical decision inputs.

### P1 — Forecast-value test

Add an experiment answering:

> Does better forecast information actually change the navigation decision?

Compare route decisions under a weaker benchmark and the more capable forecast / uncertainty pipeline.

### P2 — Vessel / fuel model

Add a clearly documented vessel-performance model if fuel optimization remains part of the final product.

Do not claim that shortest distance equals minimum fuel.

### P2 — Low-bandwidth / offline mode

Add a lightweight operational mode suitable for limited connectivity.

---

## 6. Important scientific / engineering caveats

These must not be hidden in the final demo.

### Iceberg ML residual

The current ML residual model did not improve the moved-test error sufficiently to claim a production improvement.

Therefore:

- keep the implementation and validation evidence,
- do not market it as a proven improvement,
- do not silently use it as if it were superior.

### Windage

Current shipped physics configuration uses:

```text
windage_coefficient = 0.0
```

This was a deliberate choice because the small validation gain from a windage sweep was weak operationally.

### 48-hour uncertainty

The 24-hour uncertainty radius is calibrated from data.

The 48-hour uncertainty radius is extrapolated using a sqrt-time assumption.

The UI / report should say so.

### Historical chart

USNIC/SIGRID-3 historical chart data are used for historical ice-regime / POLARIS interpretation, not as a future iceberg predictor.

### POLARIS wording

Do not convert POLARIS into a binary safety gate unless there is a separate project rule explicitly defined to do that.

### Human-in-the-loop

The final product is decision support.

The ship master / human operator remains the final authority.

---

## 7. Important repository / data setup information

The Git repository intentionally excludes large scientific datasets and local environments.

Excluded examples include:

- `.venv/`
- `data/`
- NetCDF files (`*.nc`)
- large GeoTIFFs (`*.tif`, `*.tiff`)
- GEBCO local folder
- downloaded ZIP archives
- `.cdsapirc`
- `.env*`
- frontend build output and `node_modules`

This means a fresh clone will contain the source code and configurations, but **not all local scientific data required to run every pipeline stage**.

The README and project docs should eventually contain exact data acquisition / preprocessing instructions for a clean teammate setup.

---

## 8. Local setup

### Backend

Project path on the original development machine:

```text
/Users/vihaangala/Desktop/SIH
```

Python environment:

```bash
source .venv/bin/activate
```

Use the environment's Python explicitly:

```bash
python
```

For FastAPI / Uvicorn, use:

```bash
python -m uvicorn src.api.main:app --reload
```

This avoids accidentally invoking a different system Python installation.

### Frontend

From the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

Production build:

```bash
npm run build
```

Tests:

```bash
npm test
```

Use the exact scripts in `frontend/package.json` if they change.

---

## 9. Git team workflow

Recommended branch model:

```text
main
│
├── vihaan
├── teammate-1
├── teammate-2
└── teammate-3
```

`main` should be the stable integration branch.

Each teammate should:

1. create or switch to their branch,
2. make focused changes,
3. commit frequently with clear messages,
4. push their branch,
5. open a pull request into `main`.

Do not work directly on `main` unless the team intentionally changes this policy.

Current original developer branch:

```text
vihaan
```

---

## 10. Key source files

### API

```text
src/api/main.py
src/api/forecast.py
src/api/simulate.py
src/api/world.py
src/api/geo.py
```

### Models

```text
src/models/persistence.py
src/models/train_forecast_model.py
src/models/build_forecast_dataset.py
src/models/iceberg_physics.py
src/models/iceberg_ml_residual.py
src/models/iceberg_uncertainty.py
```

### Routing

```text
src/routing/grid.py
src/routing/constraints.py
src/routing/navigation_domain.py
src/routing/astar.py
src/routing/time_astar.py
src/routing/cost.py
src/routing/navigation_cost.py
src/routing/time_navigation_cost.py
src/routing/end_to_end_route.py
src/routing/route_profiles.py
src/routing/polaris.py
src/routing/polaris_rio.py
src/routing/polaris_risk.py
src/routing/iceberg_risk.py
src/routing/iceberg_navigation_cost.py
```

### Configuration

```text
configs/
```

The JSON configuration files are important. Prefer changing project parameters there rather than hard-coding values into Python or TypeScript.

### Frontend

```text
frontend/src/App.tsx
frontend/src/components/
frontend/src/map/
frontend/src/api/
frontend/src/styles/app.css
```

---

## 11. What not to do

- Do not commit `.venv`, `node_modules`, NetCDFs, raw datasets, secrets, or large generated rasters.
- Do not fabricate missing forecast dates.
- Do not present interpolated iceberg observations as observed data.
- Do not call the ML residual model a demonstrated improvement.
- Do not describe shortest distance as fuel optimum.
- Do not describe project-specific penalty weights as POLARIS itself.
- Do not convert POLARIS special consideration into an automatic hard block.
- Do not use historical future information in a historical backtest.
- Do not make the dashboard look like every metric has equal importance.
- Do not hide forecast uncertainty.

---

# 12. AI CODING ASSISTANT PROMPT

Copy the prompt below into Claude, Cursor, Codex, Gemini, or another coding assistant before asking it to modify this repository.

```text
You are joining an existing SIH 2026 project called “Antarctic Mission Control”.

Your job is to modify the existing repository without breaking the scientific, routing, API, or test contracts that are already implemented.

PROJECT PURPOSE
Build an AI-enabled Antarctic sea-ice, iceberg trajectory, and navigation decision-support system for Antarctic expedition logistics. The product combines:
- environmental observations,
- sea-ice forecasting,
- iceberg trajectory forecasting,
- calibrated uncertainty,
- POLARIS/RIO ice-risk interpretation,
- bathymetry and land constraints,
- time-dependent navigation cost,
- time-dependent A* routing,
- route alternatives,
- explainability,
- and live iceberg-triggered replanning.

PRODUCT STORY
The final user journey should be:

OBSERVE → FORECAST → PLAN → RESPOND / REPLAN

The map is the hero. The forecast animation is the main “wow” moment. Do NOT turn this into a generic dense analytics dashboard.

CURRENT STATE
The backend is substantially complete and tested. The frontend is functional but needs UX refinement.

The existing stack is:
- Python
- FastAPI
- React 18
- TypeScript
- Vite
- OpenLayers
- proj4
- Vitest

KEY BACKEND MODULES
- src/api/main.py
- src/api/forecast.py
- src/api/simulate.py
- src/models/iceberg_physics.py
- src/models/iceberg_ml_residual.py
- src/models/iceberg_uncertainty.py
- src/models/persistence.py
- src/routing/time_astar.py
- src/routing/time_navigation_cost.py
- src/routing/end_to_end_route.py
- src/routing/route_profiles.py
- src/routing/polaris.py
- src/routing/polaris_rio.py
- src/routing/polaris_risk.py
- src/routing/iceberg_risk.py
- src/routing/iceberg_navigation_cost.py

KEY FRONTEND MODULES
- frontend/src/App.tsx
- frontend/src/components/ForecastPanel.tsx
- frontend/src/components/ForecastTimeline.tsx
- frontend/src/components/PlanDock.tsx
- frontend/src/components/SimulationControl.tsx
- frontend/src/components/WhyRouteChanged.tsx
- frontend/src/map/MissionMap.tsx
- frontend/src/map/ForecastLayers.ts
- frontend/src/map/forecast.ts
- frontend/src/api/client.ts
- frontend/src/api/types.ts
- frontend/src/styles/app.css

SCIENTIFIC / ENGINEERING FACTS YOU MUST RESPECT
1. The project uses a common Antarctic grid in EPSG:3976.
2. Sea-ice data include real calendar gaps. Do not fabricate unavailable dates.
3. Current vectors were preprocessed into the project coordinate basis.
4. ERA5 winds are handled in native east/north form and combined correctly in the physics module.
5. Iceberg physics currently ships with windage_coefficient = 0.0.
6. The ML residual model exists but did NOT demonstrate a moved-test improvement in the current experiment. Do not market it as proven superior.
7. Iceberg uncertainty is calibrated for 24h; 48h is an explicit sqrt-time extrapolation and must be labelled as such.
8. Iceberg final exposure uses great-circle distance, not raw projected Euclidean distance.
9. POLARIS is not a Go/No-Go methodology.
10. POLARIS “special consideration” must not automatically become a hard safety block.
11. Project-specific POLARIS risk penalties are project parameters, not POLARIS values.
12. Shortest-distance routing is not equivalent to minimum fuel.
13. Time-dependent A* evaluates the cost of a grid cell using the forecast corresponding to vessel arrival time at that cell.
14. Missing navigation information must not silently become permissive space.
15. BedMachine is used for the navigation mask / bathymetry workflow in the existing project.
16. USNIC/SIGRID-3 historical chart data are used for historical ice-regime / POLARIS interpretation, not future iceberg prediction.
17. The ship master / human operator remains the final authority.

CURRENT FRONTEND OBJECTIVE
Refactor the user experience so that the main screen communicates one story at a time:

1. OBSERVE
   - clean map
   - vessel
   - observed icebergs
   - basic ice/navigation context

2. FORECAST
   - clear timeline: Observed → +6h → +12h → +24h → +48h
   - animate predicted iceberg positions
   - show uncertainty cones/radii growing with time
   - visually distinguish observed from predicted
   - interpolate only for visual smoothness between backend forecast horizons
   - never imply the interpolation is additional model data

3. PLAN
   - reveal route planning after the forecast is understood
   - calculate three route objectives using the existing routing engine
   - fastest/time-oriented
   - risk-oriented/full configured risk cost
   - shortest-distance
   - never call shortest-distance “fuel optimal”

4. RESPOND / REPLAN
   - deterministic iceberg injection on/near the currently selected route
   - visible hazard impact
   - route recalculation
   - clear explanation of why the route changed

IMPORTANT DEVELOPMENT RULES
- Read the existing code before modifying it.
- Preserve existing tests unless the behavior is intentionally changed and the test is updated accordingly.
- Do not replace real backend calculations with frontend mock values.
- Do not hard-code scientific metrics into the UI.
- Keep configuration values in configs/ where appropriate.
- Keep API serialization in the API/serialization modules.
- Do not introduce fake scientific claims for visual effect.
- Prefer small, reviewable changes.
- After changes, run relevant tests and the frontend production build.
- Report exactly what you changed and what remains uncertain.

WHEN WORKING ON THE FRONTEND
Prioritize hierarchy, clarity, and animation over adding more cards.
The Antarctic map should occupy most of the viewport.
Use overlays, drawers, and contextual controls instead of displaying every section simultaneously.
The UI should feel like operational mission control rather than an admin panel.

WHEN WORKING ON REPLANNING
The existing backend already supports temporary in-memory iceberg simulation.
Do not invent a second routing engine.
Reuse the existing iceberg risk + A* routing path.
The new frontend behavior should call the existing API / simulation capability and visualize the route change.

WHEN WORKING ON ROUTE EXPLANATION
Explain route differences with actual backend metrics.
Where available, show:
- route distance,
- ETA / travel time,
- environmental cost,
- POLARIS-related cost or classification,
- iceberg exposure,
- rejected hard constraints,
- closest approach distance/time for iceberg-related route rejection.

WHEN WORKING ON DATA
The Git repository intentionally excludes large raw/generated scientific datasets.
Do not commit local datasets, NetCDFs, large rasters, secrets, virtual environments, node_modules, or build output.

FINAL SUCCESS CRITERIA
A new viewer should understand the product within seconds:

1. “This is the current Antarctic situation.”
2. “Here is what the system predicts will happen to the icebergs.”
3. “The uncertainty grows with forecast horizon.”
4. “The vessel can choose among route objectives.”
5. “A new iceberg hazard changes the route and the system replans.”
6. “The system explains why the route changed.”
7. “The system is decision support, not an autonomous replacement for the ship master.”

Before making architectural changes, identify which existing module already owns the required behavior and reuse it where possible.
```

---

## 13. Suggested first tasks for teammates

### Frontend teammate

Own:

- Observe → Forecast → Plan → Respond flow.
- Forecast animation.
- Visual distinction between observed / predicted / uncertainty.
- Deterministic iceberg injection UX.
- Route-change animation and explanation.

### Backend / routing teammate

Own:

- Deterministic route-impact simulation if API changes are needed.
- Rejected-route explanation data.
- Closest-approach evidence.
- Multiple discharge-site support.
- Departure-time comparisons.

### Data / validation teammate

Own:

- Historical voyage backtest.
- Forecast-value experiment.
- Regime-specific forecast error analysis.
- Reproducibility documentation for external datasets.

### Product / demo teammate

Own:

- India / Bharati / Maitri expedition-logistics framing.
- Demo script.
- Low-bandwidth presentation mode.
- Final explanation language and limitations.

---

## 14. Final team principle

Do not make the project look more scientifically certain than it is.

The strongest version of the demo is one that clearly shows:

**real data → forecast → uncertainty → decision → explanation → human-controlled replan**

rather than simply showing an attractive map.

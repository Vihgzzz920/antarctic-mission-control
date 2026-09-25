# SIH26059's field claims far more than it builds

**Research date: 20 September 2026. All counts, deployments and repository states are point-in-time observations on that date.**

---

## 1. Executive summary

The competitive field for **SIH26059 — "AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System" (Ministry of Earth Sciences)** is far larger and far more publicly visible than an SIH problem statement usually is, and it is almost uniformly unvalidated. We identified **roughly 20 public GitHub repositories, 6 YouTube demo videos and at least 2 deployed web apps carrying an explicit SIH26059 marker**, against a live third-party mirror showing only **21 submitted ideas of a 500 cap as of 19 September 2026** ([zaidsayyed.in/SIH26059](https://zaidsayyed.in/tools/sih-problem-statements/sih26059)). If that count is close to right, an unusually large share of the field is publicly observable — though the sets overlap in ways we could not resolve.

The decisive finding is not about scale but about substance. **Deep learning is nearly absent from this field in practice.** Across every dependency manifest we read, **exactly one repository declares `torch`** ([125112056-art/antarctic-nav-system](https://raw.githubusercontent.com/125112056-art/antarctic-nav-system/master/requirements.txt)), while ConvLSTM, U-Net, LSTM and YOLO appear constantly in READMEs and video narration. The only committed trained artifacts we found anywhere are `RandomForestRegressor` `.joblib` files. **Not one of the six video teams reports a forecast-skill number against any baseline.** Only two projects publish an evaluation capable of embarrassing them: `nawddeep/SIH26059`, whose committed `baseline_comparison.json` shows its own model **losing to persistence at +1 day (model MAE 0.0466 vs persistence 0.0205, n=300)**, and Team BYTE DRAGON's IAVNS, whose `EVALUATION.md` states a **ConvLSTM was never trained** — directly contradicting its own YouTube description selling one.

At the other extreme, the most polished pitch in the field is hollow: **PolarPath AI (Team AXIOM)** declares a model named `"ConvLSTM-SAR-Attention-v2.4"` inside a backend whose entire `requirements.txt` is **fastapi, uvicorn, pydantic, python-dotenv** — no numpy, no ML library of any kind — and whose "route optimizer" contains **no A\*, no Dijkstra and no priority queue**, only five hardcoded route builders selected by an `if/elif` ladder keyed on a literal iceberg ID.

What matters most competitively sits outside SIH entirely. Every component the field treats as its innovation already exists, free: **OpenDrift's `openberg` module** ships iceberg drift physics *with melting and rollover*; **BAS PolarRoute** (MIT) does mesh-based ice-aware fuel-optimal routing and ships a Dockerised REST server plus pre-built Antarctic meshes; **IceNet** (MIT, Nature Communications 2021) publishes downloadable U-Net ensemble sea-ice forecasting weights; **IMO POLARIS** is a published circular containing two lookup tables; and — most dangerously for an Indian team — **Mishra et al., *Polar Science* 30 (2021) 100696** already published Dijkstra-based optimum ship routing **between Bharati and Maitri**, authored from PDEU and ISRO's Space Applications Centre, and cited by BAS's own PolarRoute paper.

Structurally, SIH26059 sits at roughly the **50th percentile of the 240-statement board**, is the **most contested statement inside its own MoES polar block (SIH26059–26065)**, and will be screened down to **4–5 finalist teams** by a sponsor that the official guidelines explicitly say is **"not obligated to declare a winner"**. MoES/NCPOR has shown **zero public engagement** with the teams tagging it, and has run no mentoring session — unlike INCOIS in SIH 2025.

**Method and reliability, stated up front.** `github.com` HTML and `api.github.com` were blocked at the session proxy, so findings rest on Firecrawl scrapes and direct `raw.githubusercontent.com` fetches — meaning **no repository creation dates, no commit histories and no star counts for most repos**, and no repo-wide code search (every "absent" is "absent from the directories we enumerated"). **`sih.gov.in` returned `403 Microsoft-Azure-Application-Gateway/v2` on every attempt** across five distinct access methods, so all per-PS counts come from third-party mirrors. **LinkedIn is refused at tool level** by Firecrawl and is a closed channel, not merely an unsearched one. Overlap between the ~20 repos, the 6 videos and the deployed apps is largely unresolved.

---

## 2. Verified PS59 competitors

A project appears here only where a source **explicitly names SIH26059 / PS-26059 / PS59** in connection with an Antarctic navigation build. Evidence tags: **VERIFIED FACT** = directly observed in a primary source; **COMPETITOR CLAIM** = a team asserts it, implementation not independently confirmed; **OUR ANALYSIS** = interpretation.

### 2.1 Roster of verified competitors

| # | Project / team | Primary source | SIH26059 evidence (verbatim) | Code verified? |
|---|---|---|---|---|
| V1 | `nawddeep/SIH26059` (no team name) | [repo](https://github.com/nawddeep/SIH26059) | *"Smart India Hackathon 2026 — Problem Statement 26059 / Ministry of Earth Sciences (MoES) · National Centre for Polar and Ocean Research (NCPOR)"* | **Yes** — A\* engine + two committed metrics JSONs |
| V2 | **PolarNav** — `kanishka11082007-sys/SIH26059` (+ byte-identical mirror [`ghildiyalnitin067-a11y/SIH-2026`](https://raw.githubusercontent.com/ghildiyalnitin067-a11y/SIH-2026/main/README.md)) | [repo](https://github.com/kanishka11082007-sys/SIH26059) | *"# POLARNAV (SIH26059)"* + badge "Smart India Hackathon – PS SIH26059" | **Yes** — 93,518-byte A\*, committed `.joblib` models |
| V3 | **POLARPATH AI** — Team **AXIOM** | [repo](https://github.com/afrazhussaina5/polarpath-ai) · [video](https://www.youtube.com/watch?v=4W7-kpP6dHQ) | *"Team Name: AXIOM … SIH Problem Statement: 26059 (Smart India Hackathon 2026)"* | **Yes — and it refutes the README** |
| V4 | **IAVNS** — Team **BYTE DRAGON** | [video](https://www.youtube.com/watch?v=E9UGUEaXoUk) · [repo](https://github.com/Manthanvinzuda007/Team-SIH-) | *"SIH 2026 \| Problem Statement PS-26059 / Team: BYTE DRAGON"*; transcript: *"Our problem statement ID is 26059."* | **Yes** — on the `prod` branch |
| V5 | **SagarDrishti** (repo) — `APC2005-dev` | [repo](https://github.com/APC2005-dev/SagarDrishti) | GitHub repo description: *"SIH26059: AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System"* | **Yes** — `TimeAStar` class |
| V6 | **SagarDrishti** (video) — Team **Cinephiles**, Army Institute of Technology, Pune | [video](https://www.youtube.com/watch?v=IiyMPd4pjUY) | *"built for Smart India Hackathon 2026, Problem Statement 26059 … Team [Cinephiles] \| Army Institute of Technology, Pune"* | **No — self-declared wireframe** |
| V7 | **MitoPolarNet** — Team **MitoNaut**, Team ID **125159** | [video](https://www.youtube.com/watch?v=DdOu6ajgwco) | *"Team Name: MitoNaut / Team ID: 125159 / Problem Statement ID: SIH26059"* | **No repo found** |
| V8 | **POLAR-AI** — `2911vedant/polar-ai` | [repo](https://github.com/2911vedant/polar-ai) | *"Problem: SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory & Navigation Decision Support System"* | **Yes** — A\*, risk weights, synthetic generator |
| V9 | **POLAR-NAV-X** — `Rugved-dev18` (+ mirror [`ASIM-05`](https://github.com/ASIM-05/POLAR-NAV-X)) | [repo](https://github.com/Rugved-dev18/POLAR-NAV-X) | *"SIH Problem Statement ID: SIH26059"* | **Yes — routing directory is empty** |
| V10 | `SadhikaRana/SIH26059-ML-Trajectory` | [repo](https://github.com/SadhikaRana/SIH26059-ML-Trajectory) | Repo name encodes the PS code; README is 24 bytes | **Yes** — 15-module `src/`, no weights |
| V11 | `tanishqbafna695/sih26059-antarctic-navigation-dss` | [repo](https://github.com/tanishqbafna695/sih26059-antarctic-navigation-dss) | Repo name encodes the PS code; README positions it as PS-26059 DSS | Partial — manifest only |
| V12 | `125112056-art/antarctic-nav-system` | [repo](https://github.com/125112056-art/antarctic-nav-system) | Surfaced under GitHub search for `SIH26059`; README describes the PS system | Partial — manifest + README |
| V13 | **LIKITH Gv** — "SIH-2026 problem statement-59" | [video](https://www.youtube.com/watch?v=UpsdegTxY-c) | Description is a **verbatim copy of the official SIH26059 brief**; title names PS 59 | No content extractable |
| V14 | **POLARNAV 26059** (deployed app, unattributed) | [sih-26059-ai-enabled-antarctic-sea-pi.vercel.app](https://sih-26059-ai-enabled-antarctic-sea-pi.vercel.app/) | URL slug embeds the PS ID; page title *"POLARNAV 26059 \| Antarctic Navigation Decision Support System"* | Not fetched |
| V15 | Thin tail (verified PS marker, no substance) | [`Aviral-P/SIH26059`](https://github.com/Aviral-P/SIH26059), [`ku-prince/SIH26059`](https://github.com/ku-prince/SIH26059), [`sai16bit/SIH26059-Adaptive-Mission-Aware-Navigation`](https://github.com/sai16bit/SIH26059-Adaptive-Mission-Aware-Navigation), [`jeevanasreechinta/polar-safe`](https://github.com/jeevanasreechinta/polar-safe) | Repo name/description carries SIH26059 | **Yes — README 0 bytes, 10 bytes, 186 bytes; polar-safe 404s** |

**⚠️ Two verified-in-pass-one projects could not be re-located in pass two, and we report the conflict rather than resolve it.** The first evidence pass recorded **`Astralis-SIH26059/Sea-Ice_concentration` ("Polaris")** — verified on the basis that *the GitHub organisation is literally named after the problem statement* — and **`Pawan-19012006/BOREAS`** (originally "SETU"), quoting *"built as a response to MoES Sea-ice & Expedition problem statement (SIH26059-GREEN)"*. The second, code-level pass **could not find either repository** in GitHub repository search, and a grep of all 22 downloaded READMEs for `boreas|astralis|setu` returned zero matches. Two unconfirmed candidates exist under the name POLARIS ([`patilshrutika1803/POLARIS`](https://github.com/patilshrutika1803/POLARIS), [`CaptainHarryJones/POLARIS.ai`](https://github.com/CaptainHarryJones/POLARIS.ai)), neither of which mentions "Astralis". **OUR ANALYSIS:** either these were renamed or deleted between passes, or logged-out GitHub search (which matches only name/description/topic, capped at 10 results) missed them. Treat both as **verified-then-unverifiable**; their capability claims below are recorded as first-pass README readings only.

### 2.2 Per-competitor detail

#### V1 — `nawddeep/SIH26059` — the only team that published a result embarrassing to itself

- **Repo state (VERIFIED):** 31 commits, 1 branch, 0 stars/forks, last commit *"Add the bridge console, served by the real backend rather than its si…"* 18 Sep 2026. **No licence** (`LICENSE` 404). Top-level tree: `docs/ iceberg-drift/ integration/ local-dashboard/ model_exports/ scripts/ seaice_forecast/ shipNavigation/ telemetry-gateway/` plus `RUN_NOW.sh` and `verify.sh` ([repo](https://github.com/nawddeep/SIH26059)).
- **Objective (COMPETITOR CLAIM):** sea-ice forecasting + iceberg trajectory + fuel-efficient routing, **plus a shipboard data-acquisition pipeline** "that gets vessel telemetry ashore over a satellite link that barely exists below 70°S."
- **Sea-ice method (COMPETITOR CLAIM):** U-Net + ConvLSTM (PyTorch), next-day concentration on a 332×316 EPSG:3412 grid. **Claimed, but implementation not verified** — no `torch` in any manifest we read, and `model_exports/` could not be enumerated.
- **Sea-ice result (VERIFIED IN CODE — the single most decision-relevant number in the field):** the committed file `seaice_forecast/output/evaluation/baseline_comparison.json` gives, at **`+1d`, n_samples = 300**, ice_zone_cells 34,557, ocean_cells 83,019:

  | Predictor | MAE (ice zone) | MAE (full domain) |
  |---|---|---|
  | **Persistence** | **0.02051** | 0.00854 |
  | **Their model** | **0.04662** | 0.02059 |
  | Climatology | **0.11246** | — |

  The model is **~2.3× worse than persistence** while being ~2.4× better than climatology. At `+3d` the pattern holds (persistence 0.03641 vs model 0.05517) ([raw JSON](https://raw.githubusercontent.com/nawddeep/SIH26059/main/seaice_forecast/output/evaluation/baseline_comparison.json)).
- **The README states the loss in its own headline (VERIFIED):** *"At +1d beats climatology 2.4× (0.047 vs 0.113) but is **beaten by persistence** (0.021), n=300. Next-day only"*, and in prose *"The headline is that the forecaster loses to persistence at +1d."* It further disaggregates: **MAE 0.003 over open water vs 0.097 in the marginal ice zone**, noting *"A single overall MAE of 0.019 hides that entirely, because 97% of the domain is open water"* ([README](https://raw.githubusercontent.com/nawddeep/SIH26059/main/README.md)).
- **It also records a retraction (VERIFIED):** *"an earlier `OPEN_PROBLEM.md` claimed the model beat persistence at +5d and +7d"* — and that file now returns 404 on raw.
- **Iceberg trajectory (VERIFIED IN CODE):** `iceberg-drift/output/drift_twostage_metrics.json` gives `two_stage.pos_err_24h_km_rms = 3.9134`, `within_10km_pct = 96.716`, `skill_vs_constant = 0.09503`, against `constant.pos_err_24h_km_rms = 4.3244` and `single_stage_gbm = 4.0039`. Method claimed: two-stage HistGradientBoosting (moving/stationary gate, then u/v regressors).
- **Routing (VERIFIED IN CODE):** `shipNavigation/backend/app/route_engine.py` is **43,919 bytes**, contains `import heapq`, `A_STAR = True`, docstring *"A\* shortest path over the water cells of the LandMask grid"*, a `heapq.heappop`/`heappush` main loop, and an explicit *"Add sea ice resistance penalty so A\* seeks open water / light ice channels"*.
- **Risk modelling (COMPETITOR CLAIM):** IMO **POLARIS** Risk Index Outcome per MSC.1/Circ.1519 — deterministic, no learned parameters.
- **Fuel/ETA (COMPETITOR CLAIM):** ice-aware "speed collapse × power ramp" model, per Polar Class.
- **Uncertainty:** none found.
- **Claimed innovation / notable (VERIFIED quotes):** *"Two trained models and two published standards… Calling all four 'AI models' would be generous, so this document does not."* and *"Replacing a regulatory standard with a neural network would make the system worse, not more advanced."* A commit message records that the `local-dashboard` backend was dropped because `simulation.py` *"generated vessel speed, wind and contact bearings from `random.random()`"*, and that `/api/contacts/radar` deliberately returns an empty list rather than invented contacts.
- **Anything generic:** FastAPI + React stack, A\* routing, POLARIS lookup.
- **OUR ANALYSIS:** the most intellectually honest artifact in the set and the hardest to out-argue on rigour, with `verify.sh` wiring both metrics files to a runnable check. Its weakness is coverage: **next-day-only** sea-ice horizon, and a headline result that is a loss.

#### V2 — PolarNav (`kanishka11082007-sys/SIH26059`) — strongest build, dead backend

- **Repo state (VERIFIED):** **61 commits** (the highest in the field), 1 branch, 0 stars/forks, **MIT licence**, 2 contributors (`kanishka11082007-sys`, `ghildiyalnitin067-a11y`), last commit 18 Sep 2026 ([repo](https://github.com/kanishka11082007-sys/SIH26059)).
- **Trained artifacts ARE committed (VERIFIED IN CODE):** `backend/models/` holds `iceberg_trajectory_model.joblib`, `sea_ice_model.joblib`, `sentinel_sar_detector.joblib` plus feature configs and metrics ([models tree](https://github.com/kanishka11082007-sys/SIH26059/tree/main/backend/models)).
- **Committed metrics (VERIFIED IN CODE, verbatim):** `sea_ice_metrics.json` = `{"dataset": "NOAA/NSIDC CDR V4 (G02202)", "model_type": "RandomForestRegressor", "samples_total": 28280, "test_mae": 0.0401, "test_rmse": 0.1218, "test_r2": 0.8861, "baseline_mae": 0.0575, "training_time_seconds": 0.25}` ([raw](https://raw.githubusercontent.com/kanishka11082007-sys/SIH26059/main/backend/models/sea_ice_metrics.json)). `iceberg_metrics.json` = `{"dataset": "BYU/NIC Antarctic Iceberg Database (180+ Tracked Targets)", "model_type": "RandomForestRegressor", "total_trajectory_steps": 95696, "mean_position_error_km": 1.7, "median_position_error_km": 0.12, "baseline_mae_avg": 0.0659}` ([raw](https://raw.githubusercontent.com/kanishka11082007-sys/SIH26059/main/backend/models/iceberg_metrics.json)). **The baseline is not named or defined in either file.**
- **Routing (VERIFIED IN CODE):** `backend/src/optimization/polar_routing_engine.py` is **93,518 bytes** — the largest single routing implementation in the field — headed *"Circumpolar-Aware Geodesic A\* Pathfinding (State: node, arrival_time)"*, with `import heapq`, `def _find_polar_astar_path(` whose docstring reads *"Compute genuine discrete 2D A\* path in EPSG:3031 with polar-aware heuristics"*, and instrumentation emitting `nodes_expanded` / `astar_ms`. A companion `cost_function.py` (7,267 B) exists. The README claim *"Dynamic Conformal Polar Routing: Time-dependent A\* pathfinding on South Polar Stereographic projection (EPSG:3031)"* is **corroborated by code**.
- **Dependencies (VERIFIED):** fastapi, uvicorn, pydantic, sqlalchemy, psycopg, numpy, pandas, scipy, **xarray, netCDF4, scikit-learn, joblib, shapely, pyproj**. **No torch, no tensorflow, no xgboost** ([requirements.txt](https://raw.githubusercontent.com/kanishka11082007-sys/SIH26059/main/requirements.txt)). **Any README claim of "physics-informed machine learning", ConvLSTM or U-Net is therefore claimed, but implementation not verified** — both committed models are `RandomForestRegressor`.
- **Data sources (COMPETITOR CLAIM):** Sentinel-1 SAR, AMSR2, Copernicus GLO12, ERA5. Committed metrics name NOAA/NSIDC CDR V4 (G02202) and BYU/NIC.
- **⚠️ Deployment (VERIFIED — LIVE TEST, 20 Sep 2026):** the backend behind the README's "Live Demo" and "Swagger Docs" badges, `https://sih-2026-059.onrender.com/openapi.json`, returns **HTTP 503, page title "Service Suspended", body "This service has been suspended."** The frontend `https://frontend-pearl-nine-74.vercel.app/` returns HTTP 200 and renders as a **marketing landing page** — hero, "DATA STREAM ACTIVE•68°18'S 12°28'E", a static "ICEBERG A-17 … 24H Forecast +18.4 km … Confidence 87%" card, and three static route cards ("ROUTE A — Shortest 842 km HIGH RISK", "ROUTE B RECOMMENDED 879 km OPTIMAL RISK", "ROUTE C — Conservative 925 km VERY LOW RISK", "+4.4% distance → 52% lower ice-related risk"). Those figures render identically with the backend suspended, i.e. they are **hardcoded copy, not API output**.
- **OUR ANALYSIS:** on committed code this is the strongest build in the field — MIT-licensed, real artifacts, the largest verified routing engine, a README that itself separates "LIVE DATA" from "SIMULATION & BENCHMARK DATA". Its README is nonetheless heavier on adjectives ("production-grade", "unforgiving") than measured numbers, and it is the one project whose advertised live demo is currently a 503.

#### V3 — POLARPATH AI / Team AXIOM — the widest README-to-code gap in the field

- **Repo state (VERIFIED):** 18 commits, 2 branches, 0 stars/forks, **no licence**, last commit *"Update import path for IcebergDetection model"* 15 Sep 2026. Deployment configs present (`Dockerfile`, `railway.json`, `vercel.json`); the `backend/` tree includes a committed `__pycache__/` and a file named `test.txt` ([repo](https://github.com/afrazhussaina5/polarpath-ai)).
- **⚠️ Dependencies (VERIFIED IN CODE — the finding that reframes everything else):** `requirements.txt` **in full** is four lines — `fastapi==0.115.6`, `uvicorn[standard]==0.34.0`, `pydantic==2.10.4`, `python-dotenv==1.0.1`. **No numpy. No scikit-learn. No torch. No tensorflow. No xarray. No geospatial library of any kind** ([raw](https://raw.githubusercontent.com/afrazhussaina5/polarpath-ai/main/requirements.txt)).
- **Sea-ice "model" (VERIFIED IN CODE):** `backend/services/sea_ice_service.py` (4,494 B) line 8 reads verbatim `self.model_name = "ConvLSTM-SAR-Attention-v2.4 (Synthetic Pipeline)"`. Its only non-stdlib import is `math`. The "forecast" is a closed-form analytic field on a hardcoded 14×12 lat/lon grid (−64.2 to −70.2 / 72.0 to 80.5) with a Prydz-Bay term `bay_factor = 1.0 - 0.25 * math.exp(-((cur_lon - bay_center) ** 2) / 4.0)` and `time_wave = 0.08 * math.sin(...)`. Confidence is a hardcoded dict `{"0h": 94.2, "6h": 88.5, "12h": 81.3, "24h": 73.8, "48h": 63.5}` ([raw](https://raw.githubusercontent.com/afrazhussaina5/polarpath-ai/main/backend/services/sea_ice_service.py)).
- **⚠️ Routing (VERIFIED IN CODE):** `backend/services/route_optimizer.py` (14,435 B) imports only `math` + schemas. **There is no `heapq`, no `dijkstra`, no `a_star` and no priority queue anywhere in the file.** Routing is five hand-written builders — `_build_route_alpha`, `_build_route_beta_safety`, `_build_route_gamma_speed`, `_build_route_charlie_evasion`, `_build_route_delta_deep_detour` — selected by an `if priority == "safety" / elif priority in ["time","speed"]` ladder, with the docstring stating it *"Adapts dynamically based on whether IB-042 (simulated critical obstacle) is present"* and the branch key literally `has_hazard = "IB-042" in iceberg_service.icebergs` ([raw](https://raw.githubusercontent.com/afrazhussaina5/polarpath-ai/main/backend/services/route_optimizer.py)). `trajectory_service.py` and `risk_engine.py` likewise import only `math`.
- **README / video claims (COMPETITOR CLAIM), for contrast:** *"Human-in-the-Loop Decision Support Platform (NOT an autonomous ship-control system)"*; *"quantifies forecast uncertainty"*; *"generates Pareto-optimal multi-objective navigation routes (Safety, Fuel, Time)"*; *"explains decisions in human-interpretable maritime terms"*; horizons 6/12/24/48 h; explicit **Bharati** (69°24'S, 76°11'E) and **Maitri** (70°45'S, 11°43'E) coordinates. The video adds: *"convolutional with temporal attention"*; *"YOLOv8 and Polar SC R concept … for detecting iceberg objects from SAR imagery"*; a *"physics informed drift model"* with *"surface wind drag, ocean current drag, Coriolis force and sea-ice damping"*; *"A\*, D\* Lite along with Pareto"*; data from *"Sentinel-1 SAR, Sentinel-2 MSI, MODIS, ERA5, and HYCOM, and vessel NMEA"*; stack Python/FastAPI + React/TypeScript/Vite/Tailwind/Leaflet ([video transcript](https://www.youtube.com/watch?v=4W7-kpP6dHQ), 12 Sep 2026, 45 views, 7:35).
- **On-screen numbers (COMPETITOR CLAIM):** simulated iceberg IB042 crossing route alpha at *"approximately 14.2 hours"*; original route risk *"approximately 86.4%"* reduced to *"around 25%"* after re-route, adding *"14.2 nautical miles and 1.1 hours"*; recommended route Charlie = 276 nm, 22.6 h, fuel 45.5, risk 21.4%, safety 91.8%. Vessel "Axiom 01", ice class **PC5**, 12 knots, destination **Bharati**.
- **OUR ANALYSIS:** **every AI capability in this repository is absent as ML.** It is a deterministic simulator with AI-flavoured naming, candid in exactly one place — the string `"(Synthetic Pipeline)"` inside a Python attribute a judge reading the README would never see. The on-screen 86.4% → 25% risk reduction is an output of the team's own arithmetic with no external referent. Narratively it is the best-constructed pitch in the field; evidentially it is the weakest.

#### V4 — IAVNS / Team BYTE DRAGON — most visible, and it contradicts itself

- **Team (VERIFIED, 6 named members):** Manthan Vinzuda (leader), Swati Singh, Shreya Vyas, Ajit Chauhan, Utsav Laheru, Raj Mor. Video uploaded **13 Sep 2026**, 5:26, **129 views, 11 likes**, channel @ByteDragon-x8c (8 subscribers) ([video](https://www.youtube.com/watch?v=E9UGUEaXoUk)).
- **Repo (VERIFIED):** `main` README is a **150-byte stub** pointing to a `prod` branch. **The `prod` branch claim is TRUE** — 6 branches exist; `prod` has 14 commits, latest *"Enable predicted icebergs layer by default and display forecast trajectories on map"*, 11 Sep 2026; `prod/IAVNS/` contains `backend/ frontend/ scripts/`, **`AUDIT_REPORT.md`**, **`EVALUATION.md`**, `docker-compose.yml`, `vercel.json` ([branches](https://github.com/Manthanvinzuda007/Team-SIH-/branches/all)).
- **Routing (VERIFIED IN CODE):** `prod/IAVNS/backend/app/services/route_service.py` is **24,873 bytes**, opens `"""A* router on the shared analysis grid.`, with `import heapq`, `open_set: List[Tuple[float, Tuple[int,int]]] = []` and `heapq.heappush(open_set, (0.0, start_idx))`.
- **⚠️ `EVALUATION.md` vs the video — a direct self-contradiction (both VERIFIED):** the repo's own evaluation doc states sea ice is *"Farneback Optical Flow Advection (baseline: Persistence)"* on *"AMSR2 … (Aug 1-8 2026, 8 files)"* with **Persistence MAE 2.2% vs Optical Flow Advection MAE 1.8%**, and notes that **"A ConvLSTM was *not trained* as 8 days of data is fundamentally insufficient for generalizable spatio-temporal deep learning"** ([raw EVALUATION.md](https://raw.githubusercontent.com/Manthanvinzuda007/Team-SIH-/prod/IAVNS/EVALUATION.md)). The **YouTube description sells "a spatio-temporal ConvLSTM neural network"** fusing AMSR2/Sentinel-1/GLORYS/ERA5 ([video](https://www.youtube.com/watch?v=E9UGUEaXoUk)).
- **Iceberg LSTM (VERIFIED as a reported result):** ADE **1-Day 3.86 km vs baseline (constant velocity) 3.70 km — it loses**; 3-Day 10.34 vs 10.38; **7-Day 22.65 vs 24.91 — it wins**. Holdout 15%, *"split by iceberg (not by random points, to prevent data leakage)"*. SAR is *"Classical Computer Vision CFAR"*, with the explicit statement *"No deep learning (YOLO) precision/recall is claimed, as one unlabeled scene does not constitute a valid object detection training set."*
- **LSTM implementation not verified:** `prod/IAVNS/backend/requirements.txt` lists numpy, scipy, **xarray, netCDF4, rasterio, geopandas, shapely, pyproj, rioxarray, scikit-learn, opencv-python-headless, xgboost** — **no torch, no tensorflow, no keras**; `backend/app/ml/iceberg_lstm.py` and `backend/models/iceberg_lstm.pt` both 404. **Claimed, but implementation not verified.**
- **Data fusion (COMPETITOR CLAIM, partly corroborated):** AMSR2 SIC (6.25 km), Sentinel-1 SAR, GLORYS12 currents, ERA5 wind, GEBCO bathymetry. `AUDIT_REPORT.md` states *"Replaced all mock data stubs with real file readers"* naming `seaice_loader.py`, `gebco_loader.py`, `environment_loader.py`, `ocean_loader.py`, `iceberg_loader.py`, `sentinel1_loader.py`, and *"Fixed A\* router to use real 10km EPSG:3031 grid constraints"* — verified as a document; the loaders themselves were not individually fetched.
- **Routing UX (COMPETITOR CLAIM):** *"Risk-aware 4D A\* algorithm"* across FASTEST / SAFEST / BALANCED / CUSTOM modes with user sliders for distance, safety, fuel, currents; an "explainable risk" breakdown panel.
- **Fuel/ETA (COMPETITOR CLAIM, quantified, no derivation given):** *"⏱️ 6–14 Hours Saved… (preserving cruise speed from 4 knots up to 14 knots)"* and *"⛽ 8–15% Fuel Reduction"*. The spoken transcript garbles this to "8 to 5%".
- **Claimed innovation:** *"📄 1-Click PDF Voyage Report: Generates offline-ready, low-bandwidth (~less then 150 KB) voyage briefs for polar ship captains"*, plus local caching so the system *"keeps working even during satellite internet outages."* Targets **R/V Bharathi**. Video tags **@NCPORGoaVasco, @MoESGOI and @NarendraModi**.
- **OUR ANALYSIS:** the most publicly visible competitor, with the only polished recorded demo and a genuinely operationally-literate feature (the low-bandwidth offline voyage brief). Its `EVALUATION.md` is the second-most rigorous document in the field — and it is the document that refutes the team's own marketing.

#### V5 — SagarDrishti (repo, `APC2005-dev`) — best MLOps, narrowest scope

- **Repo state (VERIFIED):** 43 commits, 1 branch, 0 stars/forks, **no licence**, last commit "added ship" 14 Sep 2026. Tree: `backend/ data/ docs/ frontend/ ml/ models/ notebooks/`, `Makefile`, `docker-compose.yml`, `pyproject.toml`. `ml/` holds `adapters/ environment/ evaluation/ features/ inference/ models/ routing/ seaice/ tests/ training/ versioning/` plus `provenance.py` ([repo](https://github.com/APC2005-dev/SagarDrishti)).
- **⚠️ Scope mismatch (VERIFIED):** the GitHub **description** carries the full SIH26059 title, but the **README never mentions SIH** and covers **iceberg trajectory only** — no sea-ice forecasting section, no routing section — despite `ml/routing/` and `ml/seaice/` existing in the tree.
- **Routing (VERIFIED IN CODE):** `ml/routing/engine.py` (13,399 B) has `import heapq`, **`class TimeAStar:`**, `def evaluate_edge(`, `def solve(`, an inner `heuristic(point)`, and `heapq.heappop/heappush` — a real **time-dependent A\***.
- **Iceberg method (COMPETITOR CLAIM):** USNIC Antarctic Iceberg CSV ingested automatically, deduped on iceberg+Last Update, stored permanently in **PostGIS**; **GRU** over 14-entry chronological sequences producing **D+1 … D+7 in a single inference pass**. GRU weights and training loop **claimed, but implementation not verified**.
- **MLOps (COMPETITOR CLAIM — the standout):** append-only model-versioned `ml.forecasts`; forecasts scored against *later official observations* in `ml.forecast_evaluations` using **official ground truth only**; **policy-gated retraining with champion/challenger promotion and full lineage**; operator CLI; Celery + beat; Alembic migrations with append-only triggers. The presence of `evaluation/`, `training/`, `versioning/` and `provenance.py` is consistent with this, but those files were not opened.
- **UI (COMPETITOR CLAIM):** React + **Three.js** console with 1/3/7-day views filtering the same run.
- **OUR ANALYSIS:** by far the strongest engineering/model-governance story in the field — a real retraining loop with lineage, which nobody else claims. On its README's own evidence it covers roughly one of the PS's three mandated components, so a judge scoring against the PS text marks it down on coverage while a judge scoring production-readiness ranks it first.

#### V6 — SagarDrishti (video, Team Cinephiles, AIT Pune) — highest-viewed, self-declared wireframe

- **VERIFIED:** uploaded **29 Aug 2026**, 3:37, **498 views, 5 likes** — the **highest-viewed PS59 video found** — channel @Hackathon-p04 (3 subscribers), and the only video naming a college outright ([video](https://www.youtube.com/watch?v=IiyMPd4pjUY)).
- **⚠️ VERIFIED — the transcript ends:** *"So, this is just a overview of our main project. **This is the wireframe. Uh we would uh work upon it, and we will try to make our software like this.**"* **Claimed, but implementation not verified — by the team's own admission this is a mockup.**
- **Claimed features (COMPETITOR CLAIM, from the wireframe walkthrough):** captain login with *"emergency offline mode available even during satellite blackouts"*; dashboard with planned route, sea-ice heat maps, iceberg locations; panels for navigation risk, weather visibility, ocean currents, remaining fuel, route confidence; a voyage planner taking **departure port, destination port, vessel type, ice class, draft, speed, fuel capacity**; AI recommending *"safest, fastest, and most fuel-efficient routes, each with ETA, fuel consumption, ice risk, and confidence score"*; ice intelligence *"up to 7 days … +1 day, +3 days, +7 days"* with iceberg drift animations; a "captain decision center" alerting *"if an iceberg enters a planned corridor or ice compression increases"*.
- **VERIFIED absence:** **no model name, no dataset name and no metric anywhere in the video.**
- **OUR ANALYSIS:** whether this Pune team is the same as `APC2005-dev/SagarDrishti` is **unconfirmed** — "SagarDrishti" ("ocean vision") is a generic Hindi coinage and collision is plausible. The two artifacts have materially different maturity, so we list them separately.

#### V7 — MitoPolarNet / Team MitoNaut (Team ID 125159) — best framing, zero specifics

- **VERIFIED:** uploaded **15 Sep 2026**, **15:29** (longest in the field), 117 views, 5 likes. The **only team publishing its SIH team ID**. Timestamps show **~3.5 minutes of live demo** (09:03–12:45) after 9 minutes of preamble ([video](https://www.youtube.com/watch?v=DdOu6ajgwco)).
- **Architecture (COMPETITOR CLAIM):** a deliberately single connected pipeline — *"sea ice intelligence, iceberg trajectory, dynamic polar risk map, and finally navigation intelligence … we treated them as one connected system from the very start rather than trying to stitch them together later."*
- **Uncertainty (COMPETITOR CLAIM — the standout posture in the whole observed field):** *"a predicted path is still just a prediction. So, instead of saying the iceberg will definitely be here, we … account for the uncertainty in the prediction and represent it as **dynamic risk around the iceberg**. That way, we're not pretending to know more than we actually do. So, we're being honest about the uncertainty and building it directly into the system rather than **hiding it behind a confident looking line on the map**."*
- **Non-binary risk (COMPETITOR CLAIM):** *"Instead of just labeling a region safe or unsafe, we give different areas different level of predicted risk. Real conditions aren't binary, so a system that only outputs a yes or no answer is throwing away useful information."*
- **Scoping humility (COMPETITOR CLAIM):** *"MitoPolarNet is a decision support system, not a replacement for the captain"*; *"does not replace satellites, navigators or existing navigation systems."*
- **Deployment (COMPETITOR CLAIM — vague):** *"lightweight memory guided adaptive architecture … keeps compact representation of polar condition it's already learned and only activates the computation that's relevant"*; two deployment models, *"local intelligence for the vessel and cloud intelligence for fleet and mission control."* No framework, quantisation scheme or footprint named.
- **Business model (COMPETITOR CLAIM):** *"B2G + B2B"*, naming NCPOR / MoES polar research programmes and research vessel operators; revenue via institutional deployment, annual licence, analytics platform or API.
- **⚠️ VERIFIED absence:** across 15:29 the team names **no model architecture, no dataset and no numeric metric**. Everything is functional description ("our model … learns how the surrounding condition evolve"). **No repository was found.**
- **OUR ANALYSIS:** the team to beat on framing and the weakest on specifics — the exact inverse of POLARPATH. Its 15-minute runtime with 8.5 minutes of preamble also violates standard demo-length guidance badly.

#### V8 — POLAR-AI (`2911vedant/polar-ai`) — broadest feature surface, admittedly synthetic

- **Repo state (VERIFIED):** only **5 commits**, no licence, last commit *"Final verification: 25/25 live acceptance checks pass"* 13 Sep 2026. Tree includes `DATA_SOURCES.md`, `MODEL_EVALUATION.md`, `PROJECT_PLAN.md` ([repo](https://github.com/2911vedant/polar-ai)).
- **⚠️ CONTRADICTION (VERIFIED):** the README says *"Problem: SIH26059"* and, at line 6, *"developed for Smart India Hackathon **2024**"* — while SIH26059 is a 2026 code. **OUR ANALYSIS:** almost certainly an AI-generated-README or copy-paste artifact, flagged rather than silently resolved.
- **Risk weights (VERIFIED IN CODE, not just a README diagram):** `backend/app/services/risk_service.py` (12,979 B) carries the docstring `Total = 0.35*SeaIce + 0.30*Iceberg + 0.20*Weather + 0.15*Ocean`, `DEFAULT_WEIGHTS = {"sea_ice": 0.35, "iceberg": 0.30, "weather": 0.20, "ocean": 0.15}`, applies them at lines 302–305, and **renormalises user-supplied weights** at 291–294 ([raw](https://raw.githubusercontent.com/2911vedant/polar-ai/main/backend/app/services/risk_service.py)). This is the most explicit — and only code-verified — risk weighting in the field.
- **Routing (VERIFIED IN CODE):** `backend/app/services/route_service.py` (15,670 B), docstring *"Implements A\* navigation routing with multi-objective cost functions"*, `import heapq`, `def _astar(` with `heapq.heappush/heappop` and `g_score`. Produces four routes: Shortest, Safest, Fuel-Efficient, Balanced.
- **⚠️ Synthetic data admission (VERIFIED IN CODE):** `backend/app/services/demo_service.py` is **24,194 bytes**, docstring *"Generates deterministic synthetic Antarctic data using fixed random seed"*, with `RNG = np.random.default_rng(settings.DEMO_RANDOM_SEED)`, per-cell `np.random.default_rng(noise_seed).normal(0, 0.05)`, and literal `"source": "synthetic_model_v1"` / `"source": "synthetic_history"`. `risk_service.py` also carries `v = float(rng.normal(0, 0.15))`.
- **Evaluation (VERIFIED IN CODE, and self-limiting):** `MODEL_EVALUATION.md` opens *"All metrics below are computed on **DEMO/SIMULATION DATA** (synthetic, seed=42)"*, reports sea-ice RF 24 h MAE 0.0088 vs persistence 0.0467 (skill 0.811) on held-out 20% of **synthetic** data, RandomForestRegressor n_estimators=50 max_depth=8, and an 18.3 km 24 h physics drift baseline on **synthetic** trajectories. **It is internally inconsistent**: the table says skill 0.811 at 24 h while the Interpretation line says *"24h forecast is most reliable (skill = 0.52 on demo data)"* ([raw](https://raw.githubusercontent.com/2911vedant/polar-ai/main/MODEL_EVALUATION.md)).
- **Provenance (VERIFIED):** `DATA_SOURCES.md` is a genuinely good table (NSIDC G02135/G02202, NIC iceberg CSV, BYU/NIC, ERA5, CMEMS `GLOBAL_ANALYSISFORECAST_PHY_001_024`) with a Status column marking each REAL-with-credentials vs "DEMO DATA".
- **Differentiator (COMPETITOR CLAIM):** a *"Polar Navigator AI"* chat assistant with **LLM mode (OpenAI/Gemini/Groq) and a rule-based fallback requiring no API key**; 8-tab frontend including a Simulation Mode. The `polar_navigator.py` file was not fetched — **claimed, but implementation not verified**.
- **OUR ANALYSIS:** the broadest feature surface of any competitor, built on the most honestly-labelled synthetic foundation. The 2024/2026 contradiction plus seeded demo data make it the most likely to collapse under questioning about data provenance — though it is also, uniquely, the team that documented that vulnerability itself.

#### V9 — POLAR-NAV-X (`Rugved-dev18`, mirror `ASIM-05`) — the only repo with social traction, and no routing

- **Repo state (VERIFIED):** **41 commits**, 2 branches, **MIT licence**, and **7 stars / 3 forks — the only repo in the entire set with any social signal**. Last commit "Update dataset.md" 9 Sep 2026. `docs/dataset.md` is 10,809 B ([repo](https://github.com/Rugved-dev18/POLAR-NAV-X)).
- **Team (VERIFIED in pass one, unverified in pass two):** 6 named members — Asim Malik, Rugved Narkar, Sarthak Wawre, Siddhant Khedekar, Maithili Patil, Sakib Shaikh. Template placeholders `[Team Member 1]` were left in the README. No CONTRIBUTORS file was found in the code pass.
- **Self-declared stage (VERIFIED):** *"Current Development Phase — Step 1: Project Setup ✅ … establishing the foundational project structure, documentation, and configuration files."*
- **⚠️ Routing (VERIFIED ABSENT):** the `routing/` directory contains only a nested `routing/` folder and a `.gitkeep`. **No A\*/Dijkstra implementation exists there** ([routing tree](https://github.com/Rugved-dev18/POLAR-NAV-X/tree/main/routing)).
- **ML (VERIFIED IN CODE — contradicts the "Step 1" framing):** `ml/` holds `models/`, `tests/`, `features.py` and **`train_xgboost.py` (9,541 B)** importing numpy, pandas, **xgboost**, and sklearn metrics, with an argparse CLI. **No trained artifact found**; `ml/models/.gitkeep` 404s.
- **OUR ANALYSIS:** further along than its own README claims on the ML side, and further behind on routing. Python dependencies are declared nowhere at root (`requirements.txt` 404; root `package.json` is a 4-line workspace stub).

#### V10 — `SadhikaRana/SIH26059-ML-Trajectory` — better than its 24-byte README

- **VERIFIED IN CODE:** README is 24 bytes and there is no repo description, but the tree is real: `config/ data/ docs/ models/ outputs/ schemas/ src/ tests/`, 5 commits. `src/` holds 15 modules including `baseline.py`, `evaluate.py`, `feature_engineering.py`, `geo_utils.py`, `infer.py`, `predict.py`, `train.py`, `trajectory_builder.py` and **`uncertainty.py`** ([src tree](https://github.com/SadhikaRana/SIH26059-ML-Trajectory/tree/main/src)).
- **A separate baseline module exists (VERIFIED):** `src/baseline.py` imports `destination_point` from `geo_utils`; `src/evaluate.py` imports `haversine_distance_km` — i.e. a persistence/constant-heading baseline is implemented separately from the model. Few teams do this.
- **VERIFIED:** `requirements.txt` is **UTF-16 encoded** and lists pandas, numpy, scikit-learn, **xgboost**, matplotlib, seaborn, PyYAML, geopy, joblib, openpyxl, pyarrow — no torch/tensorflow. **`models/.gitkeep` is 0 bytes → no trained artifact is committed.** No licence.
- **VERIFIED ABSENT:** no routing/A\* anywhere in the tree — this is trajectory-ML only, consistent with its name. The whole tree was enumerated, so this absence is firm.

#### V11 — `tanishqbafna695/sih26059-antarctic-navigation-dss`

- **Positioning (VERIFIED as README text):** *"an uncertainty-aware, vessel-specific navigation decision layer"* producing Fastest/Safest/Balanced alternatives with human-in-the-loop ([README](https://raw.githubusercontent.com/tanishqbafna695/sih26059-antarctic-navigation-dss/main/README.md)).
- **It publishes a near-tie against persistence (COMPETITOR CLAIM — no metrics file located):** *"Seasonal forecast vs persistence \| RMSE **0.0487 vs 0.0501** at h=5 (real data)"* — a ~2.8% improvement, claimed on real data.
- **Data pipeline (VERIFIED IN CODE — the most credible manifest in the field):** numpy, pandas, **xarray, h5netcdf, h5py**, pyproj, scipy, scikit-learn, fastapi, sse-starlette, plus lazily-imported **`cdsapi>=0.7` (ERA5 via Copernicus CDS)** and **`copernicusmarine>=1.4` (GLORYS12 + OSI SAF SIC/drift via CMEMS)**, with the comment that these are *"imported lazily by `backend/data_pipeline/fetch/` so the pipeline works without them"*. **No torch/tensorflow/xgboost.** Manifest comments encode a phased plan ("Phase 4", "Phase 17 UI bundle", "Phase 18 REST API"). No licence.
- **Routing implementation: not verified** — directories not enumerated.

#### V12 — `125112056-art/antarctic-nav-system` — the only repo declaring PyTorch

- **VERIFIED:** default branch is `master`; README is 13,715 B and describes a runnable system (`run.bat`, `requirements-api.txt`, `npm run build`, `localhost:8000`). No licence.
- **⚠️ VERIFIED — the only `torch` in the field:** `requirements.txt` adds **rasterio, xarray, rioxarray, netCDF4** for *"Real satellite data ingestion (fetch_nsidc.py, data_pipeline.py)"* and **`torch>=2.2`, `scikit-learn>=1.4`** for the *"Neural forecaster (models/conv_lstm_unet.py, models/uncertainty.py)"* ([raw](https://raw.githubusercontent.com/125112056-art/antarctic-nav-system/master/requirements.txt)). The ConvLSTM file itself was not fetched, so the network is **claimed, but implementation not verified** — though this is the only repo whose dependency list could support one.
- **Routing (COMPETITOR CLAIM):** *"sparse adjacency matrix and solved with **scipy's C Dijkstra**"* — implementation file not fetched.
- **Explicit synthetic disclosure, twice (VERIFIED):** *"**The ice field that ships by default is synthetic.**"* and, under limitations, *"- The default ice field is synthetic. Covered above. **This is the big one.**"* It distinguishes three forecast modes: *"observed (cached NSIDC), neural (ConvLSTM), or the default synthetic climatology."*

#### V13–V15 — thin tail and unextractable

**LIKITH Gv** ([video](https://www.youtube.com/watch?v=UpsdegTxY-c), 7 Sep 2026, 135 views, 1:19) has a description that is a verbatim copy of the official PS26059 brief and **no transcript at all** — no architecture, team, college or metric recoverable. **POLARNAV 26059** ([app](https://sih-26059-ai-enabled-antarctic-sea-pi.vercel.app/)) is a deployed app whose URL slug and title carry the PS ID; we did not fetch the app, and whether it is the same team as V2's `frontend-pearl-nine-74.vercel.app` is **unresolved**. The verified-thin tail is genuinely thin: **`Aviral-P/SIH26059` README.md is 0 bytes**; **`ku-prince/SIH26059` README.md is 10 bytes (`# SIH26059`)**; **`sai16bit/…Adaptive-Mission-Aware-Navigation`** has a 186-byte README and **no `requirements.txt`, `package.json`, `LICENSE` or `environment.yml`** at root (language tag is Jupyter Notebook, so work may be notebook-only); **`jeevanasreechinta/polar-safe`** 404s on both branch names and could not be inspected.

---

## 3. Likely PS59 competitors

These have strong circumstantial evidence but **no explicit PS-code link**, or an explicit link to a *different* PS.

| Project / team | Source | Why "likely", not verified | What it claims |
|---|---|---|---|
| **Team Synapse core** | [video](https://www.youtube.com/watch?v=fWAjy3AHtRs), 28–29 Aug 2026, 107 views, 1:58 | Title *"SIH Problem Statement AI Enabled Antarctica iceberg trajectory forecasting software"* paraphrases the PS59 title, but **neither "26059" nor "SIH 2026" appears** in title or (empty) description | *"prototype for our live iceberg tracking platform … passively monitoring the historical drift paths of three high-priority targets, **A76, A81, and D30A**"*; *"our **U-Net segmentation** processing raw radar imagery … allows us to instantly calculate the iceberg's surface area, currently **over 4,300 square kilometres**"*; simulated LEO revisit gaps (*"Once a satellite drops over the horizon, the live feed suspends"*) |
| **POLAR-X** — `Kishore17E` | [repo](https://github.com/Kishore17E/POLAR-X) (pass one) | Explicitly entering **PS 26062**, folding 26059 in as a sub-module: *"Problem Statement ID: 26062 (Parent / Master) / Integrated Statements: 26059 (Navigation & Sea-Ice), 26060, 26061"*. **Not a head-to-head PS59 rival at judging, but building overlapping technology.** Pass two could not locate the repo | A written *"Data Truth Policy"* (`docs/DATA_TRUTH_AND_PROVENANCE.md`) — *"Never fabricates scientific data"*; Google **OR-Tools VRPTW**; 21 passing pytest tests |
| **"Polaris" / Astralis** | pass-one [repo](https://github.com/Astralis-SIH26059/Sea-Ice_concentration) | Verified in pass one via an org **named after the PS**; **not findable in pass two** | **XGBoost** 3-day SIC forecast reading NetCDF via Xarray (specifically `sic_pss25_20170101-20171231_v06r00.nc` — single-year 2017 data); EPSG:3412 lagged drift vectors + ERA5 10-m wind; **routing explicitly unbuilt** — *"Path Prediction (Coming Soon)"*; React+Vite "glassmorphic UI", custom EPSG:3031 Leaflet over NASA EOSDIS GIBS |
| **BOREAS / SETU** | pass-one [repo](https://github.com/Pawan-19012006/BOREAS) | Quoted *"SIH26059-**GREEN**"* — a suffix appearing nowhere in official SIH materials, possibly an internal college-track label; **not findable in pass two** | *"not an autopilot, a co-pilot that always shows its confidence and its reasoning"*; *"AI-generated routing recommendations with **calibrated uncertainty**"*; **CesiumJS 3D globe**; *"This README documents what is actually implemented in this repository, with an explicit real-vs-synthetic breakdown for every claim"* |
| **DAKSHIN** — Divyadarsh Pandey | [video](https://www.youtube.com/watch?v=4DfreJ9SzUg), ~11 Sep 2026, 181 views, 2:01 | **Probably NOT PS59.** Description's *"logistics, better communication, and stronger support"* framing matches **SIH26060** far better than PS59's forecast-and-route brief, despite "Navigation" in the acronym. Transcript not fetched | Antarctic prototype platform |
| Described-but-uninspected repos | [GitHub search](https://github.com/search?q=antarctic+iceberg+navigation+decision+support&type=repositories&s=updated&o=desc) | All describe PS-26059 systems and were updated within ~10 days of 20 Sep 2026; none individually verified | `anshikashivhare/offshore`, `Heer-27/antarctica-navigation-ai` (ADSS), `Smaranika2005/PenguAIn`, `amarendar10222-dev/CryoNav-AI`, `Antobenedict2006/CryoNav-…`, `ammupaleti9392-prog/polar-path-website-`, `CaptainHarryJones/POLARIS.ai`, `patilshrutika1803/POLARIS` |

**Explicitly NOT competitors** (adjacent SIH 2026 PS, recorded so they are not mistaken for rivals): `Dev-Lahrani/polar-ops-digital-twin` (PS **26060**), `deepeshaggarwal123/POLARIS-COMMAND` (PS **26062**), `avinash8386edu-eng/SIH-2026` ("POLARIS", 44th ISEA logistics, Spring Boot, subject matter points to 26062), `Uddip07/SIH26` / `parassawal/incois-3d-ocean-viz` / `KARAN-KATAKDHOND/incois-3d-ocean-viz` / `akshayaak1407/SIH_2026` (all PS **26067**, INCOIS 3D ocean), and `LALA22-7/SIH26` / `singharchit1801-gif/SIH26` (PS **70**, CycloneWatch). That last one is worth quoting because it is unusually good intelligence: its internal execution manual states verbatim **"PS59 is not part of this sprint. Do not spend build time on Antarctic sea-ice, iceberg trajectory, or navigation routing."** ([README](https://raw.githubusercontent.com/singharchit1801-gif/SIH26/main/README.md)) — a team that scoped PS59 and rejected it as too broad for an 8-day sprint.

---

## 4. Existing external systems

**These are not SIH competitors.** They are the prior art an MoES/NCPOR evaluator will test any novelty claim against, and most of them are free.

### 4.1 Routing, forecasting and risk

| System | What it is | Algorithm / method | Licence & access |
|---|---|---|---|
| **BAS PolarRoute** | Polar route planner, actively maintained (1,861 commits, latest 2 Sep 2026, 25 stars) | **Non-uniform mesh → Newton's method for the inter-cell crossing point → Dijkstra → physics-informed path smoothing**. The authors state their own novelty as *"the use of a non-uniform mesh… the use of **Newton's method within the construction of Dijkstra paths**, and the post-processing of paths to smooth them"* | **MIT**, `pip install polar-route`, [bas-logist/PolarRoute](https://github.com/bas-logist/PolarRoute), Zenodo DOI 10.5281/zenodo.21134233 |
| **PolarRoute-server** | *"A web server to manage requests for meshes and routes… implemented using **Django, Celery and Django REST framework**"*; 638 commits, latest 14 Sep 2026 | REST + Swagger, Dockerised | **MIT**, [bas-logist/PolarRoute-server](https://github.com/bas-logist/PolarRoute-server) |
| **Pre-built Antarctic vessel meshes** | `amsr_southern_SDA.json.gz`, `amsr_central_SDA`, `amsr_northern_SDA` | AMSR-based SIC + RRS *Sir David Attenborough* (Polar Class 5) vessel model | Free download, `files.bas.ac.uk/twins/polarroute/` |
| **IceNet** | Sea-ice forecasting ecosystem (BAS + Alan Turing Institute) | **Ensemble of 25 CNNs, U-Net architecture**, 50 monthly-averaged input channels, pixel-wise probabilities over three SIC classes at 25 km; beats **ECMWF SEAS5** in summer seasonal forecasts, *"particularly for extreme sea ice events"* | **MIT**, `pip install icenet`; **pretrained weights downloadable** as HDF5 from BAS RAMADDA (DOI 10.5285/71820E7D-C628-4E32-969F-464B7EFB187C) |
| **IMO POLARIS** | Polar Operational Limit Assessment Risk Indexing System, **MSC.1/Circ.1519, 6 June 2016** | **A lookup table and a weighted sum.** Risk Index Values by ice class × 12 WMO ice types (from +3 for ice-free down to −8); RIO = concentration-weighted sum of RIVs; Table 1.1 thresholds: RIO ≥ 0 normal, −10 ≤ RIO < 0 elevated risk, RIO < −10 special consideration. Table 1.4 gives a second, more permissive table for decayed ice | Published IMO circular, free |
| **OpenDrift `openberg`** | Iceberg drift module | Iceberg geometry (**sail, draft, length, width**); force terms for **ocean, wind, wave and Coriolis**; **plus melting and rollover** | GPL, `pip install opendrift`, [source](https://opendrift.github.io/_modules/opendrift/models/openberg.html) |
| **Drift+Noise IcySea** | Commercial polar ice-information app | *"near-real-time updates of ice-relevant information, including satellite imagery, ice charts, ice drift forecasts, sea temperature"*, delivered *"in a **Polar Code compliant way**"* | Commercial; **already deployed fleet-wide on Antarctic expedition vessels** (Oceanwide Expeditions); ESA InCubed-backed |
| **USCG International Ice Patrol** | Operational iceberg drift forecasting + **daily published Iceberg Limit** for shipping | *"A computer model uses ocean currents provided by the buoy and other environmental data to predict iceberg drift and deterioration"* | Free products — **but North Atlantic (40–52°N) only** |
| **OSI SAF via Copernicus Marine** | Operational sea-ice **concentration, edge, type and drift**, daily, **both hemispheres** | Passive microwave + drift retrieval | **Free** — *"They cover the global ocean, Arctic & Antarctic regions… **It's free**"* |
| **USNIC Antarctic iceberg table** | *"the global entity that **names, tracks, and documents Antarctic icebergs**"* meeting ≥20 sq NM or ≥10 NM longest axis | Observation, PDF/CSV/Shapefile + archive | **Free, no login.** As retrieved: 33 tracked bergs, all last-updated 04/09/2026 (≈ weekly-to-fortnightly cadence) |
| **BYU/NIC + Altiberg + SCAR** | Antarctic iceberg position databases | BYU: scatterometer + NIC, 1978–Apr 2025, per-berg CSV, >5 km threshold. Altiberg: **small bergs <3 km** from altimetry, v3.1, 1991–2023. SCAR: 323,520 ship-based positions | Free; **all positional history, no forecast** |
| **Commercial weather routing** | StormGeo, DTN, NAPA (4,000+ vessels), Orca AI, OneOcean, SOFAR Wayfinder | *"isochrone, dynamic programming, genetic"*; typical savings quoted **3–7%** | Commercial. **None of the material retrieved mentions ice class, POLARIS, SIC or icebergs** |
| **searoute-py / scgraph / 52°North WRT** | Free maritime routing packages | Shortest sea route (GeoJSON LineString); `scgraph` benchmarked ~0.05s vs ~0.5s; 52°North WRT *"minimize fuel consumption while considering environmental conditions"* | Free/open source |

### 4.2 Published literature that already does what PS59 asks

- **Mishra, P., Alok, S., Rajak, D. R., Beg, J. M., et al. (2021), "Investigating optimum ship route in the Antarctic in presence of sea ice and wind resistances — A case study between Bharati and Maitri", *Polar Science* 30, 100696.** *"This paper presents a technique of **ship route optimization in the Antarctic sea ice region using Dijkstra's algorithm**… It aims to optimize a safe ship route in the Antarctic sea ice **between Bharati and Maitri (Indian Research Stations in Antarctica)** coasts."* Authors from **Pandit Deendayal Energy University** and **Space Applications Centre (ISRO), Ahmedabad**. BAS's own PolarRoute paper cites and critiques it: *"the ice and wind resistance models used are sophisticated… the benefits of this sophistication may be outweighed by the simplicity of the route-planning."*
- **Zvyagina, T. & Zvyagin, P. (2022), "A model of multi-objective route optimization for a vessel in drifting ice", *Reliability Engineering & System Safety* 218 Part B, 108147.** Verbatim highlights: *"A **3D graph** provides a spatio-temporal model for a vessel traversing a water area"*; *"It is reasonable to define ice risk criteria for a route of a vessel as a **minimax**"*; *"Finding all the shortest paths on a graph helps to find the **Pareto set**"*; *"The proposed 3D graph algorithm design helps to consider **ice motion objects**."* It names icebergs explicitly: *"Let the area of interest contain dynamic ice features: ice floes, ice fields, **icebergs** or others, whose motion is directed by the current and the wind."* Method: wave algorithm + **Yen's k-shortest-paths**, lifted to a space+time graph.
- **Li et al. (2024), *Ocean Engineering* 311, 118843.** *"we developed a **sea ice collision risk index (ICRI)**, which takes into account the relative motion relationship between the sea ice and the ship as well as the sea ice area parameter. Finally, a **flexible velocity obstacle (FVO) algorithm** is proposed… provide real-time maneuvering guidance for ships."*
- **Reviews establishing the field is mature:** Tran et al. (2023), *"Pathfinding and optimization for vessels in ice: A literature review"*, *Cold Regions Science and Technology* — 32 articles reviewed, noting *"**a few studies consider dynamic ice conditions and route validation**"*; *"Arctic weather routing: a review of ship performance models and ice routing algorithms"*, *Frontiers in Marine Science* (2023); and a state-of-the-art count of *"**more than 500 papers** dedicated to the weather routing problem"*.
- **Detection benchmark already in print:** a 2026 *Earth System Science Data* paper reports deep-learning Sentinel-1 detection of grounded Antarctic icebergs at **F1 > 0.91** with a minimum detectable size of **0.016 km²**.
- **Prediction difficulty is documented:** **SIPN South** coordinates real-time Southern Ocean sea-ice predictions and focuses on *"austral summer, a season of special interest due to the **intense marine traffic** at this time of the year"*; published assessment finds *"a large spread in dynamical model–based predictions that **exceeds that of the observed climatological spread**."*

### 4.3 Two conflicts we surface rather than resolve

**IceNet's Antarctic status.** **BAS's own project page (last modified 26 Feb 2026) states: *"IceNet is a deep learning system that forecasts **Arctic** sea ice"***, and the *Nature Communications* study covers Arctic only ([BAS](https://www.bas.ac.uk/project/icenet/)). **This is contradicted by** a WWF Arctic Programme article: *"IceNet combines satellite data and weather observations to forecast **both pan-Arctic and pan-Antarctic** sea ice concentrations on a daily timescale up to several months ahead"* ([WWF Arctic](https://www.arcticwwf.org/the-circle/stories/navigating-a-changing-ice-world/)). **OUR ANALYSIS:** the codebase is hemisphere-agnostic by design, so a Southern-Hemisphere IceNet is a configuration-and-compute exercise; but the published, benchmarked results are Arctic-only. The defensible statement is that **Antarctic IceNet capability is reported by a secondary source and is best described as in development** — neither "IceNet does not work in the Antarctic" nor "IceNet gives operational Antarctic forecasts" survives both sources.

**"3D-DSP" in connection with PolarRoute.** The strings "3D", "DSP" and "3D-DSP" **do not appear anywhere in the full text of arXiv:2209.02389**, and a targeted search for `"3D-DSP" PolarRoute` returned **zero results**. What the paper *does* say is that the 3D/time extension is a difficulty it **does not solve**: *"The 2D mesh used by Dijkstra's algorithm must be extended to a third dimension to model temporal variability, which introduces extra complexity. **This complexity is avoided** by Fox et al. [2021] and Lehtola et al. [2019]."* **Do not attribute "3D-DSP" to PolarRoute.**

**One more caveat on PolarRoute that cuts both ways.** As *published* (2022 preprint), PolarRoute uses **historic AMSR-2 SIC with a 14-day temporal abstraction** and a **6-year (2005–2010) mean SOSE current field**, and the paper states plainly that *"Using this abstraction is equivalent to **assuming that conditions never change during any route**"*, with forecast ingestion named as future work. But the 2026 codebase is four years and hundreds of commits on, the current canonical citation is the **JAIR 2025 paper "Path-Planning on a Spherical Surface with Disturbances and **Exclusion Zones**" (doi:10.1613/jair.1.16746)** which we did not read, and **Jonathan Smith and James Byrne appear on both the IceNet and PolarRoute teams**. It would be unsafe to assert in 2026 that PolarRoute cannot ingest forecasts or handle discrete exclusion zones. Also note the org has migrated twice: **`antarctica/PolarRoute` → `bas-amop/PolarRoute` → `bas-logist/PolarRoute`** (all redirect; the repo's own README badges are stale and still point at `bas-amop`).

---

## 5. Detailed competitor matrix

**Legend: ✓ = explicitly present · ~ = partially or possibly present · ? = unclear · — = not found.**

**Caption — read this before the table.** Most ✓ marks in this matrix rest on **README text or video narration, not on inspected code**. Marks backed by fetched source or a committed artifact are shown in **bold**. Because `github.com` code search and the GitHub API were blocked, a `—` means "not found in the directories we enumerated", not "proved absent from the repository" — with two firm exceptions, `POLAR-NAV-X`'s empty `routing/` directory and `SadhikaRana`'s fully-enumerated tree. Rows marked † could not be re-located in the second research pass and rest on first-pass README readings alone.

| Competitor | Sea-Ice Forecast | Iceberg Prediction | Routing | Uncertainty | Risk Map | Fuel/ETA | Multi-Objective | Explainability | Data Fusion | Dashboard |
|---|---|---|---|---|---|---|---|---|---|---|
| **nawddeep/SIH26059** | **✓** (loses to persistence) | **✓** (3.91 km RMS 24 h) | **✓** (43,919 B A\*) | — | ~ (POLARIS RIO) | ~ | ? | — | ~ | **✓** |
| **PolarNav** (kanishka) | **✓** (RF, MAE 0.0401) | **✓** (RF, 1.7 km mean) | **✓** (93,518 B EPSG:3031 A\*) | — | ~ | ? | ~ | — | ~ | ~ (**backend 503**) |
| **POLARPATH AI** (AXIOM) | — (**`import math`**) | — (**`import math`**) | — (**no search algorithm**) | ~ (hardcoded % dict) | ~ | ~ | ~ (Pareto claimed) | ~ (claimed) | — | ✓ |
| **IAVNS** (BYTE DRAGON) | **✓** (optical flow, 1.8% vs 2.2%) | **✓** (LSTM ADE; framework absent) | **✓** (24,873 B A\*) | — | ✓ | ✓ (8–15% claimed) | ~ (4 modes + sliders) | ~ | ✓ (6 loaders named) | ✓ |
| **SagarDrishti** (repo) | ? (`ml/seaice/` exists) | ✓ (GRU, D+1…D+7) | **✓** (`TimeAStar`) | — | ? | ? | — | — | ~ (USNIC→PostGIS) | ✓ (Three.js) |
| **SagarDrishti** (video, Cinephiles) | ~ (+1/+3/+7 d claimed) | ~ (drift animations) | ~ | ~ ("confidence score") | ✓ (heat maps) | ✓ (ETA + fuel) | ✓ (safest/fastest/cheapest) | — | — | ✓ (**wireframe only**) |
| **MitoPolarNet** (MitoNaut) | ~ | ~ | ~ | ✓ (**core framing**) | ✓ (non-binary) | ? | ~ | ~ | ~ | ✓ |
| **POLAR-AI** (2911vedant) | ~ (RF on **synthetic**) | ~ (physics, synthetic) | **✓** (A\* verified) | — | **✓** (0.35/0.30/0.20/0.15) | ~ | **✓** (4 routes, reweighting) | ~ (LLM chat claimed) | ~ (good `DATA_SOURCES.md`) | ✓ (8 tabs) |
| **POLAR-NAV-X** (Rugved) | ~ (`train_xgboost.py`, no weights) | ? | **—** (`routing/` = `.gitkeep`) | — | — | — | — | — | ~ | ~ |
| **SIH26059-ML-Trajectory** | — | **✓** (15-module `src/`) | **—** (tree fully enumerated) | ~ (**`uncertainty.py`**) | — | — | — | — | ~ (`merge_wind.py`) | — |
| **tanishqbafna695** | ~ (0.0487 vs 0.0501 claimed) | ? | ? | ~ ("uncertainty-aware") | ? | ? | ~ (Fastest/Safest/Balanced) | ~ (human-in-the-loop) | **✓** (cdsapi + copernicusmarine) | ? |
| **125112056-art** | ~ (**only `torch` in field**) | ? | ~ (scipy Dijkstra claimed) | ~ (`models/uncertainty.py`) | ? | ? | ? | — | ~ (NSIDC ingestion) | ~ |
| **Team Synapse core** (likely) | — | ✓ (U-Net segmentation) | **—** (none shown) | — | — | — | — | — | ~ (SAR only) | ✓ |
| **"Polaris" / Astralis †** | ~ (XGBoost, 3-day, 2017 data) | ~ (lagged vectors + ERA5) | **—** (*"Coming Soon"*, their words) | — | — | — | — | — | ~ | ✓ (EPSG:3031 + GIBS) |
| **BOREAS / SETU †** | ~ | ~ | ~ | ✓ (*"calibrated uncertainty"*) | ~ | ? | ~ | ✓ (*"shows its reasoning"*) | ~ | ✓ (CesiumJS) |
| **Thin tail** (Aviral-P, ku-prince, sai16bit, polar-safe) | ? | ? | ? | — | — | — | — | — | — | — |

**What the matrix shows.** Routing and a dashboard are near-universal; **uncertainty and explainability are nearly empty columns**. Only **four teams have a verified search-based router in code** (nawddeep, PolarNav, IAVNS, POLAR-AI) and a fifth claims scipy Dijkstra. **Only two teams — nawddeep and IAVNS — have a code-committed forecast evaluation against a baseline on real data**, and in both cases the headline result is a loss at short lead. Three projects that *market* routing have none: POLARPATH (no search algorithm), POLAR-NAV-X (empty directory), Astralis (self-declared "Coming Soon"), plus Synapse core and SadhikaRana which never claimed it.

---

## 6. Common/commodity features

The convergence across independently-built projects is the clearest signal in this research. Four of six video teams and nearly every substantive repo landed on **the same architecture**: ConvLSTM (or Conv + temporal attention) for sea ice → physics or ML drift for bergs → A\*/Dijkstra-family routing over a risk raster → React/Leaflet dashboard → FastAPI backend → multi-route safest/fastest/cheapest cards with ETA, fuel and a "confidence score". **That template is the field default.**

| Feature | How common in the PS59 field | External availability |
|---|---|---|
| **A\*/Dijkstra routing on a cost raster** | **Every team that named a routing algorithm named A\* or Dijkstra.** Verified `heapq` A\* in four independent repos | NetworkX ships A\*; the raster→graph→A\* recipe is a long-answered GIS StackExchange question; least-cost path is a **built-in commercial GIS feature** |
| **React/Leaflet/MapLibre polar-stereographic hazard dashboard** | Universal — every Tier A/B project | End-to-end tutorials exist for the exact maritime case (Dash + Leaflet + Searoute; Folium + Searoute voyage appraisal) |
| **FastAPI + React + Docker Compose** | Universal | — |
| **Risk heat map with weighted layers** | Near-universal; only POLAR-AI's weights are verified in code | ABS already publishes POLARIS RIO contour maps (green/blue = safe); IcySea ships ice-chart egg-code polygons |
| **Multi-route cards (Fastest / Safest / Balanced / Fuel-efficient)** | IAVNS (4 modes), POLAR-AI (4 routes), tanishqbafna695 (3), SagarDrishti video (3), POLARPATH (5 hardcoded) | `searoute-py`, `scgraph`, 52°North Weather Routing Tool; StormGeo/DTN/NAPA commercially |
| **Physics-based iceberg drift (wind + current + Coriolis)** | POLARPATH, POLAR-AI, Astralis, nawddeep-adjacent | **`opendrift.models.openberg` does all of this free — *plus melting and rollover*, which no PS59 team claims** |
| **SAR iceberg detection/classification** | POLARPATH (YOLOv8), Synapse (U-Net), PolarNav (`sentinel_sar_detector.joblib`) | The **Statoil/C-CORE Kaggle Iceberg Classifier Challenge** is 9 years old with **93 public GitHub repositories**, and Kaggle's own metadata shows three-generation notebook copy-chains still being updated |
| **POLARIS / Polar Code badge** | nawddeep, PolarNav | **IMO MSC.1/Circ.1519 (2016)** — two lookup tables and a weighted sum; Lee et al. (2021) already routed with POLARIS |
| **"Offline mode for satellite blackouts"** | IAVNS (PDF voyage brief), SagarDrishti (emergency offline mode), MitoNaut (local/cloud split) | Genuinely less common; IAVNS's ~150 KB voyage brief is the most concrete instance |
| **Free Antarctic data claimed as scarce** | Implicit in several pitches | OSI SAF/Copernicus give **free daily SIC, edge, type and drift for both hemispheres**; USNIC gives free iceberg CSV/Shapefile |

**One important nuance against over-claiming commodity status.** GitHub repository search (metadata only, so these are floors) returns **3 repos** for `sea ice convlstm`, **6** for `"sea ice" forecasting deep learning`, **4** for `iceberg drift model` — versus **93** for `statoil iceberg classifier`. So **Antarctic ConvLSTM sea-ice forecasting is genuinely the least commodity component of PS59** — but it is also the component with the **most established public baselines** (IceNet at 106 stars; `jerrywn121/Arctic_SIC_prediction` explicitly benchmarking *"with reference to two baselines, i.e., LSTM and Damped Anomaly Persistence"*), which makes an unbaselined claim there the easiest to puncture.

---

## 7. Features we should NOT claim as innovation

This section is deliberately blunt. Each item below is something a PS59 team would plausibly present as its contribution, and each is followed by the specific evidence that would refute it.

**Routing algorithms.** **A\* and Dijkstra are not a contribution in this field; they are the baseline.** We verified genuine `heapq`-based A\* implementations in **four independent PS59 repositories** (nawddeep's 43,919-byte `route_engine.py`, PolarNav's 93,518-byte `polar_routing_engine.py`, IAVNS's 24,873-byte `route_service.py`, POLAR-AI's `_astar`), a fifth claiming scipy Dijkstra, and every single team that named a routing algorithm named one of the two. Outside SIH, NetworkX ships A\*, least-cost path across a cost raster is a built-in commercial GIS feature, and BAS PolarRoute is Dijkstra with a Newton crossing-point refinement. Dijkstra, A\*, isochrones, dynamic programming, genetic algorithms, velocity obstacles and Pareto/Yen k-shortest-paths have **all** already been applied to exactly this problem. D\* Lite (incremental replanning under changing costs) is the only routing choice we saw with a defensible rationale, and even that is a 2002 algorithm with public implementations.

**Physics-based iceberg drift.** **`opendrift.models.openberg` gives it away free** — iceberg geometry (sail, draft, length, width), force terms for ocean, wind, wave and Coriolis, **plus melting and rollover**. POLARPATH's claimed physics ("surface wind drag, ocean current drag, Coriolis force and sea-ice damping") is a strict **subset** of what `pip install opendrift` provides, minus melt and rollover. Any evaluator familiar with polar modelling asks "why not OpenDrift?" and a team with no answer loses the point.

**SAR ship-vs-iceberg classification.** The **Statoil/C-CORE Iceberg Classifier Challenge** is a nine-year-old featured Kaggle competition with a public leaderboard, **93 public GitHub repositories** matching `statoil iceberg classifier`, and Kaggle's own provenance labels documenting a three-generation copy chain of the same beginner notebook ("Notebook copied with edits from [SKS]" → Yueun Choi → hanpil), still being updated two months ago. **This cannot be claimed as innovation under any framing.** Detection (YOLOv8 on SAR) and segmentation (U-Net on radar) are a step up in task difficulty but are standard architectures on a standard modality — and a 2026 ESSD paper already reports **F1 > 0.91 at 0.016 km² minimum detectable size** for Sentinel-1 Antarctic iceberg detection. **No PS59 team quoted any detection metric at all.**

**Maritime routing, fuel-optimal weather routing and port-to-port distance.** `searoute-py` (pip-installable, returns GeoJSON), Eurostat's upstream Java `searoute` "deployed by the statistical office of the European Union", `scgraph` (benchmarked ~0.05s vs searoute's ~0.5s), and the **52°North Weather Routing Tool** ("minimize fuel consumption while considering environmental conditions") are all free. Commercially, StormGeo, DTN, NAPA (4,000+ vessels) and Orca AI have been selling isochrone/DP/genetic voyage optimisation for years at quoted savings of 3–7%. **"We built AI voyage optimisation" is refuted instantly.** The only PS59-specific part is the ice-difficulty cost field — precisely the part no team has validated.

**POLARIS integration.** POLARIS is **IMO MSC.1/Circ.1519, published 6 June 2016**: a Risk Index Value lookup matrix (ice class × 12 WMO ice types, values +3 to −8), a concentration-weighted sum, and a three-band threshold table. Implementing it is an afternoon's work given an ice chart. ABS already publishes RIO contour maps as an operational product; IcySea already advertises Polar Code compliance; **Lee et al. (2021) already used POLARIS inside a route planner**. Claiming "POLARIS compliance" as a technical differentiator is claiming credit for a lookup table and a weighted sum. (Omitting it is the opposite risk — but the correct framing is "we implement POLARIS because the Polar Code expects it", not "we invented POLARIS integration".)

**ML sea-ice forecasting.** **IceNet** — an ensemble of 25 U-Net CNNs, published in *Nature Communications* in 2021, beating ECMWF SEAS5 on summer seasonal forecasts, MIT-licensed, on PyPI, with **pretrained weights publicly downloadable** from BAS RAMADDA and a daily-resolution operational successor producing forecasts up to two weeks ahead — forecloses "first deep-learning sea-ice forecast", "first U-Net for SIC", "first probabilistic ML sea-ice forecast beating a dynamical model" and "first open-source ML sea-ice forecasting pipeline".

**Combining iceberg hazards with multi-objective route optimisation.** **Zvyagina & Zvyagin (2022), *Reliability Engineering & System Safety* 218, 108147, already did exactly this**: a 3D space-time graph over a water area containing *"ice floes, ice fields, **icebergs** or others, whose motion is directed by the current and the wind"*, with a **minimax ice-risk criterion**, objective functions over route length, risk and number of course changes, and **Pareto-optimal solution sets** via Yen's k-shortest-paths. "No system does iceberg-aware route optimisation" is false as a conceptual claim.

**A sea-ice collision risk index.** **Li et al. (2024), *Ocean Engineering* 311, 118843** publishes an **ICRI** accounting for ship–ice relative motion and ice area, plus a flexible velocity obstacle algorithm giving *"real-time maneuvering guidance for ships"*. The IIP has published an operational daily **Iceberg Limit** for the North Atlantic for a century.

**Optimum routing between Bharati and Maitri.** This is the single most dangerous piece of prior art for an Indian team on this PS. **Mishra et al., *Polar Science* 30 (2021) 100696** — *"ship route optimization in the Antarctic sea ice region using Dijkstra's algorithm… between **Bharati and Maitri** (Indian Research Stations in Antarctica)"* — was authored from **Pandit Deendayal Energy University and ISRO's Space Applications Centre**, published in a journal an MoES/NCPOR evaluator plausibly reads, and is **cited by BAS's own PolarRoute paper**. Any claim that nobody has optimised a route between India's Antarctic stations is **false and will be caught**.

**"Free Antarctic data doesn't exist."** OSI SAF via Copernicus Marine provides free, daily, both-hemisphere **sea-ice concentration, edge, type and drift**; AMSR-2 NRT is free from NASA Earthdata; USNIC publishes the Antarctic iceberg table free as PDF/CSV/Shapefile with an archive; BYU/NIC, Altiberg and SCAR are all free.

**"First open-source polar navigation software / first polar routing web API."** PolarRoute, PolarRoute-server (Django/Celery/DRF, Dockerised, Swagger) and IceNet are all MIT-licensed, and BAS hosts **pre-built Antarctic vessel meshes** for free download.

**And one more that is specific to this field's rhetoric.** Nearly everyone in the PS59 field claims a ConvLSTM, U-Net, LSTM or YOLO. **Exactly one repository in the entire field declares `torch` in any manifest we read.** The committed trained artifacts we found are `RandomForestRegressor` `.joblib` files. Claiming a deep architecture is, in this field, statistically more likely to be a README assertion than a trained model — which means an evaluator who has seen two or three of these submissions will discount the claim by default.

---

## 8. Potential competitive gaps

**This section identifies gaps only. It does not recommend what to build.**

**Validation against a baseline is the emptiest space in the field.** **Not one of the six video teams reports a forecast-skill number against any baseline** — persistence, climatology or damped anomaly persistence. Every number shown on screen is a self-generated risk or route figure. In the repositories, only **two** projects publish a baseline comparison on real, held-out data with the artifact committed (nawddeep's `baseline_comparison.json`, IAVNS's `EVALUATION.md`), and both show a **loss at short lead**. A third (PolarNav) commits a `baseline_mae` field without naming or defining the baseline; a fourth (POLAR-AI) publishes a full skill table computed on **synthetic seed=42 data** and contradicts itself within the same document (table skill 0.811, prose skill 0.52); a fifth (tanishqbafna695) asserts 0.0487 vs 0.0501 RMSE on real data in one README line with no metrics file located. Everyone else publishes **nothing**.

**A held-out-date forecast overlaid on observed ice is recommended by outside analysts and done by nobody.** SIH Buddy's PS-specific scorecard advises exactly this — *"Show a forecast ice field for a date you held out, overlay the actual observed ice for that date to prove the forecast held, and then run the router to a station"* — and warns that *"sea ice forecasting skill degrades sharply beyond a couple of weeks, so pick a horizon you can actually demonstrate skill at and **report against persistence, which is a surprisingly strong baseline**."* Zero observed teams do it.

**Vessel ice-resistance cost models are invented rather than derived.** SIH Buddy flags this directly: *"Ice-aware routing needs a cost model for how hard a vessel finds a given ice concentration, and **without a vessel ice class and resistance model your route optimises a difficulty you invented**."* POLARPATH names PC5 on screen but derives no resistance model; the rest assign difficulty scores with no stated physical basis. By contrast PolarRoute's published vessel model uses ice-resistance parameters from Li et al. (2020) and is demonstrated on a Polar Class 5 ship.

**Uncertainty is almost entirely unoccupied.** In the matrix it is the emptiest column. Only **MitoNaut** foregrounds it verbally and names no model, dataset or number across 15:29; **BOREAS** claimed "calibrated uncertainty" in a repo we could not re-locate; **POLARPATH**'s "forecast confidence" resolves in code to a hardcoded dict `{"0h": 94.2, … "48h": 63.5}`; **SadhikaRana** has an `uncertainty.py` module and no trained model; **125112056-art** references `models/uncertainty.py` in a requirements comment. Nobody publishes a calibration curve, a coverage number or a probabilistic score.

**No team benchmarks against the free alternatives.** Nobody compares their drift model to **OpenDrift `openberg`**, their router to **PolarRoute** or **searoute**, or their forecast to **IceNet**. That is the cheapest possible external referent and it is universally skipped.

**Coverage of all three mandated components is rare.** SagarDrishti's README covers iceberg trajectory only; Astralis marked routing *"Coming Soon"* in its own words; POLAR-NAV-X's routing directory is a `.gitkeep`; SadhikaRana is trajectory-only by design; Synapse core shows tracking with no forecast and no routing. An Instagram SIH-advice creator with a "Day 3/18 — SIH 2026 Theme Breakdowns" reel on SIH26059 wrote *"Zyada tar teams teen pieces mein se sirf ek solve karti hain. Asli gap inhe jodne mein hai"* ("Most teams solve only one of the three pieces. The real gap is in joining them") — **an unsourced influencer assertion that our own independent repo and video evidence corroborates.**

**Operational Antarctic iceberg drift forecasting has no public product.** Operational iceberg drift forecasting **with a published hazard limit for shipping** exists and has for a century — the **USCG International Ice Patrol** (North Atlantic, 40–52°N) and **NRC** for Grand Banks offshore operations, plus published work on NW Greenland. For the **Antarctic**, the publicly available products are **observational only**: USNIC's table (≥20 sq NM or ≥10 NM long axis, refreshed roughly weekly-to-fortnightly — all 33 entries shared a single 4 Sep 2026 update date when retrieved on 20 Sep), BYU/NIC (>5 km, last database update 13 Oct 2023), Altiberg (<3 km, but a statistical distribution product from along-track altimetry, not per-object near-real-time), and SCAR's ship-based historical archive. **We found no source contradicting this.** Note the residual unknown: whether AWI, AAD or any national Antarctic programme runs an **internal, unpublished** Antarctic iceberg drift forecast is **not determinable from open sources**.

**The small-berg size class is unserved operationally.** USNIC tracks only large bergs; bergy bits and growlers — the size class that actually hulls a resupply vessel — are not in that table, and Altiberg's coverage of <3 km bergs is climatological rather than operational.

**The tactical timescale is the least-served band.** OSI SAF is observational; IceNet is days-to-months; SIPN South is seasonal. Vessel tactical navigation needs **hours-to-days** for the Antarctic, which sits between all three.

**Satellite SIC → WMO ice type → POLARIS is not routinely automated.** POLARIS requires **ice type/stage-of-development across 12 WMO categories plus partial concentrations** — which an *ice chart* egg code provides and which passive-microwave SIC (AMSR-2, OSI SAF) does **not** provide directly. "Ice chart → POLARIS RIO" is solved (ABS ships it); "passive microwave SIC → WMO ice type → POLARIS RIO" is not.

**Integration across institutions is unclaimed.** Each component sits in a different institution: ML sea-ice forecasting at BAS/Turing, mesh routing at BAS, POLARIS at IMO, iceberg positions at USNIC/BYU/Ifremer, iceberg drift forecasting at USCG/NRC for the North Atlantic. **We found no single deployed system integrating them for Antarctic waters, and none serving India's Antarctic programme.** Two honest caveats: this is *absence of evidence*, since internal NCPOR tooling would not be web-visible; and **BAS personnel overlap between IceNet and PolarRoute (Jonathan Smith, James Byrne) makes it plausible BAS is already building the forecast-into-routing link**, which is the biggest risk to any integration-novelty claim and is unverified.

**Long-form technical writing is a completely empty channel.** **No PS59 team has published a technical article** on Medium, Dev.to or Hashnode, and no slide deck was found on any public host. The only architecture diagrams that exist are Mermaid/ASCII embedded in GitHub READMEs.

**Licensing is near-absent.** Only **2 of 12 inspected repositories carry a licence** — MIT on `kanishka11082007-sys/SIH26059` and `Rugved-dev18/POLAR-NAV-X`. Every other inspected repo returns 404 on `LICENSE`, which matters because the SIH guidelines state that IP in the winning idea *"would be split equally between industry that gave the problem statements and the winning team or will be decided on the mutual agreement."*

---

## 9. Important observations about the current PS59 landscape

### 9.1 How crowded it actually is — and the conflicts in that number

| Source | Figure for SIH26059 | Basis | Date |
|---|---|---|---|
| zaidsayyed.in (mirror, scrapes portal twice daily) | **21 ideas / 500 cap**, "+3 in 2 days", "119th of 240" | Scrape of sih.gov.in | as of **19 Sep 2026** |
| sih-gitam.vercel.app (independent mirror) | **22 submissions** | Scrape of sih.gov.in | live, 20 Sep 2026 |
| Same site, cached search snippet | **8 submissions** | Earlier crawl | early Sep 2026 |
| Earlier mirror snapshot | **2 / 500**, "+2 in 10 days" | Scrape | ~early Sep 2026 |
| **SIH Buddy (projection, not a count)** | **"Moderate — 150–360 teams expected"**, "#169 of 240 by expected field" | Regression from 2025 patterns, **self-labelled: "This is a guess, not a fact… please do not take it as the truth"**, **R² = 0.25 on held-out statements** | 2026 |

**⚠️ We surface this conflict rather than resolving it.** The observed mirror figure (21–22) and the modelled figure (150–360) are differently scoped, not directly contradictory: one is a point-in-time observation of a live counter, the other a low-R² regression of where the statement will *end up*. Both mirrors agree on order of magnitude today. **We could not verify either against sih.gov.in itself, which returned `403 Forbidden — Microsoft-Azure-Application-Gateway/v2` on every one of five access methods** (curl with desktop UA, WebFetch, Firecrawl with stealth proxy and 6–8s JS wait, Firecrawl with an Indian egress IP, and a 2025 screening-result page) — while PDFs under `sih.gov.in/letters/` serve fine, indicating the block is specifically on dynamic HTML. **The live portal counter is the only authoritative number and neither research pass could read it.**

**Context for the 21.** Across the 237 statements whose counts we could extract on 19 Sep 2026, the board held **8,103 ideas across 240 statements**, **median 21, mean 34.2, range 2–227**, with **+2,626 ideas added in the prior two days**. SIH26059 sits at almost exactly the **50th percentile** — 113 statements below, 118 above. But its **+3-in-2-days** rate is far below the board-wide surge rate (~11 per statement over the same window), so **26059 is currently losing share relative to the average statement**. For scale, SIH 2025 finished with **72,165 ideas across 271 statements ≈ 266 per statement**; SIH 2026 currently averages 34 with the deadline near.

### 9.2 The deadline is genuinely contested

| Source | Stated idea-submission deadline |
|---|---|
| Dataset/PDF mirrors (e.g. Galgotias LMS poster, 25 Aug 2026; Scribd PS list) | **20 September 2026** |
| codehuntersacademy.com PS explorer | **20 September 2026** (but its page says "226 Problem Statements" vs the live 240 — a stale snapshot) |
| zaidsayyed.in live mirror, "Updated 19 Sept, straight from sih.gov.in" | **30 September 2026**, with a countdown tile reading "10 — Days to the deadline" |
| blinknbuild.in third-party master catalogue PDF | *"Submission Window: Closes officially on **September 30, 2026** on sih.gov.in"* |
| **Official SIH 2026 Guidelines (College SPOC) PDF** | **Two different, earlier dates inside the same document**: *"the last date for team nomination and idea submission by College SPOC on SIH portal is till **30th Aug 2026**"* and, two paragraphs later, *"The last date for team nomination and idea submission by College SPOC and Team leader on SIH portal is till **15th Sept 2026 only**. No request will be entertained after the deadline."* |

**⚠️ The official rulebook contradicts itself and the official portal.** The guidelines PDF is **demonstrably stale** — it carries a section header reading **"IDEA SUBMISSION PROCESS / SIH 2025"**, i.e. it is a lightly-edited carry-over of the previous edition whose dates were not fully updated. Corroborating that 2026 deadlines have already slipped at least once: SPOC registration was publicly extended to 14 August 2026, and MoE's Innovation Cell posted about the deadline being adjusted "to facilitate smooth participation in SIH 2026" (snippet only; full post unretrievable). **Only the team leader's own portal dashboard and the college SPOC can settle this.**

### 9.3 SIH structural facts that shape the competition

| Rule | Value |
|---|---|
| Max teams nominated per institute | **50 (45 shortlisted + 5 waitlisted)**; **100 for a university**. Total, not per category |
| Team composition | **Exactly 6 members** including leader, all from the same college, **at least one female member mandatory** |
| Problem statements per team | **Maximum 2** |
| Idea cap per PS | **500**, after which the PS freezes; the counter is **intentionally public** — *"can be viewed by anyone on sih.gov.in"* |
| Teams shortlisted per PS | *"**4-5 teams per problem statement** may be selected for the grand finale"* |
| 2025 actual | **1,360 finalist teams / 271 statements = 5.02 per PS** (PIB). Verified per-PS: SIH25042 (MoES/CMLRE) = **5 shortlist + 1 waitlist**; SIH25043 = **5 shortlist + 2 waitlist** |
| Prize | **Rs 1,50,000 per problem statement** (raised from Rs 1,00,000 in the prior edition) — *"The prize money will be given by the collaborating ministry/industry **ONLY IF that organization likes the idea of the winning team**"*, and the sponsor *"isn't obligated to declare a winner unless student proposals meet their expectations"* |
| IP | *"split equally between industry that gave the problem statements and the winning team or will be decided on the mutual agreement"* |
| Grand Finale | Offline at nodal centres, **proposed for December 2026**; travel reimbursement ceiling **Rs 3,000 per person** round trip |
| Judging criteria (official, **verbatim**) | *"the ideas will be evaluated by experts. Evaluation criteria will include **novelty of the idea, complexity, clarity and details in the prescribed format, feasibility, practicability, sustainability, scale of impact, user experience and potential for future work progression**."* |
| Published weights | **None.** No marking scheme with weights exists in any source we could find, for either stage. Any "rubric" on coaching blogs is a reconstruction |

Note that **"clarity and details in the prescribed format" is itself a scored criterion** — conformance to the idea-presentation PDF template is graded — and that four of the nine criteria (feasibility, practicability, sustainability, potential for future work progression) are about whether the thing can be built and kept running.

### 9.4 The MoES polar block, and no historical precedent

The block is larger than commonly assumed: **SIH26059–SIH26065, seven consecutive statements**, all Ministry of Earth Sciences, all Software, plus adjacent SIH26057/26058.

| PS | Title | Ideas (19 Sep 2026) |
|---|---|---|
| SIH26057 | AI-Powered Automated Underwater Marine Debris and Anomaly Detection (Side-Scan Sonar) | 23 |
| SIH26058 | Low-Power, Real-Time Adaptive Software-Defined Sonar Transmitter Payload for AUVs | 8 |
| **SIH26059** | **AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System** | **21** |
| SIH26060 | Digital Platform for efficient remote management of Indian Antarctic Research Stations | 9 |
| SIH26061 | AI-Driven Smart Energy Management System for Polar Research Stations | 21 |
| SIH26062 | Integrated Polar Expedition Logistics and Asset Management System | 8 |
| SIH26063 | Integrated Polar Science Outreach, Knowledge Repository and Media Dissemination Portal | 11 |
| SIH26064 | Low-Cost Deployable Seafloor Metal Detection Sensor for Ocean Resource Exploration | 7 |
| SIH26065 | Autonomous Low-Cost Ocean Observation Platform for Polar and Southern Oceans | 12 |

**SIH26059 is the most contested statement in its own family** — more ideas than 26060, 26062, 26063, 26064 and 26065, tied with 26061. It is the glamour pick (AI + satellites + Antarctica). Its official brief, verbatim: *"Develop an AI/ML-enabled decision support platform capable of forecasting Antarctic sea-ice concentration, predicting iceberg trajectories, and identifying safe and fuel-efficient navigation routes for research vessels using satellite, oceanographic and meteorological datasets."* Transportation & Logistics is one of the smallest themes in SIH 2026, with only **7 statements**.

**We found no polar, Antarctic, cryosphere or NCPOR-sponsored problem statement in SIH 2023, 2024 or 2025.** Targeted searches for `"SIH25" Ministry of Earth Sciences NCPOR polar Antarctic` returned zero results. MoES has been a recurring sponsor through **CMLRE** (SIH25042, SIH25043) and **INCOIS** (SIH25036, SIH25039), but the seven-statement polar block appears to be **NCPOR's first significant SIH engagement**. **Practical consequence: there is no "what won last year" template for this problem** — the ~20 current repos and 6 videos are the only competitive reference material that exists. (Caveat: sih.gov.in's PS lists are unreadable to us, so this is a strong inference, not a proof.)

### 9.5 The sponsor is silent

**We found no MoES or NCPOR acknowledgement of any PS59 team, and no SIH 2026 mentoring session for the polar block.** NCPOR's public output in this window is entirely unrelated (World Ocean Science Congress 2026, Indian Arctic Expedition 2026-27 call, internship results, 26th Foundation Day, BRICS Ocean & Polar working group, COMNAP AGM). Its official accounts (@ncaor_goa on X, @ncpor.goa on Instagram, MoES on Facebook) surfaced no SIH 2026 post. **Team BYTE DRAGON's video tagging @NCPORGoaVasco and @MoESGOI shows 11 likes and no visible reply from either institution.**

This contrasts sharply with 2025: **INCOIS ran a public online mentoring session on 24 Nov 2025 for teams on SIH25039**, and that PS produced a national winner (Team 7Mod3) and a published paper. **OUR ANALYSIS:** with no mentoring session, the 277-character PS text is the *only* authoritative spec for SIH26059, so evaluator expectations are unanchored. Caveat: X is largely closed to unauthenticated scraping, so a low-visibility tweet cannot be ruled out.

### 9.6 Where this field is publicly visible, and where it is dark

GitHub and YouTube carry essentially all of it. **LinkedIn is a closed channel** — Firecrawl returns a hard tool-level refusal (*"We apologize for the inconvenience but we do not support this site"*), and a `site:linkedin.com` search returned **25 results, all individual student profile pages** (JECRC, Galgotias, Amrita, RGUKT Nuzvid, SSN, GGSIPU) with **zero PS26059 mentions in any snippet**. Those profiles keyword-match on "SIH" + "student" + "India"; treating any as a competitor would be fabrication. **Devpost, Devfolio, Medium, Dev.to, Hashnode, Reddit, Notion, X, SlideShare, Canva and Google Slides returned nothing for SIH26059** across many searches. Instagram carries real PS59 traffic but it is **influencer content, not team content** — including a third-party (not MoES) promotional post: *"MINISTRY OF EARTH SCIENCES IS CHALLENGING YOU… SIH 2026 \| SERIES 18 — PS 57… PS 58… **PS 59 – Antarctic Ice & Iceberg Navigation**."* That pump is itself intelligence: PS59 is being marketed to a national audience as an attractive, uncrowded pick, which argues for a larger and later-forming field than the early-September repo landscape suggested.

**View counts are tiny and subscriber counts are 2–8 across all six videos** (45 to 498 views). These are throwaway submission channels, not marketing; nobody in this field has an audience. The highest-viewed PS59 video in existence is **SagarDrishti at 498 views — and its own transcript ends with the presenter admitting it is a wireframe.**

### 9.7 Dates, and what they imply

Video uploads cluster in two waves — **late August** (SagarDrishti 29 Aug, Synapse 28–29 Aug) and **7–15 September** (LIKITH 7 Sep, POLARPATH 12 Sep, IAVNS 13 Sep, MitoNaut 15 Sep). Last-commit dates cluster identically: POLAR-NAV-X 9 Sep, POLAR-AI 13 Sep, SagarDrishti 14 Sep, PolarPath 15 Sep, nawddeep and PolarNav both 18 Sep. **We have no repository creation dates for any project** (GitHub API blocked, dates not rendered on repo pages), so we cannot rank by recency, distinguish active from abandoned repos, or tell which of the two duplicate mirror pairs is canonical.

---

## 10. Sources

### SIH 2026 — official and process
- SIH 2026 Guidelines (College SPOC) PDF — https://www.sih.gov.in/letters/SIH2026-Guidelines-College-SPOC.pdf
- Prior-edition SIH Guidelines (College SPOC) — https://www.sih.gov.in/letters/Guidelines-College-SPOC.pdf
- SIH 2026 problem statement portal (403 to this session) — https://www.sih.gov.in/sih2026PS
- PIB Press Release ID 2201244, Ministry of Education, 9 Dec 2025 (SIH 2025 totals) — https://www.pib.gov.in/PressReleasePage.aspx?PRID=2201244
- SIH 2025 screening results, batch 4 (MoES/CMLRE shortlists) — https://sih.gov.in/sih2025/screeningresult-batch4
- SIH 2025 screening results, batch 2 — https://sih.gov.in/sih2025/screeningresult-batch2
- SIH 2025 Grand Finale shortlist — https://sih.gov.in/sih2025/shortlisted-teams-grand-finale
- SIH 2024 Grand Finale results — https://www.sih.gov.in/sih2024/sih2024-grand-finale-result

### SIH 2026 — third-party mirrors and analysis
- zaidsayyed.in SIH26059 page (21 ideas, 30 Sep deadline) — https://zaidsayyed.in/tools/sih-problem-statements/sih26059
- zaidsayyed.in SIH 2026 trends (board distribution) — https://zaidsayyed.in/tools/sih-problem-statements/trends
- zaidsayyed.in PS index — https://zaidsayyed.in/tools/sih-problem-statements
- zaidsayyed.in internal-hackathon playbook — https://zaidsayyed.in/blog/sih-2026-internal-hackathon-guide
- SIH Buddy SIH26059 scorecard (150–360 projection, R²=0.25) — https://www.sihbuddy.in/ps/SIH26059
- SIH Buddy SIH26060 — https://www.sihbuddy.in/ps/SIH26060
- SIH Navigator (GITAM) mirror — https://sih-gitam.vercel.app/
- codehuntersacademy PS explorer (20 Sep deadline) — https://www.codehuntersacademy.com/sih-2026-ps
- blinknbuild master catalogue PDF (30 Sep deadline) — https://www.blinknbuild.in/Assets/SIH_2026_All_226_Problem_Statements_Master_Catalogue.pdf
- Galgotias University LMS PS poster, 25 Aug 2026 (0/500 snapshot) — https://gulms.galgotiasuniversity.org/pluginfile.php/14/mod_forum/attachment/19978/Poster_PS_LMS.pdf?forcedownload=1
- iQube KCT PS explorer — https://sih.iqubekct.ac.in/problems/
- Kaggle: SIH 2025 Team Outcomes dataset — https://www.kaggle.com/datasets/aryanprajapati33/smart-india-hackathon-2025-team-outcomes
- Kaggle: SIH 2024 PS with Winning Teams & Solutions — https://www.kaggle.com/datasets/adharshinikumar/sih-2024-ps-with-winning-teams-and-solutions
- SIH Winners Vault 2023–2025 (GitHub) — https://github.com/Aadiii00/SIH-Winners-PPt-and-Sources
- INCOIS SIH25039 mentoring session (Instagram) — https://www.instagram.com/p/DReojfnjVqY/

### Verified PS59 competitor repositories
- nawddeep/SIH26059 — https://github.com/nawddeep/SIH26059
  - README — https://raw.githubusercontent.com/nawddeep/SIH26059/main/README.md
  - `baseline_comparison.json` — https://raw.githubusercontent.com/nawddeep/SIH26059/main/seaice_forecast/output/evaluation/baseline_comparison.json
  - `drift_twostage_metrics.json` — https://raw.githubusercontent.com/nawddeep/SIH26059/main/iceberg-drift/output/drift_twostage_metrics.json
  - `route_engine.py` — https://raw.githubusercontent.com/nawddeep/SIH26059/main/shipNavigation/backend/app/route_engine.py
  - `verify.sh` — https://raw.githubusercontent.com/nawddeep/SIH26059/main/verify.sh
- PolarNav — https://github.com/kanishka11082007-sys/SIH26059 · mirror https://raw.githubusercontent.com/ghildiyalnitin067-a11y/SIH-2026/main/README.md
  - models tree — https://github.com/kanishka11082007-sys/SIH26059/tree/main/backend/models
  - `sea_ice_metrics.json` — https://raw.githubusercontent.com/kanishka11082007-sys/SIH26059/main/backend/models/sea_ice_metrics.json
  - `iceberg_metrics.json` — https://raw.githubusercontent.com/kanishka11082007-sys/SIH26059/main/backend/models/iceberg_metrics.json
  - `polar_routing_engine.py` — https://raw.githubusercontent.com/kanishka11082007-sys/SIH26059/main/backend/src/optimization/polar_routing_engine.py
  - `requirements.txt` — https://raw.githubusercontent.com/kanishka11082007-sys/SIH26059/main/requirements.txt
  - suspended backend — https://sih-2026-059.onrender.com/openapi.json · live frontend — https://frontend-pearl-nine-74.vercel.app/
- POLARPATH AI (Team AXIOM) — https://github.com/afrazhussaina5/polarpath-ai
  - `requirements.txt` (4 lines) — https://raw.githubusercontent.com/afrazhussaina5/polarpath-ai/main/requirements.txt
  - `sea_ice_service.py` — https://raw.githubusercontent.com/afrazhussaina5/polarpath-ai/main/backend/services/sea_ice_service.py
  - `route_optimizer.py` — https://raw.githubusercontent.com/afrazhussaina5/polarpath-ai/main/backend/services/route_optimizer.py
  - backend tree — https://github.com/afrazhussaina5/polarpath-ai/tree/main/backend
- IAVNS (Team BYTE DRAGON) — https://github.com/Manthanvinzuda007/Team-SIH-
  - branches — https://github.com/Manthanvinzuda007/Team-SIH-/branches/all · `prod/IAVNS` tree — https://github.com/Manthanvinzuda007/Team-SIH-/tree/prod/IAVNS
  - `EVALUATION.md` — https://raw.githubusercontent.com/Manthanvinzuda007/Team-SIH-/prod/IAVNS/EVALUATION.md
  - `AUDIT_REPORT.md` — https://raw.githubusercontent.com/Manthanvinzuda007/Team-SIH-/prod/IAVNS/AUDIT_REPORT.md
  - `route_service.py` — https://raw.githubusercontent.com/Manthanvinzuda007/Team-SIH-/prod/IAVNS/backend/app/services/route_service.py
  - backend `requirements.txt` — https://raw.githubusercontent.com/Manthanvinzuda007/Team-SIH-/prod/IAVNS/backend/requirements.txt
  - main README (150 B stub) — https://raw.githubusercontent.com/Manthanvinzuda007/Team-SIH-/main/README.md
- SagarDrishti — https://github.com/APC2005-dev/SagarDrishti
  - README — https://raw.githubusercontent.com/APC2005-dev/SagarDrishti/main/README.md · `ml/` tree — https://github.com/APC2005-dev/SagarDrishti/tree/main/ml
  - `ml/routing/engine.py` — https://raw.githubusercontent.com/APC2005-dev/SagarDrishti/main/ml/routing/engine.py
- POLAR-AI — https://github.com/2911vedant/polar-ai
  - README — https://raw.githubusercontent.com/2911vedant/polar-ai/main/README.md
  - `risk_service.py` — https://raw.githubusercontent.com/2911vedant/polar-ai/main/backend/app/services/risk_service.py
  - `route_service.py` — https://raw.githubusercontent.com/2911vedant/polar-ai/main/backend/app/services/route_service.py
  - `demo_service.py` — https://raw.githubusercontent.com/2911vedant/polar-ai/main/backend/app/services/demo_service.py
  - `MODEL_EVALUATION.md` — https://raw.githubusercontent.com/2911vedant/polar-ai/main/MODEL_EVALUATION.md
  - `DATA_SOURCES.md` — https://raw.githubusercontent.com/2911vedant/polar-ai/main/DATA_SOURCES.md
- POLAR-NAV-X — https://github.com/Rugved-dev18/POLAR-NAV-X · mirror https://github.com/ASIM-05/POLAR-NAV-X
  - README — https://raw.githubusercontent.com/Rugved-dev18/POLAR-NAV-X/main/README.md
  - `routing/` tree (empty) — https://github.com/Rugved-dev18/POLAR-NAV-X/tree/main/routing
  - `ml/train_xgboost.py` — https://raw.githubusercontent.com/Rugved-dev18/POLAR-NAV-X/main/ml/train_xgboost.py
- SIH26059-ML-Trajectory — https://github.com/SadhikaRana/SIH26059-ML-Trajectory
  - `src/` tree — https://github.com/SadhikaRana/SIH26059-ML-Trajectory/tree/main/src
  - `src/baseline.py` — https://raw.githubusercontent.com/SadhikaRana/SIH26059-ML-Trajectory/main/src/baseline.py
  - `requirements.txt` — https://raw.githubusercontent.com/SadhikaRana/SIH26059-ML-Trajectory/main/requirements.txt
- tanishqbafna695/sih26059-antarctic-navigation-dss — https://github.com/tanishqbafna695/sih26059-antarctic-navigation-dss
  - README — https://raw.githubusercontent.com/tanishqbafna695/sih26059-antarctic-navigation-dss/main/README.md
  - `requirements.txt` — https://raw.githubusercontent.com/tanishqbafna695/sih26059-antarctic-navigation-dss/main/requirements.txt
- 125112056-art/antarctic-nav-system — https://github.com/125112056-art/antarctic-nav-system
  - README — https://raw.githubusercontent.com/125112056-art/antarctic-nav-system/master/README.md
  - `requirements.txt` (only `torch` in the field) — https://raw.githubusercontent.com/125112056-art/antarctic-nav-system/master/requirements.txt
- Thin tail — https://github.com/Aviral-P/SIH26059 · https://github.com/ku-prince/SIH26059 · https://github.com/sai16bit/SIH26059-Adaptive-Mission-Aware-Navigation · https://github.com/jeevanasreechinta/polar-safe
- Pass-one-only (unverifiable in pass two) — https://github.com/Astralis-SIH26059/Sea-Ice_concentration · https://github.com/Pawan-19012006/BOREAS · https://github.com/Kishore17E/POLAR-X
- Unconfirmed POLARIS candidates — https://github.com/patilshrutika1803/POLARIS · https://github.com/CaptainHarryJones/POLARIS.ai
- GitHub search used — https://github.com/search?q=SIH26059&type=repositories · https://github.com/search?q=antarctic+iceberg+navigation+decision+support&type=repositories&s=updated&o=desc
- Team that rejected PS59 — https://raw.githubusercontent.com/singharchit1801-gif/SIH26/main/README.md
- Adjacent-PS repos (not competitors) — https://raw.githubusercontent.com/Dev-Lahrani/polar-ops-digital-twin/main/README.md · https://raw.githubusercontent.com/deepeshaggarwal123/POLARIS-COMMAND/master/README.md · https://raw.githubusercontent.com/avinash8386edu-eng/SIH-2026/main/README.md

### Verified PS59 competitor videos and apps
- IAVNS / Team BYTE DRAGON — https://www.youtube.com/watch?v=E9UGUEaXoUk
- SagarDrishti / Team Cinephiles, AIT Pune — https://www.youtube.com/watch?v=IiyMPd4pjUY
- MitoPolarNet / Team MitoNaut (Team ID 125159) — https://www.youtube.com/watch?v=DdOu6ajgwco
- POLARPATH AI / Team Axiom — https://www.youtube.com/watch?v=4W7-kpP6dHQ
- LIKITH Gv — https://www.youtube.com/watch?v=UpsdegTxY-c
- Team Synapse core (likely) — https://www.youtube.com/watch?v=fWAjy3AHtRs
- DAKSHIN (probably PS26060) — https://www.youtube.com/watch?v=4DfreJ9SzUg
- POLARNAV 26059 deployed app — https://sih-26059-ai-enabled-antarctic-sea-pi.vercel.app/
- Instagram PS59 theme-breakdown reel — https://www.instagram.com/reel/DcoDuV1TFBn/ · https://www.instagram.com/reel/DZh_cHmtVnm/
- Third-party MoES PS bundle promotion — https://www.instagram.com/p/DceN1MSDwNZ/

### External systems — routing, forecasting, risk
- BAS PolarRoute (canonical) — https://github.com/bas-logist/PolarRoute · legacy https://github.com/bas-amop/PolarRoute · https://github.com/antarctica/PolarRoute
- PolarRoute docs — https://bas-logist.github.io/PolarRoute/ · PyPI — https://pypi.org/project/polar-route/
- PolarRoute-server — https://github.com/bas-logist/PolarRoute-server
- PolarRoute method paper (2022 preprint) — https://arxiv.org/abs/2209.02389 · PDF https://arxiv.org/pdf/2209.02389
- MeshiPhi dataloaders — https://bas-amop.github.io/MeshiPhi/dataloaders/overview/
- IceNet ecosystem — https://icenet.ai/ · core library https://github.com/icenet-ai/icenet · research code https://github.com/tom-andersson/icenet-paper
- IceNet, BAS project page ("forecasts **Arctic** sea ice") — https://www.bas.ac.uk/project/icenet/
- IceNet, Andersson et al., *Nature Communications* 12 (2021) — https://www.nature.com/articles/s41467-021-25257-4
- WWF Arctic (claims pan-Antarctic capability) — https://www.arcticwwf.org/the-circle/stories/navigating-a-changing-ice-world/
- IceNet pretrained-weights worked example — https://acocac.github.io/environmental-ai-book/polar/modelling/polar-modelling-icenet.html
- IMO POLARIS, MSC.1/Circ.1519 — https://www.imorules.com/MSCCIRC_1519.html · RIV/RIO tables https://www.imorules.com/GUID-2C1D86CB-5D58-490F-B4D4-46C057E1D102.html
- PAME POLARIS overview — https://pame.is/ourwork/arctic-shipping/polaris/
- IMO Polar Maritime Seminar 2022, POLARIS update — https://wwwcdn.imo.org/localresources/en/About/Events/Documents/Polar%20Maritime%20Seminar%202022%20presentations/Day%201/10_James_Bond-IMO%20POLARIS%20Update%20%20Current%20Usage%20and%20Status.pdf
- ABS Polar Code case study (RIO contour maps) — https://ww2.eagle.org/content/dam/eagle/case-studies/marine-polar-oldendorff-casestudy.pdf
- OpenDrift `openberg` source — https://opendrift.github.io/_modules/opendrift/models/openberg.html · example https://opendrift.github.io/gallery/example_openberg.html
- Drift+Noise IcySea — https://driftnoise.com/icysea.html · ice charts docs https://driftnoise.com/icysea/docs/ice-charts.html
- Oceanwide Expeditions adopts IcySea — https://oceanwide-expeditions.com/press/oceanwide-expeditions-to-utilize-ice-report-software-icysea-aboard-expedition-vessels
- USNIC Antarctic iceberg data — https://usicecenter.gov/Products/AntarcIcebergs · Antarctic product suite https://usicecenter.gov/Products/AntarcHome
- BYU/NIC Antarctic Iceberg Tracking Database — https://www.scp.byu.edu/iceberg/default.html
- Altiberg (Ifremer/CERSAT) — https://cersat.ifremer.fr/fr/Data/Latest-products/Altiberg-a-database-for-small-icebergs
- SCAR International Iceberg Database — https://scar.org/library-data/data/iceberg-database
- USCG International Ice Patrol — https://www.mycg.uscg.mil/News/Article/3028040/international-ice-patrol-11-decades-of-monitoring-the-northern-atlantic-waters/ · products https://www.navcen.uscg.gov/north-american-ice-service-products · NSIDC G00874 https://nsidc.org/data/g00874/versions/1
- NRC Canada iceberg drift forecasting — https://nrc-publications.canada.ca/eng/view/ft/?id=adbcf560-222c-4d31-81c4-4e1c5edebddf
- OSI SAF sea-ice products — https://osi-saf.eumetsat.int/products/sea-ice-products · Copernicus Marine product https://data.marine.copernicus.eu/product/SEAICE_GLO_SEAICE_L4_NRT_OBSERVATIONS_011_001/description · "It's free" https://marine.copernicus.eu/news/copernicus-marine-service-redistributes-sea-ice-products-osi-saf
- SIPN South — https://fmassonn.github.io/sipn-south.github.io/ · post-season assessment https://polarmet.osu.edu/YOPP-SH/SIPN-South_2021-2022_postseason.pdf · *J. Climate* 34(15) https://journals.ametsoc.org/view/journals/clim/34/15/JCLI-D-20-0965.1.pdf
- Canadian Ice Service chart descriptions (egg code) — https://www.canada.ca/en/environment-climate-change/services/ice-forecasts-observations/latest-conditions/products-guides/chart-descriptions.html
- MET Norway cryo — https://cryo.met.no/en/understanding-ice-charts · SIGRID-3 (JCOMM TR23) — https://data-donnees.az.ec.gc.ca/api/file?path=%2Fice%2Fproducts%2Fice-charts%2Fdocumentation%2FJCOMM_TR23_SIGRID3.pdf · IICWG https://nsidc.org/iicwg/iicwg-business
- BAS Polar View — https://www.bas.ac.uk/from-ice-to-navigation/ · Esri ArcNews on USNIC use https://www.esri.com/about/newsroom/arcnews/ships-use-sea-ice-and-iceberg-maps-to-navigate-in-polar-regions

### External literature
- **Mishra et al. (2021), "Investigating optimum ship route in the Antarctic… between Bharati and Maitri", *Polar Science* 30, 100696** — https://www.researchgate.net/publication/351841705_Investigating_optimum_ship_route_in_the_Antarctic_in_presence_of_sea_ice_and_wind_resistances_-_A_case_study_between_Bharati_and_Maitri
- **Zvyagina & Zvyagin (2022), "A model of multi-objective route optimization for a vessel in drifting ice", *RESS* 218, 108147** — https://www.sciencedirect.com/science/article/abs/pii/S0951832021006359
- **Li et al. (2024), sea-ice collision risk index (ICRI) + FVO, *Ocean Engineering* 311, 118843** — https://www.sciencedirect.com/science/article/pii/S0029801824021814
- Tran et al. (2023), "Pathfinding and optimization for vessels in ice: A literature review", *Cold Reg. Sci. Technol.* — https://www.sciencedirect.com/science/article/pii/S0165232X23001064
- "Arctic weather routing: a review…", *Frontiers in Marine Science* (2023) — https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1190164/full
- Życzkowski & Śzłapczyński (2023), collision-risk-informed weather routing, *RESS* 232 — https://www.sciencedirect.com/science/article/pii/S0951832022006305
- Review of risk analysis models for ice-covered waters, *Safety Science* (2021) — https://www.sciencedirect.com/science/article/pii/S092575352100179X
- Six-year circum-Antarctic icebergs dataset (2018–2023), *ESSD* 18, 147 (2026) — https://essd.copernicus.org/articles/18/147/2026/essd-18-147-2026.pdf
- Sentinel-1 grounded-iceberg detection, F1 > 0.91, *ESSD* 18, 6017 (2026) — https://essd.copernicus.org/articles/18/6017/2026/
- Budge & Long (2017), BYU Antarctic iceberg database, *IEEE JSTARS* — https://ieeexplore.ieee.org/iel7/4609443/4609444/08247260.pdf
- "Vessel Safety Navigation Under the Influence of Antarctic Sea Ice", *JMSE* (2025) — https://www.mdpi.com/2077-1312/13/7/1267
- Sea-ice drift CDR 1991–2020, *ESSD* 15, 5807 (2023) — https://essd.copernicus.org/articles/15/5807/2023/
- VTT ice-aware maritime route optimization — https://cris.vtt.fi/en/publications/a-method-for-ice-aware-maritime-route-optimization/
- Routeview, *Int. J. Digital Earth* 15 (2022) — https://www.tandfonline.com/doi/full/10.1080/17538947.2022.2126016

### Commodity evidence
- Statoil/C-CORE Kaggle Iceberg Classifier Challenge — https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge · leaderboard https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/leaderboard · code/copy-chains https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/code
- 93 GitHub repos matching `statoil iceberg classifier` — https://github.com/search?q=statoil+iceberg+classifier&type=repositories
- `sea ice convlstm` (3 repos) — https://github.com/search?q=sea+ice+convlstm&type=repositories · `"sea ice" forecasting deep learning` (6) — https://github.com/search?q=%22sea+ice%22+forecasting+deep+learning&type=repositories · `iceberg drift model` (4) — https://github.com/search?q=iceberg+drift+model&type=repositories
- jerrywn121/Arctic_SIC_prediction (ConvLSTM with baselines) — https://github.com/jerrywn121/Arctic_SIC_prediction
- NASA-MERRA2 PolarTrax Antarctic iceberg drift tool (pre-dates SIH 2026) — https://github.com/NicolaJB/NASA-MERRA2-PolarTraxAntarctic-IcebergDriftModelTool
- searoute-py — https://github.com/genthalili/searoute-py · eurostat/searoute — https://github.com/eurostat/searoute · scgraph — https://pypi.org/project/scgraph/
- 52°North Weather Routing Tool — https://52north.org/software/software-components/weather-routing-tool/
- A\* over a raster (GIS StackExchange) — https://gis.stackexchange.com/questions/73191/python-gdal-best-way-to-read-in-raster-for-use-in-a-search-algortihm
- StormGeo weather routing — https://stormgeo.com/shipping/weather-routing-and-voyage-optimization · NAPA — https://www.napa.fi/software-and-services/ship-operations/napa-fleet-intelligence/voyage-optimization/ · industry overview (isochrone/DP/GA, 3–7%) — https://metacad.io/en/knowledge/weather-routing/
- satellite-image-deep-learning/techniques catalogue — https://github.com/satellite-image-deep-learning/techniques
- JetBrains, "How to win a hackathon: notes from the judging table" (2026) — https://blog.jetbrains.com/ai/2026/06/how-to-win-a-hackathon-notes-from-the-judging-table/

### India / NCPOR institutional context
- NCPOR — Bharati station — https://ncpor.res.in/antarcticas/display/377-bharati
- NCPOR news feed (no SIH content) — https://ncpor.res.in/news/view/980 · https://ncpor.res.in/news/view/924
- India planning its first polar research vessel — https://polarjournal.net/india-is-planning-its-own-polar-research-ship/

---

**Caveats that bound this report.** No repository creation dates, commit histories or star counts exist for most repos (GitHub API and HTML blocked; findings rest on Firecrawl scrapes and `raw.githubusercontent.com`). No repo-wide code search was possible, so every "absent" is "absent from the directories enumerated". `sih.gov.in` returned 403 on every attempt, so every per-PS count is third-party. LinkedIn is refused at tool level. Overlap between the ~20 GitHub repos, the 6 videos and the deployed apps is largely unresolved, so the competitor count is a range, not a census. Two projects verified in the first pass (Astralis/Polaris, BOREAS/SETU) could not be re-located in the second. The JAIR 2025 PolarRoute paper, the full text of Mishra et al. (2021), and whether MSC.1/Circ.1519 has been revised since 2016 all remain unread.

**Next step.** This report establishes the competitive landscape only. The user will supply their own solution next, for direct comparison against the competitors, matrix and gaps above.

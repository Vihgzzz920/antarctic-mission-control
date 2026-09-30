# Build the discharge decision, cut everything else

**Decision document for SIH 2026 PS59 / SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System (Ministry of Earth Sciences / NCPOR). Written 20 September 2026 for a six-person team of second-year CSE/Data Science students with roughly one week of build time. All competitor and dataset observations are point-in-time as of 20 September 2026.**

Build **three** things that no other team on this problem statement was found building, and refuse to build anything else. The three are: a **discharge-site decision table** grounded in NCPOR's own charter tender NCPOR/14(102)/21, which schedules **115 days of which only ~34 are transit and ~70 are spent stationary discharging cargo onto fast ice** ([41-ISEA Tender](https://www.indianembassyrome.gov.in/docs/1624279190_1285_41%20ISEA%20Tender-Ice%20Class%20Vessel.PDF)); a **vessel performance and POLARIS cost surface** ported from the British Antarctic Survey's open-source `SDA.py` and IMO [MSC.1/Circ.1519](https://www.nautinst.org/static/uploaded/2f01665c-04f7-4488-802552e5b5db62d9.pdf), so that every fuel, speed and risk number on screen has a published derivation rather than an invented weight; and a **three-run forecast-value protocol** that plans routes on a forecast and scores them on the ice that actually happened. Everything else in your proposed pipeline — the sea-ice model, the drift model, A\*, the risk raster, the dashboard — is commodity: four independent competitor repositories contain genuine `heapq` A\*, and the field's median architecture is exactly yours. The second-year constraint changes the plan materially: it downgrades the Polar Code compliance export from a feature to a slide, kills the risk-budget and Pareto-sweep modes, forbids any deep learning, and forces the interactive demo onto a **25 km grid with `pyastar2d`** and a **precomputed JSON lookup** so nothing computes live. Two verified findings restructure the build before Day 1: **folium cannot render EPSG:3031 at all**, so `streamlit-folium` is dead for the Antarctic map and the team must paste NASA's maintained Leaflet + Proj4Leaflet example instead; and **Open-Meteo's marine/wave API returns all nulls inside the ice zone**, so a wave panel would render blank on stage. The honest bottom line on Indian data: **no Indian-hosted dataset was verified as downloadable and physically valid for the Antarctic sea-ice zone within this window** — MOSDAC is behind a Keycloak SSO wall, is v1.0-beta with 2015 metadata, and is built from three inputs that are physically invalid under pack ice. The Indian dimension must come from geography, from the charter, and from a sourced ISRO-scatterometer lineage in a dataset you actually use — not from a forced data feed a domain judge can puncture in one question.

---

## STEP 1 — Select our differentiators

### How the candidates score against the eight criteria

Scored 1–5 against the user's criteria, with the second-year constraint applied as a hard filter rather than a discount. "Time" is calendar days for *this* team, i.e. roughly 1.5× what a final-year team would need.

| Candidate | 1-week feasibility | Data availability | Low data dependency | Demo impact | Technical credibility | Defensibility | Connection to PS59 | Measurable | Time | **Verdict** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Discharge-site / mission-feasibility decision** | 4 | 5 | 5 | **5** | 4 | **5** | **5** | 4 | 2.5 d | **BUILD — lead** |
| **POLARIS + PolarRoute-SDA vessel performance** | **5** | 5 | **5** | 4 | **5** | 4 | 5 | 5 | 1.5 d | **BUILD** |
| **Three-run forecast-value protocol** | 4 | 4 | 4 | 2 | **5** | **5** | 4 | **5** | 1 d | **BUILD** |
| Lead-time-dependent model selection | 5 | 4 | 4 | 2 | 4 | 3 | 3 | 5 | 0.2 d | Build as a **by-product** of the above, do not pitch separately |
| Monte Carlo route re-scoring | 5 | 5 | 5 | 4 | 3 | **2** | 4 | 4 | 1 d | Build (needed for the mission table) — **do not claim as a differentiator** |
| Risk-segment decomposition | 5 | 5 | 5 | 4 | 3 | 3 | 3 | 3 | 0.5 d | Build as a **UI feature**, not a pitched differentiator |
| Risk-budget / epsilon-constraint route modes | 2 | 5 | 5 | 3 | 4 | 3 | 3 | 3 | 2 d | **DO NOT BUILD** |
| Polar Code / Indian Antarctic Act compliance exports | 3 | 4 | 5 | 3 | 4 | 3 | 4 | 1 | 1 d | **DO NOT BUILD as code** — one two-column slide only |

The three that survive are the three the evidence points at, and the recommended selection is adopted unchanged. The reasoning for each downgrade is in Step 12; the short version is that a second-year team with four beginners has roughly **five productive engineering days** after environment setup and before rehearsal, split across six people who have never integrated a system before. Two excellent differentiators that survive Q&A beat eight half-built ones, and the only currency at an MoES-sponsored grand finale — where **4–5 teams per problem statement advance** and the sponsoring organisation "isn't obligated to declare a winner" ([SIH 2026 Guidelines, College SPOC](https://www.sih.gov.in/letters/SIH2026-Guidelines-College-SPOC.pdf)) — is a number that survives being questioned.

### Differentiator 1 — The discharge-site decision

**Why build it.** It is the only thing in this entire research programme that another team cannot reach by adding a model. It is an objective function derived from a procurement document, and copying it requires reading NCPOR/14(102)/21, ASMA 6 and a 2012 NCPOR news item and then re-architecting. Within the SIH timeline that is a rebuild, not a copy.

**The exact problem solved.** NCPOR's indicative schedule buys 115 days: **16 days Cape Town → Larsemann Hills, 40 days on station, 8 days Larsemann → India Bay, 30 days on station, 10 days home** ([41-ISEA Tender, Annexure-I](https://www.indianembassyrome.gov.in/docs/1624279190_1285_41%20ISEA%20Tender-Ice%20Class%20Vessel.PDF)). Transit is 34 days — **30% of the charter**. A system that optimises only transit is optimising the minority of the contract. Three verified facts make the discharge a genuine decision and not a fixed waypoint. The charter party requires the vessel to proceed to the point "closest to landing area **chosen by Master of the vessel(s) in consultation with the Leader of the Expedition** for overboard discharge of Charterer's cargo/equipment **on fast ice for haulage to the inland area**" — a named, joint, contractual choice. **ASMA 6 records that at the Larsemann Hills "no anchorages or barge landings are designated"**, that vessels anchor **~5 nautical miles offshore**, and that **India hauls by tracked vehicle over fast ice until mid-December and by barge after ice melt** ([ASMA No. 6 Management Plan](https://www.env.go.jp/nature/nankyoku/kankyohogo/database/jyouyaku/asma/asma_pdf_en/ASMA06_en.pdf)). And it has already gone wrong, in India: in **April 2012 a slab of shelf ice roughly 2 km × 1 km adjoining the "Indian Barrier" — the cargo unloading site about 100 km north of Maitri — broke away and drifted to sea carrying Indian tank containers and Russian cargo and fuel**, fragmenting into about ten icebergs, with RADARSAT showing "no signature of presence of metallic objects"; the same item records that **heavy sea ice had already prevented both Russian and Indian vessels from reaching their barriers in February/March 2012** ([NCPOR news/view/147](https://ncpor.res.in/news/view/147)).

**Data needed and where it realistically comes from.** Sea-ice concentration forecast (your own XGBoost output on the Bremen/NSIDC grid); live iceberg positions (USNIC CSV, `https://usicecenter.gov/File/DownloadCurrent?pId=134`); station coordinates (**Bharati 69°24.41'S, 76°11.72'E**, from ASMA 6); candidate discharge-site coordinates (**hand-digitised by you** — the exact Indian Barrier coordinates are unverified, only "Princess Astrid Coast, ~100 km north of Maitri"); the charter schedule (the tender PDF, transcribed into a Python dict). Nothing paid, nothing gated, nothing that needs an account.

**Difficulty: medium. Time: 2.5 days** for this team, of which about a day is defining the candidate sites and the scoring table and a day is wiring it to the router you already have. There is no new algorithm — you call `find_route()` once per candidate site.

**Demo impact: very high.** It is the only demo in the room that opens with the sponsor's own contract instead of a rotating globe.

**What makes it different from the common approach.** Every observed competitor's UI vocabulary is origin–destination: POLARPATH's on-screen scenario is a vessel bound for Bharati at 12 knots, IAVNS takes origin/destination inputs, SagarDrishti's wireframe takes departure and destination port. **We did not find a publicly accessible PS59 project, or any public system, that formulates the Antarctic resupply problem as discharge-site selection under a charter window** — with the standing caveat that commercial systems behind sales walls cannot be ruled out.

**What is genuinely ours versus existing method.** The *routing* is not ours — BAS PolarRoute is production, MIT-licensed, Dijkstra with Newton crossing-point refinement ([arXiv:2209.02389](https://arxiv.org/abs/2209.02389)), and **Mishra et al. (2021) already published Dijkstra route optimisation between Bharati and Maitri** from PDEU and ISRO's Space Applications Centre ([*Polar Science* 30, 100696](https://www.sciencedirect.com/science/article/pii/S1873965221000736)). What is ours is the **decision variable, the candidate-site enumeration, the comparison table and the feasibility scoring** — an operationalisation, not a discovery, and it must be presented as such.

### Differentiator 2 — Vessel performance and POLARIS as the cost surface

**Why build it.** Because it is the cheapest possible conversion of "why 0.35?" into "because the IMO says so for this ice class", and because at ~150 lines plus a lookup table it is the highest value-per-hour item available to a beginner team.

**The exact problem solved.** Every fuel and ETA number in the observed field is an assertion. IAVNS claims "8–15% fuel reduction" and "6–14 hours saved" with no derivation; POLARPATH displays "fuel 45.5" from a service whose only non-stdlib import is `math`; the only code-verified risk weights anywhere in the field are POLAR-AI's invented `{"sea_ice": 0.35, "iceberg": 0.30, "weather": 0.20, "ocean": 0.15}`. Independent PS-specific analysis names the same hole: *"without a vessel ice class and resistance model your route optimises a difficulty you invented"* ([SIH Buddy SIH26059](https://www.sihbuddy.in/ps/SIH26059)).

**Data needed.** Sea-ice concentration (you have it), a thickness field (any of the flawed ones, with its error stated), the POLARIS Table 1.3 RIV matrix (a 12 × 12 array of integers, published in MSC.1/Circ.1519), and a vessel beam and force limit (BAS's published PC5 SDA config, used as-is).

**Difficulty: easy. Time: 1.5 days** including the ice-class toggle and the sanity checks.

**Demo impact: high — and it contains the single visual peak of the whole demo.** NCPOR's tender requires ice class "**equivalent to 1-A-super or better having double skinned hull**". PolarRoute's SDA is PC5; a competitor's on-screen vessel is PC5. In **Thick First-Year Ice**, POLARIS Table 1.3 gives **PC5 an RIV of 0 and IA Super an RIV of −2**, so at 10/10 concentration a PC5 scores **RIO = 0, normal operation**, while an IA Super scores **RIO = −20, operation subject to special consideration**. Flipping one dropdown on stage visibly shrinks the accessible corridor and can change which discharge site is recommended — a ten-second, India-specific result derived from two public documents.

**What is genuinely ours.** Nothing about the equations. The resistance law, the closed-form speed inversion and the fuel polynomial are BAS's, ported and cited. POLARIS is the IMO's, from 2016, and ABS already ships RIO contour maps as a commercial product ([ABS case study](https://ww2.eagle.org/content/dam/eagle/case-studies/marine-polar-oldendorff-casestudy.pdf)). **Ours is the ice-class comparison, the honest error band, and the refusal to use Table 1.4.** Never say "we invented POLARIS integration".

### Differentiator 3 — The three-run forecast-value protocol

**Why build it.** It costs one day, it is a loop rather than an algorithm, and it is the design an earth-science-ministry judge will recognise immediately as correct.

**The exact problem solved.** MAE tells an operator nothing. A forecast can beat persistence on MAE and produce identical routing decisions, in which case it is operationally worthless. The reviewed routing literature does not run this experiment: an Arctic weather-routing review records that experiments "use actual historical data from November 2020 but **treat sea ice data as if perfectly known**", a "hindcast scenario rather than forward forecasting with uncertainty quantification" ([*Frontiers in Marine Science* 2023](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1190164/full)).

| Run | Planned on | Scored on | Meaning |
|---|---|---|---|
| **A — perfect prognosis** | Observed fields at t+1…t+n | The same observed fields | Upper bound. Not achievable operationally. |
| **B — operational** | **Your forecast** initialised at t | Observed fields at t+1…t+n | **The honest number.** |
| **C — persistence-planned** | Today's ice frozen forward | Observed fields at t+1…t+n | The no-model baseline router. |

**Value of the forecast = cost(C) − cost(B)**; **cost of forecast error = cost(B) − cost(A)**. Report alongside them the **failure rate**: the fraction of voyages where the route planned on the forecast, scored on the observed field, violates a hard constraint. A route that is cheap on average and violates a constraint 8% of the time is not deployable, and reporting that number separates an engineering team from a demo team.

**Data needed.** A held-out set of initialisation dates with matching observations — i.e. nothing you do not already have.

**Difficulty: easy-medium. Time: 1 day** given a working router and forecast. **Demo impact: medium** — it is a table, so lead with the map and hold this for Q&A, where it is decisive.

**What is genuinely ours.** The experimental design as applied to this problem. It is not a novel idea in forecast verification — the "value versus accuracy" distinction is old — but **we did not find any PS59 competitor, and did not find it in the reviewed routing literature, running a route planned on a forecast and scored on an observation.**

### The BUILD list

1. **Differentiator 1** — candidate discharge sites, comparison table, feasibility scoring.
2. **Differentiator 2** — PolarRoute-SDA resistance, speed inversion, fuel polynomial; POLARIS Table 1.3 RIO with the PC5 / PC7 / IA Super toggle; hard constraints at SIC > 80% and depth < 10 m.
3. **Differentiator 3** — the three-run protocol plus the lead-time-dependent model selector that falls out of it.
4. **Commodity core, built competently and never pitched:** data ingest and the 25 km common grid; XGBoost sea-ice forecast against four baselines; iceberg layer with a constant-velocity error cone; `pyastar2d` A\* over the POLARIS-derived cost surface; Monte Carlo route re-scoring with a spatially correlated field; the Streamlit + embedded Leaflet dashboard.
5. **Four cheap features that make the demo un-fakeable, ~half a day total:** risk-segment decomposition ("the worst 50 nm carries 43% of this route's risk, dominant driver: ice"); the **node-expansion count and re-solve latency** printed on screen; the ice-class dropdown; the uncertainty-field layer toggle.

### The DO NOT BUILD list

Polar Code 11.3 and Indian Antarctic Act exports as code (one slide instead); risk-budget / epsilon-constraint modes; a Pareto weight sweep; NAMOA\*; NSGA-II; CVaR optimisation; MPC re-planning; any Sentinel-1 SAR processing; any ConvLSTM, U-Net, LSTM or transformer; hydrodynamic vessel simulation; AIS calibration; operational-grade WMO ice classification; Lindqvist with real hull geometry; a React SPA; Docker; cloud deployment; a scheduled live data ingest. Full reasoning in Step 12.

---

## STEP 2 — Final MVP architecture

Your proposed flow put the mission decision at the end. **It moves to the front.** The discharge-site loop is the outer loop; routing is the inner function it calls. This single change is what makes the same block diagram compute something different from every competitor's identical block diagram.

```
                     ┌─ M0. Charter context (static dict, from the tender) ─┐
                     │                                                       │
M1 Ingest → M2 Grid → M3 Ice forecast ─┐                                     │
                    └→ M4 Iceberg layer ┤                                    │
                                        ├→ M5 Vessel/POLARIS cost surface ──┐│
                                        │                                   ││
              ╔═════════════════════════╧═══════════════════════════════════╧╧═════╗
              ║  M8. DISCHARGE-SITE LOOP  — for each candidate site s in S:         ║
              ║        M6 route(s) → M7 Monte Carlo re-score(s) → metrics row(s)    ║
              ╚════════════════════════════════════╤════════════════════════════════╝
                                                   ↓
                              M9. Mission comparison table + recommended site
                                                   ↓
                              M10. Dashboard (map + table + sliders)

              M11. Forecast-value protocol  — offline, runs M3/M6 three ways, produces a static table
```

| # | Module | Input | Processing | Output | Technology | Type | Essential? |
|---|---|---|---|---|---|---|---|
| **M0** | Charter context | Tender PDF (read by a human) | Hard-code the 115-day schedule, the 66–70°S / 06–80°E box, ice class, barge/crane specs | `charter.py` dict | plain Python | rule-based | **Yes** — it is Differentiator 1's spine |
| **M1** | Data ingest | URLs | `requests` fetch + retry + shape assertion; cache to `data/raw/` | NetCDF / GeoTIFF / CSV on disk | `requests`, `xarray`, `netCDF4` | — | **Yes** |
| **M2** | Common grid | Raw fields | Reproject/subset to one EPSG:3031 grid; coarsen 6.25 km → 25 km with `.mean()` for concentration and **`.max()` for hazards**; land/shallow mask from GEBCO subset | `grid.npz` (H×W float32 stack) | `numpy`, `pyproj`, `scipy.ndimage` | rule-based | **Yes** |
| **M3** | Sea-ice forecast | SIC history | Per-cell XGBoost on lagged SIC + climatology + ERA5 wind/temp; **lead-time selector** picks persistence at lead 1 | SIC(t+τ) for τ = 1,3,5,7 d | `xgboost`, `sklearn` | **ML** | **Yes** |
| **M4** | Iceberg layer | USNIC CSV + BYU tracks | Parse DMS → decimal; constant-velocity extrapolation; growing error cone σ_b(t) = √(σ₀² + (v_err·t)²); rasterise a hazard field with `.max()` | berg positions + hazard raster | `pandas`, `numpy` | physics-lite / rule-based | **Yes** |
| **M5** | Vessel + POLARIS | SIC, thickness proxy, ice class | PolarRoute `R_ice`, closed-form speed inversion, fuel polynomial; thickness → WMO stage → RIV lookup → RIO; hard mask SIC > 80%, depth < 10 m | speed(cell), fuel-rate(cell), RIO(cell), passable(cell) | pure `numpy` | **physics + documented rule** | **Yes — Differentiator 2** |
| **M6** | Routing | cost raster, start, goal | 8-connected A\* on a float32 weight array | polyline of (lat, lon), node count, solve ms | `pyastar2d` | optimisation | **Yes** |
| **M7** | Uncertainty | fixed polyline, μ and σ fields | N = 200 spatially correlated perturbations; re-score the *fixed* polyline; percentiles | ETA P10/P50/P90, P(blocked), worst-decile ETA | `numpy`, `scipy.ndimage` | statistical | **Yes** (feeds M9) |
| **M8** | Discharge-site loop | candidate sites, windows | For each site: M6 + M7; plus haul distance, mode (tracked-vehicle vs barge), iceberg exposure, ice-edge persistence | one metrics row per (site, window) | plain Python | optimisation / rule-based | **Yes — Differentiator 1** |
| **M9** | Mission table | M8 rows | Rank, flag infeasible, expose the trade-offs — **no single black-box score** | comparison table + recommendation | `pandas` | rule-based | **Yes** |
| **M10** | Dashboard | everything | Map (EPSG:3031 Leaflet in an iframe) + table + discrete sliders reading a precomputed JSON | the demo | `streamlit` + raw Leaflet | — | **Yes** |
| **M11** | Forecast-value | held-out dates | Runs A/B/C, scores all three on observation | static table + bar chart | `pandas` | evaluation | **Yes — Differentiator 3**, offline |

**Two modules deliberately absent.** There is no wave/swell module, because Open-Meteo's marine API returns nulls inside the ice zone (Step 3) and pretending otherwise would put a blank panel on stage. There is no ocean-current module, because every candidate source is either gated or physically invalid under pack ice; the current term is folded into the iceberg error cone as an uncertainty, which is the honest place for it.

---

## STEP 3 — Data

All access mechanics below are **verified live on 20 September 2026**. Where a prior assumption was wrong, the correction is marked. Use only these routes.

| Data | Example source | Resolution | Time coverage | Format | Why we need it | MVP priority |
|---|---|---|---|---|---|---|
| **Sea-ice concentration, live** | `https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/s6250/2026/sep/Antarctic/asi-AMSR2-s6250-20260919-v5.4.tif` | 6.25 km | daily, ~1-day latency | GeoTIFF, **~280 KB/day** | The live ice field on the demo map | **P0 — critical path** |
| **Sea-ice concentration, archive** | `https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/2026/` (`sic_pss25_YYYYMMDD_am2_v06r00.nc`) | 25 km | 1978?–present, **~6-day latency** | NetCDF, ~300–410 KB/day | Multi-year training archive for M3 | **P0** |
| **Sea-ice extent + climatology** | `https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v4.0.csv` + `S_seaice_extent_climatology_1981-2010_v4.0.csv` | 1-D index | daily, updated **same day** | CSV, 1.83 MB | Free 1981–2010 anomaly baseline, zero computation | **P1** |
| **Icebergs, live** | `https://usicecenter.gov/File/DownloadCurrent?pId=134` | point | **weekly**, all 33 bergs updated 18 Sep 2026 | CSV | Live berg layer; the D15 cluster | **P0** |
| **Icebergs, historical** | `https://www.scp.byu.edu/data/iceberg/consolidated_database_v8.0.zip` | point, ~daily | **1978 → 22 Apr 2025** | 4.1 MB zip of CSVs | Trajectory training and the drift baselines | **P0** |
| **Wind / temperature, live** | `https://api.open-meteo.com/v1/forecast?latitude=-67&longitude=40&hourly=wind_speed_10m,temperature_2m` | model grid | forecast | JSON, no key | Live met at route points | **P1** |
| **Wind / temperature, archive** | `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` | 0.25° | 1979–present | Zarr, **`token='anon'`** | ML features for M3 | **P1** |
| **Bathymetry / land mask** | `https://download.gebco.net/` area subset (**not** the 4 GB global file) | 15 arc-sec | static | NetCDF/GeoTIFF, few MB | The depth < 10 m hard constraint and the land mask | **P1** |
| **Ice basemap tiles** | `https://geos.polarview.aq/geoserver/wms` (EPSG:3031, no auth) | WMS | current | raster tiles | Credible operational ice basemap with **zero NetCDF processing** | **P0 — highest leverage** |
| **Antarctic basemap** | `https://gibs.earthdata.nasa.gov/wmts/epsg3031/best/` | WMTS | daily | tiles, no key | The polar map itself | **P0** |
| **ASPA/ASMA polygons** | `https://data.aad.gov.au/eds/5599/download` | vector | static | shapefile, tens of KB | Protected-area avoidance | P2 — **risky, snippet-only** |
| **Ocean currents** | OSCAR / CMEMS | 0.25° | — | — | — | **P3 — do not build** |
| **MOSDAC currents** | `https://www.mosdac.gov.in/global-ocean-surface-current` | 0.25° | daily | NetCDF | — | **RED — do not use** |

### Per-item honesty

**Bremen AMSR2 is the lowest-friction source in the whole list and belongs on the critical path.** It is a plain Apache directory listing with no key, four files per day, and **no missing days in 1–19 September 2026**; files land at ~06:00–06:40 UTC for the previous day ([Bremen s6250 listing](https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/s6250/2026/sep/Antarctic/)). A full year of daily GeoTIFFs is ~100 MB. Take the `.tif`, not the `.hdf` — same data, one-fifth the size. **Unverified:** the CRS and grid dimensions were never confirmed, because no file was opened. Run `rasterio.open(f).crs` and `.shape` on the first file and do not guess. The `_nic.png` and `_visual.png` quicklooks are instant demo visuals needing no processing at all.

**The NSIDC CDR is at V6, not V5.** `G02202_V5` **returns HTTP 404**; there is no such directory. Any tutorial, blog post or LLM-generated snippet written against V5 will fail silently in a loop. Hard-code `G02202_V6` and assert on HTTP status. Two further traps are confirmed: **9 and 10 August 2026 are missing entirely** — the listing jumps from `20260808` to `20260811` — and five August files (14, 15, 17, 18, 19) are ~80 KB smaller than their neighbours, suggesting partial coverage. A naive `for date in date_range: download(date)` loop **will crash or silently produce corrupt arrays**. Wrap fetches in try/except and sanity-check the valid fraction after load; ten lines of defensive code prevents a live-demo failure. **Unverified:** grid shape, CRS, variable names and the land/coast/lake flag values inside the NetCDF. In this product family masks are encoded as out-of-range values, and treating them as concentration will badly corrupt extent calculations. **Open one file, print `flag_values` and `flag_meanings`, and do not guess.**

**USNIC is current, not stale.** The prior finding of a uniform 4 September date reflected the previous weekly issue: as of 20 September **all 33 tracked icebergs carry Last Update = 09/18/2026**. Cadence is weekly, so do not claim daily iceberg updates. Two small parsing jobs the team should not discover at 3 a.m.: **Location is DMS** (`54° 29' S / 26° 33' W`) and must be converted to decimal degrees handling both the degree symbol and the hemisphere suffix, and **Size is an "L x W" string** while `Area (sqNM)` is already numeric and is the cleaner field for sizing map symbols. Units are nautical miles throughout (1 NM = 1.852 km). **Unverified:** the CSV's internal column names and the shapefile's CRS — neither file was downloaded.

**The BYU database's "Oct 2023" page header is a false alarm, and acting on it would have been costly.** The page header is stale but the file is not: **`consolidated_database_v8.0.zip` is 4.1 MB and covers 1978 through 22 April 2025** ([BYU Antarctic Iceberg Tracking Database](https://www.scp.byu.edu/data/iceberg/database1.html)). Columns are `ascat_1, ascat_2, date, nic_1, nic_2, nic_3, oscat_1, oscat_2, qscat_1, qscat_2, size_1, size_2`; the date is **`YYYYDDD`**, parsed with `pd.to_datetime(df.date, format='%Y%j')`; positions are decimal degrees positive north and east; sizes are km. Most cells on any row are blank, so you must **coalesce across `nic_*`, `ascat_*`, `qscat_*`, `oscat_*`** rather than reading a single lat/lon pair. The source states its own caveat plainly — gaps arise from "the occasional loss of contrast between the iceberg and surrounding area **during summer months**", with interpolation filling them — which is exactly the honest line to have ready for a domain judge, and exactly the reason the yardstick warning in Step 11 matters. Update cadence is "once or twice a year", which explains the stale header without alarm. **The architecture is: BYU for historical training, USNIC weekly for the live layer**, and the two are consistent by construction because BYU ingests NIC positions.

**ARCO ERA5 needs `token='anon'` and the `ar/` path.** The official snippet, verbatim from the [google-research/arco-era5 README](https://github.com/google-research/arco-era5):

```python
import xarray
ds = xarray.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    chunks=None,
    storage_options=dict(token='anon'),
)
```

**Do not use the `co/` stores.** They are on reduced Gaussian grids or, worse, spherical-harmonic coefficients; decoding those is a research project and would consume the entire week. The `ar/` store is the only listed option on a plain equiangular lat-lon grid. The 2.05 PB headline size is irrelevant — Zarr is chunked and `xarray` + `gcsfs` fetches only the chunks intersecting your slice; a Southern Ocean box of 10m winds and 2m temperature over a few years is tens to hundreds of MB. Because `token='anon'` works there is **zero registration latency**, which removes the biggest schedule risk in the ERA5 path. Install `xarray`, `zarr`, `gcsfs`. **ERA5 is the historical/training wind source, never "today's" winds** — use Open-Meteo for those.

**Open-Meteo is a split verdict and this is one of the most consequential findings in the whole programme.** The **forecast** API works perfectly at 67°S: a live call to `https://api.open-meteo.com/v1/forecast?latitude=-67&longitude=40&hourly=wind_speed_10m,temperature_2m&forecast_days=1` returned wind 12.4–55.0 km/h and temperature −16.3 to −22.0 °C with no key, snapping to a grid cell 0.03° away. The **marine/wave** API at the identical point returned **`wave_height` and `wave_period` as `[null, null, …]` for all 24 hourly steps**, while the same endpoint at **57°S, 40°E returned real 6.70 m rising to 12.50 m seas**. That is not an outage; it is the wave model's sea-ice mask, and it is permanent. **A wave panel in the 66–70°S box will render blank on stage in front of a judge who knows exactly why.** The correct design is to use waves only on the open-ocean approach leg north of the ice edge, where they are the real hazard, and to drive the in-ice hazard model from concentration, icebergs and wind. That split is physically correct and is a stronger design than pretending waves matter under pack ice.

**GEBCO: do not download the 4 GB global file.** The current product is GEBCO_2026 at 15 arc-seconds; the global netCDF is **4 GB zipped, 7.0 GB uncompressed** ([GEBCO Gridded Bathymetry Data](https://www.gebco.net/data-products/gridded-bathymetry-data)). On a student connection that single download could consume half a day. Use `https://download.gebco.net/` area subset or CEDA OPeNDAP and get a few MB. The grid is public domain.

**MOSDAC is RED, and this needs saying plainly to the team before anyone gets attached to it.** Its metadata does claim −90 to 90° latitude, so on paper it covers the box. In practice: fetching `https://www.mosdac.gov.in/opendata/ocean_surface_current/` **redirects to a Keycloak SSO login page**; the product is **Version 1.0 (beta) with a metadata date of 15 September 2015**; and it is derived from **three inputs none of which is valid under sea ice** — altimeter-derived absolute dynamic topography, ASCAT scatterometer winds, and Reynolds OISST — while its own stated Limitations section mentions only equatorial and near-coastal issues and **says nothing about sea ice or polar validity at all** ([MOSDAC Global Ocean Surface Current](https://www.mosdac.gov.in/global-ocean-surface-current)). Its cited validation is entirely tropical Indian Ocean. Altimetry cannot retrieve sea surface height through pack ice, ASCAT winds over ice are contaminated, and OISST under ice is filled from an ice-concentration relationship rather than measured. **The −90/+90 bounding box is a nominal grid extent, not a validity claim**, and presenting MOSDAC currents at 67°S as "an Indian data source" is precisely the claim a domain judge punctures. Registration turnaround is unverified — nobody attempted to create an account — which alone disqualifies it from a seven-day critical path.

**The honest statement on Indian data, and the honest alternative.** **No Indian-hosted dataset was verified as downloadable and physically valid for the Antarctic sea-ice zone within the project window.** That is a statement about our search, not a proof of absence — INCOIS and NCPOR's own portals were not investigated. The alternative is stronger than a forced feed. **Frame the whole system on the Maitri/Bharati corridor**: the tender's own operating box is **66–70°S, 06–80°E**, Bharati sits at **69°24.41'S, 76°11.72'E**, and the verified current iceberg cluster — **D15A at 66°38'S/81°55'E (885.59 sqNM, the largest tracked berg), D15B, D15C, D15D and D34** — sits at roughly 67°S, 79–82°E, directly adjacent to that sector, with **D23 at 69°26'S/74°43'E, C18B at 67°00'S/47°22'E, C18C at 68°28'S/39°04'E, B09G at 68°11'S/41°30'E and D37 at 69°13'S/36°22'E** inside it. That is real, current, Indian-relevant geography needing no Indian feed. And for a sourced ISRO lineage claim: **BYU's iceberg database explicitly incorporates ISRO Oceansat-2 OSCAT and ScatSat OSCAT-2 scatterometer data** — a defensible statement of Indian satellite contribution to a dataset you are actually using, rather than a badge.

### Risk register for data

| Item | Risk | Mitigation |
|---|---|---|
| CDR missing/undersized days | **High** — will crash a naive loop | try/except + valid-fraction assertion; skip and log |
| Bremen CRS / grid size | **Medium** — unverified | 30-second `rasterio.open().crs` check on Day 1 |
| CDR flag values | **High** — mis-handling corrupts extent | Print `flag_values`/`flag_meanings` before writing any mask |
| USNIC CSV columns / shapefile CRS | Medium — unverified | Fail loudly if the response is not CSV |
| AADC ASPA download | **Medium** — snippet-only, file never retrieved | P2 only; drop without consequence |
| OSI SAF | **Unverified entirely** | Treat as RED; Bremen already covers the need |
| Venue Wi-Fi on demo day | **Highest single risk** | Cache tiles and all data locally on Day 5 |

---

## STEP 4 — The ML component

**Prediction task.** Per-cell sea-ice concentration at lead times τ ∈ {1, 3, 5, 7} days over the 25 km EPSG:3031 Southern Ocean grid, restricted to cells south of ~50°S.

**Target variable.** `SIC(x, y, t+τ)` ∈ [0, 1], from **NSIDC G02202_V6** — one product, version-pinned, for both training and scoring. Never mix Bremen and CDR in one evaluation.

**Input features, per cell.** Lagged concentration `SIC(t)`, `SIC(t−1)`, `SIC(t−3)`, `SIC(t−7)`; the local spatial mean and standard deviation over a 3×3 and 5×5 neighbourhood at time t; day-of-year encoded as `sin(2πd/365)` and `cos(2πd/365)`; the day-of-year climatology value for that cell **computed on training years only**; the anomaly `SIC(t) − climatology(d)`; ERA5 10 m wind speed and direction and 2 m temperature at t, plus 3-day means; distance to the nearest ice edge (0.15 contour) at t; latitude and longitude. That is ~20 features, one row per cell per date — a wide but shallow tabular problem, which is exactly what gradient boosting is for.

**Training data.** Five to ten years of daily CDR fields. At 25 km the southern grid is roughly 10⁵ cells; subsample to ~2,000 cells per date, stratified so that the marginal ice zone is over-represented relative to its 3% area share, and you get ~3.6 million rows from ten years — comfortable for XGBoost on a laptop and trivial in Colab.

**Split — temporal, no leakage.** Train on whole seasons up to a cut date; validate on the next season; test on the most recent held-out season. **Never split randomly across dates**, because adjacent days are near-duplicates and a random split leaks tomorrow into today. Fit normalisation statistics, the climatology and the damping coefficients **on the training period only** — fitting climatology on the full record is leakage into the *baseline*, which paradoxically makes the baseline look unfairly strong and is still wrong. **Publish a `splits.json`** listing the exact initialisation dates of every held-out sample plus the code that reproduces the baseline numbers. That file is the cheapest credibility win available and it takes twenty minutes.

**Model candidates and the verdict.** `RandomForestRegressor` and `XGBRegressor`. **XGBoost/RF suffices, and deep learning is forbidden for this team.** The evidence is unambiguous. Across every competitor dependency manifest read, **exactly one repository in the entire field declares `torch`**; the only committed trained artefacts anywhere are `RandomForestRegressor` `.joblib` files; and one team's own [`EVALUATION.md`](https://raw.githubusercontent.com/Manthanvinzuda007/Team-SIH-/prod/IAVNS/EVALUATION.md) states that a **"ConvLSTM was *not trained* as 8 days of data is fundamentally insufficient"** while its YouTube description sells one. Meanwhile **IceNet** — a 25-CNN U-Net ensemble in *Nature Communications* with downloadable MIT-licensed weights ([Andersson et al. 2021](https://www.nature.com/articles/s41467-021-25257-4)) — forecloses any "first deep-learning sea-ice forecast" framing. Saying out loud *"we did not train a ConvLSTM because we do not have the data to justify one"* is worth more than a ConvLSTM you did not train, and it is a sentence a domain judge will respect.

**Baselines — four of them, and they are the point.** Climatology (day-of-year, fitted on training years); **persistence** (the reference); anomaly persistence; and **damped anomaly persistence**, climatology plus a per-cell lag-τ anomaly autocorrelation α(τ) fitted on training years only. Damped anomaly persistence dominates both persistence and climatology by construction when α is fitted honestly, which is exactly why it is the hard benchmark. It is also a legitimate talking point that **SIPN South explicitly considered and rejected damped anomaly persistence as its headline Antarctic benchmark** as "not always straightforward to implement" given the seasonal cycle ([Massonnet et al. 2023](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1148899/full)) — a student team that implements it has done something the flagship Antarctic intercomparison chose not to.

**Metrics, masks, and the masking argument that is the cheapest rigour available.** A domain-wide MAE is close to meaningless because **~97% of the Antarctic domain is open water**; a competitor's own README records **MAE 0.003 over open water against 0.097 in the marginal ice zone**, noting that "a single overall MAE of 0.019 hides that entirely". Their headline figure is therefore roughly 94% determined by trivially predictable cells. The literature's answer is to mask: **IceNet zero-weights everything outside a dynamically-sized "active grid cell region" and normalises its headline metric by that region — "Binary accuracy = (1 − IIEE / area of active grid cell region) × 100%"**. Report **four masks side by side — full domain (specifically to show it is uninformative), active region, marginal ice zone (0.15 ≤ SIC < 0.80), and an ice-edge band** — and **mask on the observation, never on the forecast**, because masking on the forecast lets the model choose its own exam. Lead with the **length-normalised IIEE in kilometres**, `D_IIEE_AVG = 2·A_IIEE / (L_O + L_M)` ([Melsom et al. 2019, *Ocean Science* 15:615](https://os.copernicus.org/articles/15/615/2019/)), because "our ice edge is on average X km out of place, persistence is Y km out of place" is legible to everyone in the room. Also area-weight every metric: a polar stereographic cell's area varies and an unweighted grid mean is subtly wrong.

| Metric | Computed on | Why |
|---|---|---|
| MAE, RMSE, R² | each of four masks | Standard, and the four-way split is the argument |
| **IIEE (10⁵ km²)** | observation mask | Integrated ice-edge error, the operational metric |
| **D_IIEE_AVG (km)** | observation mask | Length-normalised — **lead with this** |
| Skill score vs named reference | MIZ mask, per lead | The five-slot framing formula |
| **Constraint-violation rate** | route level | Feeds Differentiator 3 |

**Significance, honestly.** Use a **paired moving-block bootstrap on the per-initialisation score difference**, with block length set to the decorrelation time of the *difference* series justified from its autocorrelation function, and report the **effective sample size rather than the nominal one**. With n = 300 daily initialisations and a 20-day block you have roughly **15 independent blocks** — enough to support a per-lead comparison, not enough to support a seasonal or cross-sector generalisation claim. Say which claim your n supports and which it does not.

**Horizon.** 1–7 days. Do not go beyond 7; you have neither the data nor the need, and the charter decisions you are supporting operate on a 1–14 day horizon where the ice-edge position matters.

**Converting the negative result into an engineering decision.** Expect to lose to persistence at lead 1. The evidence says so directly: one competitor's committed [`baseline_comparison.json`](https://raw.githubusercontent.com/nawddeep/SIH26059/main/seaice_forecast/output/evaluation/baseline_comparison.json) records ice-zone MAE of **persistence 0.02051 versus model 0.04662 at +1 day (n = 300)**, with climatology at 0.11246, and the pattern holds at +3 days. Day-to-day SIC autocorrelation is near 1; **no model should beat persistence there.** So ship a **lead-time-dependent model selector**: persistence at lead 1, the learned model in the middle band, climatology at long leads, with the crossover chosen on validation data. It is roughly fifteen lines:

```python
def forecast(field_history, lead):
    if lead <= CROSSOVER_LOW:        # chosen on validation, typically 1 day
        return persistence(field_history)
    if lead >= CROSSOVER_HIGH:       # typically beyond 7 days
        return climatology(doy + lead)
    return xgb_model.predict(features(field_history, lead))
```

That single function turns "our model loses at one day" from an embarrassment into **"we use the baseline where the baseline wins"** — an engineering decision, and the fifth slot of the framing formula: reference named, region scoped, lead scoped, margin quantified with uncertainty, **failure named and explained**. Volunteering the loss before you are asked is the move that survives an MoES panel, where a team showing an unvalidated "96% accuracy" is one question away from collapse.

---

## STEP 5 — Routing system

### Grid, nodes, edges

**Representation.** A single float32 `numpy` array of shape (H, W) in **EPSG:3031**, at **25 km for the interactive demo (~325 × 325 ≈ 106,000 cells)** and at 6.25 km (~1300 × 1300 ≈ 1.69 M cells) for **one precomputed hero route** shown as a static comparison. Nodes are cell centres; edges connect the 8 neighbours. Impassable cells are marked by setting their weight to `np.inf`.

**Why 25 km, and how to defend it.** It is not an excuse. Ship routing decisions operate on scales of tens of kilometres and hours to days; 6.25 km detail exceeds the useful resolution of a route recommendation and **exceeds the skill of any short-range ice forecast**. The performance argument is secondary but decisive for a one-week build:

| Grid | Cells | `pyastar2d` (est.) | Pure-Python A\* (est.) | Verdict |
|---|---|---|---|---|
| 6.25 km full Antarctic | 1300 × 1300 | ~0.15–0.3 s | ~30 s – 3 min | Precomputed hero route only |
| 12.5 km | 650 × 650 | ~0.05 s | ~8–40 s | Comfortable |
| **25 km** | **325 × 325** | **~0.01 s** | ~1–5 s | **Recommended for the demo** |

**The solver is `pyastar2d`.** `pip install pyastar2d`; C++ A\* over a numpy array; published benchmarks of **1802 × 1802 in 0.29 s and 4008 × 4008 in 0.83 s**; wheels for Windows x86-32/64, macOS 10.9+ x86-64 and 11.0+ ARM64, Linux glibc and musl on x86-64 and ARM64; **latest release 1.1.4, 22 February 2026** ([pyastar2d on PyPI](https://pypi.org/project/pyastar2d)). That wheel coverage is exactly what a mixed Windows/Mac/Linux student team needs. Fallbacks if it fails: `tcod.path`, or `scipy.sparse.csgraph.dijkstra` over a sparse adjacency built from the grid. **Do not write your own A\* in C**; do not write your own A\* in Python either, beyond a 30-line reference implementation used once to cross-check `pyastar2d` on a toy grid.

**The coarsening correctness rule, which a domain judge may well check.** Downsample concentration with `.mean()` because averaging concentration is the right operation — but downsample **hazard and iceberg fields with `.max()`**, because averaging a hazard away is a real correctness bug:

```python
sic_25  = sic_6.coarsen(x=4, y=4, boundary='trim').mean()   # correct for concentration
hazard_25 = hazard_6.coarsen(x=4, y=4, boundary='trim').max()  # correct for hazards
```

**Crop in projected coordinates, not lat/lon.** Maitri (~70.8°S, 11.7°E) and Bharati (69.4°S, 76.2°E) span ~65° of longitude and Cape Town is at 34°S; in EPSG:3031 metres the corridor is a fan, not a rectangle. Keep the full circumpolar grid restricted to south of ~50°S and you avoid the antimeridian entirely.

### Cost function, with units

Per edge from cell *i* to cell *j*, with `d_ij` the great-circle-equivalent edge length in km (grid step, or grid step × √2 diagonally):

```
t_ij    = d_ij / v_j                              [hours]        v_j from the speed-in-ice inversion, km/h
f_ij    = fuel_rate(v_j, R_j) × t_ij / 24         [tonnes]       fuel_rate in t/day
rio_ij  = max(0, −RIO_j) × t_ij                   [RIO-hours]    POLARIS deficit, the primary risk scalar
berg_ij = p_encounter(j) × t_ij                   [dimensionless·hours]
ice_ij  = SIC_j × t_ij                            [ice-hours]    time-weighted, not distance-weighted
```

and the scalarised edge weight actually handed to `pyastar2d`:

```
w_ij = α·t_ij  +  β·f_ij  +  γ·rio_ij  +  δ·berg_ij  +  λ·σ_j·t_ij        (all weights ≥ 0)
w_ij = np.inf   if   SIC_j > 0.80  or  depth_j > −10 m  or  land_j  or  RIO_j < −10
```

Every term is normalised to its own P50 across the domain before weighting so the sliders are interpretable, and `pyastar2d` requires weights **≥ 1 and float32**, so the final array is `np.maximum(w.astype(np.float32), 1.0)` with infinities replaced by a very large finite value.

**Sea-ice exposure is time-weighted, not distance-weighted**, because a ship crawling at 4 kn through 8/10 ice is far more exposed than one crossing the same distance at 14 kn. The headline number to show a navigator is `HoursAboveThresh` — "9.4 h in ≥6/10 ice" — and the most defensible single risk scalar is `POLARIS_deficit = Σ Δt · max(0, −RIO)`, because it is anchored to an IMO methodology and a specific ice class rather than to an invented weighting. **These are our operationalisations, not standard metrics: no source gives an agreed formulation for "sea-ice exposure along a route", and they must be presented as ours.**

**Iceberg risk.** Do not use closest approach alone. Widen an error cone with lead time, `σ_b(t) = √(σ₀² + (v_err·t)²)`, compute `p_encounter = exp(−d²/(2σ_b²))` at closest approach, and combine as `P_any = 1 − Π(1 − p_b)` — **stating openly that independence across bergs is an approximation**, since bergs in one drift stream share current error.

**Weather and currents.** Wind and temperature enter through Open-Meteo at route points and feed the ML features and the wind-resistance term. **Waves do not enter inside the ice zone** — they are null there (Step 3) — and currents do not enter as a field at all; their effect is absorbed into `v_err` in the iceberg error cone, which is the honest place for an unmodelled advection error.

### A\* versus Dijkstra

Use **A\* with the admissible heuristic `h = (great-circle distance to goal) / v_max × α`**, i.e. the minimum possible remaining time cost. Because every non-time term in `w` is non-negative, scaling the heuristic by only the time coefficient keeps it admissible and A\* stays optimal on the static graph. Dijkstra is A\* with `h = 0` and is strictly slower here; there is no reason to use it. `pyastar2d` handles the heuristic internally with a grid-distance metric — which is an approximation in EPSG:3031, since the projection is approximately equal-area near 71°S but distorts elsewhere. **State that in one sentence rather than hiding it.**

**Do not make the state `(cell, arrival_time)`** in this build. Time-dependent A\* is a two-day job and a competitor already has a `TimeAStar` class, so it is not a differentiator either. If you do add it later, state the FIFO assumption: the shortest-path problem is polynomially solvable in FIFO time-dependent networks and **NP-hard in non-FIFO networks** ([Nannicini et al.](https://www.lix.polytechnique.fr/~liberti/bidirtimedepj.pdf)), and for a ship FIFO means *waiting never helps you arrive earlier* — which is **violated in real ice navigation**, where waiting twelve hours for a lead to open genuinely can get you there sooner.

### Route modes, without claiming they are novel

Four modes, produced by four weight vectors over the same cost function. **They are near-universal in this field — IAVNS ships four with sliders and real `heapq` A\* behind them — so do not differentiate on the feature.**

| Mode | Weights | Headline metrics shown |
|---|---|---|
| **Fastest** | α = 1, others ≈ 0 | ETA P10/P50/P90, distance, worst-decile ETA |
| **Fuel-efficient** | β = 1 | Fuel (t) **with its ±20–30% band**, mean/max SIC |
| **Safety-conservative** | γ, δ = 1 | POLARIS deficit (RIO-hours), min CPA to nearest berg (km), P(any encounter), hours in SIC > 0.6 |
| **Balanced** | user sliders | All of the above plus P(blocked) and the confidence driver named |

**How the cost changes when priorities change** is the thing to show on screen, not describe. Move the safety slider and `γ` rises; cells with RIO < 0 become expensive; the argmin path bends around the thick-ice corridor; the metrics table updates; **the node-expansion count and re-solve latency change**. Printing "re-solved in 180 ms, 4,812 nodes expanded" is cheap, impossible to fake convincingly, and directly rebuts the question a judge who has read one of these repositories will be asking — because at least one competitor's 14,435-byte `route_optimizer.py` claims "A\*, D\* Lite along with Pareto" while containing **no `heapq`, no priority queue and no search at all**, just five hand-written route builders selected by an if/elif ladder whose branch key is literally `"IB-042" in iceberg_service.icebergs`.

### The two things you must say about multi-objective routing

**Say "non-dominated candidates", never "the Pareto front".** Sweeping weights and minimising `w₁·time + w₂·fuel + w₃·risk` recovers only the **supported** efficient solutions on the convex hull of the achievable objective set. de Weck & Kim state it flatly of weighted sums: *"more seriously, **optimal solutions in non-convex regions are not detected**"*, and note that an evenly spaced weight sweep produces **clustered, not evenly spaced, solutions** ([de Weck & Kim, MIT](https://web.mit.edu/deweck/www/PDF_archive/2%20Refereed%20Journal/2_12_SMO_AWSMOO1_deWeck_Kim.pdf)). On a discrete graph this is not a corner case. The wording to use: *"We compute a set of non-dominated candidate routes by scalarised A\* over a sweep of preference weights. This recovers the supported (convex-hull) efficient solutions; routes in non-convex regions of the true efficient frontier are not guaranteed to be found."* One sentence acknowledging this is a credibility asset, standing next to a competitor claiming "Pareto-optimal multi-objective routes" from a file with no priority queue in it.

**The epsilon-constraint / risk-budget mode is what reaches beyond the sweep** — minimise time subject to risk ≤ B, approximated by running A\* on time with hard cell exclusion where `risk > B`, sweeping B. It maps perfectly onto navigator language: *"I will not accept more than X hours above SIC 0.6."* **For this team it is a Day-6 stretch item on a branch, not a commitment** (Step 12), because it is two days for beginners. If it ships, describe it as **"risk-budget masking"**, never "exact constrained shortest path" — exact resource-constrained shortest path is NP-hard in general.

---

## STEP 6 — Vessel performance and POLARIS

### The implementable spec, ported from PolarRoute's `SDA.py`

Roughly sixty lines of pure `numpy`, every term traceable to [BAS PolarRoute](https://github.com/bas-logist/PolarRoute).

**Ice resistance, concentration-native:**

```
R_ice = 0.5 · k · Fr^b · ρ_ice · B · h · V² · C^n
Fr    = V / √(g · C · h)
```

with hull parameters from the source: **slender hull k = 4.4, b = −0.8267, n = 2.0** (blunt: k = 16.1, b = −1.7937, n = 3). `C` is concentration as a fraction, `h` thickness in metres, `B` beam in metres, `V` speed.

**Speed in ice — closed-form inversion.** If `C = 0`, run at max speed. If `C > max_ice_conc`, the cell is **inaccessible**. Otherwise compute `R_ice` at max speed; if it exceeds the force limit, invert:

```
vexp = 2 · F_limit / ( k · ρ_ice · B · h · C^n · (g·h·C)^(−b/2) )
v    = vexp^( 1 / (2 + b) )
```

**Fuel, tonnes per day** (V in km/h, R in newtons):

```
Fuel[t/day] = 24 · ( 1.37247e-3·V² − 2.9601e-3·V + 0.25290433
                     + 7.75218178e-11·R² + 6.48113363e-6·R )
```

It is **additive-separable** — a calm-water speed term plus a resistance term, not a product — and it floors negative resistance at zero. **Two free sanity checks come with it**, and they are worth putting on a slide because they demonstrate the numbers are not invented: at V = 0 and R = 0 it gives **24 × 0.2529 ≈ 6.07 t/day**, a plausible hotel load for a 15,000 t research vessel; at the SDA's force limit it gives **≈ 32.4 t/day**, the right order for a polar research vessel.

**SDA reference configuration (verified from the source config):** `max_speed 26.5 km/h (≈14.3 kn), beam 24.0 m, hull_type "slender", force_limit 96,634.5 N, max_ice_conc 80%, min_depth 10 m`. The 80% cut-off and the 10 m depth floor are two free, citable hard constraints for the route graph.

**Wave resistance (Kreitner, ITTC-recommended for small wave heights):** `R_wave = 0.64 · ρ_w · C_B · H_w² · B² / L`, valid to ~2 m — **applicable only on the open-ocean approach leg**, since waves are null inside the ice zone.

### POLARIS

```
RIO = Σ ( Cᵢ × RIVᵢ )      with Cᵢ the concentration in TENTHS of each ice type
```

**Table 1.1 thresholds:** `RIO ≥ 0` normal operation; `−10 ≤ RIO < 0` elevated operational risk (for PC1–PC7; "special consideration" for classes below PC7); `RIO < −10` operation subject to special consideration. **Table 1.2 elevated-risk speed caps:** PC1 11 kn, PC2 8 kn, PC3–PC5 5 kn, below PC5 3 kn ([MSC.1/Circ.1519](https://www.nautinst.org/static/uploaded/2f01665c-04f7-4488-802552e5b5db62d9.pdf)).

**Table 1.3, the three rows that matter for this build** (the full 12 × 12 goes in a numpy array):

| Ice Class | Ice-Free | New | Grey | Grey-White | Thin FY 1st | Thin FY 2nd | Med FY <1 m | Med FY | **Thick FY** | 2nd Year | Light MY | Heavy MY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **PC5** | 3 | 3 | 3 | 3 | 2 | 2 | 1 | 1 | **0** | −1 | −2 | −2 |
| **PC7** | 3 | 2 | 2 | 2 | 1 | 1 | 0 | −1 | **−2** | −3 | −3 | −3 |
| **IA Super** | 3 | 2 | 2 | 2 | 2 | 1 | 0 | −1 | **−2** | −3 | −4 | −4 |

The ice-class result to demonstrate on stage: in **Thick First-Year Ice at 10/10 concentration**, PC5 gives `RIO = 10 × 0 = 0` (**normal operation**) while IA Super gives `RIO = 10 × (−2) = −20` (**special consideration**). NCPOR's tender requires "1A Super or better"; PolarRoute's SDA and a competitor's on-screen vessel are PC5. **Running POLARIS at the ice class NCPOR actually contracts for produces a visibly different accessible corridor, from two public documents.**

### The three-way separation the user asked for explicitly

This table goes on a slide, and it is the single most important credibility artefact in the deck.

| Layer | What it is | Examples in our system |
|---|---|---|
| **Real maritime standard / documented rule** | Published, citable, not ours | IMO POLARIS RIO formula, Tables 1.1/1.2/1.3; the Polar Code Ch. 11 voyage-planning duties; MARPOL Annex I Reg. 43 HFO ban south of 60°S; NCPOR's 115-day schedule and "1A Super or better" requirement; the SDA resistance law, speed inversion and fuel polynomial as published by BAS |
| **Our prototype assumption** | Reasoned, stated, defensible, **not validated** | Thickness → WMO ice-stage mapping (**ours**); treating each cell as a single ice type at the observed concentration; the SDA PC5 beam and force limit substituted for an unverified Indian charter vessel; grid-distance A\* on an EPSG:3031 raster; independence of iceberg encounters; the additive `μ + λσ` risk surrogate; candidate discharge-site coordinates hand-digitised; `v_err` as a proxy for unmodelled currents |
| **Our ML / model output** | Computed by us, with error bars | The XGBoost SIC forecast at leads 1–7; the constant-velocity iceberg extrapolation with its error cone; the route polyline; ETA P10/P50/P90 and P(blocked); the forecast-value numbers; the discharge-site feasibility rows |

Never let these three blur. When a judge asks "where does the fuel number come from?", the answer is a one-sentence walk down this table.

### The honest caveats, each of which must be stated

| Caveat | The statement to make |
|---|---|
| **Lindqvist's bending coefficient is disputed** | Two independently OCR'd sources render the leading coefficient as **37/64 and 27**, and the radical may have lost an `h³`. **Do not implement Lindqvist's R_B.** Use PolarRoute's law, or Riska/FSICR (`R_i = C₁ + C₂·v`, coefficients f₁ = 23 N/m², f₂ = 45.8 N/m, f₃ = 14.7 N/m, f₄ = 29 N/m², g₁ = 1537.3 N, g₂ = 172.3 N/m, g₃ = 398.7 N/m^1.5 — [Swedish Maritime Administration](https://www.sjofartsverket.se/globalassets/tjanster/isbrytning/pdf-regelverk/power_requirement_finnish-swedish_iceclass.pdf)). |
| **Table 1.4 may not be used** | The circular is explicit: the standard Table 1.3 values *"should be used unless ice decay is confirmed by ice information/visual observation by personnel on board qualified in accordance with chapter 12 of the Polar Code."* **A satellite-only system must use Table 1.3.** Saying this proves you read the circular rather than a blog post about it. |
| **The ice-type bridge is our approximation** | POLARIS wants twelve WMO ice types from ice charts. The best public Antarctic product, **OSI SAF OSI-403-d, gives only FYI / MYI / ambiguous**, states Southern Hemisphere ice shows "less differentiation in signature between the two ice types", **misinterprets large areas of pancake ice as multiyear ice**, and does **no classification during summer months** ([OSI SAF](https://osi-saf.eumetsat.int/community/stories/new-osi-saf-southern-hemisphere-sea-ice-type-product)) — and OSI SAF was **not verified at all in our access testing**, so treat it as unavailable. We therefore map thickness to WMO stage using the standard thickness bands (new <10 cm, grey 10–15, grey-white 15–30, thin FY 30–70, medium FY 70–120, thick FY >120 cm) and label the output **"RIO estimated from thickness-derived ice stage; not equivalent to an ice-chart-based RIO"**. **We found no published, validated method that does this mapping — it is our own approximation and must be presented as such.** It degrades gracefully: an error moves you one column, i.e. one RIV point, across a twelve-column table. |
| **Antarctic thickness is the weakest input in the chain** | **GIOMAS underestimates Antarctic sea-ice volume by ~38%**; satellite retrieval carries large snow-depth/density uncertainty; ASPeCt ship observations are biased thin because vessels avoid thick ice; satellite/in-situ correlation is only **0.69–0.73** ([Shi et al. 2022, *The Cryosphere* 16:1807](https://tc.copernicus.org/articles/16/1807/2022/)). Turn it into the sophisticated part: re-route at h × 0.5 and h × 1.5 and show the ETA/fuel spread. |
| **Absolute fuel is ±20–30%; relative ranking is much better** | And relative ranking is all a route optimiser needs. Benchmark the claim against a published full-scale comparison where a ship performance model predicted **870 kg/h against a measured 790 kg/h** over a 100 km leg, ~10% over-prediction ([ASPM manuscript](https://discovery.ucl.ac.uk/id/eprint/10137042/1/ASPM%20-%20final%20manuscript.pdf)). |
| **Power is not proportional to V³ in ice** | The cubic law follows from `P = R·V` with `R ∝ V²`, true only for viscous-dominated calm water. In ice the submersion term is near velocity-independent at leading order, and SFOC degrades sharply at low load — **155–225 g/kWh at ~85% MCR, roughly doubling at 7% load, reaching ~1,500 g/kWh at 1% load** ([Sustainable Ships](https://www.sustainable-ships.org/stories/2022/sfc)). Any competitor whose fuel figure scales as V³ has assumed away the problem. |
| **POLARIS is not go/no-go** | Quote the circular against yourself: any system based on this guidance *"should **not be interpreted as a 'Go/No Go' tool but as a decision support tool**."* The IMO is endorsing exactly the framing the problem statement asks for. |
| **Two items remain unverified** | **PolarRoute's configuration keys and objective-function names could not be confirmed** — both documentation hosts returned 404 — so clone the repo and read `examples/*.config.json` before asserting what PolarRoute does or does not do. And **whether MSC.1/Circ.1519 has been superseded is unconfirmed**; say *"we use MSC.1/Circ.1519 (2016), which was issued as interim guidance"*, never "no revision exists". |

### The vessel, and what not to assert about it

**MV *Vasiliy Golovnin* is confirmed as the 43-ISEA charter vessel** ([PIB PRID 1993769](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=1993769)), which also confirms she carried third-party cargo for **Princess Elizabeth (BELARE), Progress (Russian) and Novo Airport** — so NCPOR does not have sole authority over the itinerary, and the system must be presented as **advisory within a fixed multi-stop sequence**. **Her assigned ice class is UNVERIFIED.** The tender's "1A Super or better" *requirement* is verified; the specific vessel's class is not. **Do not assert a Polar Class for her.** Ship a **PC5 / PC7 / IA Super sensitivity dropdown** instead — which is both more honest and a better demo, because the corridor visibly changes when you flip it. Also note that **MARPOL Annex I Regulation 43 bans heavy grade oils south of 60°S** and the tender restates it — the vessel "should be using **Marine Gas Oil (MGO) / Marine Diesel Oil (MDO)**" — so there is **no fuel-switch optimisation to model** and any saved-fuel figure must be priced at MGO. And **do not invent a charter day rate**: no awarded contract value, day-hire or demurrage rate for any ISEA charter exists in the public record. Express savings in the tender's own units — **"N days of day hire plus victualling for 40 persons"** — and let NCPOR substitute its confidential rate.

---

## STEP 7 — Uncertainty

### The design, which is deliberately the cheap half of the problem

**Plan once; re-score N times.** Re-scoring a fixed polyline is `O(waypoints)` — 200 waypoints × 200 members is 40,000 array lookups, milliseconds in NumPy. **Re-planning per member is the expensive thing; do only the cheap thing.** The output is a distribution rather than a number: **ETA P10/P50/P90, P(route blocked), worst-decile ETA, and a confidence corridor** around the route.

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def perturb(mu, sigma, corr_cells, rng):
    """One spatially correlated perturbed SIC field."""
    grf = gaussian_filter(rng.standard_normal(mu.shape), sigma=corr_cells)
    grf /= grf.std()                       # renormalise to unit variance
    return np.clip(mu + sigma * grf, 0.0, 1.0)

def rescore(route_idx, mu, sigma, N=200, corr_cells=6, seed=0):
    rng, etas, blocked = np.random.default_rng(seed), [], 0
    for _ in range(N):
        field = perturb(mu, sigma, corr_cells, rng)
        v = speed_in_ice(field[route_idx])          # from Step 6
        if np.any(field[route_idx] > 0.80):         # hard constraint violated
            blocked += 1
        etas.append(np.sum(seg_len_km / np.maximum(v, 0.1)))
    etas = np.sort(np.array(etas))
    return dict(p10=np.percentile(etas, 10), p50=np.percentile(etas, 50),
                p90=np.percentile(etas, 90),
                worst_decile=etas[-int(0.1 * N):].mean(),   # CVaR, REPORTED not optimised
                p_blocked=blocked / N)
```

At 25 km, `corr_cells = 6` gives a correlation length of ~150 km. **The correlation length is the whole game.**

### The two rigour traps

**Trap 1 — the perturbation must be a spatially correlated random field.** Independent per-cell noise averages out along a route and produces an absurdly, obviously tight ETA distribution — the classic tell of a naive Monte Carlo, and a judge who has seen one will spot a P10–P90 spread of twenty minutes on a five-day voyage instantly. Use a **correlation length of order 100–200 km** via `gaussian_filter(randn(H, W), σ)`, renormalised to unit variance. Two lines. Do not skip them.

**Trap 2 — a variance penalty does not always produce detours, and can silently be a no-op.** The behaviour depends entirely on the σ field. **Detours appear** when σ is spatially heterogeneous and **not collinear with μ** — a corridor with low mean ice but high forecast disagreement gets avoided as λ rises, which is physically exactly what should happen in the marginal ice zone where ice-edge position spread is largest. **No detour** when σ is roughly uniform: `∫λσ ds ≈ λ·σ̄·L`, a constant times path length, so raising λ merely straightens the route. And **no new behaviour at all when σ ∝ μ** — which is exactly what happens if you derive "uncertainty" from the magnitude of the ice field rather than from real spread. Then `μ + λσ = μ(1 + λα)`, a pure rescale, **the argmin path is identical for every λ, and the slider is silently a no-op.** This is the most likely way a student team ships a broken control while believing it works. **Ship the unit test:**

```python
def test_lambda_actually_changes_the_route():
    r0 = find_route(cost(lam=0.0), START, GOAL)
    r2 = find_route(cost(lam=2.0), START, GOAL)
    assert r0 != r2, "risk slider is a no-op: check that sigma is not proportional to mu"
```

A fourth, perverse case is worth a one-line fix: a short route crossing a small high-σ patch can lose to a long detour, because the additive form charges λσ on *every* cell including calm ones. Use `μ_e + λ·max(0, σ_e − σ_floor)` so background uncertainty does not tax route length.

**Ship an uncertainty-field layer toggle alongside the slider.** When the route visibly bends around a bright patch the causal story is legible in two seconds; without the layer the route just moves and nobody knows why.

### The non-additivity you must state

Setting per-edge cost `c_e = μ_e + λ·σ_e` and running A\* minimises `Σμ_e + λ·Σσ_e`. The true mean-risk objective is **`μᵀx + c·√(τᵀx)`** ([Lim, Sommer, Nikolova & Rus, RSS 2012](https://www.roboticsproceedings.org/rss08/p32.pdf)), whose risk term is **non-additive** — which is exactly why mean-risk shortest path is a hard combinatorial problem rather than plain Dijkstra. Since `√(Σσ²) ≤ Σσ` for independent edges, **the additive form systematically over-penalises long routes through uncertain water**. It is nonetheless defensible here, because sea-ice and weather forecast errors are spatially correlated over scales far larger than a grid cell, so "perfectly correlated along the route" is the appropriate end of the range. **The assumption must be stated, in these words:** *"We use an additive risk-sensitive edge cost `c = mean + λ·std`. This is the standard mean-variance surrogate; it equals the true path-level objective under the assumption of perfectly correlated forecast error along the route, and is an upper bound otherwise. We do not claim to solve the exact mean-risk shortest path problem, which is non-additive and combinatorially hard."*

### Runtime and precision

| Configuration | N = 200 cost | Decision |
|---|---|---|
| 25 km, `pyastar2d`, re-score fixed polyline | **~2 s** | Could run live — but precompute anyway |
| 6.25 km, re-score fixed polyline | **~40 s** | Precompute |
| Any grid, **re-planning** per member | minutes to hours | **Do not attempt** |

Generating the perturbed fields is often the dominant cost — ~0.1–0.3 s each at 1300 × 1300, so another 20–60 s for 200. **Generate them once, save as a single `.npy` stack, and cache it.** Do not parallelise with `multiprocessing`: it needs the `if __name__ == "__main__":` guard on Windows and interacts badly with Streamlit's rerun model. Precompute instead.

**Never show a confidence to one decimal place unless N ≥ 1000.** A 200-sample Monte Carlo has ~±3% binomial standard error at p = 0.9, so show "**92% (±3%)**" or bucket into High/Medium/Low. And **confidence must fall as the voyage extends past the forecast horizon** — if the slider moves and confidence never changes, a judge will correctly assume it is hardcoded, as it demonstrably is in at least one competitor whose "confidence" resolves in code to the dictionary `{"0h": 94.2, "6h": 88.5, "12h": 81.3, "24h": 73.8, "48h": 63.5}`.

### What not to claim

**"Robust Pareto-optimum routing of ships utilising deterministic and ensemble weather forecasts" already exists as a published idea** ([record](https://www.researchgate.net/publication/232940270_Robust_Pareto-optimum_routing_of_ships_utilising_deterministic_and_ensemble_weather_forecasts)) — the full text could not be obtained, so **read it before claiming anything is new**. The analytical analogue is published too: a JMSE paper propagates ensemble means and standard deviations through First-Order Second-Moment reliability analysis, computing probability of constraint violation as `p(g ≤ 0) = Φ(−μ_g / σ_g)` with 90% confidence bands ([JMSE 9(12):1434](https://www.mdpi.com/2077-1312/9/12/1434)). **The claimable version is: "a standard robustness evaluation, which we implement and surface in the UI, and which hackathon-scale systems omit."** Say **"we report CVaR"**, never "we optimise CVaR" — optimising it over paths is a research problem.

### UI representation

Three elements, no more. A **shaded corridor** around the route polyline whose width is the P10–P90 spread at each waypoint, drawn as a semi-transparent polygon. A **small ETA fan chart** — a horizontal bar with P10, P50 and P90 ticks and the worst-decile marker — next to the route card, replacing the single ETA number every competitor shows. And a **text line with its driver named**: "P(blocked) = 8% (±3%) — dominant driver: the 62–65°S ice-edge band at day 4." The corridor is the thing a judge remembers; the driver sentence is the thing that survives the follow-up question.

---

## STEP 8 — Mission / discharge-site decision support

This is the lead differentiator and it must be designed carefully, because its failure mode is producing a black-box ranking score that looks exactly like every other team's weighted sum.

### The formulation

Inputs: a set of **candidate discharge sites** *S*; an **available time window** derived from the charter; **vessel characteristics** (ice class, beam, force limit, barge and crane capability); **environmental forecasts** (SIC at leads 1–7, wind, temperature); **route risk, fuel, ETA and uncertainty** from Modules M5–M7. Output: **one row per (site, window) pair** in a comparison table, with feasibility flags, **not a single score**.

### Grounding in the verified schedule

The charter is a fixed multi-stop sequence, and the system is advisory within it:

| Event | Date | Days | What the system computes here |
|---|---|---|---|
| ETD Cape Town | 23 Dec | — | Departure fixed |
| **ETA Larsemann Hills (Bharati)** | 08 Jan | **16 transit** | Route + ETA distribution; is the 16-day allowance realistic in this season's ice? |
| **ETD Larsemann Hills** | 17 Feb | **40 on station** | **Discharge-site choice at Bharati**; tracked-vehicle vs barge mode |
| **ETA India Bay (Maitri)** | 25 Feb | **8 inter-station** | The leg entirely inside the ice zone — the leg Mishra et al. studied |
| **ETD India Bay** | 27 Mar | **30 on station** | **Discharge-site choice at the Indian Barrier**; calving exposure |
| ETA Cape Town | 06 Apr | **10 return** | Slack against the 115-day total |
| **Total** | | **115** | |

Real coordinates to use: **Bharati 69°24.41'S, 76°11.72'E** and the Bharati helipad at 69°24.40'S, 76°11.59'E (ASMA 6); the ASMA boundary origin at 69°23'20"S, 76°31'0"E; **no designated anchorages or barge landings at Larsemann Hills, vessels anchoring ~5 nm offshore**; and the Maitri discharge point — the "Indian Barrier" on the Princess Astrid Coast — **about 100 km north of the station**. **The Indian Barrier's exact coordinates are UNVERIFIED**; only the ~100 km north bearing and the tender's 66–70°S / 06–80°E box are sourced. Say that on the slide, place the candidate sites yourself along the shelf edge, and label them "candidate sites digitised by us from the fast-ice/shelf edge; exact operational landing coordinates are not public."

### The five quantities computed per candidate site

For each site *s* and arrival window [t₀, t₁]:

1. **Reachability** — does a POLARIS-feasible, resistance-feasible route to *s* exist in the forecast ice field? Binary, plus the route's ETA distribution.
2. **Haul distance** from *s* to the station, in km. A hard operational cost — **100 km at Maitri's Indian Barrier is not a rounding error**, and it is the number the Expedition Leader actually trades against.
3. **Discharge mode** — tracked-vehicle-over-fast-ice versus barge. ASMA 6 records that **India hauls by tracked vehicle until mid-December and by barge after ice melt**; the predicted transition date is a function of fast-ice concentration persistence, which your forecast produces. This is a **binary operational mode with an economic consequence**, and it is a concrete India-specific output NCPOR would recognise as theirs.
4. **Ice-edge / fast-ice persistence at *s*** over the dwell window — the bearing-capacity proxy. Use persistence of SIC above a threshold across the forecast horizon, and **label it a proxy**, because you do not have bearing capacity and nobody publishes it.
5. **Iceberg and calving exposure at *s*** over the dwell window — from the drift model's error cones plus the count of tracked bergs whose 90% cone intersects a radius around *s*.

Then **P(both discharges complete within the charter window)** by Monte Carlo over the same perturbed fields from Step 7 — reusing the machinery, not building new machinery.

### The output, which is explicitly not a black-box score

```
CANDIDATE DISCHARGE SITES — India Bay sector, arrival window 25 Feb – 04 Mar
──────────────────────────────────────────────────────────────────────────────────
 Site   Reach?  ETA P50   ETA P90   Haul    Mode        Ice persist.  Bergs in    P(complete
                (days)    (days)    (km)                (7-day)       90% cone    both)
──────────────────────────────────────────────────────────────────────────────────
 IB-A   YES     7.9       9.4       104     tracked     0.91          0           0.84
 IB-B   YES     7.2       11.8      138     tracked     0.62          2           0.61
 IB-C   YES     9.6       10.3       88     barge       0.44          1           0.55
 IB-D   NO      —         —          96     —           0.88          0           —
                ↑ blocked: SIC > 0.80 corridor at 68.4°S persists through day 6
──────────────────────────────────────────────────────────────────────────────────
 Recommended: IB-A.  Trade-off vs IB-C: +16 km haul, −5 percentage points of
 ETA spread, +0.29 P(complete both).  IB-C is shorter to haul but sits in
 the melt-transition band, so it flips to barge mode and loses ice persistence.
```

**Why it must look like this.** The value is in exposing the trade-off so the Voyage Leader decides — the charter names **the Master, in consultation with the Leader of the Expedition**, as the people who make this call, and the system's job is to give them evidence, not to make the call for them. A single ranked score is the thing every competitor already ships and the thing a domain judge distrusts. Quote the IMO against yourself here: POLARIS *"should not be interpreted as a 'Go/No Go' tool but as a decision support tool."* And say the sentence that positions the whole system correctly: **"this is the charterer's instrument, not the bridge's"** — the Master holds ice-piloting authority; your output is what the Voyage Leader puts in front of the Master and NCPOR HQ.

### Scoping this to one week for beginners

Do **not** build a site optimiser, a scheduling solver, or a full charter simulator. Build: a hand-digitised list of **four candidate sites per station** in a JSON file; a loop that calls the existing `find_route()` and `rescore()` once per site; five derived columns; and a pandas DataFrame rendered with `st.dataframe`. That is the whole of Differentiator 1 in code, and it is roughly 150 lines. **The intellectual work is in the framing, and the framing is already done for you by the tender.**

---

## STEP 9 — Database, backend, frontend

### The stack decision, and the two findings that force it

**Decision: Streamlit on Python 3.12, with the Antarctic map as a raw Leaflet page embedded via `st.components.v1.html`. No React. No Docker. No conda. No folium.**

**Finding one: folium cannot render EPSG:3031.** `folium.Map`'s `crs` parameter accepts exactly four values — `'EPSG3857'`, `'EPSG4326'`, `'EPSG3395'` and `'Simple'` — and **there is no support for arbitrary EPSG codes** ([folium API reference](https://python-visualization.github.io/folium/latest/reference.html)); polar projections have been an open request since [issue #338](https://github.com/python-visualization/folium/issues/338). **Therefore `streamlit-folium` is dead for the Antarctic map.** If a team member starts down the `st_folium` path expecting a polar map, **that is a full day burned**. Say this out loud on Day 1. (`streamlit-folium` also has a reported failure mode where the page reloads every three seconds until it crashes — [issue #72](https://github.com/randyzwitch/streamlit-folium/issues/72) — though root cause is unconfirmed.)

**Finding two: NASA maintains a working Antarctic EPSG:3031 Leaflet example that can be pasted on Day 1.** It uses **Leaflet 1.9.4 + proj4js 2.20.2 + Proj4Leaflet 1.0.1**, with a copyright header reading "Copyright 2013 - 2026" ([source JS](https://raw.githubusercontent.com/nasa-gibs/gibs-web-examples/main/examples/leaflet/antarctic-epsg3031.js), [live demo](https://nasa-gibs.github.io/gibs-web-examples/examples/leaflet/antarctic-epsg3031.html)). The verbatim CRS definition:

```javascript
var EPSG3031 = new L.Proj.CRS(
  'EPSG:3031',
  '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 ' +
  '+ellps=WGS84 +datum=WGS84 +units=m +no_defs', {
    origin: [-4194304, 4194304],
    resolutions: [8192.0, 4096.0, 2048.0, 1024.0, 512.0, 256.0],
    bounds: L.Bounds([[-4194304, -4194304], [4194304, 4194304]])
  }
);
var map = L.map('map', { center: [-90, 0], zoom: 0, maxZoom: 5, crs: EPSG3031 });
```

**The 1-px tile-seam workaround in that same file is REQUIRED, not optional.** NASA patches `L.GridLayer.prototype._initTile` to add 1 px to tile width and height, working around fractional-transform anti-aliasing ([Leaflet#3575](https://github.com/Leaflet/Leaflet/issues/3575)). **If the team does not copy that block, their demo will have visible grid lines drawn across Antarctica and they will waste hours on it.** Proj4Leaflet is unmaintained — **last npm release 1.0.2, 14 August 2017** — which is not a reason to avoid it (NASA validating it against Leaflet 1.9.4 in a 2026-copyright repository is stronger evidence than a release date) but **is** a reason to **vendor the exact versions as local files in `static/lib/` rather than tracking "latest" on a CDN**.

**The highest-leverage find in the whole stack review: Polar View's GeoServer WMS.** `https://geos.polarview.aq/geoserver/wms` is **EPSG:3031, needs no authentication**, and serves **sea ice edge, sea ice concentration, SAR coverage and met.no ice charts** ([Polar View data access](https://www.polarview.aq/pages/index/data_access)). That gives you a credible operational ice basemap **with no NetCDF processing at all**, consumed through `L.tileLayer.wms()`. It de-risks the entire demo: your own NetCDF/ML pipeline becomes the differentiator layer *on top*, not the thing the demo depends on.

**The GIBS extent trap, and its mitigation.** NASA GIBS WMTS at `https://gibs.earthdata.nasa.gov/wmts/epsg3031/best/` is the other keyless option, but its tile matrix `bounds` are **±4,194,304 m from the pole**, which in polar stereographic terms is roughly the **low-50s °S**. **Cape Town is at ~34°S — well outside the tiled extent, so a Cape Town route runs off the basemap into blank space.** Mitigations in order of preference: **(1) frame the interactive map on the ice-affected portion of the voyage south of ~55°S**, which is where the decision-support value actually lies, and show the northern transit leg as a separate small inset or a static line; (2) render the full voyage only in a static Cartopy figure; (3) never attempt to make one interactive polar map cover both Cape Town and the pole — it will look broken. Verify the exact cut-off empirically on Day 1 by loading the NASA example and panning north. (Whether Polar View's WMS has a similar extent limit is **unverified** — check it on Day 1 too.)

### Why Streamlit, costed honestly

Every one of the required UI elements is a one-liner in Streamlit except the map: layer toggles are `st.checkbox`, the metrics table is `st.dataframe`, sliders are `st.select_slider`, the route overlay is data f-stringed into the map component, and re-running the optimiser is a cached function call. The team's Python is stronger than their JS, and Streamlit means **all six people can edit the UI** — which matters enormously when four of six are beginners. Deployment is `streamlit run app.py` on a laptop: no build step, no bundler, no CORS.

**React + Leaflet + FastAPI costs 2.5–3.5 days of the week, spent on things that score zero with judges.** The specific sinks are Node/npm/Vite setup across three operating systems; **CORS between the Vite dev server on :5173 and FastAPI on :8000** — every beginner hits this, the fix is trivial once known and costs hours once; `react-leaflet` peer-dependency mismatches with Leaflet 1.9, compounded by react-leaflet having **no first-class Proj4Leaflet binding** so you need a `useMap()` escape hatch anyway; React state management for layer toggles; and two processes to run on demo day instead of one. **Choose React only if at least two team members have already shipped a React app.** They have not.

**The middle path, named, because it is genuinely good:** **FastAPI serving one static vanilla-JS page** — a single `index.html` containing the NASA Leaflet code, `fetch('/api/route?...')` to FastAPI, no build tooling, no npm, no CORS because it is same-origin. It costs maybe half a day more than Streamlit, gives total layout control and a real "backend + frontend" story. **This is the right choice if exactly one team member is comfortable with plain JS.** Plotly Dash is worse here (its map components are Mapbox/Web-Mercator-based, so the polar problem is *harder*); Gradio is the wrong shape entirely.

**The hedge that keeps the decision reversible: build `map.html` standalone first, on Days 1–2, verified working in a browser on its own.** Then it can be dropped into Streamlit, served by FastAPI, or opened directly as a fallback. **This decouples the highest-risk component from the stack decision and means the stack choice stays reversible until Day 4.**

### Streamlit mechanics that matter

Streamlit reruns the script top to bottom on every widget interaction. `st.fragment` makes a designated section rerun independently ([Working with fragments](https://docs.streamlit.io/develop/concepts/architecture/fragments)). **The documented limitation that determines the architecture: "Using caching and fragments on the same function is unsupported."** So keep them as two functions — `compute_route(params)` decorated with `@st.cache_data`, and `render_panel()` decorated with `@st.fragment` calling it. With a warm cache, slider movement is then near-instant.

```python
@st.cache_resource                 # the opened dataset — once per process
def open_grid(): ...

@st.cache_data(show_spinner=False) # derived arrays and routes, keyed on params
def compute_route(date, ice_class, risk_level): ...

@st.fragment                       # partial rerun; must NOT also be cached
def render_panel():
    r = compute_route(date, ice_class, risk_level)
    st.components.v1.html(map_html(r), height=640)
    st.dataframe(r["metrics"])
```

`st.components.v1.html` displays an HTML string in an iframe with **full JavaScript execution** and same-origin access to the Streamlit app ([docs](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v1.html)). It is **marked deprecated in favour of V2 custom components**, which the docs describe as offering better performance and multiple callbacks without iframes — **use V1 anyway.** "Deprecated but working and backward-compatible" is exactly the right risk profile for seven days; note the deprecation in a code comment so mentors and judges see you knew. Data flow is **one-way, Python → HTML string**: f-string the route GeoJSON and layer state into the HTML each rerun, and **do not attempt bidirectional iframe↔Python messaging** — that is a rabbit hole. If you need a click-on-map interaction, fake it with an `st.selectbox` of waypoints.

**One honest flag: the Streamlit + Proj4Leaflet pairing is an inference.** Each piece is verified independently; **no published example of embedding Proj4Leaflet inside a Streamlit component was found.** Treat it as a **Day-1 spike**, not an assumption. If the spike fails by lunch on Day 1, fall back to the FastAPI-plus-static-page middle path, which has no iframe involved at all.

### Environment — the non-negotiables

**pip + venv on Python 3.12, with one pinned `requirements.txt` committed on Day 1 by the tech lead.** Not conda, not Docker, not Colab as the primary environment.

| Option | Verdict |
|---|---|
| **pip + venv, Python 3.12** | **Use this.** 3.11 acceptable; **3.13+ forbidden** — geospatial wheels lag new Python releases and on 3.13/3.14 you are gambling |
| conda / mamba | Solves install problems you no longer have, costs 30–60 min per machine, adds a new failure mode (mixing conda and pip in one env) |
| Docker | Right in principle, wrong for 2nd-years in a week. Windows needs WSL2; mounts, ports and rebuilds cost more than they save. **Exception:** if exactly one member already knows Docker, a `Dockerfile` on Day 5 as demo-day insurance is fine |
| uv | Acceptable and mildly preferable **if the tech lead already uses it**; adopt Day 1 for everyone or not at all |
| **Google Colab** | **ML training only.** Free GPU, everything preinstalled; `joblib.dump()` the model, commit the `.pkl`, load it in the local app. **Never run the demo from Colab** — no persistent filesystem, cannot host the app |

**The GDAL-free minimal dependency set**, which removes the last real install risk:

```
numpy, scipy, pandas          # core
xarray, netCDF4               # NetCDF reading
pyproj                        # lon/lat <-> EPSG:3031, PROJ bundled in the wheel
matplotlib                    # figures
scikit-learn, xgboost         # ML
streamlit                     # UI
requests                      # fetching
pyastar2d                     # fast A*
zarr, gcsfs                   # ARCO ERA5 only
```

Substitutions that make this work: **cartopy → pyproj**, since everything the *app* needs from cartopy is `Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)`; **geopandas → plain GeoJSON**, since coastlines and routes are just GeoJSON and point-in-polygon over a raster is better done as a numpy mask anyway. On rasterio there is one tension to resolve explicitly: **Bremen ships GeoTIFF**, so either install rasterio (which is now safe — `pip install rasterio` works from wheels bundling libgdal on Windows, macOS and Linux, requiring Python ≥3.10 and GDAL ≥3.5 inside the wheel, per the [rasterio installation docs](https://rasterio.readthedocs.io/en/stable/installation.html)) **or** convert the Bremen GeoTIFFs to `.npy` plus a JSON transform **once, on one machine**, and commit the result. Recommended: install rasterio, keep the `.npy` conversion as the fallback. **cartopy now ships wheels since v0.22 but requires Python 3.11+**, so keep it installed only on the one machine making static figures, not in the app's requirements.

**Four rules to announce on Day 1.** Everyone on Python 3.12, verified by a screenshot of `python --version` in the team chat. `requirements.txt` with exact `==` pins, one owner, and nobody installs anything without adding it in the same commit. Windows users run `py -3.12 -m venv .venv` and **keep the repo out of OneDrive-synced folders and paths with spaces** — OneDrive causes intermittent file-lock errors during pip and git operations. Everyone runs `pip install -r requirements.txt` and a one-line `check_env.py` that imports every library and prints versions, **before writing any code**; any failure is a whole-team problem, not the individual's.

### Storage layout and "database"

There is no database. A hackathon MVP that reaches for PostGIS spends a day on PostGIS.

```
repo/
├── requirements.txt          # pinned, one owner
├── check_env.py
├── interfaces.py             # Day 1: signatures + fakes for every cross-person function
├── charter.py                # the 115-day schedule, the box, the vessel spec
├── data/
│   ├── raw/                  # .gitignored — fetched by scripts/download.py
│   ├── grid_25km.npz         # committed: the coarsened stack
│   └── sites.json            # candidate discharge sites, hand-digitised
├── models/model.pkl          # committed, trained in Colab
├── precomputed_routes.json   # THE demo artefact — every slider combination
├── src/{load_data,predict,vessel,polaris,route,uncertainty,mission}.py
├── static/lib/               # vendored leaflet 1.9.4, proj4 2.20.2, proj4leaflet 1.0.1
├── static/tiles/             # Day 5: cached GIBS + Polar View tiles
├── map.html                  # standalone, works in a browser on its own
└── app.py                    # Streamlit
```

**Git discipline for beginners:** one file per person wherever possible, since merge conflicts are mostly a same-file problem; `main` plus short feature branches with PRs reviewed by the tech lead only; **`.gitignore` the `.venv/`, `data/raw/` and `__pycache__` on Day 1**, because a beginner committing a 500 MB venv or NetCDF archive is common and painful to undo; nobody force-pushes.

### Endpoints, if you take the FastAPI middle path

`GET /api/grid?date=…` returns the coarsened field as a compressed array; `GET /api/route?site=…&ice_class=…&risk=…` returns `{polyline, metrics, nodes_expanded, solve_ms}`; `GET /api/mission?station=…&window=…` returns the discharge-site table; `GET /api/uncertainty?route_id=…` returns the percentiles and the corridor polygon. All four read from `precomputed_routes.json` on demo day.

---

## STEP 10 — Dashboard and demo

### One scenario, and its spine is April 2012

**The spine:** the April 2012 Indian Barrier calving. A slab of shelf ice roughly **2 km × 1 km adjoining the Indian Barrier — the cargo unloading site ~100 km north of Maitri — broke away between 5 and 24 April 2012, fragmented into about ten icebergs, and carried Indian tank containers and Russian cargo and fuel to sea**; RADARSAT showed "no signature of presence of metallic objects", so the cargo most likely sank. The same NCPOR item records that **heavy sea ice had already prevented both the Russian and Indian vessels from reaching their barriers in February/March 2012** ([NCPOR news/view/147](https://ncpor.res.in/news/view/147)). That is an Indian failure, documented by the sponsor, at the exact decision point the system addresses.

### The exact Step 1 → 5 click path

**Step 1 (0:00–0:20) — the charter, not a globe.** The first screen is **NCPOR's own 115-day schedule with the 70 stationary days highlighted**, and one line beneath it: *"34 days in transit. 70 days trying to put cargo onto ice that may not be there."* Then the 2012 RADARSAT story in one sentence. **No map yet.** Within fifteen seconds the judge knows this team read the sponsor's procurement documents and news archive, while every other demo that morning opened with a rotating globe.

**Step 2 (0:20–1:00) — the situation.** The EPSG:3031 map appears, framed south of 55°S. Polar View's sea-ice-edge and concentration WMS layers are on; the USNIC bergs are plotted with area-scaled markers; **the D15A/B/C/D and D34 cluster at ~67°S, 79–82°E is visible and labelled** — real, current data in the sponsor's own sector. Toggle the forecast layer to show SIC at day +5.

**Step 3 (1:00–2:00) — the question changes.** Four candidate discharge sites appear along the Princess Astrid Coast. The mission comparison table renders. One site is flagged **NOT REACHABLE** with the reason printed: a SIC > 0.80 corridor persisting through day 6. Say the line: *"every other system asks what the shortest safe path is. This one asks which discharge site, in which window, lets us finish both discharges inside the charter."*

**Step 4 (2:00–2:40) — THE MOMENT.** Flip the ice-class dropdown from **PC5 to IA Super**. The accessible corridor visibly shrinks, the RIO layer recolours, **and the recommended discharge site changes**. Say: *"PolarRoute's reference vessel is PC5. A competing on-screen vessel is PC5. NCPOR's tender requires 1A Super or better. In thick first-year ice POLARIS gives PC5 an RIV of 0 and IA Super an RIV of −2 — so at ten-tenths, PC5 reads RIO 0, normal operation, and IA Super reads RIO −20, special consideration. That is two public documents and one dropdown."* **This is the emotional peak.** It is a regulation visibly changing an operational recommendation, it takes ten seconds, and nobody else in the room can do it.

**Step 5 (2:40–3:30) — the honesty close.** Show the route's ETA fan (P10/P50/P90 rather than a single number), P(blocked) with its ±3%, and **the node-expansion count and re-solve latency**. Then the forecast-value table: *"we do not beat persistence at one-day lead, and we do not claim to — our system uses persistence there. What we can tell you is what the forecast is worth: N hours of transit and X ice-hours per voyage versus planning on persistence."* Close on the three-way provenance slide (standard / our assumption / our model output).

### The discrete-slider trick — the single highest-value implementation decision

**The demo must never wait on computation and must never call a live API.**

```python
# Offline, on Day 5:
RISK   = [0.0, 0.5, 1.0, 1.5, 2.0]        # 5
DATES  = ["2026-01-08", "2026-02-25", "2026-03-27"]   # 3
CLASS  = ["PC5", "PC7", "IA Super"]        # 3
SITES  = ["IB-A", "IB-B", "IB-C", "IB-D"]  # 4
# 5 x 3 x 3 x 4 = 180 combinations, each ~0.01 s of A* at 25 km -> under a minute
json.dump(all_results, open("precomputed_routes.json", "w"))

# At demo time:
risk = st.select_slider("Risk tolerance", options=RISK)   # NOT st.slider — finite space
route = PRECOMPUTED[f"{date}|{ice_class}|{risk}|{site}"]  # a dict lookup
```

**Use `st.select_slider`, never `st.slider`**, so the parameter space is finite and the sweep is exhaustive. Layer `@st.cache_data(show_spinner=False)` on anything not precomputed so the UI never flashes. **Warm the cache before the demo by clicking through every control once** — Streamlit's cache is in-process and a restart empties it. **Pre-download tiles on Day 5** into `static/tiles/` and serve them locally with the remote URL as fallback.

---

## STEP 11 — Validation experiments

**Specify the experiment, not the answer. Do not invent results.** Each row below defines inputs, baseline, method, metric and the visualisation to produce; the numbers are for the team to fill in.

### E1 — Sea-ice forecast skill

**Input:** NSIDC G02202_V6, held-out season, initialisations listed in `splits.json`. **Baselines:** climatology, persistence (the reference), anomaly persistence, damped anomaly persistence. **Our method:** XGBoost per-cell. **Metrics:** MAE, RMSE, R², IIEE, **D_IIEE_AVG in km**, and skill score against each named reference, computed on **four masks side by side — full domain, active region, MIZ (0.15 ≤ SIC < 0.80), ice-edge band** — at leads 1, 3, 5, 7, with paired moving-block bootstrap CIs and the **effective** sample size. **Output:** one table, plus a line plot of skill score against lead time with the crossover where persistence stops winning marked — because that crossover is the lead-time selector's decision boundary, which is the point.

### E2 — Iceberg trajectory

**Input:** BYU consolidated v8.0 tracks, held-out bergs. **Baselines:** position persistence; **constant velocity from the last k fixes**; current-following advection; current + 2% wind (`v = v_w + 0.018·v_a`, the analytically derived coefficient **γ ≈ 0.018** from [Wagner, Dell & Eisenman 2017](https://eisenman.ucsd.edu/papers/Wagner-Dell-Eisenman-2017.pdf)). **Our method:** whichever of these you tune. **Metrics:** ADE / FDE in **geodesic kilometres, never Euclidean degrees** — 1° of longitude at 70°S is about 38 km against 111 km at the equator — plus "% within 5 / 10 / 25 km" with the radii **fixed before looking at results, and said so**. **Output:** a table by horizon plus per-berg results and a pooled number. **Physical cross-check:** because large Antarctic bergs of L > 12 km "move approximately with the ocean currents" with wind effects negligible, the current-following and current-plus-wind rows **should nearly coincide; if they differ a lot, your wind forcing or berg-size handling is wrong.**

**The yardstick warning, which must be read before publishing any 24-hour number.** The only retrievable operational figure is an iceberg drift model validated against **17 observed drift tracks growing absolute position error at about 12 km per day**, with reductions "up to 35% at 96 hour forecast periods" relative to its predecessor ([TRID 1391690](https://trid.trb.org/view/1391690)). Against that, a competitor's committed **`pos_err_24h_km_rms = 3.9134` with `within_10km_pct = 96.716`** would be **roughly three times better than an operational model, produced by a hackathon prototype**. Treat that as a red flag, not a target. The three likely explanations in order: the observed positions are **interpolated** from sparse USNIC fixes, so the model is scored against a smooth curve; the evaluation is teacher-forced, or the 24-hour step is one interpolation step in a series whose true sampling interval is about seven days; or the evaluated bergs are large, slow or grounded so any method including persistence scores well. **The one-line diagnostic: run constant velocity on identical pairs. If it also scores near-zero, your targets are interpolated** — linear interpolation between two weekly fixes is a straight line at constant speed, and constant velocity scores near-perfectly against it. **USNIC's weekly cadence makes honest evaluation-pair construction genuinely non-trivial:** build pairs **only from consecutive native observations** with the actual Δt recorded, bin errors by actual Δt, and if almost all pairs are ~7 days, say so: *"Our native observation cadence is ~7 days, so we report 7-day errors; our 24-hour numbers are model self-consistency, not validated skill."* **That sentence alone would place you ahead of every competitor in the observed field.** One further caution: IDRIFTNET's timestep length is not stated in what could be retrieved, so **do not quote its 22–49 km ADE figures as 24-hour numbers.**

### E3 — Routing

**Input:** fixed origin–destination pairs per date. **Baselines:** great-circle; shortest path avoiding land only, with no ice cost. **Our method:** risk-aware, fuel-aware, balanced. **Metrics:** distance (nm), ETA (h), fuel (t) with its band, mean/max SIC, hours in SIC > 0.6, POLARIS deficit, min CPA to nearest berg (km), ETA P10–P90, P(blocked). **Output:** one table per pair per date, plus the map overlay.

**What you may and may not claim.** Quote **the two numbers that form an honest trade-off — "+X% distance for −Y% cumulative ice-risk exposure" — and never the risk reduction alone.** Declare the risk penalty function up front and show sensitivity to it, because a risk-aware route can be made to look arbitrarily good by choosing the penalty after seeing the routes. **Do not claim the route is "safer" or "would have saved X% fuel"** — there is no counterfactual ground truth for what a ship should have done. Claim **dominance on a declared objective, Pareto position, and constraint satisfaction**: *"our route never enters SIC above the ice-class limit, whereas the great-circle route spends H hours above it"* is factual and checkable.

### E4 — Mission decision

**Input:** candidate sites × arrival windows × held-out dates. **Baseline:** the fixed-site policy — always go to the nominal discharge point, which is what happens today. **Our method:** the comparison table's recommendation. **Metric:** **P(complete both discharges within the charter window)** under the Monte Carlo, plus the **regret** — how often the fixed-site policy picks a site the system flags as infeasible on the observed field. **Output:** a two-row bar chart, our policy against the fixed-site policy, across held-out dates. **This is the experiment that gives Differentiator 1 a number**, and it is the one to build if there is time for only one beyond E1.

### E5 — Uncertainty calibration

**Input:** the Monte Carlo ETA distributions against realised transit times computed on observed fields. **Baseline:** a point-estimate ETA with no spread. **Our method:** the P10/P50/P90 fan. **Metric:** **coverage** — the fraction of realised outcomes falling inside the P10–P90 interval, which should be ~80% if calibrated — plus a reliability curve of nominal against empirical coverage. **Output:** a calibration plot. **Nobody in the observed field publishes a calibration curve, a coverage number or a probabilistic score**, so even a badly calibrated curve, honestly plotted, is more than the field has. If coverage comes out at 40%, say the spread is too tight and name the likely cause — most probably a correlation length that is too short.

---

## STEP 12 — Do not build

| Item | Verdict for this team | Why |
|---|---|---|
| **Full Sentinel-1 SAR processing** | **No** | A 2026 *ESSD* paper already publishes Antarctic grounded-iceberg detection at **F1 > 0.91, minimum detectable size 0.016 km²** ([ESSD 18, 6017](https://essd.copernicus.org/articles/18/6017/2026/)), and the Statoil/C-CORE Kaggle iceberg classifier is nine years old with **93 public GitHub repos** and documented notebook copy-chains. You cannot beat that in a week; **no PS59 team quoted any detection metric at all**, so the honest move is to not compete there |
| **ConvLSTM / U-Net / any large DL** | **No — forbidden** | **One repository in the entire field declares `torch`**; a competitor's own evaluation doc says a ConvLSTM was "not trained as 8 days of data is fundamentally insufficient"; IceNet forecloses novelty with downloadable weights. Saying "we did not train one because we do not have the data to justify one" is stronger than a model you did not train |
| **NSGA-II over route genomes** | **No** | Tuning encoding and crossover so offspring are feasible (no land crossing, no impassable ice) eats the week and yields no optimality story |
| **NAMOA\* / exact multi-objective label-setting** | **No** | Per-node non-dominated label lists explode on a dense grid with three real-valued objectives; dominance checks on insertion and expansion are where student implementations silently lose optimality |
| **CVaR optimisation** | **No** | Optimising CVaR over paths is a research problem. **Report CVaR from your Monte Carlo sample; never claim to optimise it** |
| **MPC re-planning loops / stochastic-PDE uncertainty propagation** | **No** | Not a week-scale build at demo quality |
| **Hydrodynamic vessel simulation; propeller open-water / thrust–torque equilibrium** | **No** | Needs Wageningen B-series coefficients and an installed-propeller spec you do not have |
| **Detailed hull / resistance modelling (Lindqvist with real geometry)** | **No** | Needs waterline entrance angle and rake at B_wl/4 from hull lines you do not have, and the bending coefficient is disputed across OCR'd sources |
| **AIS calibration against real voyages** | **No** | No AIS calibration, no full-scale trial. Say **"this is a physics-grounded model, not a validated one"** |
| **Operational-grade WMO ice classification, and therefore operational-grade RIO** | **No** | The best public Antarctic ice-type product gives three classes, misclassifies pancake ice and produces nothing in summer — and was **not verified at all** in our access testing |
| **Anything needing proprietary or approval-gated data** | **No** | MOSDAC (SSO wall, unknown approval time), CMEMS (account required, approval time unverified), OSCAR/Earthdata (mechanics unverified), any commercial feed |

### Added for the second-year constraint specifically

| Item | Verdict | Why the earlier assessment is downgraded |
|---|---|---|
| **Polar Code 11.3 + Indian Antarctic Act exports** | **Slide only, not code** | Rated "easy, 0.5–1 day" for a stronger team. For this team it is a day of PDF templating that produces **no measurable output** and competes directly with Differentiator 1's build time. **Build the two-column mapping slide** — eight Polar Code sub-clauses on the left, your system layer on the right — and say the compliance record is the next sprint |
| **Risk-budget / epsilon-constraint modes** | **Day-6 branch only** | Two days for beginners, and the Pareto-sweep language it supports is a Q&A asset rather than a demo asset |
| **Time-dependent A\* with `(cell, arrival_time)`** | **No** | Two days, and a competitor already ships a `TimeAStar`, so it is not a differentiator either |
| **Path diversity via the penalty method + separation number** | **No** | The attribution between two overlapping alternative-routes papers could not be cleanly separated, and tuning the tube radius is fiddly. Three route modes from three weight vectors is enough |
| **React + FastAPI + Docker Compose** | **No** | 2.5–3.5 days of non-scoring work |
| **PostGIS / any database** | **No** | JSON and `.npz` files on disk |
| **Live scheduled data ingest** | **No** | Download once, commit the derived arrays, cache the tiles |
| **Writing your own A\* in C, or your own drift PDE** | **No** | `pyastar2d` exists; OpenDrift's `openberg` ships drift physics with melting and rollover under GPL and "why not OpenDrift?" is a losing question |
| **A bigger dashboard / more tabs** | **No** | One project already has eight tabs and an LLM assistant on **admittedly synthetic seeded data** whose own evaluation doc opens "All metrics below are computed on DEMO/SIMULATION DATA (synthetic, seed=42)". Adding a ninth tab does not beat that; **having one number that survives questioning does** |

---

## STEP 13 — Final architecture decision

## OUR FINAL MVP

An Antarctic **discharge-decision support system** for NCPOR's Indian Scientific Expedition to Antarctica, built on the median PS59 pipeline — which we implement competently and refuse to pitch — and differentiated by three things: the objective function is the charterer's, every physical number has a published derivation, and the forecast is evaluated on what it is *worth* rather than on its MAE. It runs as a single Streamlit app on one laptop, on a 25 km EPSG:3031 grid, with every slider combination precomputed to a JSON lookup so nothing computes live.

### Differentiator 1 — The discharge-site decision

**What it is.** The decision variable changes from *which path between A and B* to *which discharge site, in which arrival window, with what probability of completing both discharges inside the 115-day charter*. The route becomes the sub-problem, not the product.

**Why chosen.** It is the only candidate that another team cannot reach by adding a model, because it is not a model — it is an objective function derived from NCPOR's own tender NCPOR/14(102)/21, which schedules **34 transit days against 70 stationary discharge days**, names **the Master in consultation with the Leader of the Expedition** as the people who choose the landing area, and operates where **ASMA 6 records no designated anchorages or barge landings** and where **the April 2012 Indian Barrier calving took Indian tank containers to sea**. **We did not find a publicly accessible PS59 project, or any public system, that formulates the problem this way** — commercial systems behind sales walls cannot be ruled out.

**How we implement it.** Four hand-digitised candidate sites per station in `sites.json`; a loop calling the existing `find_route()` and `rescore()` once per site; five derived columns (reachability, haul distance, discharge mode, ice-edge persistence, iceberg exposure); and a pandas table showing the trade-offs, **not a single black-box score**. About 150 lines. 2.5 days.

### Differentiator 2 — Vessel performance and POLARIS as the cost surface

**What it is.** The cost raster is not an invented weighted sum. It is BAS PolarRoute's concentration-native ice-resistance law, its closed-form speed-in-ice inversion and its published fuel polynomial, with IMO POLARIS `RIO = Σ(Cᵢ × RIVᵢ)` from Table 1.3 layered on top, run at the ice class NCPOR's tender actually requires.

**Why chosen.** It converts the field's most attackable slide — "why 0.35?" — into "because MSC.1/Circ.1519 says so for this ice class", at ~150 lines plus a lookup table. And it contains the demo's visual peak: **PC5 RIV 0 versus IA Super RIV −2 in thick first-year ice means RIO 0 against RIO −20 at ten-tenths**, a visibly different accessible corridor from two public documents.

**How we implement it.** Port `R_ice = 0.5·k·Fr^b·ρ·B·h·V²·C^n` with slender k = 4.4, b = −0.8267, n = 2.0; the inversion `v = [2F/(kρBhC^n(ghC)^(−b/2))]^(1/(2+b))`; the fuel polynomial in t/day; the SDA config (26.5 km/h, beam 24.0 m, force limit 96,634.5 N, 80% SIC cut-off, 10 m depth floor). Hard-code Table 1.3 as a 12 × 12 numpy array with a **PC5 / PC7 / IA Super dropdown**, because **MV *Vasiliy Golovnin*'s assigned ice class is unverified**. Use Table 1.3, never Table 1.4. Label the RIO **"estimated from thickness-derived ice stage; not equivalent to an ice-chart-based RIO"** and own the thickness → WMO mapping as our own approximation. State ±20–30% on absolute fuel and that relative ranking is much better. 1.5 days.

### Differentiator 3 — The three-run forecast-value protocol

**What it is.** Plan on observation and score on observation (the perfect-prognosis bound); plan on the forecast and score on observation (the honest operational number); plan on persistence and score on observation (the no-model baseline). **The gap between the last two is what the forecast is worth, in hours and ice-hours.**

**Why chosen.** It is one day, it is a loop rather than an algorithm, and **the reviewed routing literature does not run it** — an Arctic weather-routing review records that experiments "treat sea ice data as if perfectly known". It also tests, independently of MAE, whether the forecast was worth building: **if cost(C) ≈ cost(B), the model adds no decision value**, which is a genuinely interesting negative result.

**How we implement it.** Three calls to the existing router per held-out initialisation date, scored against the same observed field, plus a **constraint-violation rate** — the fraction of forecast-planned routes that enter SIC above the ice-class limit or RIO < −10 when replayed on the observation. 1 day.

### Core ML

Per-cell **XGBoost** predicting SIC at leads 1, 3, 5, 7 days on the 25 km grid, from ~20 tabular features (lagged SIC, neighbourhood statistics, day-of-year sin/cos, training-only climatology and anomaly, ERA5 winds and temperature, distance to ice edge, lat/lon). **Temporal split, no random shuffling, `splits.json` published.** Four baselines: climatology, persistence, anomaly persistence, damped anomaly persistence. Metrics on **four masks reported side by side**, masked **on the observation never on the forecast**, led by **length-normalised IIEE in kilometres**, with paired moving-block bootstrap CIs and **effective** sample size. **Deep learning is forbidden.** We expect to lose to persistence at lead 1 and we ship a **lead-time-dependent selector that uses persistence there** — the loss becomes an engineering decision.

### Core Routing

**`pyastar2d`** 8-connected A\* over a float32 cost raster at **25 km / ~325 × 325** for the interactive demo, with **one precomputed 6.25 km hero route** shown as a static comparison. Cost = `α·t + β·fuel + γ·POLARIS-deficit + δ·berg + λ·σ·t`, all terms normalised to their domain P50, with hard `np.inf` at SIC > 0.80, depth < 10 m, land, and RIO < −10. Coarsen concentration with **`.mean()`** and hazards with **`.max()`**. Admissible heuristic scaled by the time coefficient only. Three weight-vector route modes plus a user slider. **Say "non-dominated candidates", never "the Pareto front"**, and state the weighted-sum non-convexity limitation in one sentence. Print **node-expansion count and re-solve latency** on screen.

### Core Uncertainty

Monte Carlo re-scoring of a **fixed** polyline across **N = 200 spatially correlated perturbed fields** with a **100–200 km correlation length** via `gaussian_filter(randn(H,W), σ)` — **never per-cell iid noise**. Report **ETA P10/P50/P90, P(blocked) with its ±3%, worst-decile ETA (CVaR, reported not optimised)**, and a shaded confidence corridor. **Ship `assert route(λ=0) != route(λ=2)`**, because if σ ∝ μ the slider is silently a no-op. State that path standard deviation is not additive and that the additive form is an upper bound defensible only because forecast error is spatially correlated. **Attribute honestly: robust ensemble routing is published; we implement and surface it, we did not invent it.**

### Mission Decision Support

Four candidate discharge sites per station, one comparison row each: reachability with reason-for-failure printed, ETA P50/P90, haul distance in km, discharge mode (tracked-vehicle over fast ice versus barge, with the mid-December transition from ASMA 6), 7-day ice-edge persistence as a **labelled bearing-capacity proxy**, count of tracked bergs whose 90% error cone intersects the site, and **P(complete both discharges within the charter window)**. Grounded in the verified schedule (ETD Cape Town 23 Dec, ETA Larsemann Hills 8 Jan = 16 days transit, 40 on station, 8 inter-station, 30 on station, 10 return, 115 total) and the real coordinates (**Bharati 69°24.41'S, 76°11.72'E**; Maitri via India Bay with the discharge point **~100 km north of the station**, exact coordinates unverified and labelled as digitised by us). **The trade-offs are shown so the Voyage Leader decides.**

### Frontend

**Streamlit**, with the map as a **raw Leaflet 1.9.4 + proj4js 2.20.2 + Proj4Leaflet 1.0.1 EPSG:3031 page** copied from NASA's maintained `gibs-web-examples` Antarctic demo, **including the required 1-px `_initTile` seam workaround**, vendored locally in `static/lib/`, embedded via `st.components.v1.html` (V1, deprecated but backward-compatible, noted in a comment). **Polar View's keyless EPSG:3031 GeoServer WMS** (`https://geos.polarview.aq/geoserver/wms`) supplies sea-ice-edge and concentration overlays with **no NetCDF processing**; GIBS WMTS supplies the basemap, **framed south of ~55°S because its tile bounds of ±4,194,304 m cut off around the low-50s °S and Cape Town at 34°S falls off the map** — the northern leg goes in a static Cartopy inset. `st.select_slider` for finite parameter spaces; `@st.cache_data` on `compute_route()`, `@st.fragment` on `render_panel()`, **never both on the same function**. **folium and `streamlit-folium` are rejected outright: `folium.Map`'s `crs` accepts only EPSG3857/4326/3395/Simple.** Build `map.html` standalone first so the stack choice stays reversible until Day 4; the named middle path if the Streamlit iframe spike fails is **FastAPI serving one static vanilla-JS page** — no npm, no CORS, same origin.

### Backend

No database. **pip + venv on Python 3.12** (3.11 acceptable, **3.13+ forbidden**) with a pinned `requirements.txt` committed Day 1 and a `check_env.py` smoke test everyone runs before writing code. GDAL-free minimal set — **xarray + netCDF4 + numpy + scipy + pyproj + scikit-learn + xgboost + streamlit + requests + pyastar2d**, with rasterio added for the Bremen GeoTIFFs (wheels bundle libgdal and work on all three OSes) and a `.npy`-conversion fallback. Colab for ML training only, `joblib.dump()` the model, commit the `.pkl`. Storage is `.npz` grids, `sites.json`, `model.pkl` and **`precomputed_routes.json`** — the last of which *is* the backend on demo day.

### Data

Critical path, all verified key-free on 20 September 2026: **Bremen AMSR2 6.25 km GeoTIFF** (~280 KB/day, ~1-day latency, no gaps in September) for the live field; **NSIDC G02202_V6** (not V5 — **V5 404s**) for the training archive, defensively handling the **missing 9–10 August 2026** and the five undersized August files; **USNIC `DownloadCurrent?pId=134`** (weekly, all 33 bergs updated 18 Sep 2026) for the live berg layer, with DMS→decimal parsing; **BYU `consolidated_database_v8.0.zip`** (4.1 MB, **1978 → 22 April 2025** despite the 2023 page header, `YYYYDDD` dates, multi-sensor columns to coalesce) for trajectory training; **ARCO ERA5 `ar/full_37-1h-0p25deg-chunk-1.zarr-v3` with `token='anon'`** (never the `co/` stores) for historical winds; **Open-Meteo forecast API** for live wind and temperature — **but not the marine API, which returns all nulls at 67°S**; **GEBCO area subset** (never the 4 GB global file) for the land and depth mask. **MOSDAC is RED** — Keycloak SSO wall, v1.0-beta with 2015 metadata, and altimeter ADT, ASCAT winds and OISST are all physically invalid under pack ice. **No Indian-sourced dataset was verified as usable for the Antarctic in a week.** The Indian dimension is the **Maitri/Bharati corridor in the 06–80°E sector**, the verified **D15A/B/C/D and D34 cluster at ~67°S, 79–82°E**, and the sourced fact that **BYU's database itself ingests ISRO Oceansat-2 OSCAT and ScatSat OSCAT-2**.

### Validation

Five experiments, specified not answered: **E1** sea-ice skill against four baselines on four masks at four leads with bootstrap CIs; **E2** iceberg ADE/FDE in geodesic km against persistence, constant velocity and current+2%-wind, with **evaluation pairs built only from consecutive native observations** and the **constant-velocity interpolation diagnostic** run first; **E3** routing against great-circle and land-only baselines, quoting **both** trade-off numbers and claiming only dominance, Pareto position and constraint satisfaction; **E4** mission decision against the fixed-site policy, measured as P(complete both) and regret — **the experiment that gives Differentiator 1 a number**; **E5** uncertainty calibration as P10–P90 coverage against realised outcomes, plotted, which **nobody in the observed field publishes**. Publish `splits.json` and runnable evaluation code.

### DO NOT BUILD

Full Sentinel-1 SAR processing; ConvLSTM, U-Net, LSTM, YOLO or any deep learning; NSGA-II; NAMOA\* or exact multi-objective label-setting; CVaR optimisation; MPC re-planning; stochastic-PDE uncertainty propagation; hydrodynamic vessel simulation; propeller thrust–torque equilibrium; Lindqvist with real hull geometry; AIS calibration; operational-grade WMO ice classification; anything needing MOSDAC, CMEMS, Earthdata or any approval-gated source; Polar Code and Indian Antarctic Act exports as code (**slide only**); risk-budget/epsilon-constraint modes (Day-6 branch at most); time-dependent A\*; path-diversity tuning; React + FastAPI + Docker Compose; PostGIS; live scheduled ingest; a ninth dashboard tab.

### What makes our demo different

A judge who has already watched four PS59 presentations that morning has seen: a rotating globe, an origin box, a destination box, three coloured lines, a ConvLSTM claimed and not trained, a fuel number with no derivation, and a confidence percentage that does not move. **Ours is different in four specific, checkable ways.** The **first screen is not a map** — it is NCPOR's own 115-day charter schedule with 70 stationary days highlighted and the April 2012 RADARSAT story of Indian tank containers going to sea, so within fifteen seconds the judge knows this team read the sponsor's procurement documents rather than the 277-character problem statement. The **demo answers a different question** — four candidate discharge sites each with a probability of being reachable *and* still intact across the dwell window, with the route appearing as a consequence of choosing one, which is the sponsor's own question because the charter names the Master and the Expedition Leader as the people who make exactly this choice. The **numbers have provenance** — when asked where the fuel figure comes from, the field's available answers are silence, "8–15%", or "45.5", while ours is a named resistance law with published coefficients, a closed-form inversion, and a fuel polynomial that returns a plausible 6 t/day hotel load at zero speed; when asked about risk weights, the field's answer is 0.35/0.30/0.20/0.15 and ours is a POLARIS RIO computed at the ice class the tender actually requires, **with the corridor visibly shrinking when we flip PC5 to IA Super on screen**. And we **volunteer the loss before we are asked** — a table where persistence beats us at one-day lead, immediately followed by a system that *uses* persistence at one-day lead, and then the number nobody else has: what the forecast is worth in hours and ice-hours when you plan on it and score on what actually happened. In a room where a domain judge can puncture an unvalidated "96% accuracy" with a single question about how much of the domain is open water, the team that has already stratified its own metrics is **the only one with an answer ready**.

---

## The seven-day plan, with a six-person division of labour

### Roles — four beginners, and none of them on the critical integration path

| # | Role | Level | Owns | Delivers by |
|---|---|---|---|---|
| **1** | **Tech lead / integrator** | strongest | The repo, `requirements.txt`, `interfaces.py`, the Streamlit shell, all merges, final integration, `precomputed_routes.json` | **Day 1:** skeleton app running on fake data |
| **2** | **Map owner** | 2nd strongest / JS-curious | Standalone `map.html`: Leaflet + Proj4Leaflet + GIBS + Polar View; renders route and markers from a JSON blob | **Day 2:** `map.html` works standalone in a browser |
| **3** | **Data owner** | beginner | `load_data.py`: downloads, the 25 km coarsened grid, the land/depth mask, committed `.npz` | **Day 2:** committed grid + a loader function |
| **4** | **Vessel / routing owner** | beginner | `vessel.py` + `polaris.py` + `route.py`: cost surface and `find_route()`. **Owns Differentiator 2.** Pure functions, no UI | **Day 3:** works on a synthetic cost field |
| **5** | **ML owner** | beginner | `predict.py`: XGBoost trained in Colab, exported `.pkl`, plus the baseline ladder and **Differentiator 3** | **Day 3:** a `.pkl` in the repo |
| **6** | **Mission / presentation / QA** | beginner | `sites.json`, `mission.py` (**Differentiator 1**), slides, script, the backup recording, testing on a second machine, the demo runbook | **Day 4:** first full rehearsal |

The organising principle: **every beginner owns something that produces a visible artefact independently, and no beginner sits on the critical integration path.**

### The `interfaces.py` discipline — the single change that prevents a Day-6 crisis

**On Day 1, before anyone writes real code**, the tech lead commits `interfaces.py` containing the *signatures and fake implementations* of every function that crosses a person boundary:

```python
# interfaces.py — Day 1. Fakes today, real implementations by Day 4.
import numpy as np

H, W = 325, 325                        # the 25 km grid, fixed on Day 1

def load_ice_field(date: str) -> np.ndarray:            # owner 3
    """SIC in [0,1], shape (H,W), EPSG:3031, 25 km."""
    return np.clip(np.random.rand(H, W), 0, 1)

def forecast_sic(date: str, lead: int) -> np.ndarray:   # owner 5
    return load_ice_field(date)

def cost_surface(sic, thickness, ice_class: str) -> np.ndarray:   # owner 4
    return np.maximum(1.0 + 10 * sic, 1.0).astype(np.float32)

def find_route(cost, start_rc, goal_rc) -> dict:        # owner 4
    return {"polyline": [(-34.0, 18.0), (-69.4, 76.2)],
            "eta_h": 380.0, "fuel_t": 210.0, "nodes": 4812, "solve_ms": 180}

def rescore(route, mu, sigma, N=200) -> dict:           # owner 4/5
    return {"p10": 360.0, "p50": 380.0, "p90": 430.0, "p_blocked": 0.08}

def mission_table(date: str, station: str) -> "pd.DataFrame":     # owner 6
    ...
```

Everyone codes against those signatures from hour one. Real implementations are drop-in replacements. **This is the single change that converts "integration hell on Day 6" into "integration is a 20-minute merge on Day 5."** The grid shape and the polyline format are frozen on Day 1 and never renegotiated.

### Day by day

**Day 1 — environment and interfaces.** Everyone on Python 3.12 with a `python --version` screenshot in the team chat; `requirements.txt` pinned and committed; `check_env.py` passes for all six. `interfaces.py` committed with fakes. Owner 2 gets the NASA Leaflet example rendering locally **before lunch** — including the `_initTile` seam patch — and empirically finds where the GIBS extent cuts off by panning north. Owner 3 starts the downloads and runs the two 30-second checks that must not be guessed: **`rasterio.open(bremen.tif).crs` and `.shape`**, and **`xarray.open_dataset(cdr.nc)` printing variables and `flag_values`/`flag_meanings`**. Owner 1 runs the **Streamlit + Proj4Leaflet iframe spike** — if it fails by lunch, switch to the FastAPI-plus-static-page middle path. ***Gate: everyone's `check_env.py` passes and `interfaces.py` is merged.***

**Day 2 — the two hard components, separately.** `map.html` works standalone with GIBS basemap and Polar View WMS overlays, rendering a route polyline and markers from a JSON blob. The 25 km grid is committed as `.npz`, with `.mean()` for concentration and `.max()` for hazards. The Streamlit shell renders the fake route on the real map. Owner 4 ports the PolarRoute equations and runs the two sanity checks (6.07 t/day at zero, ~32.4 t/day at the force limit). Owner 5 starts training in Colab.

**Day 3 — real components, still separate.** `find_route()` works on the real cost surface with `pyastar2d`; POLARIS Table 1.3 is in a numpy array and the ice-class dropdown changes the mask. The XGBoost `.pkl` is committed and the four baselines run. Owner 6 digitises `sites.json` and drafts the mission table on fake routes. **First fake→real swap of one component.**

**Day 4 — full integration day. Everything real, nothing polished.** ***Gate: one complete run from slider to route to mission table on the real map.*** The stack decision becomes irreversible today. Run the `assert route(λ=0) != route(λ=2)` test and fix it if it fails. **First rehearsal.** Nominate the demo laptop.

**Day 5 — precompute and harden.** Sweep all 180 slider combinations offline into `precomputed_routes.json`. Cache GIBS and Polar View tiles into `static/tiles/`. Run the Monte Carlo once and commit the perturbation stack. Owner 5 runs Differentiator 3 and produces the forecast-value table. Owner 6 produces the Cartopy static figures (ice + route + bergs) as the guaranteed fallback if the web map breaks. **Second rehearsal, on the nominated laptop, offline.**

**Day 6 — polish and the safety net.** Metrics tables, the three-way provenance slide, the Polar Code 11.3 two-column mapping slide, the deck. **Record the 60-second backup video.** Stretch goals only on branches, never on `main`.

**Day 7 — demo day.** **Feature freeze at T-minus-2-hours.** Warm the cache by clicking every control once. Rehearse the exact click path **ten times** and write it down. Increase font sizes, zoom the browser, close extra tabs, disable notifications. **Hotspot ready — venue Wi-Fi is the most common single point of failure**, which is why everything is cached locally and nothing calls a live API. If the backup recording is used, **label it as a recording**. Offer judges **one live input** — a date, or an ice class — to convert a rehearsed demo into visible live validation.

### What a second-year team should be told bluntly it cannot do this week

Train a deep sea-ice forecasting model from scratch; implement a physics-based ice-drift PDE; build a production React + TypeScript SPA; deploy to cloud infrastructure with CI; ingest live satellite data on a schedule; write their own A\* in C. **Each is a week on its own, and every one of them can be gestured at in the slides as "future work" at zero cost.** The demo needs to work, not to be maximal.

---

## Carried uncertainties — say these out loud rather than hiding them

| Item | Status |
|---|---|
| **PolarRoute config keys and objective-function names** | **Unverified** — both documentation hosts 404'd. Clone the repo and read `examples/*.config.json` before asserting what PolarRoute does |
| **Lindqvist bending coefficient** | **Disputed** — 27 versus 37/64 across OCR'd sources, and the radical may have lost an `h³`. Do not implement it; use PolarRoute or Riska/FSICR |
| **MV *Vasiliy Golovnin*'s ice class** | **Unverified.** The tender's "1A Super or better" requirement is verified; the vessel's assigned class is not. Ship a PC5/PC7/IA-Super sensitivity, never an assertion |
| **Whether MSC.1/Circ.1519 is superseded** | **Unconfirmed.** Say "we use MSC.1/Circ.1519 (2016), issued as interim guidance", never "no revision exists" |
| **Charter day rate / seasonal cargo tonnage** | **Nothing public.** Do not estimate. Express savings as "N days of day hire plus victualling for 40 persons" |
| **Bremen CRS and grid size** | **Not confirmed** — no file was opened. 30-second check on Day 1 |
| **NSIDC CDR flag values, variable names, grid shape** | **Not confirmed** — no file was opened. Mis-handling will corrupt extent calculations |
| **OSI SAF** | **Entirely unverified** — no URL, no format, no access route tested; the summer-gap claim could be neither confirmed nor refuted. Treat as RED |
| **Streamlit + Proj4Leaflet pairing** | **An inference.** Each piece verified independently; the combination has no published example. **Day-1 spike required** |
| **Polar View WMS extent limits** | **Unverified** — check alongside the GIBS cut-off on Day 1 |
| **USNIC CSV columns / shapefile CRS; AADC ASPA file** | **Unverified** — snippet-level evidence only |
| **Indian Barrier exact coordinates** | **Unverified** — only "Princess Astrid Coast, ~100 km north of Maitri". Label candidate sites as digitised by us |
| **"Robust Pareto-optimum routing… ensemble forecasts"** | **Full text not obtained.** Read it before claiming anything about uncertainty routing is new |
| **Competitor observations** | All point-in-time, 20 September 2026. Every "absent" means **absent from the sources we enumerated**, not proved absent; commercial systems behind sales walls cannot be ruled out |

---

## Sources

**NCPOR operational reality and the charter**
- NCPOR/14(102)/21 — 41-ISEA Tender, Ice Class Vessel (115-day schedule, 1A Super requirement, discharge clause, barge/crane specs, MGO/MDO clause) — https://www.indianembassyrome.gov.in/docs/1624279190_1285_41%20ISEA%20Tender-Ice%20Class%20Vessel.PDF
- NCPOR news: "Shelf Ice breaks close to Indian Station Maitri" (April 2012 Indian Barrier calving, cargo lost, ~100 km north of Maitri) — https://ncpor.res.in/news/view/147
- ASMA No. 6 Larsemann Hills Management Plan (no designated anchorages, ~5 nm offshore, tracked vehicle to mid-December, Bharati 69°24.41'S 76°11.72'E) — https://www.env.go.jp/nature/nankyoku/kankyohogo/database/jyouyaku/asma/asma_pdf_en/ASMA06_en.pdf
- PIB PRID 1993769 — MV *Vasiliy Golovnin*, 43-ISEA, third-party cargo (BELARE, Progress, Novo Airport) — https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=1993769
- NCPOR ICT Division (National Polar Data Center, MET-Data Portal, hourly Maitri AWS) — https://ncpor.res.in/pages/display/117-information-communication-technology-(ict)
- Hui et al., SatSINS for RV *Xuelong* in Prydz Bay, *Remote Sensing* 9(6):518 — https://www.mdpi.com/2072-4292/9/6/518
- Mishra et al. (2021), optimum ship route Bharati–Maitri, *Polar Science* 30, 100696 — https://www.sciencedirect.com/science/article/pii/S1873965221000736

**Datasets — access mechanics verified 20 September 2026**
- University of Bremen AMSR2 ASI 6.25 km, southern daily — https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/s6250/2026/sep/Antarctic/
- NSIDC Sea Ice Index G02135 southern daily + 1981–2010 climatology — https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/
- NOAA/NSIDC CDR **G02202_V6** southern daily (V5 404s) — https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/2026/
- US National Ice Center Antarctic icebergs (CSV `pId=134`, shapefile `pId=228`) — https://usicecenter.gov/Products/AntarcIcebergs
- BYU/NIC Antarctic Iceberg Tracking Database, consolidated v8.0 (1978 → 22 Apr 2025) — https://www.scp.byu.edu/data/iceberg/database1.html
- ARCO ERA5 on Google Cloud, `token='anon'`, `ar/full_37-1h-0p25deg-chunk-1.zarr-v3` — https://github.com/google-research/arco-era5 · https://docs.cloud.google.com/storage/docs/public-datasets
- Open-Meteo forecast API (works at 67°S) — https://api.open-meteo.com/v1/forecast · marine API (**nulls at 67°S**) — https://marine-api.open-meteo.com/v1/marine
- GEBCO_2026 gridded bathymetry and the area-subset application — https://www.gebco.net/data-products/gridded-bathymetry-data · https://download.gebco.net/
- MOSDAC Global Ocean Surface Current (**SSO wall, v1.0-beta, 2015 metadata**) — https://www.mosdac.gov.in/global-ocean-surface-current
- Copernicus Marine Toolbox installation and `subset` syntax — https://help.marine.copernicus.eu/en/articles/7970514-copernicus-marine-toolbox-installation
- PO.DAAC OSCAR L4 surface currents v2.0 — https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_INTERIM_V2.0
- AADC ASPA/ASMA geometries — https://data.aad.gov.au/metadata/AAS_4296_Updated_ASPAs_2024 · https://www.data.gov.au/data/dataset/aad-protected-hsm-gis

**Stack and tooling**
- NASA GIBS Antarctic EPSG:3031 Leaflet example (Leaflet 1.9.4 + proj4js 2.20.2 + Proj4Leaflet 1.0.1, `_initTile` seam patch) — https://raw.githubusercontent.com/nasa-gibs/gibs-web-examples/main/examples/leaflet/antarctic-epsg3031.js · live demo https://nasa-gibs.github.io/gibs-web-examples/examples/leaflet/antarctic-epsg3031.html
- NASA GIBS API access basics and the four projection endpoints — https://nasa-gibs.github.io/gibs-api-docs/access-basics/
- Polar View data access — EPSG:3031 GeoServer WMS/WFS, **no authentication** — https://www.polarview.aq/pages/index/data_access
- folium API reference (**`crs` accepts only EPSG3857/4326/3395/Simple**) — https://python-visualization.github.io/folium/latest/reference.html · polar projection request https://github.com/python-visualization/folium/issues/338
- `streamlit-folium` reload/crash report — https://github.com/randyzwitch/streamlit-folium/issues/72
- Streamlit `st.components.v1.html` (V1, deprecated but backward-compatible) — https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v1.html · custom components overview https://docs.streamlit.io/develop/api-reference/custom-components
- Streamlit fragments (**caching and fragments cannot decorate the same function**) — https://docs.streamlit.io/develop/concepts/architecture/fragments
- `pyastar2d` (1802×1802 in 0.29 s, 4008×4008 in 0.83 s, wheels all platforms, v1.1.4 22 Feb 2026) — https://pypi.org/project/pyastar2d
- rasterio installation (wheels bundle libgdal; Python ≥3.10) — https://rasterio.readthedocs.io/en/stable/installation.html
- cartopy installation (wheels since v0.22; **Python 3.11+**) — https://cartopy.readthedocs.io/stable/installing.html
- GeoPandas installation (shapely/pyproj/pyogrio wheels bundle GEOS/PROJ/GDAL) — https://geopandas.org/en/stable/getting_started/install.html
- xarray I/O user guide (engines, `decode_times` failures, `open_mfdataset`) — https://docs.xarray.dev/en/stable/user-guide/io.html
- Leaflet tile-seam issue — https://github.com/Leaflet/Leaflet/issues/3575
- deck.gl GlobeView (**experimental**) — https://deck.gl/docs/api-reference/core/globe-view

**Vessel performance, POLARIS, regulation**
- BAS PolarRoute (source of the `SDA.py` resistance law, speed inversion, fuel polynomial, PC5 config) — https://github.com/bas-logist/PolarRoute · method paper https://arxiv.org/abs/2209.02389
- IMO POLARIS, MSC.1/Circ.1519 (RIV Tables 1.3/1.4, RIO formula, thresholds, speed caps, "not a Go/No Go tool") — https://www.nautinst.org/static/uploaded/2f01665c-04f7-4488-802552e5b5db62d9.pdf · https://www.imorules.com/GUID-2C1D86CB-5D58-490F-B4D4-46C057E1D102.html
- IMO Polar Code Part I-A Chapter 11 voyage planning — https://pame.is/ourwork/arctic-shipping/forum/web-portal/1-16/ · full text https://wwwcdn.imo.org/localresources/en/MediaCentre/HotTopics/Documents/POLAR%20CODE%20TEXT%20AS%20ADOPTED.pdf
- Indian Antarctic Act, 2022 — https://prsindia.org/files/bills_acts/acts_parliament/2022/The%20Indian%20Antarctic%20Act,%202022.pdf
- ABS Polar Code case study (RIO contour maps as a commercial product) — https://ww2.eagle.org/content/dam/eagle/case-studies/marine-polar-oldendorff-casestudy.pdf
- Riska et al. / FSICR power requirement coefficients — https://www.sjofartsverket.se/globalassets/tjanster/isbrytning/pdf-regelverk/power_requirement_finnish-swedish_iceclass.pdf
- Fan, Yu & Jiang (2019), Lindqvist validation errors — https://library.arcticportal.org/2707/1/A1904007.pdf
- ASPM manuscript (870 vs 790 kg/h full-scale comparison) — https://discovery.ucl.ac.uk/id/eprint/10137042/1/ASPM%20-%20final%20manuscript.pdf
- Kim et al., attainable speed in pack ice — https://www.sciencedirect.com/science/article/pii/S2092678217301851
- SFOC ranges and low-load degradation — https://www.sustainable-ships.org/stories/2022/sfc
- Shi et al. (2022), GIOMAS Antarctic volume bias ~38%, satellite/in-situ correlation 0.69–0.73 — https://tc.copernicus.org/articles/16/1807/2022/
- OSI SAF Southern Hemisphere sea-ice type (FYI/MYI/ambiguous, pancake misclassification, no summer classification) — https://osi-saf.eumetsat.int/community/stories/new-osi-saf-southern-hemisphere-sea-ice-type-product

**Validation methodology and prior art**
- Melsom et al. (2019), *Ocean Science* 15:615 — IIEE decomposition and `D_IIEE_AVG` — https://os.copernicus.org/articles/15/615/2019/
- Andersson et al. (2021), IceNet, *Nature Communications* — active-region masking, binary accuracy ≡ 1 − IIEE/area — https://www.nature.com/articles/s41467-021-25257-4
- Massonnet et al. (2023), SIPN South — climatology benchmark, rejection of damped anomaly persistence — https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1148899/full
- Arctic weather-routing review ("sea ice treated as if perfectly known") — https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1190164/full
- Wagner, Dell & Eisenman (2017), analytical iceberg drift, γ ≈ 0.018 — https://eisenman.ucsd.edu/papers/Wagner-Dell-Eisenman-2017.pdf
- Operational iceberg drift verification, **~12 km/day error growth against 17 tracks** — https://trid.trb.org/view/1391690
- de Weck & Kim, weighted-sum non-convexity and clustering — https://web.mit.edu/deweck/www/PDF_archive/2%20Refereed%20Journal/2_12_SMO_AWSMOO1_deWeck_Kim.pdf
- Lim, Sommer, Nikolova & Rus (RSS 2012), mean-risk `μᵀx + c·√(τᵀx)` — https://www.roboticsproceedings.org/rss08/p32.pdf
- FOSM ensemble uncertainty in ship routing — https://www.mdpi.com/2077-1312/9/12/1434
- Robust Pareto-optimum routing using ensemble forecasts (**full text not obtained — read before claiming novelty**) — https://www.researchgate.net/publication/232940270_Robust_Pareto-optimum_routing_of_ships_utilising_deterministic_and_ensemble_weather_forecasts
- Nannicini et al., FIFO definition and complexity — https://www.lix.polytechnique.fr/~liberti/bidirtimedepj.pdf
- OpenDrift `openberg` (drift + melting + rollover) — https://opendrift.github.io/_modules/opendrift/models/openberg.html
- Sentinel-1 Antarctic grounded-iceberg detection, F1 > 0.91 at 0.016 km² — https://essd.copernicus.org/articles/18/6017/2026/
- Statoil/C-CORE Kaggle Iceberg Classifier — https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge

**Competitor evidence (point-in-time, 20 September 2026)**
- nawddeep/SIH26059 `baseline_comparison.json` (persistence 0.02051 vs model 0.04662 at +1 day) — https://raw.githubusercontent.com/nawddeep/SIH26059/main/seaice_forecast/output/evaluation/baseline_comparison.json
- POLARPATH AI `route_optimizer.py` (no `heapq`, five hand-written builders) — https://raw.githubusercontent.com/afrazhussaina5/polarpath-ai/main/backend/services/route_optimizer.py
- IAVNS `EVALUATION.md` ("a ConvLSTM was not trained…") — https://raw.githubusercontent.com/Manthanvinzuda007/Team-SIH-/prod/IAVNS/EVALUATION.md
- POLAR-AI `risk_service.py` (the only code-verified risk weights: 0.35/0.30/0.20/0.15) — https://raw.githubusercontent.com/2911vedant/polar-ai/main/backend/app/services/risk_service.py
- 125112056-art `requirements.txt` (the only repo declaring `torch`) — https://raw.githubusercontent.com/125112056-art/antarctic-nav-system/master/requirements.txt

**SIH process and demo-day practice**
- SIH 2026 Guidelines, College SPOC (evaluation criteria; 4–5 teams per PS; sponsor not obligated to declare a winner) — https://www.sih.gov.in/letters/SIH2026-Guidelines-College-SPOC.pdf
- SIH Buddy SIH26059 ("without a vessel ice class and resistance model your route optimises a difficulty you invented") — https://www.sihbuddy.in/ps/SIH26059
- The hackathon demo that works live: a technical checklist (feature freeze at T-2h, ten rehearsals, hotspot fallback, 60-second labelled backup recording, venue Wi-Fi as the most common single point of failure) — https://dev.to/pranjulrathour/the-hackathon-demo-that-works-live-a-technical-checklist-4k1
- JetBrains, notes from the judging table — https://blog.jetbrains.com/ai/2026/06/how-to-win-a-hackathon-notes-from-the-judging-table/

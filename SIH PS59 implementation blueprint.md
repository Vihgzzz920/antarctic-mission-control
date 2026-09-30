# PS59 Implementation Blueprint

**SIH 2026 · SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory and Navigation Decision Support System**
**Ministry of Earth Sciences / NCPOR · Written 23 September 2026 · Team of 4 · ~7 days**

Build this in the order given and you have a working end-to-end demo on the evening of Day 3, with the remaining four days spent making it defensible rather than making it bigger. The system is a **mission-level decision-support tool**: given a start port, a charter window and a set of candidate discharge sites, it evaluates each site on ETA, fuel, risk, uncertainty and feasibility, and shows the navigator the trade-offs instead of a single number. The AI component is deliberately small — an XGBoost sea-ice nowcast that must beat persistence before it is allowed into the pipeline — because the differentiation is in the decision layer, not the model.

---

## ⚠️ THREE CONFLICTS WITH YOUR STATED PLAN

You asked me not to redesign unless I found concrete technical problems. I found three. Each is verified, each will cost you days if ignored.

### Conflict 1 — MOSDAC ocean currents will not work. Drop it.

Your brief lists MOSDAC currents as a data source. Verified findings:

- Access redirects to a **Keycloak SSO login wall**; approval time could not be established.
- The product is **v1.0-beta with 2015 metadata**.
- Decisively: its three inputs are **altimeter ADT, ASCAT scatterometer winds, and OISST**. All three are *physically invalid under pack ice*. Radar altimetry does not measure sea surface height through consolidated ice; scatterometer wind retrieval fails over ice; OISST under ice is an interpolated proxy. Its validation literature is entirely tropical Indian Ocean.

A judge from NCPOR will know this. Presenting MOSDAC currents in the Antarctic sea-ice zone is the single most puncturable thing in your plan.

**Decision: remove ocean currents from the MVP entirely.** Keep the architecture current-compatible (Section 9 shows where the term slots in), but do not ingest a currents dataset. If you want currents later, use **Copernicus GLORYS12**, not MOSDAC. Section 3 gives you a defensible Indian-data story that does not depend on an Indian data feed.

### Conflict 2 — React + Leaflet is a 2.5–3.5 day tax on a 4-person team

Verified: **folium cannot render EPSG:3031.** `folium.Map`'s `crs` parameter accepts only `EPSG3857`, `EPSG4326`, `EPSG3395` and `Simple`. So the usual Python escape hatch does not exist for Antarctic maps. Whatever you choose, someone writes raw Leaflet + Proj4Leaflet JavaScript.

Given that, React adds npm toolchain across three operating systems, CORS, build config and state management — 2.5–3.5 days of work that earns zero marks — and `react-leaflet` has no first-class Proj4Leaflet binding.

**Decision: build `map.html` as a standalone file on Day 1.** It is plain HTML + Leaflet + Proj4Leaflet, reads a JSON file, and renders. That single file works *unchanged* inside Streamlit (`st.components.v1.html`) **and** inside React (`<iframe>`) **and** served directly by FastAPI. You defer the framework decision to Day 4 with zero rework. My recommendation at that point will be Streamlit; Section 16 covers both paths.

### Conflict 3 — four people, not six, changes the critical path

Your split has Person 4 owning backend *and* frontend. That is the longest pole and it blocks the demo. Section 18 rebalances: the map is pulled forward to Day 1 and shared, and mission evaluation moves to the person who owns routing. Nobody owns two sequential critical-path items.

---

## 1. FREEZE THE MVP

### MUST BUILD — without these there is no demo

| Feature | Why it is essential |
|---|---|
| **Common 25 km Antarctic grid (EPSG:3031)** | Every downstream module indexes into this. Nothing works until the grid exists. It is the integration contract between all four people. |
| **Sea-ice concentration ingest (Bremen AMSR2)** | The one environmental field the whole system depends on. Key-free, ~280 KB/day, ~1-day latency. |
| **Persistence baseline forecast** | This is not a placeholder. It is the scientific control that makes every later claim meaningful, and it is 10 lines of code. |
| **XGBoost sea-ice nowcast + honest comparison vs persistence** | The ML component. It must be *compared*, not just trained. The comparison is worth more than the model. |
| **Vessel performance model (PolarRoute SDA port) + POLARIS** | Differentiator 2. Turns every fuel/speed/risk number on screen from invented into derived. ~60 lines. |
| **Risk/cost raster with hard constraints** | Feeds A\*. Must include SIC > 80% impassable and land mask. |
| **A\* routing via `pyastar2d`** | Commodity, but load-bearing. Never pitch it as novelty. |
| **Iceberg layer: constant-velocity drift from USNIC/BYU** | Real named icebergs (D15A–D, D34) in the actual Indian Ocean sector. Visually the most striking layer. |
| **Candidate discharge-site evaluation (≥3 sites)** | **Differentiator 1. This is the project.** Without it you are the median team. |
| **Mission comparison table with per-component trade-offs** | The decision surface. No magic score. |
| **Monte Carlo route re-scoring (N=200, spatially correlated perturbation)** | Produces P10/P50/P90 ETA and blockage probability. Cheap at 25 km. |
| **`map.html` — Antarctic polar map with layers** | The demo surface. Build Day 1. |
| **Precomputed `demo_cache.json`** | The demo must never compute live or call an API. Non-negotiable. |

### SHOULD BUILD — drop these before you drop anything above

| Feature | Why it is second tier |
|---|---|
| **Three-run forecast-value experiment** | Differentiator 3, and intellectually the strongest thing you can show. But it is a loop over machinery that already exists, so it *must* come after that machinery works. One day, late. |
| **Risk-segment decomposition** ("which 80 km of this route carries 60% of its risk") | Excellent explainability, genuinely cheap (a max-subarray over per-edge risk). Cut only if Day 5 runs over. |
| **Lead-time-dependent model selector** | "We ship persistence at lead 1 because persistence wins there." Converts a negative result into an engineering decision. ~20 lines, but only meaningful once validation exists. |
| **Ice-class sensitivity (PC5 vs IA Super)** | One slider, two POLARIS lookups, visibly different accessible corridor. High impact per line of code — see Section 10. |

### ONLY IF TIME — Day 6 afternoon at the earliest

- Wind/weather cost term from ARCO ERA5 (the grid join is the work, not the physics)
- Calving-risk proxy at discharge sites (distance-to-shelf-edge heuristic)
- ASPA/ASMA no-go polygons from the Australian Antarctic Data Centre
- Iceberg drift using currents instead of constant velocity
- Route-diversity penalty so alternatives look visually distinct

### DO NOT BUILD — these consume days and add nothing to the demo

| Feature | Why not |
|---|---|
| **Any deep learning** (ConvLSTM, U-Net, LSTM) | You have days of data, not years. It will lose to persistence, and you will have spent three days finding that out. Verified: across the entire competitor field, exactly **one** repository declares `torch`, while ConvLSTM appears constantly in READMEs — the gap between claim and code is where teams die under questioning. |
| **Sentinel-1 SAR processing** | Speckle filtering, calibration, geocoding. Multi-week task. Iceberg positions are already published as CSV. |
| **Ocean currents (any source)** | MOSDAC is invalid under ice; GLORYS is another auth flow and another regrid. Cut entirely — see Conflict 1. |
| **Ocean-wave hazard panel** | Verified: Open-Meteo's marine API returns **all nulls at 67°S/40°E** while returning 12 m waves at 57°S/40°E, same longitude, same instant. The wave model is ice-masked. Your wave panel would render **blank on stage**. |
| **NSGA-II / NAMOA\* / true Pareto front** | Weighted-sum sweeps give you visibly different routes for a fraction of the effort. Just never *call* it a Pareto front (Section 11). |
| **CVaR optimisation, MPC, D\* Lite replanning** | Research-grade. Report CVaR from your Monte Carlo sample if you like — one line — but never optimise it. |
| **Lindqvist ice resistance with real hull geometry** | The bending coefficient is disputed across sources (27/64 vs 37/64, both OCR'd). You have a better option that is open-source and already validated on an Antarctic vessel. |
| **Time-dependent A\* with `(cell, arrival_time)` state** | A competitor already ships a `TimeAStar`, so it is not differentiating, and it triples your state space. |
| **AIS ingestion / route validation against real tracks** | Southern Ocean research-vessel AIS coverage is too sparse to validate against. |
| **PostgreSQL / PostGIS** | Flat files and one JSON cache. A database adds ops burden and zero demo value. |
| **Docker** | Adds a platform to debug. `pip + venv` on pinned Python 3.12. |
| **User accounts, auth, multi-user state** | Nobody scores this. |
| **Polar Code / Indian Antarctic Act compliance PDF export** | A day of templating with no measurable output. Make it **one slide** instead — Section 23. |

---

## 2. FINAL END-TO-END ARCHITECTURE

```
 [ Bremen AMSR2 SIC ]  [ USNIC + BYU icebergs ]  [ IBCSO/GEBCO depth ]  [ (opt) ERA5 wind ]
            |                      |                       |                     |
            v                      v                       v                     v
 +---------------------------------------------------------------------------------+
 | 1. INGEST                    src/data/fetch.py         -> data/raw/*.tif,*.csv   |
 +---------------------------------------------------------------------------------+
            |
            v
 +---------------------------------------------------------------------------------+
 | 2. GRID        src/data/grid.py  -> EPSG:3031, 25 km, 216x216  -> data/grid.nc   |
 |                THE INTEGRATION CONTRACT. Frozen Day 1. Never changes.            |
 +---------------------------------------------------------------------------------+
            |
      +-----+--------------------+-------------------------+
      v                          v                         v
 +----------------+   +----------------------+   +----------------------+
 | 3. SEA-ICE     |   | 4. ICEBERG DRIFT     |   | 5. STATIC MASKS      |
 | persistence +  |   | constant velocity    |   | land, depth<10 m,    |
 | XGBoost        |   | + growth cone        |   | (opt) ASPA polygons  |
 | src/forecast/  |   | src/forecast/        |   | src/data/masks.py    |
 +----------------+   +----------------------+   +----------------------+
      |                          |                         |
      +-----------+--------------+-------------------------+
                  v
 +---------------------------------------------------------------------------------+
 | 6. RISK / COST RASTER        src/routing/cost.py                                 |
 |    normalised sub-risks, stored SEPARATELY for explainability                    |
 +---------------------------------------------------------------------------------+
                  |
                  v
 +---------------------------------------------------------------------------------+
 | 7. VESSEL MODEL              src/vessel/performance.py    [DIFFERENTIATOR 2]     |
 |    PolarRoute SDA ice resistance -> attainable speed -> fuel t/day; POLARIS RIO  |
 +---------------------------------------------------------------------------------+
                  |
                  v
 +---------------------------------------------------------------------------------+
 | 8. A* ROUTING                src/routing/astar.py  (pyastar2d)                   |
 +---------------------------------------------------------------------------------+
                  |
                  v
 +---------------------------------------------------------------------------------+
 | 9. MISSION EVALUATION        src/mission/evaluate.py      [DIFFERENTIATOR 1]     |
 |    for each candidate site: route + metrics + feasibility                        |
 +---------------------------------------------------------------------------------+
                  |
                  v
 +---------------------------------------------------------------------------------+
 | 10. UNCERTAINTY              src/uncertainty/montecarlo.py                       |
 |     N=200 correlated perturbations -> P10/P50/P90 ETA, P(blocked)                |
 +---------------------------------------------------------------------------------+
                  |
                  v
 +---------------------------------------------------------------------------------+
 | 11. CACHE  ->  demo_cache.json   |   12. API  backend/main.py (FastAPI)          |
 +---------------------------------------------------------------------------------+
                  |
                  v
 +---------------------------------------------------------------------------------+
 | 13. DASHBOARD   map.html (Leaflet + Proj4Leaflet, EPSG:3031) + host shell        |
 +---------------------------------------------------------------------------------+
```

### Stage detail

| # | Stage | Input | Processing | Output | Library | Module | Consumed by |
|---|---|---|---|---|---|---|---|
| 1 | Ingest | HTTPS URLs | Download, cache to disk, skip if present | `data/raw/` | `requests` | `src/data/fetch.py` | 2 |
| 2 | Grid | Raw GeoTIFF/CSV | Reproject to EPSG:3031, resample to 25 km | `data/grid.nc` | `xarray`, `pyproj`, `scipy` | `src/data/grid.py` | 3,4,5,6 |
| 3 | Sea-ice forecast | SIC history on grid | Persistence; XGBoost on lagged features | `sic_fcst[lead,y,x]` | `sklearn`, `xgboost` | `src/forecast/seaice.py` | 6 |
| 4 | Iceberg drift | USNIC positions, BYU tracks | Constant velocity + widening uncertainty cone | list of `(x,y,r)` per lead | `pandas`, `numpy` | `src/forecast/iceberg.py` | 6 |
| 5 | Masks | Bathymetry, coastline | Boolean land/shallow mask | `mask[y,x]` | `numpy` | `src/data/masks.py` | 6 |
| 6 | Risk raster | 3,4,5 | Normalise, weight, combine; keep components | `risk[y,x]`, `components{}` | `numpy` | `src/routing/cost.py` | 7,8 |
| 7 | Vessel model | SIC, thickness proxy, ice class | SDA resistance → speed → fuel; POLARIS RIO | `speed[y,x]`, `fuel[y,x]`, `rio[y,x]` | `numpy` | `src/vessel/performance.py` | 8 |
| 8 | A\* | cost raster | Grid A\* with great-circle heuristic | polyline `[(y,x)…]` | `pyastar2d` | `src/routing/astar.py` | 9 |
| 9 | Mission eval | sites, 7, 8 | Route per site, integrate metrics, feasibility | list of mission dicts | `numpy` | `src/mission/evaluate.py` | 10,12 |
| 10 | Uncertainty | 9 + perturbed fields | Re-score fixed polyline N times | P10/P50/P90, P(blocked) | `scipy.ndimage` | `src/uncertainty/montecarlo.py` | 12 |
| 11 | Cache | 9,10 | Sweep discrete params, dump JSON | `demo_cache.json` | `json` | `scripts/build_cache.py` | 12 |
| 12 | API | cache | Serve JSON | HTTP | `fastapi` | `backend/main.py` | 13 |
| 13 | Map | API JSON | Render EPSG:3031 layers | browser | Leaflet, Proj4Leaflet | `frontend/map.html` | judge |

**The single most important architectural rule:** stage 2 output (`data/grid.nc`) is frozen on Day 1 and never changes shape. Every other module reads and writes arrays of exactly that shape. That is what lets four people work in parallel without blocking.

---

## 3. DATA SOURCES — EXACTLY WHAT TO DOWNLOAD

Minimum viable set: **four datasets.** Two are mandatory, one is a small static file, one is optional.

| Dataset | Variable | Source | Resolution | Time range | Format | Download method | Why we need it |
|---|---|---|---|---|---|---|---|
| **Bremen AMSR2 ASI** | Sea-ice concentration (%) | `data.seaice.uni-bremen.de` | 6.25 km → resample to 25 km | Last 60 days + same window last 3 yrs | GeoTIFF | plain HTTPS `requests.get` | Core environmental field. Drives risk, vessel speed, POLARIS |
| **USNIC iceberg table** | Named berg positions, size | `usicecenter.gov` | point | current week | CSV | plain HTTPS | Live iceberg layer, real named bergs in the Indian Ocean sector |
| **BYU Antarctic Iceberg DB** | Historical berg tracks | `scp.byu.edu` | point | 1978 → 22 Apr 2025 | ZIP of CSV | plain HTTPS, one-time | Velocity estimation + drift validation ground truth |
| **IBCSO / GEBCO subset** *(static, optional)* | Depth | `download.gebco.net` | area subset | static | NetCDF/GeoTIFF | manual area subset | Land + <10 m shallow mask. **Do NOT download the 4 GB global file.** |
| **ARCO ERA5** *(Only-If-Time)* | 10 m wind | `gs://gcp-public-data-arco-era5` | 0.25° | any | Zarr | `xarray` + `gcsfs`, `token='anon'` | Wind cost term |

### Per-dataset detail

#### 1. Bremen AMSR2 — your critical path, and it is GREEN

- **Exact file pattern:** `https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/s6250/{YYYY}/{mon}/Antarctic/asi-AMSR2-s6250-{YYYYMMDD}-v5.4.tif`
- **Verified working example:** `.../s6250/2026/sep/Antarctic/asi-AMSR2-s6250-20260919-v5.4.tif`
- `{mon}` is a lowercase three-letter month (`sep`, `oct`).
- **No authentication.** Standard Apache directory listing, HTTP 200.
- **~280 KB/day** as GeoTIFF. The `.hdf` of identical data is 1.3 MB — use the `.tif`.
- Latency ~1 day. Spatial 6.25 km, temporal daily.
- **Use 25 km for the MVP.** 6.25 km is 1300×1300 = 1.7 M cells; 25 km is ~216×216 = 47 k cells, which makes Monte Carlo and interactive routing trivial. Keep one precomputed 6.25 km "hero" route for credibility.
- **Historical data needed:** yes, but only ~60 days for XGBoost lags, plus the same calendar window from 3 prior years if you want a climatology term. ~60–200 files × 280 KB = under 60 MB. Download in a loop; it takes minutes.
- **Preprocessing difficulty:** low-moderate. It is a GeoTIFF on a polar stereographic grid — read with `rasterio`, resample, done.
- ⚠️ **Not yet verified:** the exact CRS and grid dimensions of the Bremen file, and whether an `s3125` (3.125 km) tree exists. Both are 30-second checks on Day 1 — open one file and print `src.crs` and `src.shape`. Do not write code against a guessed CRS.
- **Fallback:** NSIDC G02202_V6 (below).

#### 2. NSIDC — fallback SIC and a free climatology

- **Sea Ice Index (G02135):** `https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v4.0.csv` — 1.83 MB, updated daily, **no Earthdata login**. This is a 1-D extent time series, not a grid, but it bundles a **1981–2010 climatology** which gives you anomaly features for free.
- **CDR (G02202):** `https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/2026/` — gridded 25 km NetCDF, ~400 KB/day, ~6-day latency.
- ⚠️ **It is V6, not V5.** The `G02202_V5/` path returns **404**. Prior guidance elsewhere says V5; that guidance is stale and will fail silently in a download loop.
- ⚠️ **Known gaps:** 9–10 August 2026 are missing entirely, and five August files (14, 15, 17, 18, 19) are ~80 KB undersized. Your loader must handle missing days rather than crash.
- ⚠️ CDR flag values (pole hole, land, missing) were **not verified** — print the unique values of one file before trusting the array.

#### 3. USNIC iceberg table — live layer

- **CSV:** `https://usicecenter.gov/File/DownloadCurrent?pId=134`
- **Shapefile:** `pId=228`
- **Verified current:** all 33 tracked bergs carried an 18 September 2026 update when checked on 20 September. Earlier concern that this source was stale was wrong — it refreshes roughly weekly.
- **Gotchas:** positions are **DMS strings** needing parsing; sizes are in **nautical miles / square nautical miles**; the `pId` URL pattern is opaque, so **fail loudly** if the response is not CSV rather than silently writing an HTML error page to disk.
- **The demo gift:** the **D15A/B/C/D and D34 cluster sits at roughly 67°S, 79–82°E** — five tracked bergs within a few degrees, directly in the sector NCPOR's Bharati resupply passes through. Use these, not a generic Weddell Sea example.

#### 4. BYU Antarctic Iceberg Database — velocity and validation

- **Direct:** `https://www.scp.byu.edu/data/iceberg/consolidated_database_v8.0.zip` — **4.1 MB**, one-time download.
- ⚠️ The page header says "last update 13 Oct 2023" — **the file itself covers 1978 to 22 April 2025.** Do not skip this source on the header date.
- This is the highest-value single file on the list: ~47 years of labelled iceberg tracks as plain CSV. It gives you velocity estimation *and* an honest validation set for drift.
- **Indian-satellite lineage, and it is sourced:** BYU's database explicitly ingests **ISRO Oceansat-2 OSCAT and ScatSat OSCAT-2** scatterometer data. That is a defensible statement of Indian satellite contribution to a dataset you actually use — far stronger than forcing in a MOSDAC feed you cannot justify.

#### 5. On Indian data — say this, honestly

**No Indian-hosted dataset was verified as downloadable and physically valid for the Antarctic sea-ice zone within your window.** Do not fake it. Your Indian dimension comes from three real places:

1. **Geography** — the Maitri/Bharati corridor, NCPOR's own stated operating box of **66–70°S, 06–80°E**, and the D15 iceberg cluster sitting inside it.
2. **The charter** — NCPOR tender NCPOR/14(102)/21, its 115-day schedule and its discharge requirements.
3. **Satellite lineage** — ISRO OSCAT/OSCAT-2 inside the BYU database.

If a judge asks "why no Indian data?", the answer is a strength: *"We checked. MOSDAC's current product is built from altimetry, scatterometer winds and OISST, none of which are valid under pack ice, so using it would have been scientifically wrong. We used the Indian operational context instead, and the ISRO scatterometer lineage in the iceberg database we do use."*

---

## 4. FROM EMPTY FOLDER TO FIRST LOADED DATASET

### 4.1 Environment — do this first, all four of you, on Day 1 morning

Use **pip + venv on Python 3.12**. Not conda, not Docker, not Colab as primary.

Rationale, verified: `rasterio` now ships wheels bundling libgdal (needs Python ≥3.10); `cartopy` ships wheels from v0.22+ **but requires Python ≥3.11**; `geopandas`/`shapely`/`pyproj`/`pyogrio` all ship wheels with dependencies included, so GDAL is no longer a separate user install. **Python 3.13+ is forbidden** — that is where the wheel availability gamble starts.

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import sys; print(sys.version)"   # must say 3.12.x
```

### 4.2 `requirements.txt` — pin it, commit it, everyone installs the same thing

```
numpy==2.1.3
pandas==2.2.3
xarray==2024.11.0
netCDF4==1.7.2
scipy==1.14.1
pyproj==3.7.0
rasterio==1.4.3
scikit-learn==1.5.2
xgboost==2.1.3
pyastar2d==1.0.6
requests==2.32.3
matplotlib==3.9.2
fastapi==0.115.5
uvicorn==0.32.1
streamlit==1.40.2
pytest==8.3.4
```

Deliberately absent: `geopandas`, `shapely`, `cartopy`, `folium`, `torch`, `tensorflow`, `xesmf`. You do not need them. `pyproj` handles every coordinate transform you have; `scipy.ndimage` handles every regrid.

If `rasterio` fights anyone's machine, there is a documented escape: one person converts the GeoTIFFs to `.npy` once and commits those. Everyone else needs only numpy. Do not let one person's GDAL problem block three people.

### 4.3 Folder structure — create it on Day 1

```bash
mkdir -p polar-nav/{data/{raw,processed,cache},models,notebooks,configs,scripts,tests}
mkdir -p polar-nav/src/{data,forecast,routing,vessel,uncertainty,mission}
mkdir -p polar-nav/{backend,frontend}
cd polar-nav && git init
printf 'data/raw/\ndata/processed/\n.venv/\n__pycache__/\n*.pyc\nmodels/*.json\n' > .gitignore
touch src/__init__.py src/{data,forecast,routing,vessel,uncertainty,mission}/__init__.py
```

**Commit `data/cache/demo_cache.json` to git.** It is the demo. Everything else under `data/` is regenerable and stays ignored.

### 4.4 `src/data/fetch.py` — download with caching

```python
"""Download raw datasets. Idempotent: skips files already on disk."""
from pathlib import Path
from datetime import date, timedelta
import requests

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BREMEN = ("https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/"
          "s6250/{y}/{mon}/Antarctic/asi-AMSR2-s6250-{ymd}-v5.4.tif")


def fetch(url: str, dest: Path) -> Path | None:
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:                       # missing day, network blip
        print(f"  SKIP {dest.name}: {e}")
        return None
    # Fail loudly if we were served an HTML error page instead of data
    if r.content[:15].lower().startswith(b"<!doctype html") or b"<html" in r.content[:200].lower():
        print(f"  SKIP {dest.name}: got HTML, not data")
        return None
    dest.write_bytes(r.content)
    print(f"  GOT  {dest.name} ({len(r.content)/1024:.0f} KB)")
    return dest


def fetch_sic_range(end: date, ndays: int = 60) -> list[Path]:
    out = []
    for i in range(ndays):
        d = end - timedelta(days=i)
        url = BREMEN.format(y=d.year, mon=d.strftime("%b").lower(),
                            ymd=d.strftime("%Y%m%d"))
        p = fetch(url, RAW / f"sic_{d:%Y%m%d}.tif")
        if p:
            out.append(p)
    return sorted(out)


def fetch_icebergs() -> Path | None:
    return fetch("https://usicecenter.gov/File/DownloadCurrent?pId=134",
                 RAW / "usnic_icebergs.csv")


def fetch_byu() -> Path | None:
    return fetch("https://www.scp.byu.edu/data/iceberg/consolidated_database_v8.0.zip",
                 RAW / "byu_icebergs.zip")


if __name__ == "__main__":
    fetch_sic_range(date.today() - timedelta(days=2), 60)
    fetch_icebergs()
    fetch_byu()
```

Run it: `python -m src.data.fetch`. Expect roughly 60 MB and a few minutes.

### 4.5 First inspection — do NOT skip this

Open a notebook and actually look at the data before writing any pipeline code. Thirty minutes here saves a day later.

```python
import rasterio, numpy as np, matplotlib.pyplot as plt

path = "data/raw/sic_20260919.tif"
with rasterio.open(path) as src:
    print("CRS        :", src.crs)          # VERIFY — do not assume EPSG:3031
    print("shape      :", src.shape)        # VERIFY — expect ~1300x1300 at 6.25 km
    print("transform  :", src.transform)    # pixel -> projected metres
    print("dtype      :", src.dtypes)
    print("nodata     :", src.nodata)
    print("bounds     :", src.bounds)
    sic = src.read(1).astype("float32")

print("min/max    :", np.nanmin(sic), np.nanmax(sic))   # 0-100? 0-1? 0-250?
print("unique tail:", np.unique(sic)[-8:])              # flag values live here
print("nan count  :", np.isnan(sic).sum())

plt.figure(figsize=(6, 6))
plt.imshow(sic, cmap="Blues_r", vmin=0, vmax=100)
plt.colorbar(label="Sea-ice concentration (%)")
plt.title("AMSR2 SIC 2026-09-19")
plt.savefig("notebooks/first_look.png", dpi=110)
```

**What you are checking for, and what to do about it:**

| Check | Why it matters | Action |
|---|---|---|
| `src.crs` | Everything downstream assumes a CRS | Write the real value into `configs/grid.yaml`. Never hardcode a guess. |
| Value range | 0–100, 0–1 and 0–250 all exist in the wild | Normalise to 0–100 once, in the loader, forever |
| Values > 100 | Flag codes (land, missing, pole hole) masquerading as concentration | Mask to NaN **before** any arithmetic |
| NaN count | Drives your gap-filling strategy | If > ~20% over ocean, check you read the right band |
| The picture | Catches flipped axes instantly | Antarctica must look like Antarctica. If it is upside down, fix it now |

**A flipped y-axis is the single most common beginner bug in this project.** Your map will look plausible and every route will be wrong. Plot it, look at it.

---

## 5. THE COMMON ANTARCTIC GRID

### 5.1 Why it must exist

Your sources disagree on everything: Bremen SIC is a 6.25 km polar stereographic raster; USNIC icebergs are DMS point strings; ERA5 is 0.25° lat/lon; bathymetry is its own grid. A\* needs **one array** where cell `(y,x)` means the same place in every layer.

The grid is also your **team integration contract**. Once four people agree that every field is `(216, 216)` float32 in EPSG:3031, they can work independently and their outputs simply stack.

### 5.2 The recommendation: EPSG:3031, 25 km, 216×216

**Projection: EPSG:3031** (Antarctic Polar Stereographic, true scale at 71°S). It is the standard for Antarctic work, it is what judges expect, and it keeps cell areas near-constant — which matters because a lat/lon grid at 70°S has cells roughly a third as wide as at the equator, so distance and risk integrals silently distort.

**Resolution: 25 km.** Justification:

| Resolution | Grid | Cells | Pure-Python A\* | `pyastar2d` | MC N=200 |
|---|---|---|---|---|---|
| 6.25 km | 1300×1300 | 1.7 M | minutes | ~0.3 s | ~40 s |
| **25 km** | **216×216** | **47 k** | **< 1 s** | **~0.01 s** | **~2 s** |

At 25 km everything is interactive even if `pyastar2d` fails to install. A ship 24 m wide does not need 6.25 km routing fidelity for a planning-horizon decision, and you should say exactly that when asked. Precompute **one** 6.25 km hero route to show the architecture scales.

**Extent:** x and y from −2,700 km to +2,700 km in EPSG:3031 covers Antarctica and the Southern Ocean to about 55°S. Note this does **not** include Cape Town at 34°S — see Section 16 for how to handle that on the map.

### 5.3 Implementation

```python
"""src/data/grid.py — the integration contract. Frozen Day 1."""
import numpy as np, xarray as xr, rasterio
from rasterio.warp import reproject, Resampling
from pyproj import CRS, Transformer

GRID_CRS  = CRS.from_epsg(3031)
RES       = 25_000.0                      # metres
HALF      = 2_700_000.0
X = np.arange(-HALF + RES/2,  HALF, RES)  # 216 cells
Y = np.arange( HALF - RES/2, -HALF, -RES) # 216, north-to-south (row 0 = top)
NY, NX = len(Y), len(X)
TRANSFORM = rasterio.transform.from_origin(-HALF, HALF, RES, RES)

_to_xy = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
_to_ll = Transformer.from_crs(GRID_CRS, "EPSG:4326", always_xy=True)


def lonlat_to_rc(lon, lat):
    """Geographic -> (row, col). The function every module uses."""
    x, y = _to_xy.transform(lon, lat)
    col = int(np.floor((x + HALF) / RES))
    row = int(np.floor((HALF - y) / RES))
    return row, col


def rc_to_lonlat(row, col):
    x = -HALF + (col + 0.5) * RES
    y =  HALF - (row + 0.5) * RES
    return _to_ll.transform(x, y)


def regrid_geotiff(path, resampling=Resampling.average):
    """Any GeoTIFF -> our (NY, NX) grid. average for fields, max for hazards."""
    with rasterio.open(path) as src:
        out = np.full((NY, NX), np.nan, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=TRANSFORM, dst_crs=GRID_CRS,
                  resampling=resampling, dst_nodata=np.nan)
    out[out > 100] = np.nan      # flag values -> NaN. VERIFY the threshold first.
    return np.clip(out, 0, 100)


def build_cube(paths, times):
    """Stack daily rasters into an xarray cube -> data/processed/sic.nc"""
    data = np.stack([regrid_geotiff(p) for p in paths])
    return xr.Dataset(
        {"sic": (("time", "y", "x"), data)},
        coords={"time": times, "y": Y, "x": X},
        attrs={"crs": "EPSG:3031", "res_m": RES},
    )
```

### 5.4 ⚠️ The coarsening rule that will otherwise cost you marks

Going from 6.25 km to 25 km averages 16 cells into one. **How you average depends on what the field means:**

- **Sea-ice concentration → `Resampling.average`.** A 25 km cell that is half 80% ice and half open water genuinely behaves like ~40%.
- **Hazards (iceberg presence, land, shallow) → `Resampling.max`.** If one 6.25 km sub-cell contains an iceberg, the 25 km cell contains an iceberg. **Averaging a hazard away is a real bug**, and it is exactly the kind of thing an NCPOR judge will catch: it silently makes dangerous cells look safe.

### 5.5 How each source maps on

| Source | Mapping | Note |
|---|---|---|
| **Bremen SIC** | `reproject(..., Resampling.average)` | Polar stereographic → polar stereographic; near-lossless |
| **Icebergs** | Parse DMS → `lonlat_to_rc()` → stamp a disc of radius `r` | Keep them as *points plus radius*, not a raster, until the risk step. Stamp with `max`. |
| **Bathymetry** | `reproject(..., Resampling.min)` | `min` is conservative: shallowest depth in the cell wins |
| **ERA5 wind** *(optional)* | `scipy.ndimage.map_coordinates` from the 0.25° lat/lon grid | Build the lon/lat of each grid cell once with `rc_to_lonlat`, then sample |
| **A\*** | Consumes `cost[y,x]` directly; neighbours are the 8 adjacent cells | Diagonal step costs `√2 × 25 km` |

---

## 6. SEA-ICE FORECASTING

### 6.1 Answer first: one global model, patch-free, XGBoost

Of the three options you raised:

- **One model per grid cell** — 47,000 models. Absurd at 25 km, impossible at 6.25 km, and each cell has only ~60 training rows.
- **One model per spatial patch** — defensible, but it adds a patch-assignment layer for a marginal gain you cannot measure in a week.
- **One global model, with position as a feature** ✅ — one model, ~2.8 M training rows (47 k cells × 60 days), trains in seconds on a laptop, and learns spatial behaviour through the position and climatology features you give it.

**Recommendation: one global XGBoost regressor.** And it is on probation: if it does not beat persistence on the masked evaluation region, it does not go into the pipeline (Section 7).

### 6.2 What one training sample looks like

One row = **one grid cell on one day**, predicting that cell's SIC at `+lead` days.

```
features:  sic_t0, sic_t-1, sic_t-2, sic_t-3, sic_t-5, sic_t-7,
           sic_mean_7d, sic_std_7d, sic_trend_3d,
           lat, lon, x_km, y_km, dist_to_edge_km,
           doy_sin, doy_cos, lead
target:    sic at (t0 + lead)
```

### 6.3 Feature engineering — what each feature is for

| Feature | Rationale |
|---|---|
| `sic_t0` | The persistence signal. It will dominate, and that is correct. |
| `sic_t-1 … t-7` | Recent trajectory — is this cell freezing or melting? |
| `sic_mean_7d`, `sic_std_7d` | Local stability. High variance = marginal ice zone = where forecasts matter. |
| `sic_trend_3d` | Simple linear slope over 3 days. |
| `lat`, `lon`, `x_km`, `y_km` | Lets one global model learn regional behaviour. |
| `dist_to_edge_km` | Distance to the 15% contour. **The most valuable engineered feature** — almost all forecast error lives near the edge. |
| `doy_sin`, `doy_cos` | Seasonal cycle, encoded cyclically so 31 Dec is adjacent to 1 Jan. |
| `lead` | One model serves all horizons. |

```python
"""src/forecast/features.py"""
import numpy as np
from scipy.ndimage import distance_transform_edt

def distance_to_edge(sic, res_km=25.0):
    ice = sic >= 15
    d_in  = distance_transform_edt(ice)      * res_km
    d_out = distance_transform_edt(~ice)     * res_km
    return np.where(ice, d_in, -d_out)       # signed: + inside pack, - outside

def build_xy(cube, leads=(1, 3, 5, 7), lags=(0, 1, 2, 3, 5, 7)):
    sic  = cube["sic"].values                      # (T, NY, NX)
    time = cube["time"].values
    T, NY, NX = sic.shape
    lat2d, lon2d = cube.attrs["lat2d"], cube.attrs["lon2d"]
    rows, targets, meta = [], [], []
    tmin, tmax = max(lags), T - max(leads)

    for t in range(tmin, tmax):
        base = [sic[t - L] for L in lags]
        win  = sic[t-6:t+1]
        base += [win.mean(0), win.std(0),
                 (sic[t] - sic[t-3]) / 3.0,
                 lat2d, lon2d,
                 np.broadcast_to(cube["x"].values / 1000, (NY, NX)),
                 np.broadcast_to(cube["y"].values[:, None] / 1000, (NY, NX)),
                 distance_to_edge(sic[t])]
        doy = int(np.datetime64(time[t], "D").astype("datetime64[D]").item().timetuple().tm_yday)
        base += [np.full((NY, NX), np.sin(2*np.pi*doy/365.25)),
                 np.full((NY, NX), np.cos(2*np.pi*doy/365.25))]

        for lead in leads:
            X = np.stack(base + [np.full((NY, NX), lead)], -1).reshape(-1, len(base) + 1)
            y = sic[t + lead].reshape(-1)
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            rows.append(X[ok]); targets.append(y[ok])
            meta.append(np.full(ok.sum(), t))     # keep t for chronological split
    return np.vstack(rows), np.concatenate(targets), np.concatenate(meta)
```

### 6.4 Missing values and normalisation

- **Missing:** drop rows where the *target* is NaN. For *features*, forward-fill up to 2 days then drop. **Never fill with zero** — zero means open water, so filling a land or sensor gap with 0 teaches the model that gaps are ocean.
- **Normalisation: not needed.** XGBoost is scale-invariant by construction. Skip it and save the bug surface.

### 6.5 Chronological split — the rule you cannot break

```python
# NEVER use train_test_split(shuffle=True). Adjacent days are near-identical;
# a shuffled split leaks the answer and gives fake 0.99 R^2.
t_train = meta <  T_CUT_1        # oldest ~70%
t_val   = (meta >= T_CUT_1) & (meta < T_CUT_2)   # next ~15%
t_test  = meta >= T_CUT_2        # most recent ~15%, never touched until the end
```

Also, because features use lags up to 7 days and targets up to +7, insert a **7-day embargo gap** between train and validation so no training row's target window overlaps a validation row's feature window.

### 6.6 Train, save, load, predict

```python
"""src/forecast/seaice.py"""
import numpy as np, xgboost as xgb, json
from pathlib import Path

MODEL = Path("models/sic_xgb.json")

def persistence(cube, t, lead):
    """Baseline. Tomorrow looks like today. Deceptively strong."""
    return cube["sic"].values[t].copy()

def train(X_tr, y_tr, X_va, y_va):
    m = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=40, eval_metric="mae",
        tree_method="hist", n_jobs=-1, random_state=42)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)
    MODEL.parent.mkdir(exist_ok=True)
    m.save_model(MODEL)
    return m

def load():
    m = xgb.XGBRegressor()
    m.load_model(MODEL)
    return m

def predict_field(model, features_2d, shape):
    return np.clip(model.predict(features_2d), 0, 100).reshape(shape)
```

Training time on 2.8 M rows × 18 features: well under two minutes on a laptop. If it is slow, cut `n_estimators` to 300 and subsample the training rows by half — it will barely move the score.

---

## 7. FORECAST VALIDATION — AND WHY IT IS A DIFFERENTIATOR

### 7.1 The masking issue that decides whether your numbers mean anything

Roughly **97% of the Antarctic domain is open water or permanent pack** — cells where SIC is 0 or 100 and stays there. A domain-wide MAE is therefore dominated by cells where *nothing happens*, and any model looks excellent.

This is not hypothetical. A competitor team's own committed metrics show MAE **0.003 over open water** versus **0.097 in the marginal ice zone** — a 32× difference hidden inside one headline number.

**Evaluate on masks, and mask on the observation, never the forecast** (masking on your own prediction lets the model choose its own exam):

```python
"""src/forecast/validate.py"""
import numpy as np

def masks(obs):
    return {
        "full_domain": np.isfinite(obs),
        "active":      np.isfinite(obs) & (obs > 0) & (obs < 100),
        "miz":         np.isfinite(obs) & (obs >= 15) & (obs < 80),   # headline
        "edge_band":   np.isfinite(obs) & (np.abs(obs - 15) < 20),
    }

def mae(pred, obs, m):  return float(np.abs(pred[m] - obs[m]).mean())
def rmse(pred, obs, m): return float(np.sqrt(((pred[m] - obs[m])**2).mean()))
def r2(pred, obs, m):
    ss_res = ((obs[m] - pred[m])**2).sum()
    ss_tot = ((obs[m] - obs[m].mean())**2).sum()
    return float(1 - ss_res/ss_tot)
def skill(mae_model, mae_ref):
    return float(1 - mae_model/mae_ref)      # >0 means better than the baseline
```

**Report the MIZ number as your headline.** That is where forecasting is hard and where ships actually operate.

### 7.2 The table you produce

| Horizon | Persistence MAE (MIZ) | XGBoost MAE (MIZ) | Improvement | Skill score | Verdict |
|---|---|---|---|---|---|
| +24 h | *fill from run* | *fill from run* | *computed* | *computed* | ship whichever wins |
| +48 h | | | | | |
| +72 h | | | | | |
| +5 d | | | | | |
| +7 d | | | | | |

**Do not pre-fill these numbers. Run it and report what happens.**

Expect persistence to win at +24 h. That is the normal, published result — this is exactly why damped-anomaly-persistence benchmarks are themselves publishable contributions in the sea-ice literature. If XGBoost beats persistence at short lead by a wide margin, **suspect a leak** before celebrating.

### 7.3 Turn the likely loss into an engineering decision

```python
"""src/forecast/selector.py — ship the baseline where the baseline wins."""
SELECTED = {}   # populated from validation, e.g. {1:"persistence", 3:"xgboost", ...}

def forecast(cube, t, lead, model):
    if SELECTED.get(lead, "persistence") == "persistence":
        return persistence(cube, t, lead)
    return predict_field(model, build_features(cube, t, lead), cube.shape[1:])
```

Said to a judge: *"Persistence wins at 24 hours, so we ship persistence at 24 hours. Our model earns its place from 72 hours out. We measured it rather than assumed it."* That is a demonstration of judgement, not an admission of failure — and across the competitor field, **zero of six teams with public demo videos reported any skill number against any baseline**.

### 7.4 The real question — does a better forecast produce a better decision?

This is Differentiator 3, and it is the strongest thing in the project. Forecast MAE is a proxy nobody outside the field cares about. NCPOR cares whether the *decision* improves.

**Three runs, same routing code, same scoring code:**

| Run | Plan the route on… | Score that route on… | What it tells you |
|---|---|---|---|
| **A — perfect prognosis** | the ice that actually happened | the ice that actually happened | Upper bound. Best achievable with a perfect forecast. |
| **B — operational** | your forecast | the ice that actually happened | The honest number. What a captain would really have got. |
| **C — no model** | persistence | the ice that actually happened | The do-nothing baseline. |

**`cost(C) − cost(B)` is the value of the forecast, in hours and tonnes of fuel.** `cost(B) − cost(A)` is what a perfect forecast would still buy you.

```python
def forecast_value(day, sites):
    obs   = observed_field(day + LEAD)
    f_xgb = model_forecast(day, LEAD)
    f_per = persistence_forecast(day, LEAD)
    A = score_on(plan_on(obs),   obs)
    B = score_on(plan_on(f_xgb), obs)
    C = score_on(plan_on(f_per), obs)
    return {"perfect": A, "ours": B, "persistence": C,
            "value_hours": C["eta_h"] - B["eta_h"],
            "value_fuel_t": C["fuel_t"] - B["fuel_t"],
            "headroom_hours": B["eta_h"] - A["eta_h"]}
```

Run it over ~10 historical days and report the distribution, not one number. **Be prepared for the value to be small or occasionally negative, and report it honestly** — a small measured value is a real result, and it is more credible than any competitor's unvalidated claim.

---

## 8. ICEBERG PREDICTION

Keep this simple. The differentiation is not here.

### 8.1 Dataset structure

**USNIC CSV** (live): iceberg name, last-update date, position as **DMS strings**, length and width in nautical miles. **BYU consolidated database** (history): per-berg CSV of dated positions, 1978 → April 2025.

### 8.2 Preprocessing and velocity

```python
"""src/forecast/iceberg.py"""
import numpy as np, pandas as pd, re

def parse_dms(s):
    """'67°13'S' or '79 25 E' -> signed decimal degrees."""
    nums = [float(v) for v in re.findall(r"[\d.]+", str(s))]
    hemi = re.search(r"[NSEW]", str(s).upper())
    deg = nums[0] + (nums[1]/60 if len(nums) > 1 else 0) + (nums[2]/3600 if len(nums) > 2 else 0)
    if hemi and hemi.group() in "SW":
        deg = -deg
    return deg

def velocity_from_track(track, days=10):
    """Mean velocity over the last `days` of a berg's track. km/day, in projected metres."""
    t = track.sort_values("date").tail(2)
    if len(t) < 2:
        return 0.0, 0.0
    (x0, y0), (x1, y1) = t[["x", "y"]].values
    dt = max((t["date"].iloc[1] - t["date"].iloc[0]).days, 1)
    return (x1 - x0) / dt, (y1 - y0) / dt          # metres/day

def predict_positions(berg, leads_days=(1, 2, 3, 5, 7)):
    """Constant velocity + a linearly widening uncertainty cone."""
    vx, vy = berg["vx"], berg["vy"]
    out = []
    for L in leads_days:
        out.append({
            "lead_days": L,
            "x": berg["x"] + vx * L,
            "y": berg["y"] + vy * L,
            # cone: berg's own radius + growth. 12 km/day is the published
            # position-error growth of an operational drift model - we use it
            # as a documented, conservative uncertainty rate.
            "radius_km": berg["radius_km"] + 12.0 * L,
        })
    return out
```

### 8.3 Validation — 12/24/48 h position error

Hold out the last 90 days of BYU tracks. For each berg and each lead, compare predicted against observed position in km.

| Lead | Persistence (berg does not move) | Constant velocity | Skill |
|---|---|---|---|
| 12 h | | | |
| 24 h | | | |
| 48 h | | | |

⚠️ **The yardstick, and a warning.** The only retrievable *operational* figure is roughly **12 km/day position-error growth**, verified against 17 real tracks. So a 24-hour RMS error around 10 km is a good, honest result.

**If your 24-hour error comes out near 3–4 km, do not celebrate — diagnose.** A competitor reports 3.91 km RMS at 24 h, which would be ~3× better than an operational model, from a hackathon prototype. The likely explanation is scoring against *interpolated* targets: USNIC updates weekly, so if you interpolate positions to daily and then "predict" them, you are scoring your model against its own smooth interpolant.

**One-line diagnostic:** run constant velocity on the identical evaluation pairs. If that *also* scores near-zero error, your targets are interpolated and the number is meaningless. Use only real observation dates as targets, never interpolated ones.

### 8.4 Icebergs → routing risk layer

```python
def iceberg_risk_layer(bergs, lead_days, grid):
    """Stamp discs of predicted position + uncertainty cone. Uses MAX, never mean."""
    risk = np.zeros((grid.NY, grid.NX), dtype="float32")
    yy, xx = np.mgrid[0:grid.NY, 0:grid.NX]
    for b in bergs:
        p = predict_positions(b, (lead_days,))[0]
        r, c = grid.xy_to_rc(p["x"], p["y"])
        rad_cells = max(p["radius_km"] / 25.0, 1.0)
        d = np.sqrt((yy - r)**2 + (xx - c)**2)
        # 1.0 at the predicted centre, decaying linearly to 0 at the cone edge
        risk = np.maximum(risk, np.clip(1.0 - d / rad_cells, 0, 1))
    return risk
```

Two properties that matter: the cone **widens with lead time**, so a route planned 7 days out is automatically more cautious about icebergs than one planned tomorrow; and stamping with `np.maximum` means overlapping bergs never average each other away.

**Do not build a neural network for this.** Constant velocity plus a documented uncertainty rate is defensible, explainable in one sentence, and you can validate it honestly in an hour.

---

## 9. ENVIRONMENTAL RISK / COST MAP

### 9.1 The formulation

```
Risk(x,y) = w_ice · R_ice(x,y)
          + w_berg · R_berg(x,y)
          + w_unc · R_unc(x,y)
          + w_wind · R_wind(x,y)      [optional, Only-If-Time]

subject to:  Risk(x,y) = ∞   if any hard constraint is violated
```

Each `R_*` is normalised to **[0, 1]** so the weights are comparable. Weights sum to 1.

### 9.2 Components

| Term | Definition | Source | Normalisation |
|---|---|---|---|
| `R_ice` | Sea-ice difficulty | Forecast SIC at the lead time the ship arrives | Piecewise: 0 below 15% (open water is free), then rising to 1.0 at 80% where the hard constraint takes over |
| `R_berg` | Iceberg encounter proximity | Predicted position + widening cone (Section 8.4) | Already 0–1 by construction |
| `R_unc` | Forecast uncertainty | Ensemble/perturbation spread of SIC, or `sic_std_7d` as a cheap proxy | `std / 30.0`, clipped to 1 |
| `R_wind` | Wind | ERA5 10 m wind speed | `(U − 10) / 15`, clipped to [0,1] — below 10 m/s costs nothing |

```python
"""src/routing/cost.py"""
import numpy as np

# ---- CONFIGURABLE PROTOTYPE ASSUMPTIONS, NOT SCIENTIFICALLY OPTIMAL VALUES ----
# These weights were chosen by the team for demonstration. They are exposed in
# configs/risk.yaml and in the UI precisely so a domain expert can change them.
# We make no claim that they are calibrated against incident data.
WEIGHTS = {"ice": 0.45, "berg": 0.30, "unc": 0.15, "wind": 0.10}

SIC_FREE, SIC_MAX = 15.0, 80.0     # SIC_MAX = PolarRoute SDA max_ice_conc (documented)
MIN_DEPTH_M = 10.0                 # PolarRoute SDA min_depth (documented)


def r_ice(sic):
    return np.clip((sic - SIC_FREE) / (SIC_MAX - SIC_FREE), 0, 1)

def r_unc(sic_std):
    return np.clip(sic_std / 30.0, 0, 1)

def r_wind(u10):
    return np.clip((u10 - 10.0) / 15.0, 0, 1)


def build_cost(sic, berg_risk, sic_std, depth, land, u10=None, weights=None,
               rio=None, rio_floor=-10.0):
    w = {**WEIGHTS, **(weights or {})}
    comp = {"ice":  w["ice"]  * r_ice(sic),
            "berg": w["berg"] * berg_risk,
            "unc":  w["unc"]  * r_unc(sic_std)}
    comp["wind"] = w["wind"] * r_wind(u10) if u10 is not None else np.zeros_like(sic)

    risk = sum(comp.values())

    # ---- HARD CONSTRAINTS: impassable, not merely expensive ----
    blocked = (
        land
        | (depth < MIN_DEPTH_M)
        | (sic > SIC_MAX)                      # documented SDA accessibility cut-off
        | ~np.isfinite(sic)
    )
    if rio is not None:
        blocked |= (rio < rio_floor)           # POLARIS "special consideration"

    risk = np.where(blocked, np.inf, risk)
    # Return components too - this is what makes the route explainable per cell
    return risk, comp, blocked
```

### 9.3 Soft penalties vs hard constraints — and why the distinction matters

| Condition | Treatment | Justification |
|---|---|---|
| SIC between 15% and 80% | **Soft** | The ship can transit, more slowly and at more cost. That is a trade-off, and trade-offs belong in the cost function. |
| **SIC > 80%** | **HARD — impassable** | PolarRoute's SDA vessel config sets `max_ice_conc: 80`. This is a documented limit from an operational Antarctic vessel model, not our invention. |
| **Depth < 10 m** | **HARD** | SDA `min_depth: 10`. |
| **Land** | **HARD** | Obviously. |
| **POLARIS RIO < −10** | **HARD** | IMO MSC.1/Circ.1519: "operation subject to special consideration". |
| −10 ≤ RIO < 0 | **Soft, plus a speed cap** | Circular Table 1.2 sets recommended speed limits for elevated-risk operation. |
| Iceberg cone | **Soft, rising steeply** | Bergs move; a cone is probabilistic, so a hard block would be overconfident. |
| High forecast uncertainty | **Soft** | Uncertainty is a reason for caution, never a wall. |

**Use `np.inf`, not a large finite number.** A finite penalty of 1e6 lets A\* "buy" its way through a wall if the alternative is long enough, and you will not notice until a judge asks why your route crosses the continent.

### 9.4 Saying this honestly to a judge

Put the weights on screen. Say: *"These are prototype weights, exposed in config and adjustable in the UI. We are not claiming they are calibrated — there is no public Antarctic incident dataset to calibrate them against. What is not arbitrary is the hard constraints: 80% concentration and 10 m depth come from BAS's published vessel model, and the POLARIS threshold comes from IMO circular MSC.1/Circ.1519."*

That distinction — arbitrary weights honestly labelled, hard limits externally sourced — is exactly what separates you from a team whose 0.35/0.30/0.20/0.15 has no answer to "why 0.35?"

---

## 10. VESSEL MODEL — DIFFERENTIATOR 2

The whole point: **every speed, fuel and risk number on your screen has a published derivation.** Competitors show "fuel 45.5" with no origin. You show a number traceable to BAS source code and an IMO circular.

### 10.1 Three-way labelling — put this exact table in your deck

| Layer | What it is | Source | Status |
|---|---|---|---|
| **Ice resistance law** | `R = 0.5·k·Fr^b·ρ·B·h·V²·C^n` | BAS PolarRoute `SDA.py`, MIT licensed | ✅ **Documented open-source implementation**, validated by BAS on RRS *Sir David Attenborough* |
| **Hull coefficients** | slender: k=4.4, b=−0.8267, n=2.0 | Same source | ✅ **Verbatim from source** — ⚠️ the source carries no citation for their origin, so cite PolarRoute, not a paper |
| **Speed inversion** | Solve R = force_limit for V | Same source | ✅ **Documented** |
| **Fuel polynomial** | t/day from speed and resistance | Same source | ✅ **Documented** |
| **Accessibility limits** | SIC ≤ 80%, depth ≥ 10 m | SDA vessel config | ✅ **Documented** |
| **POLARIS RIV table + RIO** | `RIO = Σ Cᵢ·RIVᵢ`, thresholds 0 and −10 | IMO MSC.1/Circ.1519 | ✅ **International regulatory standard** |
| **Ice thickness** | Assumed from SIC | — | ⚠️ **OUR PROTOTYPE ASSUMPTION** |
| **WMO ice stage** | Derived from thickness bands | — | ⚠️ **OUR PROTOTYPE ASSUMPTION** |
| **Vessel ice class** | Assumed 1A Super / PC5 sensitivity | NCPOR tender requires "1A Super or better"; the vessel's assigned class is unverified | ⚠️ **OUR ASSUMPTION — run it as a sensitivity, do not assert** |
| **SIC forecast** | XGBoost or persistence | — | 🔵 **OUR MODEL OUTPUT** |
| **Iceberg positions** | Constant-velocity drift | — | 🔵 **OUR MODEL OUTPUT** |

### 10.2 The implementation — about 60 lines

```python
"""src/vessel/performance.py — ported from BAS PolarRoute SDA.py (MIT).
Source: github.com/bas-logist/PolarRoute  polar_route/vessel_performance/vessels/SDA.py
"""
import numpy as np

G = 9.81
RHO_ICE = 900.0                       # kg/m3

SDA = {            # verbatim from the SDA vessel config
    "max_speed": 26.5,                # km/h  (~14.3 kn)
    "beam": 24.0,                     # m
    "hull_type": "slender",
    "force_limit": 96_634.5,          # N
    "max_ice_conc": 80.0,             # %
    "min_depth": 10.0,                # m
}
HULL = {"slender": (4.4, -0.8267, 2.0), "blunt": (16.1, -1.7937, 3.0)}


def ice_resistance(speed_kmh, sic_pct, thickness_m, cfg=SDA):
    """R_ice in newtons. speed km/h -> m/s internally."""
    k, b, n = HULL[cfg["hull_type"]]
    v = speed_kmh * 5.0 / 18.0
    c = np.clip(sic_pct, 1e-6, 100.0) / 100.0
    froude = v / np.sqrt(G * c * thickness_m)
    return 0.5 * k * froude**b * RHO_ICE * cfg["beam"] * thickness_m * v**2 * c**n


def attainable_speed(sic_pct, thickness_m, cfg=SDA):
    """Closed-form inversion of the resistance law against the force limit."""
    k, b, n = HULL[cfg["hull_type"]]
    vmax = cfg["max_speed"]
    if sic_pct <= 0:
        return vmax
    if sic_pct > cfg["max_ice_conc"]:
        return 0.0                                        # cell inaccessible
    if ice_resistance(vmax, sic_pct, thickness_m, cfg) <= cfg["force_limit"]:
        return vmax
    c = sic_pct / 100.0
    vexp = (2 * cfg["force_limit"] /
            (k * RHO_ICE * cfg["beam"] * thickness_m * c**n *
             (G * thickness_m * c) ** (-b / 2)))
    return float(vexp ** (1.0 / (2.0 + b)) * 18.0 / 5.0)   # m/s -> km/h


def fuel_tonnes_per_day(speed_kmh, resistance_n):
    """Additive separable polynomial, verbatim from SDA.py. Negative R floored at 0."""
    r = max(resistance_n, 0.0)
    return 24.0 * (1.37247e-3 * speed_kmh**2
                   - 2.9601e-3 * speed_kmh
                   + 0.25290433
                   + 7.75218178e-11 * r**2
                   + 6.48113363e-06 * r)
```

### 10.3 POLARIS — 30 lines, and the caveats matter

```python
"""src/vessel/polaris.py — IMO MSC.1/Circ.1519 Table 1.3 (standard conditions)."""
import numpy as np

# Columns: ice-free, new, grey, grey-white, thin-FY-1, thin-FY-2,
#          med-FY<1m, med-FY, thick-FY, second-year, light-MY, heavy-MY
RIV = {
    "PC5":      [3, 3, 3, 3, 2, 2,  1,  1,  0, -1, -2, -2],
    "PC6":      [3, 2, 2, 2, 2, 1,  1,  0, -1, -2, -3, -3],
    "PC7":      [3, 2, 2, 2, 1, 1,  0, -1, -2, -3, -3, -3],
    "IA Super": [3, 2, 2, 2, 2, 1,  0, -1, -2, -3, -4, -4],
    "IA":       [3, 2, 2, 2, 1, 0, -1, -2, -3, -4, -5, -5],
}

def rio(concentrations_tenths, ice_class="IA Super"):
    """RIO = sum(Ci * RIVi).  Ci in TENTHS, must sum to 10."""
    return float(np.dot(np.asarray(concentrations_tenths), RIV[ice_class]))

def category(rio_value, ice_class="IA Super"):
    polar_class = ice_class.startswith("PC")
    if rio_value >= 0:
        return "normal"
    if rio_value >= -10:
        return "elevated" if polar_class else "special_consideration"
    return "special_consideration"


def sic_to_ice_types(sic_pct, month):
    """⚠️ OUR PROTOTYPE ASSUMPTION — NOT a validated ice-type retrieval.

    POLARIS needs WMO ice STAGE. Satellite passive microwave gives CONCENTRATION.
    The best public Antarctic ice-type product (OSI SAF SH) gives only
    first-year / multi-year / ambiguous, misclassifies pancake ice as multi-year,
    and produces nothing at all in summer. So we map concentration to a
    thickness-band proxy and LABEL IT AS OURS.
    """
    c10 = np.clip(sic_pct, 0, 100) / 10.0
    out = np.zeros(12)
    out[0] = 10.0 - c10                      # remainder is ice-free
    if month in (12, 1, 2, 3):               # austral summer -> thinner ice
        out[4] = c10                         # thin FY, 1st stage
    else:
        out[7] = c10                         # medium FY
    return out
```

⚠️ **Two caveats you must state aloud, because they show you read the circular:**

1. **Table 1.4 (decayed ice) may not be used** without ice decay "confirmed by ice information/visual observation by personnel on board qualified in accordance with chapter 12 of the Polar Code". A satellite-only system must use **Table 1.3**. Using 1.4 would be a regulatory misrepresentation.
2. The circular describes POLARIS as a **decision support tool, not a Go/No-Go tool** — which is the IMO itself endorsing exactly the framing your problem statement asks for. Quote it.

### 10.4 The ice-class result — highest impact per line of code in the whole project

NCPOR's tender requires ice class **"equivalent to 1A Super or better"**. PolarRoute's SDA vessel is **PC5**, and so is at least one competitor's on-screen vessel.

Look at the **thick first-year ice** column: **PC5 → RIV 0**, **IA Super → RIV −2**.

At 10/10 thick first-year ice:
- PC5: RIO = 10 × 0 = **0** → *normal operation*
- IA Super: RIO = 10 × (−2) = **−20** → *operation subject to special consideration*

**Same ice, same day, two vessels, and one of them should not be there.** Two public documents — an NCPOR tender and an IMO circular — produce a visibly different accessible corridor on screen. It is one dropdown and two lookups.

### 10.5 How this becomes route cost

```python
def cell_costs(sic, thickness, ice_class, cell_km=25.0):
    v = attainable_speed(sic, thickness)              # km/h
    if v <= 0:
        return np.inf, np.inf, -99.0
    r = ice_resistance(v, sic, thickness)             # N
    hours = cell_km / v
    fuel_t = fuel_tonnes_per_day(v, r) * hours / 24.0
    rio_v = rio(sic_to_ice_types(sic, MONTH), ice_class)
    return hours, fuel_t, rio_v
```

### 10.6 Honest error bounds — say this before a judge asks

Published comparisons of simplified ice-resistance predictions against trials show errors roughly **0.4–9.6% ahead** and up to **26.6% astern**; an operational Arctic ship performance model predicted 870 kg/h against 790 kg/h measured (~10% over). Antarctic ice **thickness** is the weakest input — GIOMAS underestimates Antarctic sea-ice volume by ~38%, and satellite-to-in-situ correlation is only about 0.69–0.73.

**So: absolute fuel is ±20–30%. Relative ranking between routes is considerably better — and ranking is all a route optimiser actually needs.** Say that sentence before the judge says it for you.

---

## 11. A\* ROUTING

**A\* is not your novelty.** Four independent competitor repositories ship a genuine `heapq` A\*. Implement it well, then talk about something else.

### 11.1 Formulation

- **Nodes:** grid cells `(row, col)`, 216×216.
- **Edges:** 8-connected. Cardinal step 25 km, diagonal 35.36 km.
- **g(n):** accumulated cost from start.
- **h(n):** great-circle distance to goal ÷ max speed, in hours — **admissible** because no ship can beat max speed, so A\* stays optimal.
- **f(n) = g(n) + h(n)**

**Edge cost** from cell `i` into neighbour `j`:

```
cost(i→j) = (d_ij / v_j) · (1 + α · Risk_j)  +  β · fuel_ij

  d_ij   step distance (km)
  v_j    attainable speed in j from the vessel model (km/h)
  fuel_ij  tonnes burned crossing j
  α      risk aversion  (0 = ignore risk, 3 = very cautious)
  β      fuel weight, converting tonnes into hours-equivalent
  cost = ∞ if j is blocked
```

The first term is **time in hours**, inflated by risk. The second converts fuel into the same unit via β. Units are consistent, which is what makes the weighted sum legitimate.

### 11.2 Implementation

```python
"""src/routing/astar.py"""
import numpy as np, heapq
try:
    import pyastar2d
    HAVE_PYASTAR = True
except ImportError:
    HAVE_PYASTAR = False


def build_edge_cost(speed_kmh, fuel_t, risk, blocked, alpha=1.0, beta=2.0, cell_km=25.0):
    """Per-cell traversal cost in hours-equivalent. pyastar2d needs float32 >= 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        hours = cell_km / np.where(speed_kmh > 0, speed_kmh, np.nan)
    cost = hours * (1.0 + alpha * np.nan_to_num(risk, nan=1.0)) + beta * fuel_t
    cost = np.where(blocked | ~np.isfinite(cost), np.inf, cost)
    return np.maximum(cost.astype("float32"), 1.0)      # pyastar2d requires >= 1


def route(cost, start_rc, goal_rc):
    if HAVE_PYASTAR:
        grid = np.where(np.isfinite(cost), cost, np.float32(1e9)).astype("float32")
        path = pyastar2d.astar_path(grid, start_rc, goal_rc, allow_diagonal=True)
        if path is None:
            return None
        path = [tuple(p) for p in path]
        if any(not np.isfinite(cost[r, c]) for r, c in path):
            return None                                 # forced through a wall
        return path
    return _astar_python(cost, start_rc, goal_rc)


def _astar_python(cost, start, goal, cell_km=25.0, vmax=26.5):
    NY, NX = cost.shape
    NB = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
          (-1,-1,1.4142),(-1,1,1.4142),(1,-1,1.4142),(1,1,1.4142)]
    def h(n):
        return np.hypot(n[0]-goal[0], n[1]-goal[1]) * cell_km / vmax
    openq = [(h(start), 0.0, start)]
    g = {start: 0.0}; came = {}
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == goal:
            p = [cur]
            while cur in came:
                cur = came[cur]; p.append(cur)
            return p[::-1]
        if gc > g.get(cur, np.inf):
            continue
        for dr, dc, w in NB:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0 <= nr < NY and 0 <= nc < NX):
                continue
            c = cost[nr, nc]
            if not np.isfinite(c):
                continue
            ng = gc + c * w
            if ng < g.get((nr, nc), np.inf):
                g[(nr, nc)] = ng; came[(nr, nc)] = cur
                heapq.heappush(openq, (ng + h((nr, nc)), ng, (nr, nc)))
    return None
```

### 11.3 How constraints are actually enforced

| Constraint | Mechanism |
|---|---|
| Land / shallow / SIC > 80% / RIO < −10 | `cost = ∞` in `build_edge_cost`. A\* never expands them. |
| Iceberg proximity | Soft, via `Risk` — steep but finite, because cones are probabilistic. |
| pyastar2d's finite-cost requirement | It cannot take `inf`, so blocked cells become `1e9` — **then the returned path is re-validated against the true cost array and rejected if it crossed one.** Without that check A\* can tunnel through a "wall" if the detour is long enough. |
| Route modes | Change `alpha` and `beta`, re-solve. Same code. |

### 11.4 Route modes — define them, never call them novel

| Mode | α | β | Behaviour |
|---|---|---|---|
| **Fastest** | 0.0 | 0.0 | Minimum transit time; ignores risk beyond hard limits |
| **Fuel-efficient** | 0.5 | 6.0 | Prefers open water and lower speed-in-ice |
| **Safety-conservative** | 3.0 | 1.0 | Wide berth from ice and iceberg cones |
| **Balanced** | 1.0 | 2.0 | Default |

⚠️ **The claim you must not make.** Sweeping weights and presenting the results as "the Pareto front" is **an overclaim**: weighted-sum scalarisation provably cannot reach solutions on non-convex regions of the front. Correct wording:

> *"Non-dominated candidate routes found by scalarised A\* over a weight sweep. This recovers supported (convex-hull) efficient solutions only."*

If you want the honest upgrade, add a **risk-budget mode** (ε-constraint): minimise time subject to `total_risk ≤ B`. It *can* reach non-convex regions, and "keep me under this much risk exposure" is how a navigator actually thinks. One extra constraint check in the cost function.

---

## 12. MULTIPLE CANDIDATE DISCHARGE SITES — DIFFERENTIATOR 1

**This is the project.** Everything above is infrastructure for this section.

### 12.1 Why this reframes the problem

From NCPOR's own charter tender NCPOR/14(102)/21:

| Leg | Days |
|---|---|
| Cape Town → Larsemann Hills (Bharati) | **16 transit** |
| At Larsemann Hills | **40 on station** |
| Larsemann Hills → India Bay (Maitri) | **8 transit** |
| At India Bay | **30 on station** |
| India Bay → Cape Town | **10 transit** |
| **Total charter** | **115 days** |

**34 days of transit. 70 days stationary, discharging cargo.** A system that optimises only the sailing leg is optimising 30% of the charter.

And the destination is **not a point**. The charter party requires the vessel to reach the point *"closest to landing area chosen by Master of the vessel(s) in consultation with the Leader of the Expedition for overboard discharge of Charterer's cargo/equipment on fast ice for haulage to the inland area."* The ASMA 6 management plan for Larsemann Hills states **"no anchorages or barge landings are designated"**; vessels anchor roughly 5 nautical miles offshore. For Maitri, the discharge point — the "Indian Barrier" — is a shelf-ice edge about **100 km north of the station**.

And it can fail. In **April 2012** a slab of shelf ice roughly **2 km × 1 km** adjoining the Indian Barrier calved away **carrying Indian tank containers and Russian cargo and fuel out to sea**; RADARSAT found no metallic signature afterwards, meaning the cargo most likely sank. In the same season, *"both Russian and Indian expedition vessels could not reach their respective barriers in February/March 2012."*

So the real question is not *"what is the shortest safe path to a pin?"* It is:

> **Which candidate discharge site can we reach, and hold, inside the window we have paid for?**

### 12.2 Representing candidate sites

```python
"""configs/sites.py"""
CANDIDATE_SITES = [
    {"id": "BHARATI_N",  "name": "Bharati — north approach",
     "lon": 76.196, "lat": -69.407,           # 69°24.41'S 76°11.72'E
     "haul_km": 5,   "station": "Bharati",
     "note": "ASMA 6: no designated anchorage; vessels anchor ~5 nm offshore"},

    {"id": "BHARATI_W",  "name": "Bharati — western fast-ice edge",
     "lon": 75.90,  "lat": -69.35,
     "haul_km": 18,  "station": "Bharati",
     "note": "Longer haul, historically more stable fast ice"},

    {"id": "INDIAN_BARRIER", "name": "Indian Barrier (India Bay)",
     "lon": 11.60,  "lat": -70.00,            # approximate: ~100 km N of Maitri
     "haul_km": 100, "station": "Maitri",
     "note": "2012 calving event lost cargo here — elevated calving risk",
     "calving_risk": "documented"},

    {"id": "INDIA_BAY_E", "name": "India Bay — eastern alternative",
     "lon": 12.40,  "lat": -69.90,
     "haul_km": 130, "station": "Maitri",
     "note": "Alternative shelf edge, longer overland haul"},
]

START = {"id": "CAPE_TOWN", "lon": 18.42, "lat": -33.92}

MISSION = {
    "charter_days": 115,            # from the NCPOR tender schedule
    "window_days": 16,              # planned Cape Town -> Larsemann transit
    "fuel_budget_t": 900,           # ⚠️ OUR PROTOTYPE ASSUMPTION
    "ice_class": "IA Super",        # tender requires "1A Super or better"
    "haul_rate_km_per_day": 40,     # ⚠️ OUR PROTOTYPE ASSUMPTION
}
```

⚠️ **Honesty note.** Exact coordinates of the Indian Barrier discharge point are **unverified** — public sources give only "Princess Astrid Coast, ~100 km north of Maitri" and the tender's 66–70°S / 06–80°E operating box. Label these as **indicative candidate sites derived from published descriptions**, not surveyed positions. A judge will respect that far more than false precision.

### 12.3 `evaluate_mission()`

```python
"""src/mission/evaluate.py"""
import numpy as np
from src.routing.astar import route, build_edge_cost
from src.vessel.performance import attainable_speed, ice_resistance, fuel_tonnes_per_day
from src.vessel.polaris import rio, category, sic_to_ice_types


def evaluate_mission(site, start, fields, mission, alpha=1.0, beta=2.0, grid=None):
    """One candidate discharge site -> full decision record. No magic score."""
    speed, fuel_cell, rio_cell = fields["speed"], fields["fuel"], fields["rio"]
    cost = build_edge_cost(speed, fuel_cell, fields["risk"], fields["blocked"], alpha, beta)

    s_rc = grid.lonlat_to_rc(start["lon"], start["lat"])
    g_rc = grid.lonlat_to_rc(site["lon"],  site["lat"])
    path = route(cost, s_rc, g_rc)

    if path is None:
        return {"site_id": site["id"], "name": site["name"], "reachable": False,
                "feasibility": "INFEASIBLE",
                "reasons": ["No ice-free-enough corridor exists to this site"]}

    # ---- integrate along the path ----
    steps = np.array([np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])])
    d_km  = float((steps * 25.0).sum())
    hours = float(sum(25.0 * s / max(speed[r, c], 1e-6) for s, (r, c) in zip(steps, path[1:])))
    fuel  = float(sum(fuel_cell[r, c] * s for s, (r, c) in zip(steps, path[1:])))

    sic_path = np.array([fields["sic"][r, c] for r, c in path])
    rio_path = np.array([rio_cell[r, c]      for r, c in path])
    risk_path = np.array([fields["risk"][r, c] for r, c in path])

    eta_days  = hours / 24.0
    haul_days = site["haul_km"] / mission["haul_rate_km_per_day"]
    total_days = eta_days + haul_days

    # ---- feasibility: transparent, rule-based, every reason enumerated ----
    reasons, status = [], "FEASIBLE"
    if total_days > mission["window_days"]:
        status = "INFEASIBLE"
        reasons.append(f"{total_days:.1f} d needed vs {mission['window_days']} d window")
    elif total_days > 0.85 * mission["window_days"]:
        status = "MARGINAL"
        reasons.append(f"Uses {100*total_days/mission['window_days']:.0f}% of the window")
    if fuel > mission["fuel_budget_t"]:
        status = "INFEASIBLE"; reasons.append(f"Fuel {fuel:.0f} t over budget {mission['fuel_budget_t']} t")
    if (rio_path < 0).any():
        n = int((rio_path < 0).sum())
        if status == "FEASIBLE":
            status = "MARGINAL"
        reasons.append(f"POLARIS elevated risk in {n} cells ({n*25} km)")
    if site.get("calving_risk") == "documented":
        if status == "FEASIBLE":
            status = "MARGINAL"
        reasons.append("Documented calving event at this site (April 2012, cargo lost)")

    return {
        "site_id": site["id"], "name": site["name"], "reachable": True,
        "distance_km": round(d_km),
        "eta_hours": round(hours, 1),
        "eta_days": round(eta_days, 2),
        "haul_days": round(haul_days, 2),
        "total_days": round(total_days, 2),
        "fuel_tonnes": round(fuel, 1),
        "mean_sic": round(float(sic_path.mean()), 1),
        "max_sic": round(float(sic_path.max()), 1),
        "hours_above_sic70": round(float(sum(
            25.0 * s / max(speed[r, c], 1e-6)
            for s, (r, c) in zip(steps, path[1:]) if fields["sic"][r, c] > 70)), 1),
        "min_rio": round(float(rio_path.min()), 1),
        "polaris_category": category(float(rio_path.min()), mission["ice_class"]),
        "risk_mean": round(float(risk_path.mean()), 3),
        "risk_max": round(float(risk_path.max()), 3),
        "feasibility": status,
        "reasons": reasons,
        "path_rc": [[int(r), int(c)] for r, c in path],
        "path_lonlat": [list(grid.rc_to_lonlat(r, c)) for r, c in path],
    }
```

### 12.4 Comparing sites — no black-box score

```python
def compare_sites(sites, start, fields, mission, **kw):
    out = [evaluate_mission(s, start, fields, mission, **kw) for s in sites]
    rank = {"FEASIBLE": 0, "MARGINAL": 1, "INFEASIBLE": 2}
    return sorted(out, key=lambda m: (rank[m["feasibility"]], m.get("total_days", 1e9)))
```

Sorting is by **feasibility class first, then time** — both fully explainable. The dashboard shows every column. There is deliberately no weighted composite "site score", because the trade-off between 1.5 extra days and 95 km less overland haul is a **command decision**, not an arithmetic one.

**This is the sentence for the judge:** *"We do not rank the sites for the Voyage Leader. We show what each one costs in days, tonnes, ice exposure and regulatory margin, and let them decide — which is what a decision-support system is, and what the IMO circular explicitly says POLARIS is for."*

---

## 13. UNCERTAINTY / MONTE CARLO

### 13.1 The design — re-score, do not re-plan

```
forecast field
      |
      +-- perturb N=200 times (spatially correlated noise)
      |
      +-- for each: re-SCORE the FIXED route (do not re-run A*)
      |
      v
distribution of ETA, fuel, blockage  ->  P10 / P50 / P90, P(blocked)
```

**Re-scoring a fixed polyline is O(waypoints), not O(grid).** Re-planning 200 times would be 200 A\* runs; re-scoring is 200 array lookups. At 25 km this is ~2 seconds. That distinction is what makes the whole thing feasible.

### 13.2 ⚠️ The perturbation must be spatially correlated

```python
"""src/uncertainty/montecarlo.py"""
import numpy as np
from scipy.ndimage import gaussian_filter


def perturb_sic(sic, sigma_pct=10.0, corr_km=150.0, cell_km=25.0, rng=None):
    """Spatially correlated SIC perturbation.

    ⚠️ CRITICAL: the noise MUST be spatially correlated. Per-cell iid noise
    averages out along a route of ~200 cells, producing an absurdly tight ETA
    spread (minutes on a 16-day voyage) — the classic tell of naive Monte Carlo.
    Forecast errors are correlated over ~100-200 km, so we smooth white noise
    to that scale and re-standardise.
    """
    rng = rng or np.random.default_rng()
    field = gaussian_filter(rng.standard_normal(sic.shape), sigma=corr_km / cell_km)
    field /= (field.std() + 1e-9)
    return np.clip(sic + sigma_pct * field, 0, 100)


def mc_route(path_rc, sic, thickness, ice_class, n=200, sigma_pct=10.0, seed=42):
    from src.vessel.performance import attainable_speed, ice_resistance, fuel_tonnes_per_day
    rng = np.random.default_rng(seed)
    etas, fuels, blocked_runs = [], [], 0

    for _ in range(n):
        s = perturb_sic(sic, sigma_pct, rng=rng)
        h = f = 0.0
        blocked = False
        for (r, c) in path_rc[1:]:
            v = attainable_speed(s[r, c], thickness[r, c])
            if v <= 0:                       # SIC pushed above the 80% limit
                blocked = True
                break
            dt = 25.0 / v
            h += dt
            f += fuel_tonnes_per_day(v, ice_resistance(v, s[r, c], thickness[r, c])) * dt / 24.0
        if blocked:
            blocked_runs += 1
        else:
            etas.append(h); fuels.append(f)

    if not etas:
        return {"p_blocked": 1.0, "n": n, "note": "blocked in every realisation"}
    e, fu = np.array(etas), np.array(fuels)
    return {
        "eta_p10_h": round(float(np.percentile(e, 10)), 1),
        "eta_p50_h": round(float(np.percentile(e, 50)), 1),
        "eta_p90_h": round(float(np.percentile(e, 90)), 1),
        "eta_spread_h": round(float(np.percentile(e, 90) - np.percentile(e, 10)), 1),
        "fuel_p50_t": round(float(np.percentile(fu, 50)), 1),
        "fuel_p90_t": round(float(np.percentile(fu, 90)), 1),
        "p_blocked": round(blocked_runs / n, 3),
        "cvar_90_h": round(float(e[e >= np.percentile(e, 90)].mean()), 1),
        "n": n,
    }
```

### 13.3 Mission feasibility under uncertainty

```python
def p_feasible(mc, window_days):
    """Fraction of realisations completing inside the window."""
    if mc.get("p_blocked", 0) >= 1.0:
        return 0.0
    return round((1 - mc["p_blocked"]) * float(mc["eta_p90_h"] / 24.0 <= window_days), 3)
```

### 13.4 The slider trap — write this test on Day 5

A variance penalty does **not** always make routes more conservative. If your uncertainty field σ is proportional to your risk field μ — which happens naturally if you derive "uncertainty" from ice magnitude rather than real spread — then `μ + λσ = μ(1 + λα)`, and the argmin path is **identical for every λ**. Your slider moves and nothing happens, silently.

```python
def test_risk_slider_actually_changes_route():
    r0 = route(build_edge_cost(**f, alpha=0.0), s, g)
    r3 = route(build_edge_cost(**f, alpha=3.0), s, g)
    assert r0 != r3, "Risk slider is a no-op — sigma is probably collinear with mu"
```

Also be honest about one more thing: summing per-cell standard deviations along a path is **not** the true path standard deviation (the correct objective is `μᵀx + c·√(τᵀx)`). The additive form is an upper bound, and it is defensible **only because** forecast errors are spatially correlated. State the assumption rather than hiding it.

### 13.5 What not to claim

**Robust and ensemble-based weather routing is published prior art.** Do not call Monte Carlo route evaluation novel. What you can say is narrower and true: *"we propagate forecast uncertainty into the mission feasibility decision, not just into the route geometry."*

### 13.6 On the UI

- **ETA fan** — a P10–P90 band behind the route line, widening with distance.
- **Percentile row** in the comparison table: `P10 / P50 / P90`.
- **`P(blocked)`** as a bar, with any value above ~0.2 in red.
- **Never draw a single deterministic iceberg position.** Always the cone.

---

## 14. MISSION FEASIBILITY

### 14.1 Definition

> **Mission feasibility** is the estimated probability, under our stated model assumptions, that the vessel can reach a given discharge site, complete cargo transfer, and do so inside the available charter window and fuel budget, without violating a hard operating limit.

It is **not** "safe". It is a probability under assumptions, and the assumptions are listed on screen.

### 14.2 The rule

```
FEASIBLE    — all of:
                total_days ≤ 0.85 × window_days
                fuel_p90   ≤ fuel_budget_t
                min RIO    ≥ 0          (POLARIS normal operation)
                P(blocked) ≤ 0.10
                no hard constraint violated anywhere on the route

MARGINAL    — reachable, but one or more of:
                total_days in (0.85, 1.00] × window
                fuel_p90 within 10% of budget
                min RIO in [−10, 0)      (elevated operational risk)
                P(blocked) in (0.10, 0.30]
                documented site-specific hazard (e.g. calving history)

INFEASIBLE  — any of:
                no route exists
                total_days > window_days
                fuel_p90 > budget
                min RIO < −10            (special consideration)
                P(blocked) > 0.30
```

`total_days = ETA_p50/24 + haul_km / haul_rate`. Including the **overland haul** is what makes this a mission model rather than a routing model — a site 1.5 days closer by sea but 95 km further overland may well be the worse choice, and only this formulation can show that.

### 14.3 Estimated feasibility vs guaranteed safety — put this on a slide

| | Our system says | Our system does **not** say |
|---|---|---|
| Ice | Estimated concentration from a validated forecast with stated skill | That the ice will be as forecast |
| Route | No cell exceeds documented operating limits **in our model** | That the route is safe |
| Fuel | ±20–30% estimate; ranking more reliable than absolutes | An operational bunkering figure |
| POLARIS | RIO from Table 1.3 using **our** ice-type proxy | A certified POLARIS assessment |
| Feasibility | Probability under our assumptions | A guarantee |
| Decision | Support for the Master and Voyage Leader | A replacement for their judgement |

The charter party names the decision-makers explicitly: the landing area is *"chosen by Master of the vessel(s) in consultation with the Leader of the Expedition."* Your system informs those two people. Say so — it demonstrates you understood who the user is.

---

## 15. BACKEND

Thin by design. It serves precomputed JSON. It does **not** compute during the demo.

```python
"""backend/main.py"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json, pathlib

app = FastAPI(title="PolarNav PS59")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CACHE = json.loads(pathlib.Path("data/cache/demo_cache.json").read_text())
app.mount("/static", StaticFiles(directory="frontend"), name="static")


class MissionRequest(BaseModel):
    start_id: str = "CAPE_TOWN"
    date: str = "2026-09-19"
    lead_days: int = 3
    alpha: float = 1.0          # risk aversion
    beta: float = 2.0           # fuel weight
    ice_class: str = "IA Super"


def key(r: MissionRequest) -> str:
    return f"{r.date}|{r.lead_days}|{r.alpha}|{r.beta}|{r.ice_class}"
```

| Endpoint | Request | Response | Calls |
|---|---|---|---|
| `GET /candidate-sites` | — | site list with lon/lat, haul_km, notes | `configs/sites.py` |
| `GET /forecast/sea-ice?date&lead` | query | `{grid_meta, sic[][], contour_geojson}` | `forecast.seaice` |
| `GET /forecast/iceberg?date&lead` | query | `{bergs:[{name,lon,lat,radius_km,cone_km}]}` | `forecast.iceberg` |
| `POST /route` | `{start, goal, alpha, beta}` | `{path_lonlat, distance_km, eta_hours, fuel_tonnes}` | `routing.astar` |
| `POST /mission/evaluate` | `MissionRequest` | `{missions:[…]}` — the comparison table | `mission.evaluate` |
| `POST /uncertainty` | `{site_id, n}` | `{eta_p10,p50,p90, p_blocked, cvar_90}` | `uncertainty.montecarlo` |
| `GET /validation` | — | forecast skill tables + forecast-value result | precomputed |
| `GET /health` | — | `{status, cache_keys, build_time}` | — |

```python
@app.post("/mission/evaluate")
def evaluate(req: MissionRequest):
    k = key(req)
    if k not in CACHE["missions"]:
        raise HTTPException(404, f"Not precomputed: {k}")
    return {"request": req.model_dump(), "missions": CACHE["missions"][k]}
```

**Sample response** — this is the shape the frontend renders:

```json
{
  "request": {"date": "2026-09-19", "lead_days": 3, "alpha": 1.0, "ice_class": "IA Super"},
  "missions": [
    {
      "site_id": "BHARATI_W", "name": "Bharati — western fast-ice edge",
      "reachable": true, "distance_km": 4712,
      "eta_hours": 231.4, "eta_days": 9.64, "haul_days": 0.45, "total_days": 10.09,
      "fuel_tonnes": 486.2,
      "mean_sic": 34.8, "max_sic": 71.2, "hours_above_sic70": 3.1,
      "min_rio": 2.0, "polaris_category": "normal",
      "risk_mean": 0.221, "risk_max": 0.640,
      "uncertainty": {"eta_p10_h": 219.0, "eta_p50_h": 232.6, "eta_p90_h": 258.9,
                      "p_blocked": 0.02, "cvar_90_h": 264.1},
      "feasibility": "FEASIBLE", "reasons": [],
      "path_lonlat": [[18.42,-33.92], "…"]
    },
    {
      "site_id": "INDIAN_BARRIER", "name": "Indian Barrier (India Bay)",
      "reachable": true, "distance_km": 4344,
      "eta_days": 8.9, "haul_days": 2.5, "total_days": 11.4,
      "fuel_tonnes": 502.7, "min_rio": -4.0, "polaris_category": "elevated",
      "uncertainty": {"eta_p90_h": 289.0, "p_blocked": 0.14},
      "feasibility": "MARGINAL",
      "reasons": ["POLARIS elevated risk in 6 cells (150 km)",
                  "Documented calving event at this site (April 2012, cargo lost)"]
    }
  ]
}
```

---

## 16. FRONTEND

### 16.1 The map — build `map.html` on Day 1, before choosing a framework

`folium` **cannot** render EPSG:3031, so a Python map wrapper is not an option. Write plain Leaflet once, reuse everywhere.

NASA maintains a working Antarctic EPSG:3031 Leaflet example using **Leaflet 1.9.4 + proj4js 2.20.2 + Proj4Leaflet 1.0.1** (`nasa-gibs/gibs-web-examples`, `examples/leaflet/antarctic-epsg3031.js`). Copy its CRS definition rather than deriving your own.

```html
<!-- frontend/map.html — framework-agnostic. Works in Streamlit, React and FastAPI. -->
<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/proj4js/2.20.2/proj4.js"></script>
<script src="https://unpkg.com/proj4leaflet@1.0.2/src/proj4leaflet.js"></script>
<style>html,body,#map{height:100%;margin:0}</style></head>
<body><div id="map"></div><script>
// EPSG:3031 Antarctic Polar Stereographic
const crs = new L.Proj.CRS('EPSG:3031',
  '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
  { origin: [-4194304, 4194304],
    resolutions: [8192, 4096, 2048, 1024, 512, 256] });

const map = L.map('map', { crs, center: [-90, 0], zoom: 1, maxZoom: 5 });

// REQUIRED: 1-px tile seam workaround from NASA's example. Without this you get
// grid lines across Antarctica and it looks like a rendering bug on stage.
const _origInit = L.GridLayer.prototype._initTile;
L.GridLayer.include({ _initTile: function (tile) {
  _origInit.call(this, tile);
  const s = this.getTileSize();
  tile.style.width  = s.x + 1 + 'px';
  tile.style.height = s.y + 1 + 'px';
}});

// Free, keyless, EPSG:3031 operational ice layers — no NetCDF processing needed
L.tileLayer.wms('https://geos.polarview.aq/geoserver/wms',
  { layers: 'seaice_conc', format: 'image/png', transparent: true, opacity: 0.75 }
).addTo(map);

function render(data) {                       // called with the API's JSON
  (window._layers || []).forEach(l => map.removeLayer(l));
  window._layers = [];
  const colour = { FEASIBLE: '#2e7d32', MARGINAL: '#f9a825', INFEASIBLE: '#c62828' };

  data.missions.forEach(m => {
    if (!m.reachable) return;
    const line = L.polyline(m.path_lonlat.map(p => [p[1], p[0]]),
      { color: colour[m.feasibility], weight: 3,
        dashArray: m.feasibility === 'MARGINAL' ? '8,6' : null })
      .bindPopup(`<b>${m.name}</b><br>${m.total_days} d · ${m.fuel_tonnes} t<br>
                  POLARIS: ${m.polaris_category}<br>P(blocked) ${m.uncertainty.p_blocked}`);
    line.addTo(map); window._layers.push(line);
  });

  (data.bergs || []).forEach(b => {           // cone, never a point
    const c = L.circle([b.lat, b.lon], { radius: b.cone_km * 1000,
      color: '#5e35b1', fillOpacity: 0.18, weight: 1 }).bindTooltip(b.name);
    c.addTo(map); window._layers.push(c);
  });
}
window.addEventListener('message', e => { if (e.data?.missions) render(e.data); });
</script></body></html>
```

⚠️ **Two map gotchas to plan around:**

1. **Cape Town falls off the tile grid.** The GIBS EPSG:3031 tile matrix bounds are about ±4,194,304 m, which reaches only into the low-50s °S. Cape Town at 34°S is outside it. Options: start your demo route at the **ice edge (~60°S)** and show the Cape Town leg as a labelled straight line; or add a plain lat/lon inset for the open-ocean leg; or simply state that the map covers the operational area and the northern transit is not ice-constrained. The third is honest and costs nothing.
2. **Proj4Leaflet is unmaintained** (last release 2017), though NASA validates it against Leaflet 1.9.4 in a repo updated through 2026. **Vendor exact versions locally** rather than hot-linking to a CDN `latest`.

### 16.2 The shell — Streamlit recommended, React possible

**Recommended: Streamlit**, because it costs hours instead of days and the map file is identical either way.

```python
"""app.py"""
import streamlit as st, requests, pandas as pd, pathlib

st.set_page_config(page_title="PolarNav — PS59", layout="wide")
st.title("Antarctic Mission Decision Support — SIH26059")

with st.sidebar:
    st.header("Mission")
    date  = st.selectbox("Forecast date", ["2026-09-19", "2026-09-12", "2026-09-05"])
    lead  = st.select_slider("Lead time (days)", [1, 3, 5, 7], value=3)
    alpha = st.select_slider("Risk aversion", [0.0, 1.0, 2.0, 3.0], value=1.0)
    beta  = st.select_slider("Fuel weight",   [0.0, 2.0, 4.0, 6.0], value=2.0)
    iclass = st.selectbox("Ice class", ["IA Super", "PC5", "PC7"])
    st.caption("Tender requires '1A Super or better'. PC5 shown for comparison.")

@st.cache_data
def evaluate(d, l, a, b, c):
    return requests.post("http://localhost:8000/mission/evaluate",
                         json={"date": d, "lead_days": l, "alpha": a,
                               "beta": b, "ice_class": c}, timeout=10).json()

data = evaluate(date, lead, alpha, beta, iclass)

left, right = st.columns([3, 2])
with left:
    html = pathlib.Path("frontend/map.html").read_text()
    st.components.v1.html(
        html + f"<script>render({data});</script>", height=620)
with right:
    st.subheader("Candidate discharge sites")
    df = pd.DataFrame([{
        "Site": m["name"],
        "Total days": m.get("total_days"),
        "Fuel (t)": m.get("fuel_tonnes"),
        "Max SIC %": m.get("max_sic"),
        "POLARIS": m.get("polaris_category"),
        "P(blocked)": m.get("uncertainty", {}).get("p_blocked"),
        "Feasibility": m["feasibility"],
    } for m in data["missions"]])
    st.dataframe(df, use_container_width=True, hide_index=True)

    for m in data["missions"]:
        if m.get("reasons"):
            st.warning(f"**{m['name']}** — " + "; ".join(m["reasons"]))
```

Two Streamlit specifics that will bite otherwise: **use discrete `st.select_slider`, not continuous `st.slider`** — a finite parameter space is what lets you precompute every combination; and **`@st.cache_data` and `@st.fragment` cannot decorate the same function**, so cache the fetch and call it from the fragment.

**If you insist on React:** keep `map.html` in an `<iframe>` and `postMessage` the JSON into it. Everything else — table, sidebar — is ordinary React. Budget 2.5–3.5 days and decide on **Day 4**, not Day 1.

### 16.3 What the frontend receives

Exactly the `/mission/evaluate` payload in Section 15 — one array of mission records. The map draws `path_lonlat`, colours by `feasibility`, and plots iceberg cones. The table renders scalar fields. The warnings render `reasons`. **One request, one render.** No streaming, no websockets, no live computation.

---

## 17. PROJECT FOLDER STRUCTURE

```text
polar-nav/
├── configs/
│   ├── grid.yaml              # CRS, resolution, extent — VERIFY values Day 1
│   ├── sites.py               # CANDIDATE_SITES, START, MISSION
│   ├── risk.yaml              # weights (labelled prototype assumptions)
│   └── vessel.yaml            # SDA config + ice class
├── data/
│   ├── raw/                   # .gitignore — downloaded GeoTIFF/CSV/ZIP
│   ├── processed/             # .gitignore — sic.nc, fields.npz
│   └── cache/demo_cache.json  # COMMITTED. This is the demo.
├── models/sic_xgb.json        # trained model (committed, it is small)
├── notebooks/
│   ├── 01_inspect_data.ipynb
│   ├── 02_train_forecast.ipynb
│   └── 03_validation_tables.ipynb
├── src/
│   ├── interfaces.py          # ⭐ Day 1: signatures + fakes for EVERY module
│   ├── data/
│   │   ├── fetch.py           # downloads, idempotent
│   │   ├── grid.py            # ⭐ the integration contract
│   │   └── masks.py           # land, depth, ASPA polygons
│   ├── forecast/
│   │   ├── features.py        # lag/spatial/seasonal feature builder
│   │   ├── seaice.py          # persistence + XGBoost train/load/predict
│   │   ├── iceberg.py         # DMS parse, velocity, drift, risk layer
│   │   ├── validate.py        # masks + MAE/RMSE/R2/skill
│   │   └── selector.py        # lead-time-dependent model choice
│   ├── routing/
│   │   ├── cost.py            # risk raster + hard constraints
│   │   └── astar.py           # pyastar2d with pure-Python fallback
│   ├── vessel/
│   │   ├── performance.py     # SDA resistance, speed, fuel
│   │   └── polaris.py         # RIV table, RIO, category
│   ├── uncertainty/montecarlo.py
│   └── mission/evaluate.py    # ⭐ Differentiator 1
├── scripts/
│   ├── build_cache.py         # sweep all discrete params -> demo_cache.json
│   └── run_validation.py      # produces the tables for the deck
├── backend/main.py
├── frontend/map.html          # ⭐ Day 1, framework-agnostic
├── app.py                     # Streamlit shell
├── tests/
├── requirements.txt
└── README.md
```

### 17.1 `src/interfaces.py` — the single highest-value file in the repo

Write this on **Day 1 morning**, with fake implementations of every cross-person function. Then four people develop against stable signatures, and integration on Day 5 is a merge rather than a crisis.

```python
"""src/interfaces.py — contracts + fakes. Replace fakes as real code lands."""
import numpy as np
NY = NX = 216

def fake_sic():     return np.clip(np.random.rand(NY, NX) * 100, 0, 100).astype("float32")
def fake_mask():    return np.zeros((NY, NX), bool)
def fake_speed():   return np.full((NY, NX), 20.0, "float32")
def fake_fuel():    return np.full((NY, NX), 0.5,  "float32")
def fake_risk():    return np.random.rand(NY, NX).astype("float32")
def fake_rio():     return np.full((NY, NX), 1.0,  "float32")

def fake_fields():
    return {"sic": fake_sic(), "speed": fake_speed(), "fuel": fake_fuel(),
            "risk": fake_risk(), "rio": fake_rio(), "blocked": fake_mask(),
            "sic_std": np.full((NY, NX), 5.0, "float32")}

# CONTRACT — every real implementation must match these exactly:
#   grid.lonlat_to_rc(lon, lat)              -> (row, col)
#   grid.rc_to_lonlat(row, col)              -> (lon, lat)
#   forecast.seaice.predict(date, lead)      -> float32 (NY, NX), 0-100, NaN allowed
#   forecast.iceberg.risk_layer(date, lead)  -> float32 (NY, NX), 0-1
#   vessel.performance.fields(sic, thick)    -> dict(speed, fuel, rio)
#   routing.cost.build_cost(...)             -> (risk, components, blocked)
#   routing.astar.route(cost, s_rc, g_rc)    -> [(r,c), ...] or None
#   mission.evaluate.evaluate_mission(...)   -> dict (schema in Section 12.3)
```

---

## 18. TEAM DIVISION — FOUR PEOPLE

Your proposed split puts backend **and** frontend on one person, which makes them the critical path and blocks the demo. Rebalanced:

| | **P1 — Data & Grid** | **P2 — Forecasting** | **P3 — Routing, Vessel & Mission** | **P4 — Map, App & Integration** |
|---|---|---|---|---|
| **Owns** | `src/data/*`, `configs/grid.yaml` | `src/forecast/*`, `models/` | `src/routing/*`, `src/vessel/*`, `src/mission/*`, `src/uncertainty/*` | `frontend/map.html`, `app.py`, `backend/main.py`, `scripts/build_cache.py` |
| **Day-1 deliverable** | `grid.py` frozen + one real SIC array | persistence baseline running on fakes | A\* on a fake cost grid | `map.html` drawing a hardcoded line |
| **Depends on** | nothing | P1's grid (uses fakes until Day 2) | P1's grid, P2's forecast (fakes until Day 3) | P3's mission JSON (fakes until Day 4) |
| **Blocks** | everyone | P3 | P4 | the demo |
| **Key output** | `data/processed/sic.nc` | `sic_fcst[lead,y,x]` + skill tables | `missions[]` JSON | `demo_cache.json` + running app |
| **Day 6–7** | validation data prep, fallback datasets | forecast-value experiment, skill slides | risk-segment decomposition | rehearsal, backup video, feature freeze |

**Why P3 carries vessel + mission:** they are tightly coupled — mission evaluation calls the vessel model per cell along a route. Splitting them creates a chatty interface between two people for no benefit.

**Why P4 gets the map on Day 1:** it is the longest-lead item, it is the only JavaScript in the project, and everything else can be demonstrated through it. It must not start on Day 4.

**Integration rule:** nobody merges to `main` without their module passing its test against `interfaces.py` fakes. Fifteen-minute standup at 09:00 and 18:00 — what landed, what is blocked, what changed in a shared signature.

---

## 19. DEVELOPMENT ORDER

The governing principle: **an ugly end-to-end pipeline on Day 3 beats nine beautiful disconnected modules on Day 7.**

| Phase | What to code | Working at the end | Test that proves it | Skippable? |
|---|---|---|---|---|
| **1 — Route prototype** | `grid.py`, `astar.py`, `map.html` on **synthetic** SIC | A line drawn across Antarctica on a real polar map | `test_route_reaches_goal` | ❌ Never |
| **2 — Real data** | `fetch.py`, regrid, land/depth mask | The same route on **real AMSR2 ice** | `test_no_nan_in_cost`, visual check | ❌ Never |
| **3 — Forecast** | persistence, features, XGBoost, `validate.py` | Skill table, persistence vs XGBoost by lead | `test_chronological_split_no_leak` | ❌ Never (persistence alone suffices) |
| **4 — Iceberg** | DMS parse, velocity, drift cones, risk layer | Real named bergs (D15A–D, D34) with widening cones | `test_cone_grows_with_lead` | ⚠️ Could use USNIC positions statically |
| **5 — Vessel** | SDA port, POLARIS, cost integration | Per-cell speed/fuel/RIO; ice-class toggle changes the corridor | `test_speed_zero_above_80pct` | ❌ It is Differentiator 2 |
| **6 — Mission** | `sites.py`, `evaluate_mission`, `compare_sites` | **Three routes, three metric sets, one comparison table** | `test_infeasible_when_window_exceeded` | ❌ **It is the project** |
| **7 — Uncertainty** | correlated perturbation, MC re-scoring | P10/P50/P90, P(blocked) per site | `test_correlated_noise_gives_wide_spread` | ⚠️ Could ship P50 only |
| **8 — Dashboard** | Streamlit shell, cache builder | Full click-through on cached data | `test_cache_covers_all_slider_combos` | ❌ No demo without it |
| **9 — Validation & demo** | forecast-value runs, slides, rehearsal | Skill + forecast-value tables; 10 clean rehearsals | Run the demo end to end, offline | ❌ Never |

---

## 20. SEVEN-DAY EMERGENCY PLAN

### Day 1 — Skeleton and contracts. Everything fake, everything connected.

| Who | Task | Output |
|---|---|---|
| All | Repo, venv 3.12, `requirements.txt`, `interfaces.py` | Four machines running identical stacks |
| P1 | `grid.py` frozen; **open one Bremen GeoTIFF and verify CRS, shape, value range** | `configs/grid.yaml` with **real** values |
| P2 | `persistence()` + `validate.py` skeleton on fake cube | A number printed |
| P3 | `astar.py` + `build_edge_cost` on `fake_fields()` | A path returned |
| P4 | `map.html` with the NASA CRS block, Polar View WMS, one hardcoded polyline | **A polar map with a line on it** |

**Checkpoint (18:00):** a line renders on an Antarctic map from fake data.
**Fallback:** if Bremen is unreachable, start on NSIDC G02202_V6; if `pyastar2d` will not install, use `_astar_python` — at 25 km it is fast enough.

### Day 2 — Real ice.

| Who | Task |
|---|---|
| P1 | `fetch.py` pulls 60 days; build `sic.nc`; land + depth mask |
| P2 | Feature builder; chronological split with 7-day embargo; first XGBoost fit |
| P3 | Wire real SIC into the cost raster; hard constraints live |
| P4 | FastAPI `/route`; map reads from the API instead of a hardcoded line |

**Checkpoint:** a route on **real ice** that visibly avoids the pack.
**Fallback:** if fewer than 30 usable days download, train on 30 and say so.

### Day 3 — END-TO-END. This is the day that matters.

| Who | Task |
|---|---|
| P1 | Iceberg ingest: USNIC CSV parsed, positions on the grid |
| P2 | XGBoost trained; **first honest skill table by lead time** |
| P3 | **SDA vessel model + POLARIS**; per-cell speed/fuel/RIO |
| P4 | `/mission/evaluate` returns three sites; table renders beside the map |

**Checkpoint (18:00): three candidate sites, three routes, one comparison table, real data, clicking works.**
**This is your minimum submittable product.** Everything after is improvement.
**Fallback:** if XGBoost is not ready, ship persistence — the pipeline does not care.

### Day 4 — Uncertainty and the ice-class result.

- P3: Monte Carlo re-scoring, N=200, **spatially correlated** perturbation → P10/P50/P90, P(blocked). Write the slider no-op test.
- P2: lead-time model selector; finalise skill tables.
- P1: iceberg drift + cones wired into risk.
- P4: **Decide Streamlit vs React today.** Build `build_cache.py`, sweep all discrete slider combinations.
- P3: ice-class toggle (IA Super vs PC5) — the two-document result.

**Checkpoint:** moving a slider changes the recommended site, and the ETA band widens with lead time.

### Day 5 — Differentiator 3 and explainability.

- P2 + P3: **three-run forecast-value experiment** over ~10 historical days.
- P3: risk-segment decomposition.
- P4: warnings panel rendering `reasons`; polish.
- P1: verify every number on screen traces to a source; build the assumptions slide.

**Checkpoint:** you can state, in hours and tonnes, what the forecast is worth.
**Fallback:** if the experiment will not converge by 18:00, cut it and lead with the ice-class result instead.

### Day 6 — Freeze, cache, rehearse.

- **Feature freeze at 12:00.** No new features after noon. Bugs only.
- P4: regenerate `demo_cache.json`; **run the entire demo with Wi-Fi off**.
- All: rehearse the click path **ten times**. Record a **60-second backup video**.
- Write the 10-line assumptions slide and the one-slide Polar Code / Indian Antarctic Act mapping.

### Day 7 — Buffer and submission.

Morning: fix only what broke in rehearsal. Afternoon: README, submission, final run-through. **Do not add anything.**

**Demo-day discipline:** feature freeze 2 hours before judging; everything runs from local cache; venue Wi-Fi is the most common single point of failure, so assume it fails; if you fall back to the recording, say it is a recording.

---

## 21. FAILURE / FALLBACK PLAN

| Component | Primary | Fallback 1 | Fallback 2 | Decide by |
|---|---|---|---|---|
| **Sea-ice data** | Bremen AMSR2 6.25 km GeoTIFF | NSIDC **G02202_V6** 25 km NetCDF (note V5 404s; 9–10 Aug 2026 missing) | NSIDC G02135 extent CSV + climatology, coarse field | Day 1, 14:00 |
| **rasterio / GDAL** | `pip install rasterio` (wheels bundle libgdal) | One person converts GeoTIFF → `.npy` once, commits | Use NetCDF sources only, `xarray` + `netCDF4` | Day 1, 12:00 |
| **XGBoost underperforms** | XGBoost, tuned | **Ship persistence at the leads where it wins** — this is the selector, and it is a feature | Persistence everywhere; pitch the validation, not the model | Day 3 |
| **XGBoost too slow** | 600 trees, full rows | 300 trees, 50% row subsample | RandomForest, depth 8 | Day 2 |
| **Temporal leakage suspected** (R² > 0.98) | Re-check the split | Widen the embargo to 14 days | Report the suspicion openly — *"we found and fixed a leak"* is a credibility gain | Day 3 |
| **Iceberg data insufficient** | USNIC live + BYU history | USNIC positions only, static, with cones | Three hand-placed bergs at published D15 coordinates, clearly labelled illustrative | Day 3 |
| **Ocean currents** | **Cut from MVP** (see Conflict 1) | — | Architecture keeps the slot; say it is future scope | Already decided |
| **POLARIS ice-type bridge** | Thickness-band proxy, labelled ours | Two-class (FY/MY) split by month | Drop RIO to a display-only metric; keep the 80% SIC hard constraint | Day 3 |
| **Vessel model disputed** | PolarRoute SDA port | Speed = `vmax × (1 − SIC/100)^2`, labelled a prototype assumption | Constant speed, fuel ∝ distance — and **say so plainly** | Day 3 |
| **`pyastar2d` install fails** | `pip install pyastar2d` | `_astar_python` at 25 km (sub-second) | `scipy.sparse.csgraph.dijkstra` on an 8-connected graph | Day 1 |
| **A\* too slow** | 25 km, 216×216 | Crop to a regional box around the corridor | 50 km grid | Day 2 |
| **Monte Carlo too slow** | N=200 at 25 km (~2 s) | N=50 | Analytic ±1σ band from the forecast error stats | Day 4 |
| **Map will not render** | Leaflet + Proj4Leaflet EPSG:3031 | Static Matplotlib polar PNG + HTML table beside it | Plain lat/lon Leaflet with a stated distortion caveat | Day 1, 18:00 |
| **React behind schedule** | — | **Streamlit** — `map.html` is unchanged | FastAPI serving `map.html` directly | Day 4, 12:00 |
| **Polar View WMS down** | `geos.polarview.aq` | NASA GIBS EPSG:3031 WMTS | No basemap; render your own SIC raster as a PNG overlay | Day 1 |
| **Cache build too slow** | Sweep all slider combos | Cut to 3 dates × 2 leads × 2 alphas | One frozen scenario | Day 6, 12:00 |
| **Everything breaks on stage** | Live cached demo | Recorded 60-second video | Slides with screenshots | Day 6 |

---

## 22. TESTING

```python
"""tests/test_pipeline.py — pytest. Run before every merge."""
import numpy as np, pytest
from src.data.grid import lonlat_to_rc, rc_to_lonlat, NY, NX
from src.routing.cost import build_cost
from src.routing.astar import route, build_edge_cost
from src.vessel.performance import attainable_speed, fuel_tonnes_per_day
from src.vessel.polaris import rio, category
from src.interfaces import fake_fields


# ---- data & grid ----
def test_grid_roundtrip():
    lon, lat = 76.196, -69.407                       # Bharati
    r, c = lonlat_to_rc(lon, lat)
    assert 0 <= r < NY and 0 <= c < NX
    lon2, lat2 = rc_to_lonlat(r, c)
    assert abs(lat2 - lat) < 0.5 and abs(lon2 - lon) < 1.0   # within one 25 km cell

def test_sic_physically_valid():
    from src.data.grid import regrid_geotiff
    sic = regrid_geotiff("data/raw/sic_20260919.tif")
    ok = np.isfinite(sic)
    assert ok.mean() > 0.3, "more than 70% NaN — wrong band or wrong flag threshold?"
    assert sic[ok].min() >= 0 and sic[ok].max() <= 100

def test_antarctica_is_the_right_way_up():
    """Catches the flipped-y bug, which otherwise silently breaks every route."""
    r_south, _ = lonlat_to_rc(0, -85)
    r_north, _ = lonlat_to_rc(0, -60)
    assert r_south != r_north
    # In EPSG:3031, -85 deg is nearer the pole, i.e. nearer the grid centre
    assert abs(r_south - NY // 2) < abs(r_north - NY // 2)


# ---- cost map ----
def test_no_nan_in_cost():
    f = fake_fields()
    cost = build_edge_cost(f["speed"], f["fuel"], f["risk"], f["blocked"])
    assert not np.isnan(cost).any(), "NaN in cost -> A* behaves unpredictably"

def test_blocked_cells_are_infinite_not_expensive():
    f = fake_fields(); f["blocked"][10, 10] = True
    cost = build_edge_cost(f["speed"], f["fuel"], f["risk"], f["blocked"])
    assert not np.isfinite(cost[10, 10]), "finite penalty lets A* buy through a wall"


# ---- routing ----
def test_route_reaches_goal():
    f = fake_fields()
    cost = build_edge_cost(f["speed"], f["fuel"], f["risk"], f["blocked"])
    p = route(cost, (20, 20), (180, 180))
    assert p and p[0] == (20, 20) and p[-1] == (180, 180)

def test_route_never_crosses_blocked_cells():
    f = fake_fields()
    f["blocked"][:, 100] = True; f["blocked"][100, 100] = False   # one gate
    cost = build_edge_cost(f["speed"], f["fuel"], f["risk"], f["blocked"])
    p = route(cost, (20, 20), (180, 180))
    assert p is not None
    assert all(np.isfinite(cost[r, c]) for r, c in p)
    assert (100, 100) in p, "should have gone through the only gate"

def test_risk_slider_actually_changes_route():
    """Guards the sigma-collinear-with-mu no-op described in 13.4."""
    f = fake_fields()
    r0 = route(build_edge_cost(f["speed"], f["fuel"], f["risk"], f["blocked"], alpha=0.0), (20,20), (180,180))
    r3 = route(build_edge_cost(f["speed"], f["fuel"], f["risk"], f["blocked"], alpha=3.0), (20,20), (180,180))
    assert r0 != r3, "risk slider is a no-op"


# ---- vessel ----
def test_speed_zero_above_max_ice_conc():
    assert attainable_speed(85.0, 1.0) == 0.0          # SDA max_ice_conc = 80
    assert attainable_speed(0.0, 1.0) == 26.5          # max speed in open water

def test_speed_decreases_monotonically_with_ice():
    v = [attainable_speed(s, 1.0) for s in (0, 20, 40, 60, 79)]
    assert all(a >= b for a, b in zip(v, v[1:])), "speed must not rise with more ice"

def test_fuel_positive_and_increases_with_resistance():
    assert fuel_tonnes_per_day(15.0, 0.0) > 0
    assert fuel_tonnes_per_day(15.0, 5e4) > fuel_tonnes_per_day(15.0, 0.0)


# ---- POLARIS ----
def test_polaris_ice_free_is_normal():
    assert rio([10,0,0,0,0,0,0,0,0,0,0,0], "IA Super") == 30.0
    assert category(30.0, "IA Super") == "normal"

def test_ice_class_changes_the_answer():
    """The two-document result from Section 10.4: 10/10 thick first-year ice."""
    c = [0,0,0,0,0,0,0,0,10,0,0,0]
    assert rio(c, "PC5") == 0.0 and category(0.0, "PC5") == "normal"
    assert rio(c, "IA Super") == -20.0
    assert category(-20.0, "IA Super") == "special_consideration"


# ---- mission ----
def test_infeasible_when_window_exceeded():
    from src.mission.evaluate import evaluate_mission
    from configs.sites import CANDIDATE_SITES, START
    import src.data.grid as grid
    m = evaluate_mission(CANDIDATE_SITES[0], START, fake_fields(),
                         {"window_days": 0.1, "fuel_budget_t": 1e9,
                          "ice_class": "IA Super", "haul_rate_km_per_day": 40},
                         grid=grid)
    assert m["feasibility"] == "INFEASIBLE" and m["reasons"]

def test_every_mission_field_is_json_serialisable():
    import json
    from src.mission.evaluate import evaluate_mission
    from configs.sites import CANDIDATE_SITES, START, MISSION
    import src.data.grid as grid
    m = evaluate_mission(CANDIDATE_SITES[0], START, fake_fields(), MISSION, grid=grid)
    json.dumps(m)      # numpy float32 is NOT serialisable — catches it early


# ---- uncertainty ----
def test_correlated_noise_gives_realistic_spread():
    from src.uncertainty.montecarlo import perturb_sic
    sic = np.full((216, 216), 50.0, "float32")
    p = perturb_sic(sic, sigma_pct=10.0, corr_km=150.0)
    assert 3.0 < p.std() < 20.0
    # neighbours must be correlated, else it is iid noise
    assert np.corrcoef(p[:, :-1].ravel(), p[:, 1:].ravel())[0, 1] > 0.8


# ---- API ----
def test_api_contract():
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    r = c.post("/mission/evaluate", json={"date": "2026-09-19", "lead_days": 3,
                                          "alpha": 1.0, "beta": 2.0, "ice_class": "IA Super"})
    assert r.status_code == 200
    for m in r.json()["missions"]:
        assert {"site_id", "feasibility"} <= m.keys()
        assert m["feasibility"] in {"FEASIBLE", "MARGINAL", "INFEASIBLE"}
```

**Forecast leakage test — write it on Day 2, not Day 6:**

```python
def test_chronological_split_has_no_leakage():
    X_tr, y_tr, m_tr, X_te, y_te, m_te = load_split()
    assert m_tr.max() + EMBARGO <= m_te.min(), "train/test overlap in time"
    r2 = score(model, X_te, y_te)
    assert r2 < 0.995, f"R2={r2:.4f} is implausibly high — check for leakage"
```

---

## 23. FINAL DEMO SCENARIO — 3 TO 5 MINUTES

**Scenario: 43rd Indian Scientific Expedition to Antarctica. The vessel has departed Cape Town. The Voyage Leader must choose where to discharge cargo for Maitri.**

| Time | What you do | What you say |
|---|---|---|
| **0:00–0:30** | Map open, real AMSR2 ice, four candidate sites pinned | *"NCPOR charters an ice-class vessel for 115 days. Only 34 of those are sailing — 70 are spent stationary, putting cargo onto fast ice at a site the Master chooses with the Expedition Leader. There are no designated anchorages. So the question is not 'what is the shortest route'. It is 'which discharge site can we reach and hold'."* |
| **0:30–1:15** | Toggle the forecast layer; step lead 1 → 3 → 7 days; iceberg cones widen | *"Sea-ice concentration from AMSR2. Our forecast at three days. These are real tracked icebergs — D15A through D, and D34, sitting at 67 South, 80 East, directly in the Bharati corridor. We draw cones, not points, because we do not know where a berg will be."* |
| **1:15–2:15** | **The comparison table.** Four sites, four routes, full metrics | *"Every site gets a route and a full cost. Bharati West: 10.1 days, 486 tonnes, POLARIS normal — feasible. Indian Barrier: shorter by sea, but 100 km of overland haul, POLARIS elevated for 150 km, and a documented 2012 calving event that carried Indian cargo out to sea. We mark it marginal and we say why. We do not rank them with a single score — that is the Voyage Leader's call."* |
| **2:15–2:50** | Show P10/P50/P90 fan and P(blocked) | *"We perturb the forecast 200 times with spatially correlated noise and re-score each route. Bharati West arrives between 219 and 259 hours, with a 2% chance of blockage. Indian Barrier has a 14% chance. That difference is the decision."* |
| **2:50–3:30** | 🎯 **THE MOMENT.** Switch ice class PC5 → IA Super | *"The vessel in most routing models is PC5. NCPOR's tender requires 1A Super. In thick first-year ice, POLARIS gives PC5 a risk index of zero and 1A Super minus two. Watch."* **The accessible corridor visibly shrinks and the recommended site changes.** *"Same ice, same day. Two public documents — an NCPOR tender and IMO circular 1519. One of those vessels should not be there."* |
| **3:30–4:10** | Validation slide | *"Persistence beats our model at 24 hours, so we ship persistence at 24 hours. Our model earns its place from 72 hours out. And here is what that is worth: planning on our forecast instead of persistence, then scoring on the ice that actually happened, saves X hours and Y tonnes."* |
| **4:10–4:40** | Assumptions slide | *"Our risk weights are prototype values and adjustable. Fuel is ±20–30% absolute, better for ranking. Our ice-type mapping is our own approximation because no public Antarctic product gives WMO stages. The hard limits are not ours: 80% concentration and 10 metre depth come from BAS's vessel model, the POLARIS thresholds from the IMO."* |
| **4:40–5:00** | Close | *"This is decision support, not autonomy. The charter names the decision-makers — the Master and the Expedition Leader. We give them the trade-offs."* |

**The single visual peak is the ice-class switch at 2:50.** It is two lookups in code, it comes from two public documents, and it visibly changes the answer. Rehearse that transition until it is seamless.

**If asked "why no Indian data?"** — *"We checked. MOSDAC's current product is built from altimetry, scatterometer winds and OISST, none of which are valid under pack ice. Using it would have been scientifically wrong. Our Indian context is NCPOR's operating box, their charter schedule, their stations — and the iceberg database we use ingests ISRO's OSCAT and OSCAT-2 scatterometers."*

---

## 24. WHAT ARE WE ACTUALLY BUILDING — READ THIS TO YOUR TEAM

**What are we building?**
A mission-level decision-support system for Indian Antarctic resupply. Given a departure point, a charter window and several candidate cargo-discharge sites, it tells the Voyage Leader what each site would cost — in days, tonnes of fuel, ice exposure, regulatory margin and probability of failure — and lets them choose. It is not a route planner with a map. It is a *which-site-and-can-we-make-it* planner that happens to compute routes along the way.

**What are the inputs?**
Sea-ice concentration from AMSR2 (Bremen, free, daily). Iceberg positions from the US National Ice Center and historical tracks from BYU. A bathymetric land and shallow mask. Optionally wind from ERA5. Plus the mission parameters that actually matter: the charter window, the fuel budget, the vessel's ice class, and the overland haul distance from each candidate site to the station.

**What does the AI do?**
One thing, deliberately: it forecasts sea-ice concentration 1 to 7 days ahead using XGBoost on lagged, spatial and seasonal features. It is on probation — it only gets used at lead times where it measurably beats persistence, and we publish the comparison either way. The AI is a component, not the product.

**What does A\* do?**
It finds the least-cost path across a 25 km Antarctic grid, where cost is transit time inflated by risk plus weighted fuel, and where cells above 80% ice concentration, under 10 m depth, on land, or below the POLARIS threshold are simply impassable. **A\* is not our innovation.** Four other teams have it. We use it because it is the right tool.

**What does the vessel model do?**
It converts ice into consequences. Using BAS's open-source PolarRoute vessel model, it computes the ship's attainable speed in each cell, the resistance it faces, and the fuel it burns — and using IMO circular MSC.1/Circ.1519, it computes a POLARIS risk index for the ship's ice class. This is why our numbers have a derivation instead of a weight somebody invented.

**What is the differentiator?**
Three things, in order:
1. **We optimise the discharge decision, not the sailing leg.** NCPOR's own charter spends 70 of 115 days stationary at a discharge site, and the destination is a *choice among fast-ice edges*, not a pin on a chart. No competitor was found addressing this.
2. **Every performance number is sourced.** Speed, fuel and risk come from published BAS code and an IMO circular. Our assumptions are labelled as assumptions, on screen.
3. **We measure what the forecast is worth.** We plan routes on the forecast, score them on the ice that actually happened, and report the difference in hours and tonnes — including when it is small.

**What does the final dashboard show?**
An Antarctic polar map with sea ice, iceberg drift cones and four candidate discharge sites, each with its own route coloured by feasibility. Beside it, a comparison table: days, fuel, maximum ice, POLARIS category, probability of blockage, feasibility verdict — and for anything marginal, the specific reasons why. Sliders for risk aversion, fuel weight, forecast lead and ice class. Change one, and the recommendation changes in front of you.

---

## Carried uncertainties — verify these, do not assert them

| Item | Status | Action |
|---|---|---|
| Bremen GeoTIFF CRS and grid dimensions | **Not verified** | Open one file Day 1, print `src.crs` and `src.shape` |
| Bremen flag values (>100 = land/missing?) | **Not verified** | Print `np.unique(sic)[-10:]` before masking |
| NSIDC CDR flag values | **Not verified** | Same check if you fall back to CDR |
| Bremen 3.125 km (`s3125`) path | **Not verified** | List the parent directory before coding a guessed path |
| PolarRoute k/b/n coefficient provenance | Verbatim from source, **but source carries no citation** | Cite **PolarRoute itself**, never a paper |
| Whether MSC.1/Circ.1519 has been superseded | **Not verified** | Check before calling it "current" in the deck |
| MV *Vasiliy Golovnin*'s assigned ice class | **Not verified** (tender requires "1A Super or better"; vessel's class unconfirmed) | Present as a **sensitivity**, never an assertion |
| Exact Indian Barrier discharge coordinates | **Not verified** — only "~100 km N of Maitri" | Label sites "indicative, derived from published descriptions" |
| OSI SAF summer ice-type gap | **Entirely unverified** | Do not cite it as fact; your proxy stands on its own |
| Streamlit + Proj4Leaflet pairing | **Inference** — each verified separately, the combination is not | **Day-1 spike.** Thirty minutes. |
| Polar View WMS layer names and extent | Service verified keyless; **layer names not** | Hit `GetCapabilities` Day 1 |
| Charter day rate | **No public figure exists** | Express savings as "N days of day-hire", let NCPOR substitute their rate |

---

## Sources

**Data**
- Bremen AMSR2 — https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/s6250/
- NSIDC Sea Ice Index G02135 — https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/
- NSIDC CDR G02202_V6 — https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/2026/
- USNIC iceberg table (CSV) — https://usicecenter.gov/File/DownloadCurrent?pId=134
- BYU Antarctic Iceberg Database v8.0 — https://www.scp.byu.edu/data/iceberg/consolidated_database_v8.0.zip
- ARCO ERA5 — `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` — https://github.com/google-research/arco-era5
- GEBCO area subset — https://download.gebco.net

**Vessel, regulation and operations**
- BAS PolarRoute (MIT) — https://github.com/bas-logist/PolarRoute · `polar_route/vessel_performance/vessels/SDA.py`
- IMO MSC.1/Circ.1519 (POLARIS) — https://www.nautinst.org/static/uploaded/2f01665c-04f7-4488-802552e5b5db62d9.pdf
- NCPOR 41-ISEA ice-class vessel charter tender — https://www.indianembassyrome.gov.in/docs/1624279190_1285_41%20ISEA%20Tender-Ice%20Class%20Vessel.PDF
- ASMA No. 6 Larsemann Hills management plan — https://www.env.go.jp/nature/nankyoku/kankyohogo/database/jyouyaku/asma/asma_pdf_en/ASMA06_en.pdf
- NCPOR: shelf ice breaks near Maitri (April 2012) — https://ncpor.res.in/news/view/147
- PIB, 43rd ISEA departure — https://www.pib.gov.in/PressReleasePage.aspx?PRID=1993769

**Mapping**
- NASA GIBS Antarctic EPSG:3031 Leaflet example — https://github.com/nasa-gibs/gibs-web-examples/blob/main/examples/leaflet/antarctic-epsg3031.js
- Polar View GeoServer WMS (keyless, EPSG:3031) — https://geos.polarview.aq/geoserver/wms
- NASA GIBS WMTS EPSG:3031 — https://gibs.earthdata.nasa.gov/wmts/epsg3031/best/

**Libraries**
- pyastar2d — https://github.com/hjweide/pyastar2d
- Proj4Leaflet — https://kartena.github.io/Proj4Leaflet/

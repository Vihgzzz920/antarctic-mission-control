# polar-nav — SIH26059

## Step 1: Antarctic sea-ice concentration (SIC) ingestion + inspection

Downloads one or more daily AMSR2 Antarctic SIC GeoTIFFs from University of
Bremen and reports everything downstream code must not assume: CRS, shape,
pixel size, bounds, dtype, nodata, value scale, flag codes, missing fraction
and grid orientation.

### Setup

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

### Run

    python -m src.data.fetch_sic                 # newest available day
    python -m src.data.fetch_sic --days 5        # 5 most recent days
    python -m src.data.inspect_sic               # inspect newest file

### Outputs

    data/raw/sic_YYYYMMDD.tif       downloaded GeoTIFF (gitignored)
    notebooks/sic_first_look.png    quick-look map + histogram
    configs/grid_observed.json      observed facts, for configs/grid.yaml

### No network?

    python scripts/make_test_fixture.py          # SYNTHETIC data, never for analysis
    python -m src.data.inspect_sic data/raw/sic_TESTFIXTURE.tif

### Before moving on

Open `notebooks/sic_first_look.png`. It must look like Antarctica: continent in
the middle, ice ring around it, grey where land/missing is masked. If it is
upside down or inside out, fix that now — a flipped axis is silent and will
make every later route wrong.

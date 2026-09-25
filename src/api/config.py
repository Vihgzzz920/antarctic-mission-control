"""
Paths and the one validated demonstration this prototype ships with.

Everything here is a POINTER to data the backend already produced. No cost, no
penalty, no route and no metric is defined in this package.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BEDMACHINE = ROOT / "data" / "processed" / "bedmachine" / \
    "bedmachine_land_exclusion_3976.tif"
#  the classified mask, used ONLY to draw land and no-coverage apart on screen
BEDMACHINE_CLASSES = ROOT / "data" / "processed" / "bedmachine" / \
    "bedmachine_mask_3976.tif"
USNIC = ROOT / "data" / "processed" / "usnic"
ICEBERGS = ROOT / "data" / "processed" / "icebergs"
SIC_DIR = ROOT / "data" / "raw"
COMP_CONFIG = ROOT / "configs" / "navigation_cost_composition.json"
LAYER_DIR = ROOT / "frontend" / "public" / "layers"

#  The demonstration described in the project notes: 2025-01-08 environment,
#  PC6 on IMO Table 1.3, the shipped iceberg exposure weight, and the 2020-12-24
#  USNIC chart that only combines with it under an explicit override.
DEMO = {
    "environment_date": "2025-01-08",
    "departure_time": "2025-01-08T00:00:00",
    "start": [948, 503],
    "goal": [922, 497],
    "vessel_speed_mps": 5.0,
    "polaris_ice_class": "PC6",
    "polaris_riv_table": "1.3",
    "exposure_horizons": [("06h", 0), ("12h", 1)],
    "bucket_hours": 6,
}

#  the project CRS, read from the rasters at import time rather than retyped
GRID_EPSG = 3976


def demo_departure() -> datetime:
    return datetime.fromisoformat(DEMO["departure_time"])

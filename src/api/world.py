"""
Assemble the real 2025-01-08 routing world, once, and hand it to the router.

This module WIRES existing layers together. It computes nothing: every number
that reaches the API comes out of a module that already owned it --
build_cost_grid, compose_grid, build_polaris_penalties, the iceberg exposure
rasters and compare_profiles. There is no second A*, no second cost model and
no second POLARIS calculation anywhere in src/api.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np
import rasterio

from src.data.rasterize_sigrid3 import COVERAGE_TIF
from src.routing.cost import build_cost_grid
from src.routing.end_to_end_route import build_polaris_penalties
from src.routing.grid import RoutingGrid
from src.routing.iceberg_navigation_cost import (CONFIG, ExposureField,
                                                 IcebergExposureProvider,
                                                 IcebergTimeNavigationCost,
                                                 load_config)
from src.routing.iceberg_risk import IcebergForecast
from src.routing.navigation_cost import compose_grid, load_composition_config
from src.routing.navigation_domain import NavigationDomain
from src.routing.time_navigation_cost import (ComposedField,
                                              TimeIndexedNavigationCost)

from src.api.config import (BEDMACHINE, COMP_CONFIG, DEMO, ICEBERGS, USNIC)

HOUR = 3600.0


class WorldError(RuntimeError):
    """A real input this prototype needs is missing from data/processed."""


def _require(path):
    if not path.exists():
        raise WorldError(
            f"missing real input: {path}. The API serves measured data only and "
            f"will not substitute anything for it.")
    return path


def chart_date() -> date:
    """The USNIC chart date, read from the raster's own source_chart tag."""
    with rasterio.open(_require(USNIC / COVERAGE_TIF)) as src:
        tag = src.tags()["source_chart"]            # e.g. ANTARC20201224
    return date(int(tag[-8:-4]), int(tag[-4:-2]), int(tag[-2:]))


def _exposure_fields(shape, day: date):
    fields, meta = [], []
    for label, bucket in DEMO["exposure_horizons"]:
        path = _require(ICEBERGS / f"iceberg_exposure_{label}_3976.tif")
        with rasterio.open(path) as src:
            band = src.read(1).astype("float64")
            tags = src.tags()
        forecast = IcebergForecast(
            iceberg_id=f"exposure_raster_{label}_max_over_"
                       f"{tags['iceberg_count']}_icebergs",
            position_source="ascat",
            forecast_start_date=tags["forecast_start_date"],
            horizon_seconds=float(tags["horizon_hours"]) * HOUR,
            horizon_hours=float(tags["horizon_hours"]), latitude=-70.0,
            longitude=0.0, radius_km=float(tags["radius_km"]),
            extrapolated_uncertainty=tags["extrapolated_uncertainty"] == "true",
            physics_position_is_extrapolated=False, calibration_quantile=0.9,
            calibration_n=1197, growth_law="power",
            interpolation_quality="observed_to_observed")
        fields.append(ExposureField(
            bucket=bucket, exposure=band,
            dominant=np.where(band > 0, 0, -1).astype("int32"),
            forecasts=(forecast,), forecast_start_date=day,
            horizon_hours=float(tags["horizon_hours"]), label=label))
        meta.append({"label": label, "bucket": bucket,
                     "horizon_hours": float(tags["horizon_hours"]),
                     "radius_km": float(tags["radius_km"]),
                     "iceberg_count": int(tags["iceberg_count"]),
                     "extrapolated_uncertainty":
                         tags["extrapolated_uncertainty"] == "true",
                     "non_zero_cells": int((band > 0).sum()),
                     "source": str(path)})
    return fields, meta


@lru_cache(maxsize=1)
def get_world():
    """Everything a mission request needs, built once and reused."""
    day = date.fromisoformat(DEMO["environment_date"])
    grid = RoutingGrid.for_date(day)
    domain = NavigationDomain(grid).add_geotiff("land_mask", _require(BEDMACHINE))
    constraints = domain.to_constraints()
    excluded = constraints.combined(tuple(grid.shape))

    polaris = build_polaris_penalties(ice_class=DEMO["polaris_ice_class"],
                                      riv_table=DEMO["polaris_riv_table"])
    env = build_cost_grid(grid, constraints=constraints).cost
    composition = compose_grid(env, polaris.penalties,
                               load_composition_config(COMP_CONFIG), USNIC)
    chart = chart_date()

    fields = [ComposedField(
        bucket=b, composition=composition, environment_date=day, chart_date=chart,
        epsg=grid.crs.to_epsg(), shape=tuple(grid.shape),
        transform=tuple(grid.transform)[:6],
        label=f"{b * DEMO['bucket_hours']}-{(b + 1) * DEMO['bucket_hours']} h")
        for _, b in DEMO["exposure_horizons"]]

    #  the 2020 chart under 2025 fields: the provider refuses this pairing
    #  unless it is told to allow it, and so does plan_route afterwards
    base = TimeIndexedNavigationCost.from_fields(
        fields, bucket_seconds=DEMO["bucket_hours"] * HOUR,
        allow_historical_demo_mismatch=True)
    exposure_fields, exposure_meta = _exposure_fields(tuple(grid.shape), day)
    provider = IcebergTimeNavigationCost.from_providers(
        base, IcebergExposureProvider.from_fields(exposure_fields,
                                                  tuple(grid.shape)),
        load_config(CONFIG), allow_historical_demo_mismatch=True)

    return {
        "day": day, "chart_date": chart, "grid": grid, "domain": domain,
        "constraints": constraints, "excluded": excluded,
        "composition": composition, "polaris": polaris, "provider": provider,
        "exposure_fields": exposure_fields, "exposure_meta": exposure_meta,
        "dates_aligned": chart == day,
    }


def fresh_grid() -> RoutingGrid:
    """plan_route writes the constraint mask onto the grid it is given, so each
    profile run gets its own grid rather than a shared one."""
    return RoutingGrid.for_date(get_world()["day"])


def grid_info() -> dict:
    w = get_world()
    g = w["grid"]
    return {
        "epsg": g.crs.to_epsg(),
        "proj4": g.crs.to_proj4(),
        "wkt": g.crs.to_wkt(),
        "shape": [int(g.height), int(g.width)],
        "transform": list(tuple(g.transform)[:6]),
        "pixel_size": list(g.pixel_size),
        "extent": [g.bounds[0], g.bounds[1], g.bounds[2], g.bounds[3]],
        "environment_date": w["day"].isoformat(),
        "usnic_chart_date": w["chart_date"].isoformat(),
        "dates_aligned": w["dates_aligned"],
        "excluded_cells": int(w["excluded"].sum()),
    }

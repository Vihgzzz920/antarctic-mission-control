"""
Assemble the real 2025-01-08 routing world, once, and hand it to the router.

This module WIRES existing layers together. It computes nothing: every number
that reaches the API comes out of a module that already owned it --
build_cost_grid, compose_grid, build_polaris_penalties, the iceberg exposure
rasters and compare_profiles. There is no second A*, no second cost model and
no second POLARIS calculation anywhere in src/api.
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import numpy as np
import rasterio

from src.data.rasterize_sigrid3 import COVERAGE_TIF
from src.routing.cost import build_cost_grid
from src.routing.end_to_end_route import build_polaris_penalties
from src.routing.grid import CURRENTS_DIR, SIC_DIR, RoutingGrid
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
from src.api.environment_timeline import (ON_MISSING_ERROR, POLICY_PERSISTENCE,
                                          plan_environment_timeline,
                                          timeline_summary)

HOUR = 3600.0

#  WHICH ENVIRONMENT EACH ROUTING BUCKET IS PRICED WITH.
#
#  The shipped setting is persistence: every bucket is priced against the
#  departure day's analysis. That is what this API has always done -- what is
#  new is that it now SAYS so, per bucket, in the route's own provenance,
#  instead of registering one composed field for every bucket and leaving the
#  assumption unstated.
#
#  It is also exact for the validated demonstration, whose two buckets both
#  close inside the analysis day: persistence carries nothing forward there,
#  and the timeline records fallback=None for both.
#
#  The alternative policy (dated_observation_hindcast) reads each bucket's own
#  observed field. It is deliberately NOT reachable from the API: an
#  observation later than the departure analysis is perfect foresight, not a
#  forecast, and it belongs to an offline baseline comparison rather than to a
#  mission response. See src/api/environment_timeline.py.
ENVIRONMENT_POLICY = POLICY_PERSISTENCE
ENVIRONMENT_ON_MISSING = ON_MISSING_ERROR


class WorldError(RuntimeError):
    """A real input this prototype needs is missing from data/processed."""


def _require(path):
    if not path.exists():
        raise WorldError(
            f"missing real input: {path}. The API serves measured data only and "
            f"will not substitute anything for it.")
    return path


def environment_dates() -> set:
    """The dates the environment archive can actually serve.

    A date counts only when BOTH rasters RoutingGrid.for_date needs are
    present, because a day with SIC and no current is not a day this router can
    be run on. Nothing is inferred from a filename beyond its date.
    """
    sic = {p.name[4:12] for p in SIC_DIR.glob("sic_????????.tif")}
    cur = {p.name[9:17] for p in CURRENTS_DIR.glob("currents_????????.tif")}
    out = set()
    for stamp in sic & cur:
        try:
            out.add(date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8])))
        except ValueError:                       # not a date; not an input
            continue
    return out


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
    comp_config = load_composition_config(COMP_CONFIG)
    env = build_cost_grid(grid, constraints=constraints).cost
    composition = compose_grid(env, polaris.penalties, comp_config, USNIC)
    chart = chart_date()

    #  ---- which environment each bucket is entitled to, stated per bucket ----
    #  The POLARIS term is deliberately NOT part of this: it comes from one
    #  operational USNIC chart and is not a forecast, so every bucket carries
    #  the same penalties and only the environmental term can vary.
    buckets = [b for _, b in DEMO["exposure_horizons"]]
    timeline = plan_environment_timeline(
        buckets=buckets, bucket_seconds=DEMO["bucket_hours"] * HOUR,
        departure_time=datetime.fromisoformat(DEMO["departure_time"]),
        policy=ENVIRONMENT_POLICY, available_dates=environment_dates(),
        analysis_date=day, on_missing=ENVIRONMENT_ON_MISSING)

    #  one composition per DISTINCT environment date; under persistence that is
    #  exactly one, and it is the composition built above rather than a second
    #  pass over the same raster
    compositions = {day: composition}
    for entry in timeline:
        if entry.environment_date in compositions:
            continue
        other = RoutingGrid.for_date(entry.environment_date)
        if tuple(other.shape) != tuple(grid.shape) or \
                tuple(other.transform)[:6] != tuple(grid.transform)[:6]:
            raise WorldError(
                f"the environment for {entry.environment_date} is not on the "
                f"routing grid ({other.shape} / {tuple(other.transform)[:6]} "
                f"against {tuple(grid.shape)} / {tuple(grid.transform)[:6]}); "
                f"nothing is resampled here.")
        compositions[entry.environment_date] = compose_grid(
            build_cost_grid(other, constraints=constraints).cost,
            polaris.penalties, comp_config, USNIC)

    fields = [ComposedField(
        bucket=entry.bucket, composition=compositions[entry.environment_date],
        #  the ANALYSIS this field is issued from, which is what the iceberg
        #  layer matches its own forecast_start_date against. The raster that
        #  was actually read is in `environment` below.
        environment_date=day, chart_date=chart,
        epsg=grid.crs.to_epsg(), shape=tuple(grid.shape),
        transform=tuple(grid.transform)[:6],
        label=f"{entry.bucket * DEMO['bucket_hours']}-"
              f"{(entry.bucket + 1) * DEMO['bucket_hours']} h",
        environment=entry.to_dict())
        for entry in timeline]

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
        "environment_timeline": timeline,
        "environment_summary": timeline_summary(timeline),
    }


def sic_for_bucket():
    """The sea-ice field the fuel proxy should read, per arrival-time bucket.

    The shipped world resolves ONE environmental field for every bucket
    (environment_is_time_varying is False), so that single field is the honest
    answer for any bucket and this says so rather than assuming it. If the
    world ever becomes time-varying, a per-bucket field must be supplied by
    whoever resolved it; returning the departure-day analysis for every bucket
    would then be a silent persistence assumption, so None is returned instead
    and the fuel profile is simply absent.
    """
    w = get_world()
    if w["environment_summary"].get("environment_is_time_varying"):
        return None
    sic = w["grid"].sic
    return lambda bucket: sic


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

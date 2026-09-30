"""
A temporary "what if there were one more iceberg here" scenario.

    POST /api/mission/simulate

NOTHING IS WRITTEN ANYWHERE
  The simulated iceberg lives in one dict for the length of one request. No
  BYU/NIC observation, ERA5 field, GLORYS current, SIGRID raster, POLARIS
  table, config or processed file is opened for writing by this module, and
  the cached world the API serves is never mutated -- the scenario is built
  from copies and discarded when the response is sent. Clearing the simulation
  needs no undo because nothing was changed.

NO NEW SCIENCE
  The simulated berg's exposure comes from iceberg_risk.build_exposure(), the
  same function that produced the shipped rasters, with the same configured
  exposure function and the same calibrated radius the real forecast carries
  for that horizon. This module contains no distance rule, no cone, no decay
  and no calibration of its own. Combining it with the existing field uses
  iceberg_risk's own documented rule -- the MAXIMUM over icebergs, never a sum.

NO NEW SEARCH
  Replanning calls route_profiles.compare_profiles(), which calls the existing
  plan_route() and the frozen time_astar(). There is no second A*, no second
  cost model, and no route arithmetic here.

WHEN THE BERG APPEARS
  A simulated berg that exists for all time would be a different question from
  the one a planner asks. It is placed at the bucket in which the CURRENT route
  is predicted to be nearest to it -- read off that route's own arrival times --
  and exists from that bucket onward. Before then it contributes nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.routing.iceberg_navigation_cost import (CONFIG as ICEBERG_COST_CONFIG,
                                                 ExposureField,
                                                 IcebergExposureProvider,
                                                 IcebergTimeNavigationCost,
                                                 load_config as load_cost_config)
from src.routing.iceberg_risk import (CONFIG as RISK_CONFIG, IcebergForecast,
                                      RoutingGridGeometry, build_exposure,
                                      load_config as load_risk_config)

SIMULATION_LABEL = "SIMULATED ICEBERG"
SIMULATED_ID = "simulated_iceberg"


class SimulationError(ValueError):
    """The requested simulation cannot be built from the real world."""


@dataclass(frozen=True)
class Injection:
    """One simulated iceberg, and the moment the current route meets it."""

    latitude: float
    longitude: float
    row: int
    col: int
    radius_km: float
    from_bucket: int
    encounter_arrival_s: float
    encounter_datetime: str
    encounter_cell: tuple[int, int]
    distance_to_route_m: float
    horizon_hours: float

    def to_dict(self) -> dict:
        return {
            "label": SIMULATION_LABEL,
            "iceberg_id": SIMULATED_ID,
            "simulated": True,
            "temporary": True,
            "written_to_any_dataset": False,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "row": self.row,
            "col": self.col,
            "radius_km": self.radius_km,
            "radius_source": "the calibrated uncertainty radius the real "
                             "forecast carries for this horizon; no new "
                             "calibration was performed",
            "from_bucket": self.from_bucket,
            "encounter_arrival_s": self.encounter_arrival_s,
            "encounter_datetime": self.encounter_datetime,
            "encounter_cell": list(self.encounter_cell),
            "distance_to_route_m": self.distance_to_route_m,
            "horizon_hours": self.horizon_hours,
        }


@lru_cache(maxsize=1)
def _geometry() -> RoutingGridGeometry:
    """The routing grid's cached per-cell lat/lon. Opened read-only, once."""
    return RoutingGridGeometry()


def simulated_forecast(latitude: float, longitude: float, radius_km: float,
                       horizon_hours: float,
                       forecast_start_date: str) -> IcebergForecast:
    """The simulated berg as the SAME record type the real forecasts use."""
    return IcebergForecast(
        iceberg_id=SIMULATED_ID, position_source="operator_simulation",
        forecast_start_date=forecast_start_date,
        horizon_seconds=horizon_hours * 3600.0, horizon_hours=horizon_hours,
        latitude=float(latitude), longitude=float(longitude),
        radius_km=float(radius_km), extrapolated_uncertainty=False,
        physics_position_is_extrapolated=False, calibration_quantile=0.9,
        calibration_n=1197, growth_law="power",
        interpolation_quality="operator_simulation")


def simulated_exposure_raster(forecast: IcebergForecast) -> np.ndarray:
    """One berg's exposure, from iceberg_risk's own builder. Read-only."""
    exposure, _dominant, _stats = build_exposure(
        [forecast], _geometry(), load_risk_config(RISK_CONFIG))
    return exposure


def inject(fields: list[ExposureField], raster: np.ndarray, *,
           from_bucket: int, label: str = SIMULATION_LABEL,
           simulated_forecast_record: IcebergForecast | None = None
           ) -> list[ExposureField]:
    """Fold a simulated exposure raster into copies of the existing fields.

    The combination is iceberg_risk's rule: the maximum over icebergs. The
    inputs are not touched -- every field returned is a new object over a new
    array, so the cached world stays exactly as it was.
    """
    out: list[ExposureField] = []
    for field in fields:
        if field.bucket < from_bucket:
            out.append(field)
            continue
        base = np.asarray(field.exposure, dtype="float64")
        if raster.shape != base.shape:
            raise SimulationError(
                f"the simulated exposure raster is {raster.shape} but the "
                f"forecast field is {base.shape}")
        merged = np.maximum(base, raster)
        wins = raster > base
        dominant = np.asarray(field.dominant, dtype="int32").copy()
        forecasts = tuple(field.forecasts)
        if simulated_forecast_record is not None:
            forecasts = forecasts + (simulated_forecast_record,)
        dominant[wins] = len(forecasts) - 1
        out.append(ExposureField(
            bucket=field.bucket, exposure=merged, dominant=dominant,
            forecasts=forecasts, forecast_start_date=field.forecast_start_date,
            horizon_hours=field.horizon_hours,
            label=f"{field.label or field.bucket} + {label}"))
    return out


def encounter(route: dict, row: int, col: int, pixel_size: float
              ) -> tuple[int, float, str, tuple[int, int], float]:
    """Where and when the CURRENT route comes nearest the simulated position.

    The bucket and the timestamp are the route's own predicted arrival at that
    cell, not a guess: this is what makes the berg appear when the vessel would
    actually be there.
    """
    cells = route.get("cells") or []
    if not cells:
        raise SimulationError(
            "the selected route has no priced cells, so there is no predicted "
            "arrival time to place a simulated iceberg against")
    best = min(cells, key=lambda c: math.hypot(c["row"] - row, c["col"] - col))
    distance = math.hypot(best["row"] - row, best["col"] - col) * pixel_size
    return (int(best["bucket"]), float(best["arrival_time"]),
            str(best["arrival_datetime"]), (int(best["row"]), int(best["col"])),
            distance)


def simulated_provider(base_provider: IcebergTimeNavigationCost,
                       fields: list[ExposureField], shape: tuple[int, int],
                       *, allow_mismatch: bool) -> IcebergTimeNavigationCost:
    """A provider over the injected fields, on the SAME composed-cost base and
    the SAME shipped exposure weight. Nothing about the cost model changes."""
    return IcebergTimeNavigationCost.from_providers(
        base_provider.base,
        IcebergExposureProvider.from_fields(fields, shape),
        load_cost_config(ICEBERG_COST_CONFIG),
        allow_historical_demo_mismatch=allow_mismatch)


# ------------------------------------------------------------ the change
def _cells_touched(profile: dict, raster: np.ndarray) -> int:
    """Route cells the simulated berg puts non-zero exposure on."""
    return sum(1 for cell in profile.get("cells", [])
               if raster[cell["row"], cell["col"]] > 0.0)


def describe_change(before: dict, after: dict, raster: np.ndarray,
                    injection: Injection) -> dict:
    """Before/after, per profile, from the two real comparisons only.

    Every number here is read off a route the backend returned. Nothing is
    modelled, ranked or predicted, and no claim is made that the replanned
    route avoids anything.
    """
    profiles = {}
    for name, was in before["profiles"].items():
        now = after["profiles"].get(name)
        if not now:
            continue
        entry = {
            "path_changed": was.get("path") != now.get("path"),
            "routed_before": bool(was.get("success")),
            "routed_after": bool(now.get("success")),
            "cells_with_simulated_exposure":
                _cells_touched(was, raster) if was.get("success") else None,
        }
        for key in ("distance_m", "travel_time_s", "configured_cost",
                    "iceberg_exposure_contribution", "max_iceberg_exposure",
                    "polaris_contribution", "special_consideration_cells"):
            old, new = was.get(key), now.get(key)
            entry[key] = {
                "before": old, "after": new,
                "delta": (new - old) if isinstance(old, (int, float))
                         and isinstance(new, (int, float)) else None,
            }
        profiles[name] = entry
    return {
        "simulation": injection.to_dict(),
        "profiles": profiles,
        "claims": {
            "is_a_collision_probability": False,
            "guarantees_avoidance": False,
            "ranks_the_routes": False,
            "note": "the replanned route is the one that minimises the same "
                    "configured cost under one added simulated iceberg. It is "
                    "not a claim that any route avoids anything.",
        },
    }

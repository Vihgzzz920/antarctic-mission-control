"""
Build one bucket's composed routing cost from whichever environment the
resolver says is legitimate for that bucket's arrival-time interval.

    fields = forecast_aware_fields(
        timeline=timeline, resolver=resolver, grid=grid,
        constraints=constraints, polaris_penalties=polaris.penalties,
        comp_config=cfg, usnic_dir=USNIC, chart_date=chart, analysis_date=day)
    provider = TimeIndexedNavigationCost.from_fields(fields, bucket_seconds=...)

WHAT THIS ADDS, AND NOTHING ELSE
  src/api/environment_timeline.py already decides WHICH SOURCE each routing
  bucket is entitled to, and src/api/environment_forecast.py is the authority
  that answers with observed_analysis / model_forecast / explicit_persistence /
  unavailable. What was missing was the last step: turning that answer into the
  environmental cost array the existing composition expects. That is all this
  module does.

  It runs no search, prices no cell, and contains no cost formula: the arrays
  come from cost.build_cost_grid and navigation_cost.compose_grid exactly as
  src/api/world.py already uses them.

THE RESOLVER IS THE AUTHORITY
  This module never opens a forecast raster it chose itself. It asks the
  resolver for the bucket's own window and reads the field through
  environment_forecast.read_environment, so the lead, the validity interval and
  the refusals are decided in one place. No file name is parsed, no "nearest"
  forecast is considered, and a bucket the resolver did not answer with a
  forecast is priced from the observation the timeline named.

POLARIS IS NOT A FORECAST
  Every bucket carries the SAME POLARIS penalties, from the one operational
  chart, exactly as before. Only the environmental term varies with the source.

CURRENTS HAVE NO FORECAST, AND THAT IS SAID OUT LOUD
  There is no predicted current field in this project. A forecast bucket
  therefore carries the ANALYSIS day's currents alongside the predicted SIC.
  At the shipped current_weight of 0.0 the current term contributes nothing to
  any cost, so this is a carrier, not a claim. If a caller raises
  current_weight above zero, this module REFUSES rather than pricing a future
  bucket's currents from an analysis field without saying so.

  `environment_date` on every field stays the ANALYSIS date -- the quantity the
  iceberg layer matches its own forecast_start_date against. The valid time,
  the lead and the model live in the field's `environment` provenance, where a
  route can report them without changing what the date checks mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

from src.api.environment_forecast import (SOURCE_FORECAST, SOURCE_OBSERVED,
                                          SOURCE_PERSISTENCE,
                                          EnvironmentResolution,
                                          read_environment)
from src.api.environment_timeline import BucketEnvironment
from src.routing.cost import CostParams, build_cost_grid
from src.routing.grid import RoutingGrid
from src.routing.navigation_cost import compose_grid
from src.routing.time_navigation_cost import ComposedField


class EnvironmentProviderError(RuntimeError):
    """A bucket's environment cannot be assembled as specified."""


@dataclass(frozen=True)
class BucketSource:
    """One bucket, the answer the resolver gave for it, and the field used."""

    bucket: int
    entry: BucketEnvironment
    resolution: EnvironmentResolution | None
    sic: np.ndarray
    origin: str                      # 'forecast' | 'observation'

    def provenance(self) -> dict:
        """What the route will report for this bucket."""
        record = dict(self.entry.to_dict())
        if self.resolution is not None:
            #  the resolver's own words, carried through unchanged
            record["resolution"] = self.resolution.to_dict()
            record["source_type"] = self.resolution.source_type
            record["model"] = self.resolution.model
            record["forecast_origin"] = self.resolution.forecast_origin.isoformat()
            record["valid_time"] = (self.resolution.valid_time.isoformat()
                                    if self.resolution.valid_time else None)
            record["lead_hours"] = self.resolution.lead_hours
        return record


def resolve_bucket_sources(timeline, *, resolver, grid: RoutingGrid,
                           sic_for_date=None) -> list[BucketSource]:
    """Ask the resolver for each bucket's window and fetch the field it names.

    `resolver(window_from, window_to)` is the SAME callable the timeline was
    planned with -- normally functools.partial(resolve_environment, ...). It is
    called again here rather than having the timeline carry a path, so that the
    field this module reads is the one the authority returned, not one inferred
    from a recorded name.
    """
    loader = sic_for_date or (lambda day: RoutingGrid.for_date(day))
    out: list[BucketSource] = []
    observed: dict[date, np.ndarray] = {}

    for entry in timeline:
        if entry.source_type == SOURCE_FORECAST:
            answer = resolver(entry.valid_from, entry.valid_to)
            if getattr(answer, "source_type", None) != SOURCE_FORECAST:
                raise EnvironmentProviderError(
                    f"bucket {entry.bucket} was planned as a forecast bucket "
                    f"but the resolver now answers "
                    f"{getattr(answer, 'source_type', answer)!r}: "
                    f"{getattr(answer, 'reason', '')} Nothing is substituted.")
            field = np.asarray(read_environment(answer), dtype="float32")
            if field.shape != tuple(grid.shape):
                raise EnvironmentProviderError(
                    f"bucket {entry.bucket}: the forecast field is "
                    f"{field.shape}, the routing grid is {tuple(grid.shape)}; "
                    f"nothing is resampled here.")
            out.append(BucketSource(bucket=entry.bucket, entry=entry,
                                    resolution=answer, sic=field,
                                    origin="forecast"))
            continue

        if entry.source_type not in (SOURCE_OBSERVED, SOURCE_PERSISTENCE):
            raise EnvironmentProviderError(
                f"bucket {entry.bucket} has source type "
                f"{entry.source_type!r}, which carries no field to price with.")
        day = entry.environment_date
        if day not in observed:
            observed[day] = np.asarray(loader(day).sic, dtype="float32")
        out.append(BucketSource(bucket=entry.bucket, entry=entry,
                                resolution=None, sic=observed[day],
                                origin="observation"))
    return out


def forecast_aware_fields(*, timeline, resolver, grid: RoutingGrid,
                          constraints, polaris_penalties: dict, comp_config,
                          usnic_dir: Path, chart_date: date,
                          analysis_date: date, bucket_hours: float,
                          cost_params: CostParams | None = None,
                          sic_for_date=None) -> list[ComposedField]:
    """One ComposedField per bucket, each built from its own resolved source."""
    params = cost_params or CostParams()
    #  checked BEFORE any field is read: a refusal should not cost the caller
    #  a raster load first
    if params.current_weight > 0 and any(
            getattr(e, "source_type", None) == SOURCE_FORECAST for e in timeline):
        raise EnvironmentProviderError(
            "current_weight is above zero and at least one bucket is priced "
            "from a sea-ice forecast, but this project has no predicted "
            "current field. Pricing a future bucket's currents from the "
            "analysis day would be an unstated assumption, so it is refused "
            "here rather than done quietly.")

    sources = resolve_bucket_sources(timeline, resolver=resolver, grid=grid,
                                     sic_for_date=sic_for_date)
    fields: list[ComposedField] = []
    #  Buckets that share a source share one composition. Composing the full
    #  grid is the expensive step, and two buckets reading the same field can
    #  only produce the same array -- so it is computed once per SOURCE, the
    #  way src/api/world.py already caches per environment date.
    composed: dict = {}
    for source in sources:
        key = (source.origin,
               source.resolution.path.name if source.resolution
               else source.entry.environment_date.isoformat())
        if key not in composed:
            #  a grid carrying THIS source's sea ice, the analysis day's
            #  currents (weight 0.0, so they price nothing) and the shared mask
            bucket_grid = RoutingGrid(
                transform=grid.transform, crs=grid.crs, shape=tuple(grid.shape),
                sic=source.sic,
                current_u=np.asarray(grid.current_u, dtype="float32"),
                current_v=np.asarray(grid.current_v, dtype="float32"),
                blocked_mask=np.asarray(grid.blocked_mask, dtype=bool),
                day=analysis_date)
            env = build_cost_grid(bucket_grid, constraints=constraints,
                                  params=params).cost
            composed[key] = compose_grid(env, polaris_penalties, comp_config,
                                         usnic_dir)
        fields.append(ComposedField(
            bucket=source.bucket,
            composition=composed[key],
            #  the analysis this field is issued from, not its valid time
            environment_date=analysis_date, chart_date=chart_date,
            epsg=grid.crs.to_epsg(), shape=tuple(grid.shape),
            transform=tuple(grid.transform)[:6],
            label=f"{source.bucket * bucket_hours:g}-"
                  f"{(source.bucket + 1) * bucket_hours:g} h",
            environment=source.provenance()))
    return fields


def source_summary(fields) -> dict:
    """What a route can say about where each bucket's environment came from."""
    by_bucket = {int(f.bucket): dict(f.environment or {}) for f in fields}
    kinds = {b: e.get("source_type") for b, e in by_bucket.items()}
    models = sorted({e.get("model") for e in by_bucket.values() if e.get("model")})
    leads = sorted({e.get("lead_hours") for e in by_bucket.values()
                    if e.get("source_type") == SOURCE_FORECAST
                    and e.get("lead_hours") is not None})
    return {
        "source_type_by_bucket": kinds,
        "buckets_from_model_forecast": sorted(
            b for b, k in kinds.items() if k == SOURCE_FORECAST),
        "models": models,
        "forecast_leads_hours": leads,
        "environment_is_time_varying": len(
            {(e.get("source_type"), e.get("valid_time"),
              e.get("environment_date")) for e in by_bucket.values()}) > 1,
    }

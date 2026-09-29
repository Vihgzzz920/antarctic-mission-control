"""
Serve the composed navigation cost to time_astar, indexed by ARRIVAL time.

    provider = TimeIndexedNavigationCost.from_fields(fields, bucket_seconds=3600)
    res = time_astar(grid, start, goal,
                     cost_fn=provider.cost_fn("conservative"),
                     min_cost=provider.min_cost("conservative"),
                     vessel_speed_mps=5.0, time_step_s=3600.0,
                     max_horizon_s=provider.horizon_s)

THE LOOKUP, STATED ONCE
    arrival_time (s)  ->  bucket = floor((t - start_time) / bucket_seconds)
                      ->  the ComposedField registered for that bucket
                      ->  field.conservative_cost[row, col]  (or optimistic)

  time_astar prices an edge as
      distance * 0.5 * (cost_fn(A, t_depart) + cost_fn(B, t_arrive))
  so cell B is already evaluated at the moment the ship reaches it. This adapter
  therefore does exactly one thing with the time it is handed: it looks it up.
  It never uses the route start time for a later cell and never uses a
  departure time for the cell being entered.

NOTHING IS COMPUTED HERE
  Every field arrives already composed by navigation_cost.compose_grid, which
  in turn takes the environmental term from cost.py and the POLARIS penalty from
  polaris_risk.py. No SIC formula, no current formula, no RIO, no penalty value
  and no coverage-uncertainty formula appears in this file. It is a lookup table
  with a clock.

AN ABSENT FORECAST IS NEVER FILLED IN
  A bucket with no registered field raises ForecastUnavailable. This adapter
  will not reuse the newest field, fall back to yesterday, interpolate between
  buckets or extrapolate past the horizon: each of those invents a forecast the
  project does not have. A caller who would rather the search simply stop at the
  horizon can pass on_unavailable="impassable", which returns +inf -- the same
  meaning cost.py already gives an unpriceable cell, and the meaning
  time_astar documents for a non-finite return. That is NOT a hard navigation
  block on the cell; it says this layer cannot price it at that time.

TEMPORAL ALIGNMENT IS CHECKED, NOT ASSUMED
  Every field records the date of its environmental layer and the date of the
  USNIC chart behind its POLARIS term. Mixing dates is refused by default. The
  only way past that is allow_historical_demo_mismatch=True, which is named for
  what it is, stamps `historical_demo_override` on the provider and every field,
  and is never the default. A 2020 ice chart under a 2025 environment is a
  plumbing demonstration, not an operational assessment.

BOTH BRANCHES SURVIVE
  Conservative and optimistic costs are held side by side and chosen at call
  time. Asking for the other branch is a different cost_fn over the same fields;
  no RIO is recomputed and nothing is collapsed.

NOT IN SCOPE
  No route selection, no iceberg trajectory, no hard constraint, no mask, and
  no change to time_astar, which is used exactly as published.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np

from src.routing.navigation_cost import GridComposition

ROOT = Path(__file__).resolve().parents[2]

BRANCHES = ("conservative", "optimistic")
ON_UNAVAILABLE = ("error", "impassable")


class TimeCostError(ValueError):
    """The time-indexed cost field cannot be built or served as specified."""


class ForecastUnavailable(TimeCostError):
    """No forecast field exists for the requested time. Nothing is substituted."""


@dataclass(frozen=True)
class ComposedField:
    """One bucket's composed cost, with the provenance that justifies its date.

    `environment_date` is the date of the ANALYSIS this field is issued from --
    the same quantity the iceberg layer matches its forecast_start_date
    against. It is not necessarily the valid time of the bucket: a bucket four
    hours after departure is priced from the departure-day analysis unless the
    caller resolved a different field for it.

    `environment` optionally carries that resolution, as
    src.api.environment_timeline.BucketEnvironment.to_dict(): which raster was
    actually read for this bucket, the window it is valid for, and whether
    anything was carried forward. It is provenance only -- no lookup, cost or
    refusal in this module reads it -- and an empty dict simply means the
    caller did not record one.
    """

    bucket: int
    composition: GridComposition
    environment_date: date
    chart_date: date
    epsg: int
    shape: tuple[int, int]
    transform: tuple[float, ...]
    label: str = ""
    environment: dict = dc_field(default_factory=dict)

    def cost(self, branch: str) -> np.ndarray:
        if branch == "conservative":
            return self.composition.conservative_cost
        if branch == "optimistic":
            return self.composition.optimistic_cost
        raise TimeCostError(f"branch must be one of {BRANCHES}, got {branch!r}")

    @property
    def dates_aligned(self) -> bool:
        return self.environment_date == self.chart_date


@dataclass
class TimeIndexedNavigationCost:
    """Buckets of composed cost, served by arrival time."""

    fields: dict[int, ComposedField]
    bucket_seconds: float
    start_time: float = 0.0
    on_unavailable: str = "error"
    historical_demo_override: bool = False
    lookups: dict = dc_field(default_factory=dict)

    # ------------------------------------------------------------ building
    @classmethod
    def from_fields(cls, fields: list[ComposedField], bucket_seconds: float,
                    start_time: float = 0.0, on_unavailable: str = "error",
                    allow_historical_demo_mismatch: bool = False
                    ) -> "TimeIndexedNavigationCost":
        if not fields:
            raise TimeCostError("no forecast fields supplied")
        if not (isinstance(bucket_seconds, (int, float))
                and math.isfinite(bucket_seconds) and bucket_seconds > 0):
            raise TimeCostError(f"bucket_seconds must be finite and > 0, "
                                f"got {bucket_seconds!r}")
        if on_unavailable not in ON_UNAVAILABLE:
            raise TimeCostError(f"on_unavailable must be one of {ON_UNAVAILABLE}, "
                                f"got {on_unavailable!r}")

        ref = fields[0]
        by_bucket: dict[int, ComposedField] = {}
        for f in fields:
            if f.bucket in by_bucket:
                raise TimeCostError(f"two fields claim bucket {f.bucket}")
            # --- geospatial identity, checked not assumed ------------------
            if f.epsg != ref.epsg:
                raise TimeCostError(f"bucket {f.bucket}: CRS EPSG:{f.epsg} != "
                                    f"EPSG:{ref.epsg}")
            if tuple(f.shape) != tuple(ref.shape):
                raise TimeCostError(f"bucket {f.bucket}: shape {f.shape} != {ref.shape}")
            for i, (a, b) in enumerate(zip(f.transform, ref.transform)):
                if abs(float(a) - float(b)) > 1e-6:
                    raise TimeCostError(f"bucket {f.bucket}: transform[{i}] {a} != {b}")
            for branch in BRANCHES:
                arr = f.cost(branch)
                if arr.shape != tuple(ref.shape):
                    raise TimeCostError(f"bucket {f.bucket}: {branch} cost shape "
                                        f"{arr.shape} != {ref.shape}")
                finite = np.isfinite(arr)
                if finite.any() and (arr[finite] < 0).any():
                    raise TimeCostError(f"bucket {f.bucket}: negative {branch} cost")
                if np.isnan(arr).any():
                    raise TimeCostError(
                        f"bucket {f.bucket}: {branch} cost contains NaN; the project "
                        f"convention for an unpriceable cell is +inf, not NaN")
            # --- temporal alignment ---------------------------------------
            if not f.dates_aligned and not allow_historical_demo_mismatch:
                raise TimeCostError(
                    f"bucket {f.bucket}: environment dated {f.environment_date} but "
                    f"the USNIC chart behind its POLARIS term is dated "
                    f"{f.chart_date}. Combining them is a HISTORICAL DEMONSTRATION, "
                    f"not an operational assessment. Pass "
                    f"allow_historical_demo_mismatch=True to do it anyway; it will "
                    f"be stamped on every result.")
            by_bucket[f.bucket] = f

        mismatched = any(not f.dates_aligned for f in fields)
        return cls(fields=by_bucket, bucket_seconds=float(bucket_seconds),
                   start_time=float(start_time), on_unavailable=on_unavailable,
                   historical_demo_override=bool(mismatched))

    # -------------------------------------------------------------- lookup
    def bucket_for(self, arrival_time: float) -> int:
        """arrival_time (seconds) -> bucket index. The whole temporal rule."""
        t = float(arrival_time)
        if not math.isfinite(t):
            raise TimeCostError(f"arrival_time must be finite, got {t}")
        return int(math.floor((t - self.start_time) / self.bucket_seconds))

    def field_for(self, arrival_time: float) -> ComposedField:
        """The field for an arrival time. Raises rather than substituting."""
        b = self.bucket_for(arrival_time)
        f = self.fields.get(b)
        if f is None:
            raise ForecastUnavailable(
                f"no forecast field for bucket {b} (arrival t={float(arrival_time):g} s). "
                f"Available buckets: {sorted(self.fields)}. This adapter does not "
                f"reuse the newest field, fall back a day, interpolate or extrapolate.")
        return f

    @property
    def buckets(self) -> list[int]:
        return sorted(self.fields)

    @property
    def horizon_s(self) -> float:
        """The last arrival time this provider can price, for max_horizon_s.

        time_astar prunes an edge with `t_arrive > deadline`, so the deadline is
        INCLUSIVE. The first instant of the bucket after the last registered
        field would therefore still be evaluated, and it has no field. This
        returns the largest float strictly below that boundary, so that passing
        max_horizon_s=provider.horizon_s can never ask for a bucket that does
        not exist. It is not a claim that anything beyond it is impassable.
        """
        boundary = self.start_time + (max(self.fields) + 1) * self.bucket_seconds
        return math.nextafter(boundary, -math.inf)

    # ------------------------------------------------------- the cost_fn
    def cost_fn(self, branch: str = "conservative") -> Callable[[int, int, float], float]:
        """A cost_fn(row, col, arrival_time) for time_astar. Lookup only."""
        if branch not in BRANCHES:
            raise TimeCostError(f"branch must be one of {BRANCHES}, got {branch!r}")
        self.lookups.setdefault(branch, 0)

        def _cost(row: int, col: int, t: float) -> float:
            self.lookups[branch] += 1
            try:
                f = self.field_for(t)
            except ForecastUnavailable:
                if self.on_unavailable == "impassable":
                    # cost.py's own convention for a cell it cannot price. NOT a
                    # navigation prohibition on the cell itself.
                    return math.inf
                raise
            return float(f.cost(branch)[row, col])

        return _cost

    def min_cost(self, branch: str = "conservative") -> float:
        """The caller's promise to time_astar: a true lower bound over all buckets."""
        if branch not in BRANCHES:
            raise TimeCostError(f"branch must be one of {BRANCHES}, got {branch!r}")
        lows = []
        for f in self.fields.values():
            arr = f.cost(branch)
            finite = np.isfinite(arr)
            if finite.any():
                lows.append(float(arr[finite].min()))
        if not lows:
            raise TimeCostError("no finite cost in any bucket; min_cost is undefined")
        return min(lows)

    # ------------------------------------------------------------- audit
    def explain_lookup(self, row: int, col: int, arrival_time: float,
                       branch: str = "conservative") -> str:
        b = self.bucket_for(arrival_time)
        try:
            f = self.field_for(arrival_time)
        except ForecastUnavailable as exc:
            return (f"arrival t={float(arrival_time):g}s -> bucket {b} -> "
                    f"UNAVAILABLE\n  {exc}")
        cell = f.composition.cell(row, col, branch)
        lines = [
            f"arrival t={float(arrival_time):g}s -> bucket {b} -> field "
            f"{f.label or f.bucket!r}",
            f"  environment {f.environment_date}   USNIC chart {f.chart_date}"
            + ("   [HISTORICAL DEMO OVERRIDE: dates do not match]"
               if not f.dates_aligned else "   [dates aligned]"),
            f"  branch: {branch}",
        ]
        return "\n".join(lines) + "\n" + "\n".join(
            "  " + ln for ln in cell.explain().splitlines())

    def provenance(self) -> dict:
        return {
            "bucket_seconds": self.bucket_seconds,
            "start_time": self.start_time,
            "buckets": self.buckets,
            "horizon_s": self.horizon_s,
            "on_unavailable": self.on_unavailable,
            "historical_demo_override": self.historical_demo_override,
            "fields": {f.bucket: {"label": f.label,
                                  "environment_date": str(f.environment_date),
                                  "chart_date": str(f.chart_date),
                                  "dates_aligned": f.dates_aligned,
                                  "environment": dict(f.environment)}
                       for f in self.fields.values()},
            "lookups": dict(self.lookups),
        }

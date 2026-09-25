"""
Fold the iceberg exposure index into the existing time-indexed navigation cost.

    final_cost = existing_composed_cost + iceberg_exposure_weight * exposure

That is the whole arithmetic. One addition and one multiplication on top of
what is already there.

NOTHING BELOW IS RECOMPUTED
  The composed cost -- environmental + POLARIS penalty + coverage uncertainty --
  comes from time_navigation_cost.TimeIndexedNavigationCost, used as published.
  The exposure field comes from iceberg_risk, whose build_exposure and
  forecast_icebergs are imported rather than restated. No SIC formula, no RIO,
  no penalty value, no cone geometry and no bucket arithmetic is written in
  this file.

ARRIVAL TIME, AND THE SAME BUCKET RULE
  Exposure is selected by the vessel's ARRIVAL time at the cell, through the
  provider's own `bucket_for`. This module never spells out
  floor((t - start)/bucket_seconds): it asks the existing provider, so the two
  layers cannot drift into different clocks. time_astar prices an edge as
  distance * 0.5 * (cost(A, t_depart) + cost(B, t_arrive)), so the destination
  cell is already being asked about the moment the ship reaches it.

MISSING EXPOSURE IS NOT ZERO EXPOSURE
  A bucket with no exposure field does not silently become open water. Under
  the default policy the lookup raises IcebergExposureUnavailable. Under
  `omit_iceberg_term` the iceberg term contributes nothing but the cell is
  stamped `iceberg_exposure_unavailable`, so an audit can tell "no modeled
  iceberg here" apart from "nobody looked". Neither policy invents a number.

TEMPORAL MISMATCH IS REFUSED BY DEFAULT
  An exposure field forecast from one date cannot be laid over an environment
  from another without saying so. The check mirrors the one
  time_navigation_cost already performs, down to the override flag name.

NO MASK
  Every term added here is finite and >= 0. This layer produces no blocked,
  no-go, navigable or exclusion field; hard constraints remain with
  NavigationDomain and constraints.py. An infinite composed cost inherited from
  cost.py passes through unchanged, because adding to it would still be
  infinite and suppressing it would hand A* a finite price for a cell it must
  not enter.

Usage:
    provider = IcebergTimeNavigationCost.from_providers(base, exposure, cfg)
    res = time_astar(grid, start, goal, cost_fn=provider.cost_fn("conservative"),
                     min_cost=provider.min_cost("conservative"), ...)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field as dc_field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np

from src.routing.iceberg_risk import IcebergForecast
from src.routing.time_navigation_cost import (BRANCHES, ForecastUnavailable,
                                              TimeCostError,
                                              TimeIndexedNavigationCost)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "iceberg_navigation_cost.json"

STATE_PRICED = "iceberg_exposure_priced"
STATE_UNAVAILABLE = "iceberg_exposure_unavailable"
ON_MISSING = ("error", "omit_iceberg_term")


class IcebergCostError(ValueError):
    """The iceberg-aware navigation cost cannot be built or served."""


class IcebergExposureUnavailable(IcebergCostError):
    """No exposure field for that arrival time. Nothing is substituted."""


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class IcebergCostConfig:
    raw: dict
    path: str

    @property
    def weight(self) -> float:
        return float(self.raw["iceberg_exposure_weight"])

    @property
    def on_missing(self) -> str:
        return str(self.raw["missing_exposure"]["policy"])

    def with_weight(self, weight: float) -> "IcebergCostConfig":
        raw = json.loads(json.dumps(self.raw))
        raw["iceberg_exposure_weight"] = weight
        return _validated(raw, self.path)


def _validated(raw: dict, path: str) -> IcebergCostConfig:
    if "iceberg_exposure_weight" not in raw:
        raise IcebergCostError(f"{path} has no 'iceberg_exposure_weight'")
    w = raw["iceberg_exposure_weight"]
    if isinstance(w, bool) or not isinstance(w, (int, float)):
        raise IcebergCostError(f"iceberg_exposure_weight is {w!r}, not a number")
    w = float(w)
    if not math.isfinite(w):
        raise IcebergCostError(
            f"iceberg_exposure_weight is {w}. An infinite weight is a hard "
            f"navigation block wearing a number; hard constraints belong to "
            f"NavigationDomain, not to a cost.")
    lim = raw.get("limits", {})
    lo, hi = float(lim.get("min_weight", 0.0)), float(lim.get("max_weight", math.inf))
    if not lo <= w <= hi:
        raise IcebergCostError(
            f"iceberg_exposure_weight {w} is outside the configured limits "
            f"[{lo}, {hi}]")
    pol = raw.get("missing_exposure", {}).get("policy")
    if pol not in ON_MISSING:
        raise IcebergCostError(
            f"missing_exposure.policy is {pol!r}; it must be one of {ON_MISSING}")
    if raw.get("missing_exposure", {}).get("state_when_unavailable") != STATE_UNAVAILABLE:
        raise IcebergCostError(
            f"missing_exposure.state_when_unavailable must be "
            f"{STATE_UNAVAILABLE!r}")
    if raw.get("temporal", {}).get("mismatch_policy") != "reject":
        raise IcebergCostError(
            "temporal.mismatch_policy must be 'reject'; an exposure field from "
            "another date is never laid over an environment silently")
    if raw.get("no_hard_mask") is not True:
        raise IcebergCostError("no_hard_mask must be true")
    if raw.get("exposure_source", {}).get("is_a_probability") is not False:
        raise IcebergCostError(
            "exposure_source.is_a_probability must be false")
    return IcebergCostConfig(raw=raw, path=path)


def load_config(path: Path | str = CONFIG) -> IcebergCostConfig:
    """Read and validate the routing parameter. No built-in default."""
    path = Path(path)
    if not path.exists():
        raise IcebergCostError(
            f"No iceberg-cost configuration at {path}. The exposure weight is "
            f"a project routing parameter and must be stated in a file, not "
            f"chosen in code.")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise IcebergCostError(f"{path.name} is not valid JSON: {exc}") from None
    return _validated(raw, str(path))


# ------------------------------------------------------------- exposure
@dataclass(frozen=True)
class ExposureField:
    """One bucket's exposure surface, with the forecast that produced it."""

    bucket: int
    exposure: np.ndarray
    dominant: np.ndarray                  # index into `forecasts`, -1 = none
    forecasts: tuple[IcebergForecast, ...]
    forecast_start_date: date
    horizon_hours: float
    label: str = ""

    def iceberg_at(self, row: int, col: int) -> IcebergForecast | None:
        idx = int(self.dominant[row, col])
        return self.forecasts[idx] if idx >= 0 else None


@dataclass
class IcebergExposureProvider:
    """Exposure surfaces indexed by the SAME bucket rule as the base provider."""

    fields: dict[int, ExposureField]
    shape: tuple[int, int]

    @classmethod
    def from_fields(cls, fields: list[ExposureField],
                    shape: tuple[int, int]) -> "IcebergExposureProvider":
        by_bucket: dict[int, ExposureField] = {}
        for f in fields:
            if f.bucket in by_bucket:
                raise IcebergCostError(f"two exposure fields claim bucket {f.bucket}")
            if f.exposure.shape != tuple(shape):
                raise IcebergCostError(
                    f"bucket {f.bucket}: exposure shape {f.exposure.shape} != "
                    f"{shape}")
            if f.dominant.shape != tuple(shape):
                raise IcebergCostError(
                    f"bucket {f.bucket}: dominant shape {f.dominant.shape} != "
                    f"{shape}")
            arr = f.exposure
            if not np.isfinite(arr).all():
                raise IcebergCostError(
                    f"bucket {f.bucket}: exposure contains a non-finite value")
            if (arr < 0.0).any() or (arr > 1.0).any():
                raise IcebergCostError(
                    f"bucket {f.bucket}: exposure leaves [0, 1] "
                    f"(min {arr.min()}, max {arr.max()})")
            by_bucket[f.bucket] = f
        if not by_bucket:
            raise IcebergCostError("no exposure fields supplied")
        return cls(fields=by_bucket, shape=tuple(shape))

    def field_for_bucket(self, bucket: int) -> ExposureField:
        f = self.fields.get(bucket)
        if f is None:
            raise IcebergExposureUnavailable(
                f"no iceberg exposure field for bucket {bucket}. Available "
                f"buckets: {sorted(self.fields)}. A missing field is NOT read "
                f"as zero exposure.")
        return f


# ------------------------------------------------------------- the cell
@dataclass(frozen=True)
class IcebergCellCost:
    """Every term behind one cell's final cost, at one arrival time."""

    row: int
    col: int
    branch: str
    arrival_time: float
    bucket: int
    state: str
    environmental_cost: float
    polaris_penalty: float
    coverage_uncertainty_cost: float
    composed_cost: float
    iceberg_exposure: float | None
    iceberg_cost: float
    final_cost: float
    iceberg_exposure_weight: float
    iceberg_id: str | None = None
    predicted_center_latitude: float | None = None
    predicted_center_longitude: float | None = None
    radius_km: float | None = None
    forecast_horizon_hours: float | None = None
    extrapolated_uncertainty: bool | None = None

    def explain(self) -> str:
        ice = ("unavailable" if self.iceberg_exposure is None
               else f"{self.iceberg_exposure:.6g}")
        berg = self.iceberg_id or "none"
        extra = ""
        if self.extrapolated_uncertainty:
            extra = "   [EXTRAPOLATED uncertainty: beyond the 24 h calibration]"
        return "\n".join([
            f"cell (row {self.row}, col {self.col})  branch={self.branch}  "
            f"state={self.state}",
            f"  arrival t={self.arrival_time:g}s -> bucket {self.bucket}",
            f"  composed cost        : {self.composed_cost:g}  "
            f"(env {self.environmental_cost:g} + POLARIS "
            f"{self.polaris_penalty:g} + coverage "
            f"{self.coverage_uncertainty_cost:g})",
            f"  iceberg exposure     : {ice}  from {berg}"
            + (f"  centre ({self.predicted_center_latitude:.4f}, "
               f"{self.predicted_center_longitude:.4f})  r="
               f"{self.radius_km:.3f} km  horizon "
               f"{self.forecast_horizon_hours:g} h" if berg != "none" else "")
            + extra,
            f"  iceberg cost         : {self.iceberg_exposure_weight:g} x "
            f"{ice} = {self.iceberg_cost:g}",
            f"  final cost           : {self.composed_cost:g} + "
            f"{self.iceberg_cost:g} = {self.final_cost:g}",
        ])


# ---------------------------------------------------------- the provider
@dataclass
class IcebergTimeNavigationCost:
    """The composed cost plus a weighted iceberg-exposure term."""

    base: TimeIndexedNavigationCost
    exposure: IcebergExposureProvider
    config: IcebergCostConfig
    historical_demo_override: bool = False
    lookups: dict = dc_field(default_factory=dict)

    @classmethod
    def from_providers(cls, base: TimeIndexedNavigationCost,
                       exposure: IcebergExposureProvider,
                       config: IcebergCostConfig,
                       allow_historical_demo_mismatch: bool = False
                       ) -> "IcebergTimeNavigationCost":
        ref = base.fields[min(base.fields)]
        if tuple(ref.shape) != tuple(exposure.shape):
            raise IcebergCostError(
                f"exposure shape {exposure.shape} != composed-cost shape "
                f"{ref.shape}; nothing is resampled here")
        mismatched = []
        for bucket, ef in sorted(exposure.fields.items()):
            cf = base.fields.get(bucket)
            if cf is None:
                continue
            if ef.forecast_start_date != cf.environment_date:
                mismatched.append(
                    f"bucket {bucket}: iceberg forecast dated "
                    f"{ef.forecast_start_date} but the environment is dated "
                    f"{cf.environment_date}")
        if mismatched and not allow_historical_demo_mismatch:
            raise IcebergCostError(
                "the iceberg exposure and the environment come from different "
                "dates:\n  " + "\n  ".join(mismatched)
                + "\nCombining them is a HISTORICAL DEMONSTRATION, not an "
                  "operational assessment. Pass "
                  "allow_historical_demo_mismatch=True to do it anyway; it "
                  "will be stamped on the provider.")
        return cls(base=base, exposure=exposure, config=config,
                   historical_demo_override=bool(mismatched))

    # -------------------------------------------------------- the lookup
    def bucket_for(self, arrival_time: float) -> int:
        """Delegated, never restated: one clock for both layers."""
        return self.base.bucket_for(arrival_time)

    def cell(self, row: int, col: int, arrival_time: float,
             branch: str = "conservative") -> IcebergCellCost:
        """Every term behind one cell at one ARRIVAL time."""
        if branch not in BRANCHES:
            raise IcebergCostError(f"branch must be one of {BRANCHES}, got "
                                   f"{branch!r}")
        bucket = self.bucket_for(arrival_time)
        base_field = self.base.field_for(arrival_time)
        comp = base_field.composition.cell(row, col, branch)
        weight = self.config.weight
        try:
            ef = self.exposure.field_for_bucket(bucket)
        except IcebergExposureUnavailable:
            if self.config.on_missing == "error":
                raise
            return IcebergCellCost(
                row=row, col=col, branch=branch,
                arrival_time=float(arrival_time), bucket=bucket,
                state=STATE_UNAVAILABLE,
                environmental_cost=comp.environmental_cost,
                polaris_penalty=comp.polaris_penalty,
                coverage_uncertainty_cost=comp.coverage_uncertainty_cost,
                composed_cost=comp.composed_cost,
                iceberg_exposure=None, iceberg_cost=0.0,
                final_cost=comp.composed_cost,
                iceberg_exposure_weight=weight)
        exposure = float(ef.exposure[row, col])
        berg = ef.iceberg_at(row, col)
        ice_cost = weight * exposure
        return IcebergCellCost(
            row=row, col=col, branch=branch, arrival_time=float(arrival_time),
            bucket=bucket, state=STATE_PRICED,
            environmental_cost=comp.environmental_cost,
            polaris_penalty=comp.polaris_penalty,
            coverage_uncertainty_cost=comp.coverage_uncertainty_cost,
            composed_cost=comp.composed_cost,
            iceberg_exposure=exposure, iceberg_cost=ice_cost,
            final_cost=comp.composed_cost + ice_cost,
            iceberg_exposure_weight=weight,
            iceberg_id=berg.iceberg_id if berg else None,
            predicted_center_latitude=berg.latitude if berg else None,
            predicted_center_longitude=berg.longitude if berg else None,
            radius_km=berg.radius_km if berg else None,
            forecast_horizon_hours=berg.horizon_hours if berg else None,
            extrapolated_uncertainty=(berg.extrapolated_uncertainty
                                      if berg else None))

    def cost_fn(self, branch: str = "conservative"
                ) -> Callable[[int, int, float], float]:
        """A cost_fn(row, col, arrival_time) for time_astar. Lookup and add."""
        if branch not in BRANCHES:
            raise IcebergCostError(f"branch must be one of {BRANCHES}, got "
                                   f"{branch!r}")
        self.lookups.setdefault(branch, 0)
        base_fn = self.base.cost_fn(branch)

        def _cost(row: int, col: int, t: float) -> float:
            self.lookups[branch] += 1
            composed = base_fn(row, col, t)
            if not math.isfinite(composed):
                # cost.py already declared this cell unpriceable; adding a
                # finite preference to an infinity would say nothing.
                return composed
            try:
                ef = self.exposure.field_for_bucket(self.bucket_for(t))
            except IcebergExposureUnavailable:
                if self.config.on_missing == "error":
                    raise
                return composed
            return composed + self.config.weight * float(ef.exposure[row, col])

        return _cost

    def min_cost(self, branch: str = "conservative") -> float:
        """The base lower bound. The iceberg term only ever adds."""
        return self.base.min_cost(branch)

    def explain_lookup(self, row: int, col: int, arrival_time: float,
                       branch: str = "conservative") -> str:
        head = self.base.explain_lookup(row, col, arrival_time, branch)
        return head + "\n" + self.cell(row, col, arrival_time, branch).explain()

    def provenance(self) -> dict:
        return {
            "iceberg_exposure_weight": self.config.weight,
            "weight_is_a_project_routing_parameter": True,
            "weight_is_a_polaris_or_physical_quantity": False,
            "formula": self.config.raw["formula"],
            "missing_exposure_policy": self.config.on_missing,
            "state_when_unavailable": STATE_UNAVAILABLE,
            "historical_demo_override": self.historical_demo_override,
            "no_hard_mask": True,
            "exposure_buckets": sorted(self.exposure.fields),
            "exposure_horizons_hours": {
                b: f.horizon_hours for b, f in sorted(self.exposure.fields.items())},
            "exposure_extrapolated_by_bucket": {
                b: bool(f.forecasts and f.forecasts[0].extrapolated_uncertainty)
                for b, f in sorted(self.exposure.fields.items())},
            "base": self.base.provenance(),
            "lookups": dict(self.lookups),
        }

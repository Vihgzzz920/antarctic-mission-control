"""
Historical forecast -> mission decision backtest.

    python -m src.evaluation.historical_forecast_decision

THE QUESTION
  Does forecast information CHANGE a routing decision, and how does the changed
  decision compare against the sea ice that actually occurred? Three things are
  measured and reported SEPARATELY, never combined into one number:

    LAYER 1  forecast accuracy   -- how close the +24 h / +48 h fields were to
                                    the observation that later arrived
    LAYER 2  decision effect     -- did the route change, and by how much
    LAYER 3  realized outcome    -- what each ALREADY-PLANNED path would have
                                    encountered given what actually happened

  There is no composite "forecast value score" here, and no route is called
  better, safer, optimal or recommended.

CAUSAL ORDER, WHICH IS THE WHOLE POINT
  1. the forecast is produced from the origin day's observation only
  2. both plans are routed -- one on persistence, one on the forecast
  3. ONLY THEN is the observation for the target days opened, and both
     already-fixed paths are priced against it
  Nothing is re-routed after the realization is seen, and the forecast is never
  scored against itself. The ordering is enforced in code (see `_Realizer`)
  and asserted by tests.

HELD-OUT ORIGINS ONLY
  The origins come from the intersection of the two models' own recorded TEST
  base ranges, read from the training metrics files rather than retyped. A
  training- or validation-period origin cannot enter the primary backtest; it
  is skipped with a structured reason.

NOTHING HERE IS A VOYAGE
  The geometry is the already-validated synthetic long-horizon demonstration,
  unchanged. is_operational_mission is false on every case, and the iceberg
  term is omitted beyond its configured 12 h horizon exactly as the long-horizon
  evaluation already omits it.

NOT IN THIS MODULE
  No A*, no POLARIS, no fuel model, no SIC model and no cost model. Every one of
  those is called, not reimplemented.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio

from src.api.config import BEDMACHINE, COMP_CONFIG, DEMO, USNIC
from src.api.mission_decision import (LONG_HORIZON_SITE, LONG_HORIZON_START,
                                      MODE_LONG_HORIZON_EVALUATION,
                                      evaluate_mission)
from src.api.environment_timeline import (POLICY_LONG_HORIZON_EVALUATION,
                                          POLICY_PERSISTENCE)
from src.api.world import chart_date, environment_dates
from src.data.preprocess import load_sic
from src.models.forecast_sic_raster import (TRAINED_LEAD_HOURS,
                                            ForecastRasterError,
                                            hindcast_validation, predict_sic,
                                            raster_path, write_forecast_raster)
from src.routing.cost import build_cost_grid
from src.routing.fuel_cost import load_fuel_params, route_fuel
from src.routing.grid import RoutingGrid
from src.routing.navigation_domain import NavigationDomain
from src.routing.end_to_end_route import build_polaris_penalties
from src.routing.route_profiles import FUEL_CONFIG, PROFILE_RISK

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
METRICS = {24: ROOT / "data" / "processed" / "forecast_metrics.json",
           48: ROOT / "data" / "processed" / "forecast_metrics_48h.json"}
#  a SEPARATE directory: the canonical shipped rasters are never overwritten
EVAL_DIR = ROOT / "data" / "processed" / "evaluation"
EVAL_RASTERS = EVAL_DIR / "sic_forecast"
RESULT_PATH = EVAL_DIR / "historical_forecast_decision_backtest.json"
#  One JSON per case. The backtest is deterministic, so a cached case and a
#  recomputed one must be identical -- which is asserted rather than assumed
#  (tests/test_historical_backtest.py). The cache exists so a long run can be
#  produced in pieces, not to make the result depend on what ran before.
CASE_DIR = EVAL_DIR / "cases"

HOUR = 3600.0
PROFILE = PROFILE_RISK
HORIZON_BUCKETS = 16
BUCKET_HOURS = float(DEMO["bucket_hours"])

#  The candidate rule, fixed before the backtest was ever run: every fifth day
#  of the held-out December month. Small, spread across the period, and
#  reproducible from this line alone.
CANDIDATE_RULE = "every 5th day of 2025-12, from 2025-12-01"
CANDIDATE_ORIGINS = tuple(date(2025, 12, d) for d in range(1, 32, 5))

SKIP_OUTSIDE_TEST = "origin_is_not_in_the_held_out_test_period"
SKIP_NO_TARGET = "target_observation_is_not_in_the_archive"
SKIP_NO_ORIGIN = "origin_observation_is_not_in_the_archive"
SKIP_NO_REALIZED = "an_observation_the_route_would_cross_is_not_in_the_archive"


class BacktestError(ValueError):
    """The backtest cannot be run as specified."""


class RealizationUnavailable(BacktestError):
    """A planned path cannot be scored against reality without inventing data.

    Raised when the observation that actually arrived has no value at a cell
    the plan crosses. The plan itself is still valid and its decision effect is
    still measurable; only LAYER 3 is unavailable for that case.
    """


# ------------------------------------------------------- held-out periods
def test_base_range(lead_hours: int) -> tuple[date, date]:
    """The model's OWN recorded test base range, read from its metrics file."""
    path = METRICS[int(lead_hours)]
    try:
        rec = json.loads(path.read_text())["split_information"]["test"]
        lo, hi = rec["base_range"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BacktestError(
            f"{path.name} does not record a test base range: {exc}") from exc
    return date.fromisoformat(lo), date.fromisoformat(hi)


def held_out_window() -> tuple[date, date]:
    """Origins safely held out for BOTH leads: the intersection."""
    ranges = [test_base_range(l) for l in TRAINED_LEAD_HOURS]
    return max(r[0] for r in ranges), min(r[1] for r in ranges)


def observed(day: date) -> Path:
    return RAW_DIR / f"sic_{day:%Y%m%d}.tif"


# --------------------------------------------------------------- the case
@dataclass(frozen=True)
class Case:
    origin: date
    included: bool
    reason: str | None = None
    targets: dict = field(default_factory=dict)      # lead -> date

    def to_dict(self) -> dict:
        return {"origin": self.origin.isoformat(), "included": self.included,
                "skip_reason": self.reason,
                "targets": {str(k): v.isoformat() for k, v in self.targets.items()}}


def build_cases(origins=CANDIDATE_ORIGINS) -> list[Case]:
    """Decide, per candidate, whether it can be backtested -- and if not, why.

    Nothing is fabricated: a missing observation skips the case with a named
    reason rather than being filled, interpolated or carried forward.
    """
    lo, hi = held_out_window()
    dates = environment_dates()
    out = []
    for origin in origins:
        targets = {int(l): origin + timedelta(hours=int(l)) for l in TRAINED_LEAD_HOURS}
        if not (lo <= origin <= hi):
            out.append(Case(origin, False,
                            f"{SKIP_OUTSIDE_TEST}: the held-out window for both "
                            f"leads is {lo} to {hi}", targets))
            continue
        if origin not in dates or not observed(origin).exists():
            out.append(Case(origin, False, SKIP_NO_ORIGIN, targets))
            continue
        missing = [f"+{l}h -> {d}" for l, d in targets.items()
                   if not observed(d).exists()]
        if missing:
            out.append(Case(origin, False,
                            f"{SKIP_NO_TARGET}: {', '.join(missing)}", targets))
            continue
        #  the realized evaluation opens every day the route can cross
        span = int(HORIZON_BUCKETS * BUCKET_HOURS // 24) + 1
        absent = [str(origin + timedelta(days=k)) for k in range(span)
                  if not observed(origin + timedelta(days=k)).exists()]
        if absent:
            out.append(Case(origin, False,
                            f"{SKIP_NO_REALIZED}: {', '.join(absent)}", targets))
            continue
        out.append(Case(origin, True, None, targets))
    return out


# ------------------------------------------------------- a world for a day
def day_world(day: date) -> dict:
    """The routing world for ONE historical day.

    The same builders src/api/world.py uses, on a different date. POLARIS, the
    land mask, the composition config and the USNIC chart are the SHIPPED ones
    and are identical for every case, so only the sea ice differs between
    cases. No iceberg provider: this evaluation omits that term.
    """
    grid = RoutingGrid.for_date(day)
    domain = NavigationDomain(grid).add_geotiff("land_mask", BEDMACHINE)
    constraints = domain.to_constraints()
    polaris = build_polaris_penalties(ice_class=DEMO["polaris_ice_class"],
                                      riv_table=DEMO["polaris_riv_table"])
    return {"day": day, "grid": grid, "domain": domain,
            "constraints": constraints, "polaris": polaris,
            "chart_date": chart_date(),
            "excluded": constraints.combined(tuple(grid.shape)),
            "environment_summary": {"policy": POLICY_PERSISTENCE}}


# ---------------------------------------------------------- the forecasts
def ensure_rasters(origin: date, *, out_dir: Path = EVAL_RASTERS) -> dict:
    """Produce this origin's +24 h and +48 h rasters from the shipped
    artifacts, into the EVALUATION directory. The canonical shipped rasters are
    not touched."""
    made = {}
    for lead in TRAINED_LEAD_HOURS:
        forecast = predict_sic(origin, lead_hours=int(lead))
        path = raster_path(origin, Path(out_dir), int(lead))
        if path.exists():
            #  Already produced by an earlier run. It is NOT overwritten --
            #  it is CHECKED against this run's prediction, so a reused file
            #  can never be a stale or different field wearing the same name.
            with rasterio.open(path) as ds:
                have = ds.read(1).astype("float64")
            want = np.asarray(forecast.sic, dtype="float64")
            same_mask = (np.isfinite(have) == np.isfinite(want)).all()
            both = np.isfinite(have) & np.isfinite(want)
            if not (same_mask and bool((have[both] == want[both]).all())):
                raise BacktestError(
                    f"{path.name} already exists but does not equal the field "
                    f"this run predicts for the same origin and lead. It is "
                    f"not overwritten; move or remove it deliberately.")
            reused = True
        else:
            path = write_forecast_raster(forecast, out_dir=out_dir,
                                         overwrite=False)
            reused = False
        made[int(lead)] = {"path": str(path.relative_to(ROOT)),
                           "forecast": forecast, "reused": reused}
    return made


def accuracy(made: dict) -> dict:
    """LAYER 1. The forecast scored against the observation that arrived."""
    #  the four figures this project already reports, under the names the
    #  validator itself uses. Nothing is recomputed here.
    keys = (("mae", "mae"), ("rmse", "rmse"), ("bias", "bias"),
            ("miz_mae", "mae_miz"))

    def take(block):
        if not block:
            return None
        return {out_key: float(block[src]) for out_key, src in keys}

    out = {}
    for lead, rec in sorted(made.items()):
        v = hindcast_validation(rec["forecast"])
        label, blocks = v["model_label"], v.get("metrics") or {}
        #  the block is keyed by the model's short name; `model_label` is the
        #  human-readable "HGB +<lead>h" that goes in the report
        key = next((k for k in blocks if k != "persistence"), None)
        if key is None:
            raise BacktestError(
                f"the validator reported no model metrics block; it has "
                f"{sorted(blocks)}")
        #  the validator's OWN leakage verdict, carried through rather than
        #  recomputed: an origin inside the training period is not held out
        leaked = bool(v.get("origin_is_inside_the_training_period"))
        if leaked:
            raise BacktestError(
                f"origin {v['forecast_origin']} is inside the training period "
                f"for the +{lead}h model; it cannot enter the held-out "
                f"backtest. The validator reported this, not this module.")
        out[f"+{lead}h"] = {
            "valid_time": v["valid_time"], "scored_cells": v["scored_cells"],
            "model_label": label,
            "origin_is_inside_the_training_period": leaked,
            "observation": v.get("observation"),
            "model": take(blocks[key]),
            "persistence": take(blocks.get("persistence")),
        }
    return out


# -------------------------------------------------- realized-history pricing
class _Realizer:
    """Prices an ALREADY-FIXED path against the observation that occurred.

    It is constructed only after both plans exist, and it exposes no routing of
    any kind: it can price a path and nothing else. That is the code-level
    guarantee that no plan is re-optimised once the outcome is known.
    """

    def __init__(self, origin: date, world: dict):
        self._origin = origin
        self._world = world
        self._cache: dict = {}
        self._fuel = load_fuel_params(FUEL_CONFIG)
        self._sic: dict = {}

    def _day_for(self, bucket: int) -> date:
        return self._origin + timedelta(
            days=int((bucket * BUCKET_HOURS) // 24))

    def _observed_grid(self, day: date):
        if day not in self._cache:
            grid = RoutingGrid.for_date(day)
            cost = build_cost_grid(grid, constraints=self._world["constraints"]).cost
            self._cache[day] = cost
            self._sic[day] = np.asarray(
                load_sic(observed(day)).sic, dtype="float64")
        return self._cache[day], self._sic[day]

    def price(self, path, arrival_times) -> dict:
        """Environmental cost and fuel this path would have met, in reality."""
        px = self._world["grid"].pixel_size
        dx, dy = abs(float(px[0])), abs(float(px[1]))
        vals, days, buckets, unpriceable = [], [], [], []
        for (r, c), t in zip(path, arrival_times):
            b = int(np.floor(float(t) / (BUCKET_HOURS * HOUR)))
            day = self._day_for(b)
            cost, _sic = self._observed_grid(day)
            v = float(cost[int(r), int(c)])
            if not np.isfinite(v):
                unpriceable.append((int(r), int(c), day.isoformat()))
            vals.append(v)
            days.append(day)
            buckets.append(b)
        if unpriceable:
            #  The plan was made on the analysis available at the origin, where
            #  these cells were priceable. The observation that later arrived
            #  has no value there (a swath gap or the pole hole). Nothing is
            #  substituted, carried forward or read as ice-free, so this path
            #  simply cannot be scored against reality.
            raise RealizationUnavailable(
                f"{len(unpriceable)} of {len(path)} cells on this path have no "
                f"observed sea ice on the day the vessel would reach them "
                f"(first: cell {unpriceable[0][0], unpriceable[0][1]} on "
                f"{unpriceable[0][2]}). The realized evaluation substitutes "
                f"nothing, so this path is not scored.")
        total = 0.0
        for a, b_, va, vb in zip(path, path[1:], vals, vals[1:]):
            dist = float(np.hypot((b_[1] - a[1]) * dx, (b_[0] - a[0]) * dy))
            total += dist * 0.5 * (va + vb)
        fuel = route_fuel(
            path, arrival_times,
            sic_for_bucket=lambda bb: self._sic[self._day_for(int(bb))],
            bucket_of=lambda t: int(np.floor(float(t) / (BUCKET_HOURS * HOUR))),
            pixel_size=px, params=self._fuel)
        per_day = {}
        for d, v in zip(days, vals):
            per_day.setdefault(d.isoformat(), []).append(v)
        return {
            "realized_environmental_cost": total,
            "realized_estimated_fuel": fuel.estimated_fuel,
            "realized_estimated_fuel_units": fuel.units,
            "realized_estimated_fuel_per_km": fuel.fuel_per_km,
            "observations_used": sorted({d.isoformat() for d in days}),
            "buckets_used": sorted(set(buckets)),
            "mean_observed_cost_per_day": {
                k: float(np.mean(v)) for k, v in sorted(per_day.items())},
        }


# ------------------------------------------------------------- one case
def run_case(case: Case) -> dict:
    """Forecast, then plan both, then -- and only then -- realize."""
    if not case.included:
        return {"case": case.to_dict(), "ran": False}
    origin_dt = datetime.combine(case.origin, datetime.min.time())
    world = day_world(case.origin)

    #  STEP 1: the forecast, from the origin day's observation only
    made = ensure_rasters(case.origin)
    layer1 = accuracy(made)

    #  STEP 2: both plans, neither of which has seen a target-day observation
    plans = {}
    for policy in (POLICY_PERSISTENCE, POLICY_LONG_HORIZON_EVALUATION):
        matrix = evaluate_mission(
            sites=[LONG_HORIZON_SITE], departure_times=[origin_dt],
            #  this case's OWN origin is the reference, so the route starts at
            #  offset 0 in its own provider timeline rather than at the demo's
            reference_departure=origin_dt,
            start=LONG_HORIZON_START, environment_policy=policy,
            profiles=(PROFILE,), horizon_buckets=HORIZON_BUCKETS,
            grid_factory=lambda d=case.origin: RoutingGrid.for_date(d),
            forecast_dir=EVAL_RASTERS, world=world)
        plans[policy] = matrix.options[0]

    a, b = plans[POLICY_PERSISTENCE], plans[POLICY_LONG_HORIZON_EVALUATION]
    out = {"case": case.to_dict(), "ran": True,
           "forecast_accuracy": layer1,
           "forecast_rasters": {str(k): v["path"] for k, v in made.items()},
           "plans": {}, "decision": {}, "realized": {}}
    for name, option in (("persistence", a), ("forecast", b)):
        p = option.profiles.get(PROFILE, {})
        out["plans"][name] = {
            "policy": option.scenario.environment_policy,
            "evaluation_mode": option.scenario.evaluation_mode,
            "is_operational_mission": option.scenario.is_operational_mission,
            "is_operational_assessment": option.scenario.is_operational_assessment,
            "iceberg_exposure": option.scenario.iceberg_exposure,
            "feasible": option.feasible,
            "refusal_outcome": option.refusal_outcome,
            "refusal_reason": option.refusal_reason,
            "metrics": option.objective_values().get(PROFILE),
            "environment": {k: option.environment.get(k) for k in
                            ("source_types", "models", "forecast_origins",
                             "forecast_leads", "validity_rules", "artifacts",
                             "buckets_used")},
            "forecast_buckets": [
                {"bucket": x["bucket"], "artifact": (x.get("resolution") or {}).get("path"),
                 "lead_hours": (x.get("resolution") or {}).get("lead_hours"),
                 "model_artifact_sha256": x.get("model_artifact_sha256"),
                 "dataset_sha256": x.get("dataset_sha256")}
                for x in option.environment.get("buckets", [])
                if x.get("source_type") == "model_forecast"],
        }
    if not (a.feasible and b.feasible):
        out["decision"] = {"both_planned": False}
        return out

    pa = [tuple(c) for c in a.profiles[PROFILE]["path"]]
    pb = [tuple(c) for c in b.profiles[PROFILE]["path"]]
    va, vb = out["plans"]["persistence"]["metrics"], out["plans"]["forecast"]["metrics"]
    out["decision"] = {
        "both_planned": True,
        "route_changed": pa != pb,
        "cells_only_in_persistence": len(set(pa) - set(pb)),
        "cells_only_in_forecast": len(set(pb) - set(pa)),
        "shared_cells": len(set(pa) & set(pb)),
        "first_divergence_index": next(
            (i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), None),
        "distance_km_delta": vb["distance_km"] - va["distance_km"],
        "travel_time_h_delta": vb["travel_time_h"] - va["travel_time_h"],
        "configured_cost_delta": vb["configured_cost"] - va["configured_cost"],
        "estimated_fuel_delta": vb["estimated_fuel"] - va["estimated_fuel"],
    }

    #  STEP 3: only now is the target-day observation opened
    realizer = _Realizer(case.origin, world)
    try:
        ra = realizer.price(pa, a.profiles[PROFILE]["arrival_times"])
        rb = realizer.price(pb, b.profiles[PROFILE]["arrival_times"])
    except RealizationUnavailable as exc:
        out["realized"] = {
            "evaluated": False,
            "reason": str(exc),
            "layers_still_valid": ["forecast_accuracy", "decision"],
            "nothing_was_substituted": True,
        }
        return out
    denv = rb["realized_environmental_cost"] - ra["realized_environmental_cost"]
    dfuel = rb["realized_estimated_fuel"] - ra["realized_estimated_fuel"]
    out["realized"] = {
        "evaluated": True,
        "persistence_plan": ra, "forecast_plan": rb,
        "realized_environmental_cost_delta": denv,
        "realized_estimated_fuel_delta": dfuel,
        #  a statement about THIS case only. Not a verdict, not a ranking.
        "which_plan_met_lower_realized_environmental_cost": (
            "identical" if pa == pb else
            "forecast_plan" if denv < 0 else
            "persistence_plan" if denv > 0 else "tied"),
        "is_a_claim_that_one_plan_is_better": False,
        "scope": "this origin, this geometry, this parameter set only",
    }
    return out


# ------------------------------------------------------------- the backtest
def case_path(origin: date, cache_dir: Path = CASE_DIR) -> Path:
    return Path(cache_dir) / f"case_{origin:%Y%m%d}.json"


def run_case_cached(case: Case, cache_dir: Path = CASE_DIR,
                    refresh: bool = False) -> dict:
    """Compute one case, or reuse an identical earlier computation of it."""
    path = case_path(case.origin, cache_dir)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    result = run_case(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    #  read back, so the cached and in-memory shapes can never differ
    return json.loads(path.read_text())


def run(origins=CANDIDATE_ORIGINS, *, cache_dir: Path = CASE_DIR,
        refresh: bool = False) -> dict:
    cases = build_cases(origins)
    results = [run_case_cached(c, cache_dir, refresh) for c in cases]
    ran = [r for r in results if r["ran"] and r["decision"].get("both_planned")]
    changed = [r for r in ran if r["decision"]["route_changed"]]
    realized = [r for r in ran if r["realized"].get("evaluated")]
    unrealized = [r for r in ran if not r["realized"].get("evaluated")]
    lower_f = [r for r in realized
               if r["realized"]["which_plan_met_lower_realized_environmental_cost"]
               == "forecast_plan"]
    lower_p = [r for r in realized
               if r["realized"]["which_plan_met_lower_realized_environmental_cost"]
               == "persistence_plan"]
    return {
        "what_this_is": (
            "a historical forecast-value / decision-value backtest. It measures "
            "whether forecast information changed a routing decision and how "
            "the already-planned paths compare against the sea ice that "
            "actually occurred. It is NOT a claim that either plan is better, "
            "safer, optimal or operationally recommended."),
        "is_operational_assessment": False,
        "is_operational_mission": False,
        "candidate_rule": CANDIDATE_RULE,
        "held_out_window": [d.isoformat() for d in held_out_window()],
        "held_out_source": "the test base_range recorded in each model's own "
                           "training metrics file, read not retyped",
        "scenario": {
            "geometry": "the already-validated synthetic long-horizon "
                        "demonstration; not a voyage replay",
            "start": list(LONG_HORIZON_START),
            "goal": list(LONG_HORIZON_SITE.cell),
            "profile": PROFILE, "horizon_buckets": HORIZON_BUCKETS,
            "bucket_hours": BUCKET_HOURS,
            "vessel_speed_mps": DEMO["vessel_speed_mps"],
            "polaris_ice_class": DEMO["polaris_ice_class"],
            "polaris_riv_table": DEMO["polaris_riv_table"],
            "iceberg_exposure": "omitted beyond its configured 12 h horizon",
            "evaluation_mode": MODE_LONG_HORIZON_EVALUATION,
        },
        "causal_order": [
            "1. forecast produced from the origin day's observation only",
            "2. both plans routed, neither having seen a target-day observation",
            "3. only then is the target-day observation opened and both "
            "already-fixed paths priced against it",
        ],
        "aggregate": {
            "cases_attempted": len(cases),
            "cases_included": sum(1 for c in cases if c.included),
            "cases_skipped": sum(1 for c in cases if not c.included),
            "skip_reasons": {c.origin.isoformat(): c.reason
                             for c in cases if not c.included},
            "cases_both_planned": len(ran),
            "route_changed": len(changed),
            "route_unchanged": len(ran) - len(changed),
            "cases_realized": len(realized),
            "cases_not_realized": len(unrealized),
            "not_realized_reasons": {
                r["case"]["origin"]: r["realized"]["reason"]
                for r in unrealized},
            "forecast_plan_lower_realized_environmental_cost": len(lower_f),
            "persistence_plan_lower_realized_environmental_cost": len(lower_p),
            "tied_or_identical_route": len(realized) - len(lower_f) - len(lower_p),
            "this_is_not_a_universal_accuracy_claim": True,
            "scope": "these origins, this geometry, this parameter set only",
        },
        "cases": results,
    }


def save(result: dict, path: Path = RESULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    return path


def main() -> int:
    result = run()
    path = save(result)
    agg = result["aggregate"]
    print("=" * 78)
    print("HISTORICAL FORECAST -> MISSION DECISION BACKTEST")
    print("  not a claim that either plan is better, safer or recommended")
    print("=" * 78)
    print(f"  held-out window {result['held_out_window'][0]} .. "
          f"{result['held_out_window'][1]}   rule: {result['candidate_rule']}")
    print(f"  attempted {agg['cases_attempted']}, included "
          f"{agg['cases_included']}, skipped {agg['cases_skipped']}")
    for o, why in agg["skip_reasons"].items():
        print(f"    SKIP {o}: {why[:88]}")
    print()
    head = (f"{'origin':<12}{'+24 MAE':>9}{'+48 MAE':>9}{'chg':>5}{'cost Δ':>14}"
            f"{'fuel Δ':>13}{'realized env Δ':>16}  lower")
    print(head)
    print("-" * len(head))
    for r in result["cases"]:
        if not r["ran"] or not r["decision"].get("both_planned"):
            continue
        a1 = r["forecast_accuracy"]
        d, z = r["decision"], r["realized"]
        realized = (f"{z['realized_environmental_cost_delta']:>16,.1f}"
                    if z.get("evaluated") else f"{'not evaluated':>16}")
        verdict = (z["which_plan_met_lower_realized_environmental_cost"]
                   if z.get("evaluated") else "no observation on the path")
        print(f"{r['case']['origin']:<12}"
              f"{a1['+24h']['model']['mae']:>9.3f}"
              f"{a1['+48h']['model']['mae']:>9.3f}"
              f"{('yes' if d['route_changed'] else 'no'):>5}"
              f"{d['configured_cost_delta']:>14,.1f}"
              f"{d['estimated_fuel_delta']:>13,.0f}"
              f"{realized}  {verdict}")
    print()
    print(f"  route changed {agg['route_changed']}, unchanged "
          f"{agg['route_unchanged']}")
    print(f"  forecast plan lower realized env cost: "
          f"{agg['forecast_plan_lower_realized_environmental_cost']}")
    print(f"  persistence plan lower:                "
          f"{agg['persistence_plan_lower_realized_environmental_cost']}")
    print(f"  tied / identical route:                "
          f"{agg['tied_or_identical_route']}")
    print(f"  realized {agg['cases_realized']}, not realized "
          f"{agg['cases_not_realized']}")
    for o, why in agg["not_realized_reasons"].items():
        print(f"    {o}: {why[:88]}")
    print(f"\n  written: {path.relative_to(ROOT)}")
    print(f"  sha256 : {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

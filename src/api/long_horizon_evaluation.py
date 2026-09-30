"""
An EVALUATION harness: does swapping persistence for the SIC forecasts change
a long-horizon routing decision?

    python -m src.api.long_horizon_evaluation

WHAT THIS IS, AND WHAT IT IS NOT
  It is an offline comparison of two runs of the SAME mission through the SAME
  unmodified time_astar, differing in one variable: which environmental field
  each arrival-time bucket is priced from. It is NOT the production demo, it
  does not change the production demo, and nothing in src/routing or
  src/api/world.py imports it.

THE ICEBERG TERM IS OMITTED, DELIBERATELY AND LOUDLY
  The routing configuration loads iceberg exposure for buckets 0 and 1 only --
  a 12 h horizon. Any route past 12 h therefore cannot be priced with that
  layer, and the layer's own policy for a missing bucket is to refuse.

  This harness resolves that by running WITHOUT the iceberg layer at all,
  rather than by loading the 24 h and 48 h exposure rasters (which would mean
  deciding which six-hour buckets a "max over 0-24 h" raster represents -- an
  interpretation of the iceberg product that belongs to the iceberg layer, not
  to an evaluation script) and rather than by letting a missing bucket quietly
  contribute zero.

  Every result records `iceberg_exposure: "omitted"` with the reason. A route
  produced here is therefore NOT comparable with a production route, and is
  not a navigational assessment of anything.

WHAT IS PRICED
  environmental cost (sea ice, per bucket) + POLARIS penalty + coverage
  uncertainty, composed by navigation_cost.compose_grid exactly as the shipped
  world composes them. Hard constraints are the domain's, unchanged. No weight,
  penalty or formula is touched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from src.api.config import COMP_CONFIG, DEMO, USNIC
from src.api.environment_forecast import (OUT_DIR as FORECAST_DIR,
                                          SOURCE_FORECAST, resolver_for_policy)
from src.api.environment_provider import forecast_aware_fields
from src.api.environment_timeline import (ON_MISSING_PERSIST,
                                          POLICY_LONG_HORIZON_EVALUATION,
                                          POLICY_PERSISTENCE,
                                          plan_environment_timeline,
                                          timeline_summary)
from src.api.world import environment_dates, get_world
from src.routing.end_to_end_route import plan_route
from src.routing.navigation_cost import load_composition_config
from src.routing.time_navigation_cost import TimeIndexedNavigationCost

HOUR = 3600.0

ICEBERG_OMITTED = "omitted"
ICEBERG_OMISSION_REASON = (
    "the routing configuration loads iceberg exposure for buckets 0-1 only "
    "(a 12 h horizon) and refuses a missing bucket; this evaluation runs past "
    "12 h, so the iceberg term is omitted entirely rather than reinterpreting "
    "the 24 h / 48 h exposure rasters or reading a missing bucket as zero")

#  A DEMONSTRATION SCENARIO, not an operational mission. See `provenance()`.
SCENARIO = {
    "label": "SYNTHETIC LONG-HORIZON DEMONSTRATION SCENARIO",
    "is_an_operational_mission": False,
    "start": tuple(DEMO["start"]),
    #  chosen deterministically: the navigable cell whose shortest-distance
    #  transit from the documented demo start is closest to 60 h, first in
    #  row-major order among ties
    "goal": (934, 336),
    "goal_selection_rule":
        "the navigable cell whose 8-connected shortest-distance transit from "
        "the documented demo start is closest to 60 h at the demonstration's "
        "own vessel speed; ties broken row-major. Not a port, station or "
        "documented waypoint.",
    "why_synthetic":
        "the repository documents exactly one endpoint pair (src/api/config.py "
        "DEMO, about 9.9 h apart) and one other real coordinate (Bharati, in "
        "another sector with no documented partner). Neither gives a leg long "
        "enough to cross 24 h, and inventing a coordinate and calling it real "
        "would be worse than labelling this one synthetic.",
    "departure_time": DEMO["departure_time"],
    "vessel_speed_mps": DEMO["vessel_speed_mps"],
    "polaris_ice_class": DEMO["polaris_ice_class"],
    "polaris_riv_table": DEMO["polaris_riv_table"],
    "bucket_hours": DEMO["bucket_hours"],
    "buckets": 16,                      # 0 .. 96 h
}


#  Where a forecast raster's training lineage is recorded. The raster names
#  the model artifact and its SHA; each training metrics file records the SHA
#  of the artifact it produced and the dataset it was fitted on. The link is
#  made by matching those SHAs -- never by filename, lead number or order.
METRICS_DIR = Path("data") / "processed"


def _artifact_lineage(raster: Path | None) -> dict:
    """The model artifact and training dataset behind one forecast raster.

    Read from the raster's own tags and from the training metrics file whose
    recorded artifact SHA equals the raster's. If nothing matches, the dataset
    is reported as unresolved with the reason -- it is never guessed.
    """
    out = {
        "model_artifact": None, "model_artifact_sha256": None,
        "sklearn_version_at_prediction": None,
        "dataset": None, "dataset_sha256": None,
        "dataset_link": None,
    }
    if raster is None or not Path(raster).exists():
        out["dataset_link"] = "no raster to read"
        return out
    import rasterio
    with rasterio.open(raster) as ds:
        tags = ds.tags()
    out["model_artifact"] = tags.get("model_artifact")
    out["model_artifact_sha256"] = tags.get("model_artifact_sha256")
    out["sklearn_version_at_prediction"] = tags.get("sklearn_version_at_prediction")
    sha = out["model_artifact_sha256"]
    if not sha:
        out["dataset_link"] = "the raster records no model artifact SHA"
        return out
    matches = []
    for m in sorted(METRICS_DIR.glob("forecast_metrics*.json")):
        try:
            rec = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        if rec.get("artifact_sha256") == sha:
            matches.append((m, rec))
    if len(matches) != 1:
        out["dataset_link"] = (
            f"{len(matches)} training metrics files record artifact SHA "
            f"{sha[:16]}...; the dataset is left unresolved rather than guessed")
        return out
    m, rec = matches[0]
    out["dataset"] = rec.get("dataset")
    out["dataset_sha256"] = rec.get("dataset_sha256")
    out["dataset_link"] = (
        f"{m.name} records artifact_sha256 == the raster's model_artifact_sha256")
    return out


@dataclass
class EvaluationRun:
    """One routed mission under one environmental policy."""

    policy: str
    route: object
    fields: list
    timeline: list
    forecast_dir: Path = FORECAST_DIR
    iceberg_exposure: str = ICEBERG_OMITTED
    iceberg_omission_reason: str = ICEBERG_OMISSION_REASON

    @property
    def success(self) -> bool:
        return bool(getattr(self.route, "success", False))

    def bucket_provenance(self) -> list:
        """Per bucket: interval, source and, for a forecast, its full lineage."""
        out = []
        for f in sorted(self.fields, key=lambda x: x.bucket):
            e = dict(f.environment or {})
            resolution = e.get("resolution") or {}
            out.append({
                "bucket": int(f.bucket),
                "arrival_interval": [e.get("valid_from"), e.get("valid_to")],
                "source_type": e.get("source_type"),
                "model": e.get("model"),
                "lead_hours": e.get("lead_hours"),
                "forecast_origin": e.get("forecast_origin"),
                "valid_time": e.get("valid_time"),
                "validity_rule": resolution.get("validity_rule"),
                "validity_from": resolution.get("validity_from"),
                "validity_to": resolution.get("validity_to"),
                "artifact": resolution.get("path"),
                "fallback": e.get("fallback"),
                "iceberg_exposure": self.iceberg_exposure,
            })
            if e.get("source_type") == SOURCE_FORECAST:
                name = resolution.get("path")
                out[-1].update(_artifact_lineage(
                    None if not name else Path(self.forecast_dir) / name))
        return out


def _timeline(policy: str, departure: datetime, analysis_day: date,
              buckets: list, forecast_dir: Path | None):
    common = dict(buckets=buckets, bucket_seconds=SCENARIO["bucket_hours"] * HOUR,
                  departure_time=departure, analysis_date=analysis_day,
                  available_dates=environment_dates())
    if policy == POLICY_PERSISTENCE:
        return plan_environment_timeline(policy=POLICY_PERSISTENCE, **common)
    kw = {} if forecast_dir is None else {"forecast_dir": Path(forecast_dir)}
    return plan_environment_timeline(
        policy=policy, on_missing=ON_MISSING_PERSIST,
        resolver=resolver_for_policy(policy, origin_time=departure, **kw),
        **common)


def run(policy: str, *, scenario: dict = SCENARIO,
        forecast_dir: Path | None = None,
        max_expansions: int = 5_000_000) -> EvaluationRun:
    """Route the scenario once, under one environmental policy."""
    world = get_world()
    departure = datetime.fromisoformat(scenario["departure_time"])
    analysis_day = world["day"]
    buckets = list(range(int(scenario["buckets"])))

    timeline = _timeline(policy, departure, analysis_day, buckets, forecast_dir)
    fields = forecast_aware_fields(
        timeline=timeline,
        resolver=(None if policy == POLICY_PERSISTENCE
                  else resolver_for_policy(
                      policy, origin_time=departure,
                      **({} if forecast_dir is None
                         else {"forecast_dir": Path(forecast_dir)}))),
        grid=world["grid"], constraints=world["constraints"],
        polaris_penalties=world["polaris"].penalties,
        comp_config=load_composition_config(COMP_CONFIG), usnic_dir=USNIC,
        chart_date=world["chart_date"], analysis_date=analysis_day,
        bucket_hours=scenario["bucket_hours"],
        sic_for_date=lambda day: world["grid"])

    #  NO iceberg layer: see ICEBERG_OMISSION_REASON
    provider = TimeIndexedNavigationCost.from_fields(
        fields, bucket_seconds=scenario["bucket_hours"] * HOUR,
        allow_historical_demo_mismatch=True)

    route = plan_route(
        world["grid"], scenario["start"], scenario["goal"], provider=provider,
        departure_time=departure, vessel_speed_mps=scenario["vessel_speed_mps"],
        constraints=world["constraints"],
        polaris_ice_class=scenario["polaris_ice_class"],
        polaris_riv_table=scenario["polaris_riv_table"],
        polaris_penalties=world["polaris"],
        allow_historical_demo_override=True, max_expansions=max_expansions)
    return EvaluationRun(
        policy=policy, route=route, fields=fields, timeline=timeline,
        forecast_dir=(FORECAST_DIR if forecast_dir is None
                      else Path(forecast_dir)))


def compare(a: EvaluationRun, b: EvaluationRun) -> dict:
    """Measured differences between two runs. No verdict, no score."""
    ra, rb = a.route, b.route
    if not (a.success and b.success):
        return {"both_routed": False,
                "outcomes": {a.policy: ra.outcome, b.policy: rb.outcome},
                "reasons": {a.policy: ra.reason, b.policy: rb.reason}}
    pa, pb = [tuple(c) for c in ra.path], [tuple(c) for c in rb.path]
    ma, mb = ra.metrics, rb.metrics
    changed = pa != pb
    shared = set(pa) & set(pb)
    out = {
        "both_routed": True,
        "path_changed": changed,
        "cells": {a.policy: len(pa), b.policy: len(pb)},
        "cells_only_in_first": len(set(pa) - shared),
        "cells_only_in_second": len(set(pb) - shared),
        "shared_cells": len(shared),
        "first_divergence_index": next(
            (i for i, (x, y) in enumerate(zip(pa, pb)) if x != y),
            None if pa == pb else min(len(pa), len(pb))),
    }
    for name, attr in (("distance_km", "distance_km"),
                       ("travel_time_h", "travel_time_h"),
                       ("configured_cost", "total_cost"),
                       ("environmental_contribution", "environmental_contribution"),
                       ("polaris_contribution", "polaris_contribution"),
                       ("coverage_uncertainty_contribution",
                        "coverage_uncertainty_contribution")):
        x, y = getattr(ma, attr, None), getattr(mb, attr, None)
        out[name] = {a.policy: x, b.policy: y,
                     "delta": (y - x) if (x is not None and y is not None) else None}
    out["buckets_used"] = {a.policy: sorted({c.bucket for c in ra.cells}),
                           b.policy: sorted({c.bucket for c in rb.cells})}
    out["forecast_priced_buckets"] = {
        r.policy: [e["bucket"] for e in r.bucket_provenance()
                   if e["source_type"] == SOURCE_FORECAST] for r in (a, b)}
    return out


def provenance(runs) -> dict:
    return {
        "kind": "OFFLINE FORECAST-vs-PERSISTENCE ROUTING EVALUATION",
        "is_an_operational_assessment": False,
        "scenario": dict(SCENARIO),
        "iceberg_exposure": ICEBERG_OMITTED,
        "iceberg_omission_reason": ICEBERG_OMISSION_REASON,
        "priced_terms": ["environmental cost (sea ice)", "POLARIS penalty",
                         "coverage uncertainty"],
        "hard_constraints": "NavigationDomain / BedMachine, unchanged",
        "runs": {r.policy: {"outcome": r.route.outcome,
                            "buckets": r.bucket_provenance(),
                            "timeline": timeline_summary(r.timeline)}
                 for r in runs},
    }


def main() -> int:
    from src.api.environment_timeline import POLICY_PERSISTENCE as PP
    print("=" * 78)
    print(f"{SCENARIO['label']}")
    print("=" * 78)
    print(f"  start {SCENARIO['start']} -> goal {SCENARIO['goal']}   "
          f"departure {SCENARIO['departure_time']}   "
          f"{SCENARIO['vessel_speed_mps']} m/s")
    print(f"  iceberg exposure: {ICEBERG_OMITTED.upper()} -- {ICEBERG_OMISSION_REASON}")
    runs = []
    for policy in (PP, POLICY_LONG_HORIZON_EVALUATION):
        print(f"\n---- {policy} ----")
        r = run(policy)
        runs.append(r)
        if r.success:
            m = r.route.metrics
            print(f"  {r.route.outcome}: {len(r.route.path)} cells, "
                  f"{m.distance_km:,.2f} km, {m.travel_time_h:,.2f} h, "
                  f"cost {m.total_cost:,.1f}")
        else:
            print(f"  {r.route.outcome}: {r.route.reason}")
        for e in r.bucket_provenance():
            if e["source_type"] == SOURCE_FORECAST:
                print(f"    bucket {e['bucket']:>2}  {e['arrival_interval'][0][5:16]}"
                      f" .. {e['arrival_interval'][1][5:16]}  {e['source_type']:<20}"
                      f" +{int(e['lead_hours'])}h  {e['artifact']}")
    print("\n---- comparison ----")
    print(json.dumps(compare(*runs), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

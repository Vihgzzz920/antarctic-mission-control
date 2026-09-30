"""
End-to-end route orchestration around the UNMODIFIED time_astar.

One entry point, plan_route(), which wires together layers that already exist
and adds no model of its own:

    hard constraints   <- NavigationDomain / NavigationConstraints   (masks only)
    edge cost          <- TimeIndexedNavigationCost or
                          IcebergTimeNavigationCost                  (cost_fn)
    search             <- time_astar                                 (unchanged)
    audit              <- this module                                (bookkeeping)

    from src.routing.end_to_end_route import plan_route
    res = plan_route(grid, (r0, c0), (r1, c1), provider=provider,
                     departure_time=datetime(2025, 1, 8), vessel_speed_mps=5.0,
                     constraints=domain)
    print(res.explain())

WHAT THIS MODULE IS NOT
  It is not an optimiser, a cost model, a forecast, or a ranking of routes by
  risk. It returns the route that MINIMISES THE CONFIGURED COST and stores the
  metrics that produced that number, so a later layer can compare route
  profiles against each other. Calling the result "the safe route" would be a
  claim this module cannot support: the cost is a preference ordering built
  from project parameters, and the iceberg term in particular is weighted by a
  project routing parameter with no physical or POLARIS meaning.

HARD CONSTRAINTS COME FROM EXACTLY ONE PLACE
  The only cells a route may not enter are the ones the constraint layer
  already excludes -- land / grounded ice, insufficient draught, and any hard
  mask the caller supplies there. This module builds no mask, and asserts at
  the end of every search that the mask time_astar actually used is bit for bit
  the mask the constraint layer produced.

  POLARIS special consideration, iceberg exposure and forecast uncertainty are
  NEVER converted into blocks. They are priced, and a priced cell stays
  reachable however expensive it becomes. A route through a costly cell is a
  route the planner chose; a route through a blocked cell is a bug.

FIVE OUTCOMES, KEPT APART ON PURPOSE
  route_found                  a path exists and was priced
  no_feasible_path             the search exhausted every option
  hard_navigation_constraint   an endpoint sits inside an excluded cell
  forecast_unavailable         a bucket the route needed has no field
  historical_date_mismatch     e.g. a 2020 USNIC chart under 2025 environment
  search_limit_reached         the expansion cap stopped the search early

  An ORDINARY HIGH COST is none of these. It is a success with a large number,
  and this module never downgrades it to a failure, nor substitutes a route
  when one of the failures above occurs.

TEMPORAL MISMATCH IS REFUSED BY DEFAULT
  The composed-cost and iceberg layers already refuse to mix a chart date with
  a different environment date unless the caller opts in. This module refuses a
  second time, at route level: a provider that was built with the override is
  rejected unless plan_route was ALSO asked for it, so an override cannot leak
  in from however the provider happened to be constructed. When it is requested
  it is stamped as historical_demo_override=true in the route provenance, and
  it is never the default.

POLARIS IS SELECTED, NEVER DEFAULTED
  polaris_ice_class and polaris_riv_table are the operator's, and must be given
  together or not at all. Given, the class is validated against the RIV config
  itself, the table is resolved through configs/sigrid3_to_polaris_policy.json
  (by its policy key, or by the IMO table number the config's own `source`
  declares), and the RIO the SIGRID/RIO pipeline already computed for that
  (table, class) pair is read out of the chart sidecar and handed to the
  existing polaris_risk.assess() and navigation_cost.penalties_from_risk().

  Not one RIV value, RIO sum, Table 1.1 band, SIGRID mapping or penalty number
  is computed or transcribed here. Table 1.4 is never inferred from date,
  season or latitude -- the policy file forbids exactly that, and this module
  only ever passes on what the caller asked for.

  Every route cell's charged POLARIS penalty is then checked against what the
  declared class and table produce for that cell's polygon, so the selection
  named in the result is the one the cost actually used rather than a label.
  Charted water takes no RIO and no penalty; an unresolvable regime keeps the
  existing `unknown` behaviour. None of special consideration, indeterminate or
  unknown ever becomes a block: they are penalties, and the config refuses a
  non-finite one.

NOT IN THIS MODULE
  Pareto route sets, route selection, fuel or bunker estimation, departure-time
  optimisation, multi-site selection, frontend, iceberg injection, route
  ranking. Distance and travel time are reported because the search produces
  them; neither is a fuel figure and neither is labelled as one.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.routing.constraints import ConstraintError, NavigationConstraints
from src.routing.iceberg_navigation_cost import (IcebergCostError,
                                                 IcebergExposureUnavailable,
                                                 IcebergTimeNavigationCost)
from src.routing.navigation_cost import penalties_from_risk
from src.routing.navigation_domain import NavigationDomain
from src.routing.polaris import (OperationalCategory, PolarisError,
                                 load_riv_table)
from src.routing.polaris_rio import (CLASSIFICATION_INDETERMINATE, Branch,
                                     Endpoint, RIOCell)
from src.routing.polaris_risk import RiskConfigError, assess, load_risk_config
from src.routing.time_astar import TimeAStarError, time_astar
from src.routing.time_navigation_cost import (ForecastUnavailable,
                                              TimeIndexedNavigationCost)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIDECAR = ROOT / "data" / "processed" / "usnic" / \
    "sigrid3_cell_metadata.json"
DEFAULT_POLARIS_POLICY = ROOT / "configs" / "sigrid3_to_polaris_policy.json"
DEFAULT_POLARIS_PENALTIES = ROOT / "configs" / "polaris_risk_penalties.json"

#  the classification vocabulary is owned by polaris.py and polaris_rio.py and
#  is re-exported here, never re-spelled
SPECIAL_CONSIDERATION = OperationalCategory.SPECIAL_CONSIDERATION.value
INDETERMINATE = CLASSIFICATION_INDETERMINATE

#  The sidecar records each endpoint's RIO and its classification, which is
#  everything polaris_risk.assess() reads. It does NOT record the endpoint's
#  tenths breakdown or its residual ice-free share, and those are not
#  reconstructed here -- inventing them would be fabricating chart data. The
#  fields are stamped with this sentinel so an audit can tell "absent from the
#  sidecar" from a real zero.
NOT_CARRIED_BY_SIDECAR = -1

OUTCOME_FOUND = "route_found"
OUTCOME_NO_PATH = "no_feasible_path"
OUTCOME_HARD_CONSTRAINT = "hard_navigation_constraint"
OUTCOME_FORECAST_UNAVAILABLE = "forecast_unavailable"
OUTCOME_DATE_MISMATCH = "historical_date_mismatch"
OUTCOME_SEARCH_LIMIT = "search_limit_reached"

OUTCOMES = (OUTCOME_FOUND, OUTCOME_NO_PATH, OUTCOME_HARD_CONSTRAINT,
            OUTCOME_FORECAST_UNAVAILABLE, OUTCOME_DATE_MISMATCH,
            OUTCOME_SEARCH_LIMIT)

OBJECTIVE = "minimum configured cost"

#  the per-term sums are rebuilt from the same trapezoid rule time_astar uses,
#  so they must reconcile with its total to floating-point noise, not to a
#  tolerance chosen to hide a discrepancy
RECONCILE_REL_TOL = 1e-9


class RouteError(ValueError):
    """The route request itself is malformed; not a routing failure."""


# --------------------------------------------------------------- POLARIS
#  The vessel ice class and the RIV table are the OPERATOR's choice. Nothing
#  below supplies a default for either, infers one from date, season or
#  latitude, or falls back when one is missing. What this section does is
#  plumbing only: it reads the RIO the SIGRID/RIO pipeline already computed for
#  the chosen (table, class) pair out of the chart sidecar, hands it to the
#  existing polaris_risk.assess(), and hands that to the existing
#  navigation_cost.penalties_from_risk(). No RIV value, no RIO sum, no Table
#  1.1 band and no penalty number is computed, transcribed or defaulted here.


@dataclass(frozen=True)
class PolarisSelection:
    """Exactly which class, table, chart and penalty config were used."""

    ice_class: str
    riv_table_requested: str
    riv_table_key: str
    riv_table_path: str
    riv_table_source: str
    sidecar_path: str
    sidecar_chart: str
    sidecar_rio_key: str
    penalties_config: str
    penalty_values_are_project_parameters: bool = True
    penalty_values_are_polaris_quantities: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PolarisPenalties:
    """The penalty table for compose_grid, plus the assessment behind each one."""

    selection: PolarisSelection
    penalties: dict                 # polygon_id -> the compose_grid record
    records: dict                   # polygon_id -> RiskAssessment
    polygons: int
    with_rio: int
    without_rio: int
    water: int
    mapping_sensitive: int
    by_classification: dict

    def to_dict(self) -> dict:
        return {"selection": self.selection.to_dict(), "polygons": self.polygons,
                "with_rio": self.with_rio, "without_rio": self.without_rio,
                "water": self.water, "mapping_sensitive": self.mapping_sensitive,
                "by_classification": dict(self.by_classification)}


def resolve_riv_table(requested: str,
                      policy_path=DEFAULT_POLARIS_POLICY) -> tuple[str, Path, str]:
    """Which RIV config the operator meant, decided by the policy file.

    Accepts the policy's own key ("default", "decayed_ice") or the IMO table
    number ("1.3", "1.4"). The number is matched against the table file's OWN
    declared source, so no table number is written into this module.
    """
    policy_path = Path(policy_path)
    try:
        tables = json.loads(policy_path.read_text())["riv_tables"]
    except (OSError, KeyError, ValueError) as exc:
        raise RouteError(f"cannot read riv_tables from {policy_path}: {exc}") from exc
    want = str(requested).strip()
    #  riv_tables also carries prose and the selection policy, so a candidate
    #  is only an entry that actually names a config file on disk
    options = {k: v for k, v in tables.items()
               if isinstance(v, str) and v.endswith(".json") and (ROOT / v).is_file()}
    if not options:
        raise RouteError(f"{policy_path.name} names no readable RIV table config")
    hits = []
    for key, rel in options.items():
        path = ROOT / rel
        try:
            source = json.loads(path.read_text())["source"]
        except (OSError, KeyError, ValueError) as exc:
            raise RouteError(f"cannot read {path}: {exc}") from exc
        #  every RIV config's source names its OWN table first and the Table 1.1
        #  band criteria (and sometimes the other RIV table) after it, so the
        #  first "Table N.N" in the source is the table that file carries
        named = re.search(r"Table (\d+\.\d+)", source)
        if want == key or (named is not None and named.group(1) == want):
            hits.append((key, path, source))
    if len(hits) != 1:
        raise RouteError(
            f"polaris_riv_table={requested!r} matched {len(hits)} of the tables "
            f"in {policy_path.name} ({sorted(options)}). Name the policy key or "
            f"the IMO table number exactly; nothing is guessed.")
    return hits[0]


def _sidecar_branch(label: str, entry: dict, prefix: str) -> Branch:
    """A Branch rebuilt from the RIO the pipeline already stored. Read, not run."""
    rio, cls = entry[f"{prefix}_rio"], entry[f"{prefix}_classification"]
    ends = tuple(Endpoint(rio=float(r), classification=str(c),
                          total_tenths=NOT_CARRIED_BY_SIDECAR,
                          residual_ice_free=NOT_CARRIED_BY_SIDECAR,
                          iceberg_assigned_type=None, terms=())
                 for r, c in zip(rio, cls))
    return Branch(label=label, low=ends[0], high=ends[1],
                  classification=str(entry[f"{prefix}_branch_classification"]),
                  infeasible=())


def build_polaris_penalties(*, ice_class: str, riv_table: str,
                            sidecar_path=DEFAULT_SIDECAR,
                            policy_path=DEFAULT_POLARIS_POLICY,
                            risk_config_path=DEFAULT_POLARIS_PENALTIES
                            ) -> PolarisPenalties:
    """The per-polygon POLARIS penalty table for one explicit class and table."""
    if not ice_class or not str(ice_class).strip():
        raise RouteError("polaris_ice_class must be a vessel ice class named in "
                         "the RIV table; there is no default.")
    key, table_path, source = resolve_riv_table(riv_table, policy_path)

    #  the authoritative table decides whether this class exists at all
    table = load_riv_table(table_path)
    try:
        table.ice_class(str(ice_class))
    except PolarisError as exc:
        raise RouteError(
            f"polaris_ice_class={ice_class!r} is not in {table_path.name}: {exc}"
        ) from exc
    try:
        risk_config = load_risk_config(risk_config_path)
    except (RiskConfigError, OSError) as exc:
        raise RouteError(f"cannot load {risk_config_path}: {exc}") from exc

    sidecar_path = Path(sidecar_path)
    try:
        side = json.loads(sidecar_path.read_text())
    except (OSError, ValueError) as exc:
        raise RouteError(f"cannot read the chart sidecar {sidecar_path}: "
                         f"{exc}") from exc
    available = set(side.get("ice_classes", ()))
    if available and str(ice_class) not in available:
        raise RouteError(
            f"the sidecar {sidecar_path.name} carries no RIO for ice class "
            f"{ice_class!r} (it has {sorted(available)}). Re-run the SIGRID "
            f"rasteriser for that class; a RIO is never computed here.")
    rio_key = f"{key}|{ice_class}"
    label = f"{key} ({table_path.name})"

    records, counts = {}, {}
    water = mapping_sensitive = with_rio = 0
    for pid, poly in side["polygons"].items():
        entry = (poly.get("rio") or {}).get(rio_key)
        cons = opt = None
        if entry is not None:
            cons = _sidecar_branch("conservative", entry, "conservative")
            opt = _sidecar_branch("optimistic", entry, "optimistic")
        cell = RIOCell(
            index=int(pid), cell_state=str(poly.get("interpretation_state", "")),
            table_used=label, ice_class=str(ice_class), conservative=cons,
            optimistic=opt, mapping_sensitive=bool(poly.get("mapping_sensitive")),
            mapping_classification_difference=bool(
                (entry or {}).get("mapping_classification_difference", False)),
            iceberg_fraction_tenths=poly.get("iceberg_tenths"),
            iceberg_trace_present=bool(poly.get("icebergs_trace_present")),
            warnings=tuple(poly.get("warnings", ())), interpretation=None)
        record = assess(cell, risk_config)
        records[int(pid)] = record
        water += bool(poly.get("is_water"))
        mapping_sensitive += bool(record.mapping_sensitive)
        with_rio += bool(record.has_rio)
        if not poly.get("is_water"):
            k = record.classification_used_conservative
            counts[k] = counts.get(k, 0) + 1

    selection = PolarisSelection(
        ice_class=str(ice_class), riv_table_requested=str(riv_table),
        riv_table_key=key, riv_table_path=str(table_path),
        riv_table_source=source, sidecar_path=str(sidecar_path),
        sidecar_chart=str(side.get("chart", "")), sidecar_rio_key=rio_key,
        penalties_config=str(risk_config_path))
    return PolarisPenalties(
        selection=selection,
        penalties=penalties_from_risk(records.values(), sidecar_path),
        records=records, polygons=len(records), with_rio=with_rio,
        without_rio=len(records) - with_rio, water=water,
        mapping_sensitive=mapping_sensitive,
        by_classification=dict(sorted(counts.items())))


# ------------------------------------------------------------- audit rows
@dataclass(frozen=True)
class RouteCell:
    """One cell of the route, priced at the time the vessel reaches it."""

    row: int
    col: int
    arrival_time: float                 # seconds on the provider's clock
    arrival_datetime: str
    bucket: int
    state: str
    environmental_cost: float
    polaris_penalty: float
    coverage_uncertainty_cost: float
    composed_cost: float
    iceberg_exposure: float | None
    iceberg_cost: float
    final_cost: float
    iceberg_id: str | None = None
    radius_km: float | None = None
    forecast_horizon_hours: float | None = None
    extrapolated_uncertainty: bool | None = None
    chart_state: str | None = None
    polygon_id: int | None = None
    polaris_classification: str | None = None
    polaris_mapping_sensitive: bool | None = None
    polaris_indeterminate: bool | None = None
    polaris_special_consideration: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RouteMetrics:
    """Everything a later route-profile comparison needs, and nothing derived."""

    cells: int
    total_cost: float
    distance_m: float
    travel_time_s: float
    environmental_contribution: float
    polaris_contribution: float
    coverage_uncertainty_contribution: float
    iceberg_exposure_contribution: float
    max_iceberg_exposure: float | None
    contributing_iceberg_ids: tuple[str, ...]
    any_extrapolated_uncertainty: bool
    polaris_special_consideration_cells: int
    polaris_indeterminate_cells: int
    polaris_indeterminate_used_branch_cells: int
    polaris_mapping_sensitive_cells: int
    polaris_unknown_regime_cells: int
    polaris_charted_water_cells: int
    polaris_charted_ice_cells: int
    polaris_uncharted_cells: int
    buckets_used: tuple[int, ...]
    expanded_nodes: int
    labels_created: int
    labels_retired: int
    reconciliation_residual: float

    @property
    def distance_km(self) -> float:
        return self.distance_m / 1000.0

    @property
    def travel_time_h(self) -> float:
        return self.travel_time_s / 3600.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contributing_iceberg_ids"] = list(self.contributing_iceberg_ids)
        d["buckets_used"] = list(self.buckets_used)
        d["distance_km"] = self.distance_km
        d["travel_time_h"] = self.travel_time_h
        return d


@dataclass(frozen=True)
class RouteResult:
    """The auditable answer: what was found, what it cost, and on what data."""

    success: bool
    outcome: str
    reason: str
    objective: str
    start: tuple[int, int] | None
    goal: tuple[int, int] | None
    departure_time: str
    arrival_time: str | None
    path: tuple[tuple[int, int], ...] = ()
    arrival_times: tuple[float, ...] = ()
    cells: tuple[RouteCell, ...] = ()
    metrics: RouteMetrics | None = None
    constraint_summary: dict = field(default_factory=dict)
    temporal_provenance: dict = field(default_factory=dict)
    polaris: dict = field(default_factory=dict)

    @property
    def polaris_selected(self) -> bool:
        return bool(self.polaris.get("selected", False))

    @property
    def historical_demo_override(self) -> bool:
        return bool(self.temporal_provenance.get("historical_demo_override", False))

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "outcome": self.outcome,
            "reason": self.reason,
            "objective": self.objective,
            "start": list(self.start) if self.start else None,
            "goal": list(self.goal) if self.goal else None,
            "departure_time": self.departure_time,
            "arrival_time": self.arrival_time,
            "path": [list(p) for p in self.path],
            "arrival_times": list(self.arrival_times),
            "cells": [c.to_dict() for c in self.cells],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "constraint_summary": self.constraint_summary,
            "temporal_provenance": self.temporal_provenance,
            "polaris": self.polaris,
        }

    def explain(self) -> str:
        head = (f"route {self.outcome}: {self.reason}" if not self.success
                else f"route {self.outcome} ({self.objective})")
        lines = [head,
                 f"  start {self.start} -> goal {self.goal}",
                 f"  departure {self.departure_time}"
                 + (f"   arrival {self.arrival_time}" if self.arrival_time else "")]
        if self.metrics:
            m = self.metrics
            lines += [
                f"  {m.cells} cells   {m.distance_km:,.2f} km   "
                f"{m.travel_time_h:,.2f} h   cost {m.total_cost:,.3f}",
                f"  environment {m.environmental_contribution:,.3f}"
                f"   POLARIS {m.polaris_contribution:,.3f}"
                f"   coverage-uncertainty {m.coverage_uncertainty_contribution:,.3f}"
                f"   iceberg {m.iceberg_exposure_contribution:,.3f}",
                f"  max iceberg exposure "
                + ("n/a" if m.max_iceberg_exposure is None
                   else f"{m.max_iceberg_exposure:.6g}")
                + f"   icebergs {list(m.contributing_iceberg_ids) or 'none'}"
                + ("   [EXTRAPOLATED uncertainty on the route]"
                   if m.any_extrapolated_uncertainty else ""),
                f"  buckets used {list(m.buckets_used)}   "
                f"expanded {m.expanded_nodes:,}   labels {m.labels_created:,}",
            ]
        tp = self.temporal_provenance
        if tp:
            lines.append(
                f"  dates: environment {tp.get('environment_dates')}   "
                f"USNIC chart {tp.get('chart_dates')}   "
                + ("[HISTORICAL DEMO OVERRIDE]" if self.historical_demo_override
                   else "[aligned]" if tp.get("all_dates_aligned") else "[MISMATCH]"))
        pol = self.polaris
        if pol.get("selected"):
            sel = pol["selection"]
            lines.append(
                f"  POLARIS: class {sel['ice_class']}   RIV table "
                f"{sel['riv_table_requested']} -> {sel['riv_table_key']} "
                f"({Path(sel['riv_table_path']).name})   chart "
                f"{sel['sidecar_chart']}")
            lines.append(
                f"    special consideration {pol['special_consideration_cells']}"
                f"   indeterminate {pol['indeterminate_cells']} "
                f"(overlapping; {pol['indeterminate_used_branch_cells']} on the "
                f"priced branch)"
                f"   mapping-sensitive {pol['mapping_sensitive_cells']}"
                f"   unknown regime {pol['unknown_regime_cells']}"
                f"   charted water {pol['charted_water_cells']}"
                f"   [penalties are project parameters, not POLARIS quantities]")
        elif pol:
            lines.append(f"  POLARIS: {pol.get('reason', 'not selected')}")
        cs = self.constraint_summary
        if cs:
            lines.append(
                f"  constraints: supplied {cs.get('supplied')}   "
                f"missing {cs.get('missing')}   "
                f"excluded cells {cs.get('excluded_cells')}")
        return "\n".join(lines)


# --------------------------------------------------------------- helpers
def _base_of(provider) -> TimeIndexedNavigationCost:
    """The time-indexed composed-cost provider underneath, whatever was passed."""
    if isinstance(provider, IcebergTimeNavigationCost):
        return provider.base
    if isinstance(provider, TimeIndexedNavigationCost):
        return provider
    raise RouteError(
        "provider must be a TimeIndexedNavigationCost or an "
        f"IcebergTimeNavigationCost, got {type(provider).__name__}. The route "
        "layer prices cells through an existing provider; it has no cost model.")


def _as_constraints(constraints, grid) -> NavigationConstraints:
    if isinstance(constraints, NavigationDomain):
        return constraints.to_constraints()
    if isinstance(constraints, NavigationConstraints):
        return constraints
    raise RouteError(
        "constraints must be a NavigationDomain or NavigationConstraints, got "
        f"{type(constraints).__name__}. Hard constraints come only from that "
        "layer; this module does not accept a raw mask and does not build one.")


def _decompose(provider, row: int, col: int, t: float, branch: str) -> dict:
    """One cell's cost split into the terms that made it. Lookup only.

    The composed layer is where the POLARIS penalty and the polygon behind it
    live, so both providers are read through it -- the iceberg provider's base
    is consulted for the polygon id and classification it does not carry, and
    the two POLARIS penalties are cross-checked against each other.
    """
    base = _base_of(provider)
    cc = base.field_for(t).composition.cell(row, col, branch)
    #  cc.state is the USNIC chart state (uncharted / water / unknown regime /
    #  ice). The iceberg layer overwrites `state` with its OWN pricing state, so
    #  the chart state is carried separately instead of being lost behind it.
    chart = dict(chart_state=cc.state, polygon_id=cc.polygon_id,
                 polaris_classification=cc.polaris_classification)
    if isinstance(provider, IcebergTimeNavigationCost):
        c = provider.cell(row, col, t, branch)
        if c.polaris_penalty != cc.polaris_penalty:
            raise RouteError(
                f"cell ({row}, {col}) at t={t:g}: the iceberg layer reports a "
                f"POLARIS penalty of {c.polaris_penalty!r} while the composed "
                f"layer underneath reports {cc.polaris_penalty!r}")
        return dict(state=c.state, environmental_cost=c.environmental_cost,
                    polaris_penalty=c.polaris_penalty,
                    coverage_uncertainty_cost=c.coverage_uncertainty_cost,
                    composed_cost=c.composed_cost,
                    iceberg_exposure=c.iceberg_exposure,
                    iceberg_cost=c.iceberg_cost, final_cost=c.final_cost,
                    iceberg_id=c.iceberg_id, radius_km=c.radius_km,
                    forecast_horizon_hours=c.forecast_horizon_hours,
                    extrapolated_uncertainty=c.extrapolated_uncertainty, **chart)
    return dict(state=cc.state, environmental_cost=cc.environmental_cost,
                polaris_penalty=cc.polaris_penalty,
                coverage_uncertainty_cost=cc.coverage_uncertainty_cost,
                composed_cost=cc.composed_cost, iceberg_exposure=None,
                iceberg_cost=0.0, final_cost=cc.composed_cost, iceberg_id=None,
                radius_km=None, forecast_horizon_hours=None,
                extrapolated_uncertainty=None, **chart)


def _temporal_provenance(provider, *, departure_time: datetime, t0: float,
                         max_horizon_s: float, requested_override: bool) -> dict:
    """Every date the route depends on, and whether they agree."""
    base = _base_of(provider)
    prov = provider.provenance()
    base_prov = prov.get("base", prov)
    fields = base_prov.get("fields", {})
    aligned = {int(b): bool(v["dates_aligned"]) for b, v in fields.items()}
    env = {int(b): v["environment_date"] for b, v in fields.items()}
    chart = {int(b): v["chart_date"] for b, v in fields.items()}
    dep_date = departure_time.date().isoformat()
    out = {
        "departure_time": departure_time.isoformat(),
        "departure_date": dep_date,
        "route_start_time_s": t0,
        "provider_start_time_s": base.start_time,
        "bucket_seconds": base.bucket_seconds,
        "buckets": base.buckets,
        "provider_horizon_s": base.horizon_s,
        "max_horizon_s_passed_to_time_astar": max_horizon_s,
        "environment_dates": sorted(set(env.values())),
        "chart_dates": sorted(set(chart.values())),
        "dates_aligned_by_bucket": aligned,
        "all_dates_aligned": all(aligned.values()) if aligned else True,
        "departure_date_matches_environment_dates":
            set(env.values()) == {dep_date} if env else True,
        "historical_demo_override": bool(requested_override),
        "provider_historical_demo_override":
            bool(base_prov.get("historical_demo_override", False)),
        "override_is_never_the_default": True,
        "provider_provenance": prov,
    }

    #  Which environmental field each bucket was actually built from, when the
    #  caller recorded one (src.api.environment_timeline). Read-only: nothing
    #  here validates, selects or substitutes a field, and a provider that
    #  recorded nothing simply reports that it did not.
    by_bucket = {int(b): dict(v.get("environment") or {})
                 for b, v in fields.items()}
    recorded = {b: e for b, e in by_bucket.items() if e}
    out["environment_provenance_recorded"] = bool(recorded)
    if recorded:
        field_dates = sorted({e["environment_date"] for e in recorded.values()})
        out["environment_by_bucket"] = by_bucket
        out["environment_policies"] = sorted(
            {e["policy"] for e in recorded.values()})
        #  the honest headline: did the priced environment change with the
        #  arrival time, or was one field served to every bucket?
        out["environment_field_dates"] = field_dates
        out["environment_is_time_varying"] = len(field_dates) > 1
        out["environment_persisted_buckets"] = sorted(
            b for b, e in recorded.items() if e.get("persisted"))
        out["environment_uses_future_observations"] = any(
            e.get("uses_future_observation") for e in recorded.values())
    if isinstance(provider, IcebergTimeNavigationCost):
        out["iceberg_exposure_weight"] = prov["iceberg_exposure_weight"]
        out["iceberg_weight_is_a_project_routing_parameter"] = True
        out["iceberg_weight_is_a_polaris_or_physical_quantity"] = False
        out["iceberg_exposure_horizons_hours"] = prov["exposure_horizons_hours"]
        out["iceberg_exposure_extrapolated_by_bucket"] = \
            prov["exposure_extrapolated_by_bucket"]
        out["iceberg_provider_historical_demo_override"] = \
            bool(prov.get("historical_demo_override", False))
    return out


def _fail(outcome: str, reason: str, *, start, goal, departure_time: datetime,
          constraint_summary: dict, temporal: dict,
          polaris: dict | None = None) -> RouteResult:
    return RouteResult(success=False, outcome=outcome, reason=reason,
                       objective=OBJECTIVE, start=start, goal=goal,
                       departure_time=departure_time.isoformat(),
                       arrival_time=None, constraint_summary=constraint_summary,
                       temporal_provenance=temporal, polaris=polaris or {})


# ------------------------------------------------------------ the entry point
def plan_route(grid,
               start: tuple[int, int],
               goal: tuple[int, int],
               *,
               provider,
               departure_time: datetime,
               vessel_speed_mps: float,
               constraints=None,
               branch: str = "conservative",
               departure_offset_s: float = 0.0,
               polaris_ice_class: str | None = None,
               polaris_riv_table: str | None = None,
               polaris_penalties: PolarisPenalties | None = None,
               polaris_sidecar=DEFAULT_SIDECAR,
               polaris_policy=DEFAULT_POLARIS_POLICY,
               polaris_risk_config=DEFAULT_POLARIS_PENALTIES,
               allow_historical_demo_override: bool = False,
               allow_unconstrained: bool = False,
               allow_corner_cutting: bool = False,
               max_expansions: int = 5_000_000,
               time_step_s: float | None = None) -> RouteResult:
    """Plan one route and return an auditable result. Never raises on a
    routing failure -- only on a malformed request.

    constraints is a NavigationDomain or NavigationConstraints and is the ONLY
    source of hard exclusions. Supply it; allow_unconstrained=True is the
    explicit way to say that routing over an unmasked grid is intended.

    polaris_ice_class and polaris_riv_table are the operator's choice and must
    be given TOGETHER or not at all. Given, they are resolved against the
    chart sidecar and the RIV configs, and every route cell's POLARIS penalty
    is checked against the penalty that selection produces for its polygon --
    so the class and table named in the result are the ones the cost actually
    used, not a label. Omitted, POLARIS is simply not selected and the result
    says so; nothing is defaulted, inferred or silently applied.
    """
    if not isinstance(departure_time, datetime):
        raise RouteError(
            f"departure_time must be a datetime, got {type(departure_time).__name__}")
    if not np.isfinite(vessel_speed_mps) or vessel_speed_mps <= 0:
        raise RouteError(
            f"vessel_speed_mps must be finite and > 0, got {vessel_speed_mps!r}")
    if not np.isfinite(departure_offset_s) or departure_offset_s < 0:
        raise RouteError(
            f"departure_offset_s must be finite and >= 0, got {departure_offset_s!r}")

    given = [polaris_ice_class is not None, polaris_riv_table is not None]
    if any(given) and not all(given):
        raise RouteError(
            "polaris_ice_class and polaris_riv_table must be given together. "
            "A POLARIS penalty is meaningless without both, and neither has a "
            "default: MSC.1/Circ.1519 leaves the table choice to qualified "
            "personnel and the class to the vessel.")
    if polaris_penalties is not None and not all(given):
        raise RouteError(
            "polaris_penalties was supplied without polaris_ice_class and "
            "polaris_riv_table; name the selection the penalties belong to.")
    if all(given):
        if polaris_penalties is None:
            polaris_penalties = build_polaris_penalties(
                ice_class=polaris_ice_class, riv_table=polaris_riv_table,
                sidecar_path=polaris_sidecar, policy_path=polaris_policy,
                risk_config_path=polaris_risk_config)
        sel = polaris_penalties.selection
        if (sel.ice_class != str(polaris_ice_class)
                or sel.riv_table_requested != str(polaris_riv_table)):
            raise RouteError(
                f"polaris_penalties were built for {sel.ice_class!r} / "
                f"{sel.riv_table_requested!r}, but this route asked for "
                f"{polaris_ice_class!r} / {polaris_riv_table!r}")
        polaris_audit = dict(selected=True, selection=sel.to_dict(),
                             chart=polaris_penalties.to_dict())
    else:
        polaris_penalties = None
        polaris_audit = dict(
            selected=False,
            reason="no polaris_ice_class / polaris_riv_table was supplied, so no "
                   "POLARIS penalty was selected by this route. Any POLARIS term "
                   "in the cost came from however the provider was composed.")

    base = _base_of(provider)
    t0 = float(base.start_time) + float(departure_offset_s)
    step = float(base.bucket_seconds if time_step_s is None else time_step_s)
    max_horizon_s = float(base.horizon_s) - t0

    temporal = _temporal_provenance(
        provider, departure_time=departure_time, t0=t0,
        max_horizon_s=max_horizon_s,
        requested_override=allow_historical_demo_override)

    # ---- hard constraints, from the constraint layer and nowhere else -------
    if constraints is None:
        if not allow_unconstrained:
            raise RouteError(
                "no constraints supplied, so land and shallow water would both "
                "be navigable. Pass a NavigationDomain / NavigationConstraints, "
                "or allow_unconstrained=True if that is genuinely intended.")
        mask_used = np.asarray(grid.blocked_mask, dtype=bool)
        constraint_summary = {"supplied": [], "missing": list(("land_mask",
                              "shallow_mask", "polaris_no_go_mask")),
                              "excluded_cells": int(mask_used.sum()),
                              "source": "grid.blocked_mask as supplied by the caller",
                              "applied_by_route_layer": False}
    else:
        nc = _as_constraints(constraints, grid)
        try:
            applied = nc.apply_to(grid, allow_unconstrained=allow_unconstrained)
        except ConstraintError as exc:
            raise RouteError(str(exc)) from exc
        mask_used = np.asarray(applied, dtype=bool)
        constraint_summary = dict(nc.summary(tuple(grid.shape)))
        #  NavigationConstraints calls it combined_blocked, NavigationDomain
        #  calls it combined_excluded. Normalise so the route audit has one
        #  name, without renaming anything in either of those layers.
        constraint_summary["excluded_cells"] = int(
            constraint_summary.get("combined_blocked",
                                   constraint_summary.get("combined_excluded",
                                                          int(mask_used.sum()))))
        constraint_summary["applied_by_route_layer"] = True
        constraint_summary["masks_added_by_route_layer"] = 0

    # ---- temporal mismatch: refuse by default, at route level too ----------
    if not allow_historical_demo_override:
        if temporal["provider_historical_demo_override"] or \
                temporal.get("iceberg_provider_historical_demo_override"):
            return _fail(
                OUTCOME_DATE_MISMATCH,
                "the cost provider was built with the historical-demo override "
                "but plan_route was not asked for it. An override is never "
                "inherited silently: pass allow_historical_demo_override=True "
                "to route on mismatched dates deliberately.",
                start=start, goal=goal, departure_time=departure_time,
                constraint_summary=constraint_summary, temporal=temporal,
                polaris=polaris_audit)
        if not temporal["all_dates_aligned"]:
            return _fail(
                OUTCOME_DATE_MISMATCH,
                f"USNIC chart dates {temporal['chart_dates']} do not match the "
                f"environment dates {temporal['environment_dates']}. They are "
                f"not combined by default.",
                start=start, goal=goal, departure_time=departure_time,
                constraint_summary=constraint_summary, temporal=temporal,
                polaris=polaris_audit)
        if not temporal["departure_date_matches_environment_dates"]:
            return _fail(
                OUTCOME_DATE_MISMATCH,
                f"departure date {temporal['departure_date']} is not one of the "
                f"environment dates {temporal['environment_dates']}.",
                start=start, goal=goal, departure_time=departure_time,
                constraint_summary=constraint_summary, temporal=temporal,
                polaris=polaris_audit)

    if not np.isfinite(max_horizon_s) or max_horizon_s <= 0:
        return _fail(
            OUTCOME_FORECAST_UNAVAILABLE,
            f"departure_offset_s={departure_offset_s:g} starts the route at or "
            f"after the provider horizon {base.horizon_s:g} s; there is no "
            f"bucket left to price.",
            start=start, goal=goal, departure_time=departure_time,
            constraint_summary=constraint_summary, temporal=temporal,
            polaris=polaris_audit)

    # ---- the search: time_astar, unchanged ---------------------------------
    try:
        cost_fn = provider.cost_fn(branch)
        min_cost = provider.min_cost(branch)
        res = time_astar(grid, start, goal, cost_fn=cost_fn, min_cost=min_cost,
                         vessel_speed_mps=float(vessel_speed_mps),
                         time_step_s=step, max_horizon_s=max_horizon_s,
                         start_time=t0, allow_corner_cutting=allow_corner_cutting,
                         max_expansions=max_expansions)
    except IcebergExposureUnavailable as exc:
        return _fail(OUTCOME_FORECAST_UNAVAILABLE,
                     f"iceberg exposure unavailable: {exc}", start=start, goal=goal,
                     departure_time=departure_time,
                     constraint_summary=constraint_summary, temporal=temporal,
                polaris=polaris_audit)
    except ForecastUnavailable as exc:
        return _fail(OUTCOME_FORECAST_UNAVAILABLE,
                     f"composed-cost forecast unavailable: {exc}", start=start,
                     goal=goal, departure_time=departure_time,
                     constraint_summary=constraint_summary, temporal=temporal,
                polaris=polaris_audit)
    except (IcebergCostError,) as exc:
        return _fail(OUTCOME_DATE_MISMATCH, str(exc), start=start, goal=goal,
                     departure_time=departure_time,
                     constraint_summary=constraint_summary, temporal=temporal,
                polaris=polaris_audit)
    except TimeAStarError as exc:
        text = str(exc)
        if "blocked cell" in text or "outside the grid" in text:
            return _fail(OUTCOME_HARD_CONSTRAINT, text, start=start, goal=goal,
                         departure_time=departure_time,
                         constraint_summary=constraint_summary, temporal=temporal,
                polaris=polaris_audit)
        raise RouteError(text) from exc

    if not res.success:
        outcome = (OUTCOME_SEARCH_LIMIT if res.expanded_nodes >= max_expansions
                   else OUTCOME_NO_PATH)
        return _fail(outcome, res.reason, start=res.start, goal=res.goal,
                     departure_time=departure_time,
                     constraint_summary=constraint_summary, temporal=temporal,
                polaris=polaris_audit)

    # ---- price the route we were given, term by term -----------------------
    dx, dy = grid.pixel_size
    recs = polaris_penalties.records if polaris_penalties else {}
    rows = []
    for (r, c), t in zip(res.path, res.arrival_times):
        d = _decompose(provider, r, c, t, branch)
        rec = recs.get(d["polygon_id"]) if d["polygon_id"] is not None else None
        if rec is not None:
            #  the penalty the composed cost actually charged must be the one
            #  THIS class and table produce for THIS polygon, or the selection
            #  named in the result would be a label over someone else's number
            want = (rec.optimistic_penalty if branch == "optimistic"
                    else rec.conservative_penalty)
            is_water = bool(polaris_penalties.penalties.get(
                d["polygon_id"], {}).get("is_water"))
            charged = 0.0 if is_water else float(want)
            if abs(d["polaris_penalty"] - charged) > 1e-9:
                raise RouteError(
                    f"cell ({r}, {c}) sits in polygon {d['polygon_id']} and was "
                    f"charged a POLARIS penalty of {d['polaris_penalty']!r}, but "
                    f"{polaris_ice_class} on table {polaris_riv_table} gives "
                    f"{charged!r} for that polygon. The cost was composed with a "
                    f"different selection than the one this route declares.")
            d.update(
                polaris_mapping_sensitive=bool(rec.mapping_sensitive),
                polaris_indeterminate=bool(rec.indeterminate),
                polaris_special_consideration=(
                    d["polaris_classification"] == SPECIAL_CONSIDERATION))
        rows.append(RouteCell(
            row=int(r), col=int(c), arrival_time=float(t),
            arrival_datetime=(departure_time
                              + timedelta(seconds=float(t) - t0)).isoformat(),
            bucket=int(base.bucket_for(t)), **d))

    def trapezoid(values) -> float:
        total = 0.0
        for a, b, va, vb in zip(res.path, res.path[1:], values, values[1:]):
            dist = math.hypot((b[1] - a[1]) * dx, (b[0] - a[0]) * dy)
            total += dist * 0.5 * (va + vb)
        return total

    distance_m = sum(math.hypot((b[1] - a[1]) * dx, (b[0] - a[0]) * dy)
                     for a, b in zip(res.path, res.path[1:]))
    env_c = trapezoid([x.environmental_cost for x in rows])
    pol_c = trapezoid([x.polaris_penalty for x in rows])
    cov_c = trapezoid([x.coverage_uncertainty_cost for x in rows])
    ice_c = trapezoid([x.iceberg_cost for x in rows])
    residual = (env_c + pol_c + cov_c + ice_c) - res.total_cost
    scale = max(abs(res.total_cost), 1.0)
    if abs(residual) > RECONCILE_REL_TOL * scale:
        raise RouteError(
            f"the per-term contributions {env_c + pol_c + cov_c + ice_c!r} do "
            f"not reconcile with the search total {res.total_cost!r} (residual "
            f"{residual!r}). The composed cost is documented as additive; this "
            f"module will not report a breakdown that does not add up.")

    exposures = [x.iceberg_exposure for x in rows if x.iceberg_exposure is not None]
    metrics = RouteMetrics(
        cells=len(rows), total_cost=float(res.total_cost),
        distance_m=float(distance_m),
        travel_time_s=float(res.arrival_times[-1] - res.arrival_times[0]),
        environmental_contribution=float(env_c),
        polaris_contribution=float(pol_c),
        coverage_uncertainty_contribution=float(cov_c),
        iceberg_exposure_contribution=float(ice_c),
        max_iceberg_exposure=(max(exposures) if exposures else None),
        contributing_iceberg_ids=tuple(sorted(
            {x.iceberg_id for x in rows if x.iceberg_id})),
        any_extrapolated_uncertainty=any(bool(x.extrapolated_uncertainty)
                                         for x in rows),
        polaris_special_consideration_cells=sum(
            1 for x in rows if x.polaris_special_consideration),
        polaris_indeterminate_cells=sum(1 for x in rows if x.polaris_indeterminate),
        polaris_indeterminate_used_branch_cells=sum(
            1 for x in rows if x.polaris_classification == INDETERMINATE),
        polaris_mapping_sensitive_cells=sum(
            1 for x in rows if x.polaris_mapping_sensitive),
        polaris_unknown_regime_cells=sum(
            1 for x in rows if x.chart_state == "charted_unknown_regime"),
        polaris_charted_water_cells=sum(1 for x in rows if x.chart_state == "charted_water"),
        polaris_charted_ice_cells=sum(1 for x in rows if x.chart_state == "charted_ice"),
        polaris_uncharted_cells=sum(1 for x in rows if x.chart_state == "usnic_uncharted"),
        buckets_used=tuple(sorted({x.bucket for x in rows})),
        expanded_nodes=int(res.expanded_nodes),
        labels_created=int(res.labels_created),
        labels_retired=int(res.labels_retired),
        reconciliation_residual=float(residual))

    # ---- the route stayed inside the domain the constraint layer defined ---
    now = np.asarray(grid.blocked_mask, dtype=bool)
    if not np.array_equal(now, mask_used):
        raise RouteError(
            "grid.blocked_mask changed during the search; the route layer adds "
            "no exclusions and cannot vouch for a mask it did not see.")
    inside = [(r, c) for r, c in res.path if bool(mask_used[r, c])]
    if inside:
        raise RouteError(
            f"route entered excluded cells {inside[:5]}; a priced cell may be "
            f"expensive but an excluded cell is never traversable.")
    constraint_summary = dict(constraint_summary)
    constraint_summary["route_cells_inside_excluded"] = 0
    constraint_summary["mask_unchanged_during_search"] = True

    polaris_audit = dict(polaris_audit)
    polaris_audit.update(
        total_contribution=float(pol_c),
        special_consideration_cells=metrics.polaris_special_consideration_cells,
        indeterminate_cells=metrics.polaris_indeterminate_cells,
        indeterminate_used_branch_cells=(
            metrics.polaris_indeterminate_used_branch_cells),
        definitions={
            "special_consideration_cells":
                "cells whose classification on the branch this route was priced "
                "with is special consideration",
            "indeterminate_cells":
                "cells whose polygon has an indeterminate RIO interval on EITHER "
                "mapping branch. This OVERLAPS special_consideration_cells: one "
                "branch can straddle a Table 1.1 boundary while the other does "
                "not, so the two counts must not be added together",
            "indeterminate_used_branch_cells":
                "the subset whose priced branch is itself indeterminate",
            "mapping_sensitive_cells":
                "cells whose polygon has at least one aggregate SIGRID stage, so "
                "the conservative and optimistic POLARIS types differ"},
        mapping_sensitive_cells=metrics.polaris_mapping_sensitive_cells,
        unknown_regime_cells=metrics.polaris_unknown_regime_cells,
        charted_water_cells=metrics.polaris_charted_water_cells,
        charted_ice_cells=metrics.polaris_charted_ice_cells,
        uncharted_cells=metrics.polaris_uncharted_cells,
        per_cell=[{"row": x.row, "col": x.col, "polygon_id": x.polygon_id,
                   "state": x.state, "chart_state": x.chart_state,
                   "classification": x.polaris_classification,
                   "penalty": x.polaris_penalty,
                   "mapping_sensitive": x.polaris_mapping_sensitive,
                   "indeterminate": x.polaris_indeterminate,
                   "special_consideration": x.polaris_special_consideration}
                  for x in rows],
        penalties_are_a_preference_never_an_exclusion=True,
        no_hard_block_from_special_consideration_indeterminate_or_unknown=True)

    return RouteResult(
        success=True, outcome=OUTCOME_FOUND, reason=res.reason,
        objective=OBJECTIVE, start=res.start, goal=res.goal,
        departure_time=departure_time.isoformat(),
        arrival_time=(departure_time + timedelta(
            seconds=float(res.arrival_times[-1]) - t0)).isoformat(),
        path=tuple(tuple(p) for p in res.path),
        arrival_times=tuple(float(t) for t in res.arrival_times),
        cells=tuple(rows), metrics=metrics,
        constraint_summary=constraint_summary, temporal_provenance=temporal,
        polaris=polaris_audit)

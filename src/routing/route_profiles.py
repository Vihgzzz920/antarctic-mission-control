"""
Three route profiles over the UNMODIFIED end-to-end orchestrator.

    from src.routing.route_profiles import compare_profiles
    cmp = compare_profiles(grid_factory, risk_provider=provider, usnic_dir=USNIC,
                           start=(r0, c0), goal=(r1, c1),
                           departure_time=datetime(2025, 1, 8),
                           vessel_speed_mps=5.0, constraints=domain,
                           polaris_ice_class="PC6", polaris_riv_table="1.3",
                           polaris_penalties=pp)
    print(cmp.table())

WHAT A PROFILE IS
  One objective, expressed as a cost composition, handed to the SAME
  plan_route() and therefore the same time_astar. Nothing here searches,
  prices a cell, or decides anything about navigation:

    fastest            environmental cost = 1 / vessel_speed_mps SECONDS per
                       metre, uniform. The edge rule then makes the search
                       total literally the travel time in seconds, so "minimise
                       cost" and "minimise travel time" are the same statement
                       rather than a proxy for one another. No POLARIS penalty
                       table, iceberg exposure weight 0.

    risk_oriented      the caller's own complete configured cost -- environment
                       + POLARIS + coverage uncertainty + iceberg exposure at
                       the shipped weight -- priced at the vessel's ARRIVAL
                       time, exactly as end_to_end_route already does it.

    shortest_distance  environmental cost = 1.0 per metre, uniform, so the
                       search total is the path length in metres. No POLARIS
                       penalty table, iceberg exposure weight 0.

  FASTEST AND SHORTEST_DISTANCE HAVE THE SAME OPTIMUM HERE, and the comparison
  says so rather than hiding it. time_astar advances the clock by
  `dist / vessel_speed_mps`, so with one constant speed the travel time of a
  path is a strictly increasing function of its length and the two objectives
  order every path identically. They are still two different objectives, in
  different units, and they separate the moment the speed stops being a single
  constant -- a vessel model that slows in ice, a current-assisted speed, or a
  per-cell speed limit. Until such a model exists, the comparison reports the
  measured fact that the two routes coincided.

WHAT shortest_distance IS NOT
  It is NOT a fuel, bunker or emissions figure, and must never be presented as
  one. This project has no validated vessel consumption model, and distance
  alone does not become one: consumption depends on speed, ice resistance,
  hull and machinery, none of which is modelled anywhere in this codebase. The
  objective is stored under the name `shortest_distance` for that reason.

THE THREE PROFILES SHARE EVERYTHING EXCEPT THE OBJECTIVE
  Same start, goal, departure time, vessel speed, branch, hard constraints,
  forecast buckets and dates, and the same date-mismatch rules. The objective
  compositions are built from the risk provider's OWN per-bucket cost, so a
  cell the shared forecast cannot price stays unpriceable for all three and the
  feasible set is identical. compare_profiles() records the shared inputs and
  refuses to return a comparison whose profiles disagree about any of them.

REPORTING IS NOT OPTIMISING
  fastest and shortest_distance never see a POLARIS penalty or a weighted
  exposure while choosing their path -- their compositions carry neither. Their
  environmental, POLARIS, coverage-uncertainty and iceberg figures are measured
  AFTERWARDS, by pricing the route they chose through the risk provider. That
  re-pricing uses the edge rule time_astar documents, and every comparison
  checks it against plan_route's own numbers on the risk profile, so it cannot
  drift away from the real one unnoticed.

NOT IN THIS MODULE
  No fuel model, no Pareto dominance, no ranking, no recommendation, no
  multi-site optimisation, no frontend, no replanning. Three objectives, three
  answers, side by side, in three different units. Which one matters is not a
  question this layer answers.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.routing.end_to_end_route import (SPECIAL_CONSIDERATION, RouteResult,
                                          plan_route)
from src.routing.iceberg_navigation_cost import (IcebergTimeNavigationCost,
                                                 load_config)
from src.routing.navigation_cost import compose_grid, load_composition_config
from src.routing.time_navigation_cost import (ComposedField,
                                              TimeIndexedNavigationCost)

PROFILE_FASTEST = "fastest"
PROFILE_RISK = "risk_oriented"
PROFILE_SHORTEST = "shortest_distance"
PROFILES = (PROFILE_FASTEST, PROFILE_RISK, PROFILE_SHORTEST)

OBJECTIVES = {
    PROFILE_FASTEST: {
        "label": "travel time",
        "minimises": "travel time",
        "units": "seconds",
        "cost_composition": "uniform 1 / vessel_speed_mps seconds per metre over "
                            "the cells the shared forecast can price",
        "polaris_penalties_applied": False,
        "iceberg_exposure_weight_applied": 0.0,
    },
    PROFILE_RISK: {
        "label": "configured cost",
        "minimises": "the configured navigation cost: environment + POLARIS + "
                     "coverage uncertainty + iceberg exposure",
        "units": "configured cost units",
        "cost_composition": "the caller's own provider, priced at the arrival time",
        "polaris_penalties_applied": True,
        "iceberg_exposure_weight_applied": "the shipped weight",
    },
    PROFILE_SHORTEST: {
        "label": "distance",
        "minimises": "travelled distance",
        "units": "metres",
        "cost_composition": "uniform 1.0 per metre over the cells the shared "
                            "forecast can price",
        "polaris_penalties_applied": False,
        "iceberg_exposure_weight_applied": 0.0,
        "is_a_fuel_or_consumption_figure": False,
        "why_not": "no validated vessel consumption model exists in this project; "
                   "distance is distance",
    },
}

#  the re-priced terms must reproduce plan_route's own numbers on the risk
#  profile to floating-point noise, not to a tolerance picked to hide a gap
REPRICE_REL_TOL = 1e-9


class ProfileError(ValueError):
    """The profile request is malformed, or the profiles disagree on an input."""


# --------------------------------------------------------------- results
@dataclass(frozen=True)
class RouteProfileResult:
    """One objective's answer, with the metrics of the route it chose."""

    profile: str
    objective: str
    objective_units: str
    objective_value: float | None
    objective_is_the_configured_cost: bool
    success: bool
    outcome: str
    reason: str
    start: tuple[int, int] | None
    goal: tuple[int, int] | None
    departure_time: str
    arrival_time: str | None
    path: tuple[tuple[int, int], ...] = ()
    arrival_times: tuple[float, ...] = ()
    distance_m: float | None = None
    travel_time_s: float | None = None
    configured_cost: float | None = None
    environmental_contribution: float | None = None
    polaris_contribution: float | None = None
    coverage_uncertainty_contribution: float | None = None
    iceberg_exposure_contribution: float | None = None
    max_iceberg_exposure: float | None = None
    contributing_iceberg_ids: tuple[str, ...] = ()
    iceberg_provenance: dict = field(default_factory=dict)
    special_consideration_cells: int | None = None
    indeterminate_cells: int | None = None
    mapping_sensitive_cells: int | None = None
    any_extrapolated_uncertainty: bool | None = None
    buckets_used: tuple[int, ...] = ()
    cells: tuple = ()
    constraint_summary: dict = field(default_factory=dict)
    temporal_provenance: dict = field(default_factory=dict)
    polaris: dict = field(default_factory=dict)

    @property
    def distance_km(self) -> float | None:
        return None if self.distance_m is None else self.distance_m / 1000.0

    @property
    def travel_time_h(self) -> float | None:
        return None if self.travel_time_s is None else self.travel_time_s / 3600.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = [list(p) for p in self.path]
        d["cells"] = [c.to_dict() for c in self.cells]
        d["contributing_iceberg_ids"] = list(self.contributing_iceberg_ids)
        d["buckets_used"] = list(self.buckets_used)
        d["distance_km"] = self.distance_km
        d["travel_time_h"] = self.travel_time_h
        return d


@dataclass(frozen=True)
class ProfileComparison:
    """Three objectives side by side. Not a ranking and not a recommendation."""

    profiles: dict
    shared_inputs: dict
    provenance: dict
    objectives: dict = field(default_factory=lambda: dict(OBJECTIVES))

    def get(self, profile: str) -> RouteProfileResult:
        try:
            return self.profiles[profile]
        except KeyError:
            raise ProfileError(f"no profile {profile!r}; have {sorted(self.profiles)}")

    @property
    def succeeded(self) -> tuple[str, ...]:
        return tuple(p for p in PROFILES if self.profiles[p].success)

    def to_dict(self) -> dict:
        return {"profiles": {k: v.to_dict() for k, v in self.profiles.items()},
                "shared_inputs": self.shared_inputs, "provenance": self.provenance,
                "objectives": self.objectives}

    def table(self) -> str:
        head = (f"{'profile':<18}{'minimises':<18}{'value':>16}{'dist km':>10}"
                f"{'time h':>9}{'cost':>16}{'iceberg':>12}{'max exp':>9}"
                f"{'spec':>6}{'indet':>7}")
        lines = [head, "-" * len(head)]
        for name in PROFILES:
            r = self.profiles[name]
            if not r.success:
                lines.append(f"{name:<18}{OBJECTIVES[name]['label']:<18}"
                             f"{'FAILED: ' + r.outcome:>16}")
                continue
            #  the POLARIS counts are None when no per-polygon assessments were
            #  supplied; that is "not measured", which is not the same as 0
            spec = ("n/a" if r.special_consideration_cells is None
                    else str(r.special_consideration_cells))
            indet = ("n/a" if r.indeterminate_cells is None
                     else str(r.indeterminate_cells))
            lines.append(
                f"{name:<18}{OBJECTIVES[name]['label']:<18}"
                f"{r.objective_value:>16,.2f}"
                f"{r.distance_km:>10,.2f}{r.travel_time_h:>9,.2f}"
                f"{r.configured_cost:>16,.0f}"
                f"{r.iceberg_exposure_contribution:>12,.0f}"
                f"{(r.max_iceberg_exposure or 0.0):>9.3f}"
                f"{spec:>6}{indet:>7}")
        p = self.provenance
        lines += ["",
                  f"objective units differ by design: "
                  + ", ".join(f"{n}={OBJECTIVES[n]['units']}" for n in PROFILES),
                  f"shortest_distance is a distance, NOT a fuel or consumption "
                  f"figure",
                  f"environment {p.get('environment_dates')}  USNIC chart "
                  f"{p.get('chart_dates')}  "
                  + ("[HISTORICAL DEMO OVERRIDE]"
                     if p.get("historical_demo_override") else "[aligned]"),
                  f"POLARIS {p.get('polaris_ice_class')} / table "
                  f"{p.get('polaris_riv_table')}   iceberg exposure weight "
                  f"{p.get('iceberg_exposure_weight')}",
                  "three objectives, three answers; this table is not a ranking"]
        return "\n".join(lines)


# --------------------------------------------------------------- internals
def _uniform_fields(risk_base: TimeIndexedNavigationCost, value: float,
                    usnic_dir, comp_config) -> list[ComposedField]:
    """One ComposedField per risk bucket, carrying a uniform cost per metre.

    The finite/non-finite pattern is copied from the risk provider's OWN cost
    for that bucket, so a cell the shared forecast cannot price stays
    unpriceable here too and all three profiles search the same feasible set.
    No POLARIS penalty table is supplied: these objectives do not price POLARIS.
    """
    cfg = load_composition_config(comp_config)
    out = []
    for bucket in risk_base.buckets:
        f = risk_base.fields[bucket]
        real = np.asarray(f.cost("conservative"), dtype="float64")
        env = np.where(np.isfinite(real), float(value), np.inf)
        out.append(ComposedField(
            bucket=bucket, composition=compose_grid(env, {}, cfg, usnic_dir),
            environment_date=f.environment_date, chart_date=f.chart_date,
            epsg=f.epsg, shape=f.shape, transform=f.transform,
            label=f"{f.label or bucket} [objective]"))
    return out


def _objective_provider(risk_provider, value: float, usnic_dir, comp_config,
                        iceberg_config_path, *, allow_mismatch: bool):
    """A provider whose cost IS the objective, sharing the risk provider's
    buckets, dates and exposure fields. Exposure is carried at weight 0, so it
    is still REPORTED for every cell and charged on none of them."""
    risk_base = (risk_provider.base if isinstance(risk_provider,
                                                  IcebergTimeNavigationCost)
                 else risk_provider)
    base = TimeIndexedNavigationCost.from_fields(
        _uniform_fields(risk_base, value, usnic_dir, comp_config),
        bucket_seconds=risk_base.bucket_seconds,
        start_time=risk_base.start_time,
        on_unavailable=risk_base.on_unavailable,
        allow_historical_demo_mismatch=allow_mismatch)
    if not isinstance(risk_provider, IcebergTimeNavigationCost):
        return base
    return IcebergTimeNavigationCost.from_providers(
        base, risk_provider.exposure,
        load_config(iceberg_config_path).with_weight(0.0),
        allow_historical_demo_mismatch=allow_mismatch)


def _reprice(risk_provider, route: RouteResult, branch: str,
             pixel_size) -> dict:
    """Price a route the risk objective did not choose, through the risk cost.

    Pure lookups plus the edge rule time_astar documents:
        edge = distance * 0.5 * (cost(A, t_depart) + cost(B, t_arrive))
    Every comparison checks this against plan_route's own metrics on the risk
    profile, where the two must agree exactly, so it cannot drift silently.
    """
    dx, dy = pixel_size
    cells = [risk_provider.cell(r, c, t, branch)
             for (r, c), t in zip(route.path, route.arrival_times)]

    def trapezoid(values) -> float:
        total = 0.0
        for a, b, va, vb in zip(route.path, route.path[1:], values, values[1:]):
            dist = math.hypot((b[1] - a[1]) * dx, (b[0] - a[0]) * dy)
            total += dist * 0.5 * (va + vb)
        return total

    exposures = [c.iceberg_exposure for c in cells if c.iceberg_exposure is not None]
    env = trapezoid([c.environmental_cost for c in cells])
    pol = trapezoid([c.polaris_penalty for c in cells])
    cov = trapezoid([c.coverage_uncertainty_cost for c in cells])
    ice = trapezoid([c.iceberg_cost for c in cells])
    return {
        "configured_cost": env + pol + cov + ice,
        "environmental_contribution": env, "polaris_contribution": pol,
        "coverage_uncertainty_contribution": cov,
        "iceberg_exposure_contribution": ice,
        "max_iceberg_exposure": (max(exposures) if exposures else None),
        "contributing_iceberg_ids": tuple(sorted(
            {c.iceberg_id for c in cells if c.iceberg_id})),
        "any_extrapolated_uncertainty": any(bool(c.extrapolated_uncertainty)
                                            for c in cells),
    }


def _check_reprice(got: dict, route: RouteResult) -> None:
    m = route.metrics
    for key, want in (("configured_cost", m.total_cost),
                      ("environmental_contribution", m.environmental_contribution),
                      ("polaris_contribution", m.polaris_contribution),
                      ("coverage_uncertainty_contribution",
                       m.coverage_uncertainty_contribution),
                      ("iceberg_exposure_contribution",
                       m.iceberg_exposure_contribution)):
        scale = max(abs(want), 1.0)
        if abs(got[key] - want) > REPRICE_REL_TOL * scale:
            raise ProfileError(
                f"the profile layer re-priced the risk route's {key} as "
                f"{got[key]!r} while plan_route reported {want!r}. The edge rule "
                f"used for the other profiles no longer matches the real one.")


def _counts(route: RouteResult, records: dict | None) -> dict:
    m = route.metrics
    if records is None:
        return dict(special_consideration_cells=None, indeterminate_cells=None,
                    mapping_sensitive_cells=None)
    if route.polaris_selected:
        return dict(
            special_consideration_cells=m.polaris_special_consideration_cells,
            indeterminate_cells=m.polaris_indeterminate_cells,
            mapping_sensitive_cells=m.polaris_mapping_sensitive_cells)
    #  this profile did not price POLARIS, so the counts are measured after the
    #  fact from the same per-polygon assessments the risk profile used
    spec = indet = mapsens = 0
    for cell in route.cells:
        rec = records.get(cell.polygon_id)
        if rec is None:
            continue
        spec += int(rec.classification_used_conservative == SPECIAL_CONSIDERATION)
        indet += int(bool(rec.indeterminate))
        mapsens += int(bool(rec.mapping_sensitive))
    return dict(special_consideration_cells=spec, indeterminate_cells=indet,
                mapping_sensitive_cells=mapsens)


def _result(profile: str, route: RouteResult, *, priced: dict | None,
            records: dict | None) -> RouteProfileResult:
    obj = OBJECTIVES[profile]
    if not route.success:
        return RouteProfileResult(
            profile=profile, objective=obj["minimises"],
            objective_units=obj["units"], objective_value=None,
            objective_is_the_configured_cost=(profile == PROFILE_RISK),
            success=False, outcome=route.outcome, reason=route.reason,
            start=route.start, goal=route.goal,
            departure_time=route.departure_time, arrival_time=None,
            constraint_summary=route.constraint_summary,
            temporal_provenance=route.temporal_provenance, polaris=route.polaris)
    m = route.metrics
    return RouteProfileResult(
        profile=profile, objective=obj["minimises"], objective_units=obj["units"],
        objective_value=float(m.total_cost),
        objective_is_the_configured_cost=(profile == PROFILE_RISK),
        success=True, outcome=route.outcome, reason=route.reason,
        start=route.start, goal=route.goal, departure_time=route.departure_time,
        arrival_time=route.arrival_time, path=route.path,
        arrival_times=route.arrival_times, distance_m=m.distance_m,
        travel_time_s=m.travel_time_s, buckets_used=m.buckets_used,
        cells=route.cells, constraint_summary=route.constraint_summary,
        temporal_provenance=route.temporal_provenance, polaris=route.polaris,
        iceberg_provenance={
            "contributing_iceberg_ids": list(priced["contributing_iceberg_ids"]),
            "max_iceberg_exposure": priced["max_iceberg_exposure"],
            "any_extrapolated_uncertainty": priced["any_extrapolated_uncertainty"],
            "exposure_is_a_probability": False,
            "weight_used_while_choosing_this_route":
                obj["iceberg_exposure_weight_applied"]},
        **{k: priced[k] for k in ("configured_cost", "environmental_contribution",
                                  "polaris_contribution",
                                  "coverage_uncertainty_contribution",
                                  "iceberg_exposure_contribution",
                                  "max_iceberg_exposure",
                                  "contributing_iceberg_ids",
                                  "any_extrapolated_uncertainty")},
        **_counts(route, records))


# ------------------------------------------------------------ entry point
def compare_profiles(grid_factory, *, risk_provider, usnic_dir,
                     start: tuple[int, int], goal: tuple[int, int],
                     departure_time: datetime, vessel_speed_mps: float,
                     constraints_factory=None,
                     comp_config=None, iceberg_config=None,
                     branch: str = "conservative",
                     polaris_ice_class: str | None = None,
                     polaris_riv_table: str | None = None,
                     polaris_penalties=None,
                     allow_historical_demo_override: bool = False,
                     allow_unconstrained: bool = False,
                     **plan_kwargs) -> ProfileComparison:
    """Run all three profiles on identical inputs and return them side by side.

    grid_factory() must return a FRESH RoutingGrid each call, and
    constraints_factory(grid) the constraints for it, because plan_route writes
    the constraint layer's mask onto the grid it is given.

    The POLARIS selection is applied to the risk profile only: fastest and
    shortest_distance do not price POLARIS, so declaring a class and table for
    them would be a label over a penalty they never paid. Their POLARIS figures
    are measured afterwards instead, from the same per-polygon assessments.
    """
    if not isinstance(departure_time, datetime):
        raise ProfileError(
            f"departure_time must be a datetime, got {type(departure_time).__name__}")
    if not np.isfinite(vessel_speed_mps) or vessel_speed_mps <= 0:
        raise ProfileError(
            f"vessel_speed_mps must be finite and > 0, got {vessel_speed_mps!r}")
    if not callable(grid_factory):
        raise ProfileError("grid_factory must be callable and return a fresh "
                           "RoutingGrid; plan_route writes a mask onto the grid")
    comp_config = comp_config or (Path(__file__).resolve().parents[2] / "configs"
                                  / "navigation_cost_composition.json")
    iceberg_config = iceberg_config or (Path(__file__).resolve().parents[2]
                                        / "configs"
                                        / "iceberg_navigation_cost.json")
    risk_base = (risk_provider.base if isinstance(risk_provider,
                                                  IcebergTimeNavigationCost)
                 else risk_provider)
    mismatch = bool(risk_base.historical_demo_override)

    #  one objective per profile, all three built from the SAME risk provider
    providers = {
        PROFILE_FASTEST: _objective_provider(
            risk_provider, 1.0 / float(vessel_speed_mps), usnic_dir, comp_config,
            iceberg_config, allow_mismatch=mismatch),
        PROFILE_RISK: risk_provider,
        PROFILE_SHORTEST: _objective_provider(
            risk_provider, 1.0, usnic_dir, comp_config, iceberg_config,
            allow_mismatch=mismatch),
    }

    shared = dict(start=list(start), goal=list(goal),
                  departure_time=departure_time.isoformat(),
                  vessel_speed_mps=float(vessel_speed_mps), branch=branch,
                  allow_historical_demo_override=bool(
                      allow_historical_demo_override))
    results, routes = {}, {}
    for name in PROFILES:
        grid = grid_factory()
        kw = dict(plan_kwargs)
        if constraints_factory is not None:
            kw["constraints"] = constraints_factory(grid)
        if name == PROFILE_RISK:
            kw.update(polaris_ice_class=polaris_ice_class,
                      polaris_riv_table=polaris_riv_table,
                      polaris_penalties=polaris_penalties)
        route = plan_route(
            grid, start, goal, provider=providers[name],
            departure_time=departure_time, vessel_speed_mps=vessel_speed_mps,
            branch=branch, allow_unconstrained=allow_unconstrained,
            allow_historical_demo_override=allow_historical_demo_override, **kw)
        routes[name] = route
        pixel = grid.pixel_size
        priced = None
        if route.success:
            priced = _reprice(risk_provider, route, branch, pixel)
            if name == PROFILE_RISK:
                _check_reprice(priced, route)
        records = polaris_penalties.records if polaris_penalties else None
        results[name] = _result(name, route, priced=priced, records=records)

    #  every profile must have been asked the same question
    for name in PROFILES:
        r = results[name]
        for field_name, want in (("start", tuple(start)), ("goal", tuple(goal)),
                                 ("departure_time", departure_time.isoformat())):
            got = getattr(r, field_name)
            got = tuple(got) if isinstance(got, (list, tuple)) else got
            if r.start is not None and got != want:
                raise ProfileError(
                    f"profile {name} ran with {field_name}={got!r} but the "
                    f"comparison asked for {want!r}")
        tp = r.temporal_provenance
        if tp.get("buckets") != routes[PROFILE_RISK].temporal_provenance.get("buckets"):
            raise ProfileError(
                f"profile {name} used buckets {tp.get('buckets')} while the "
                f"risk profile used "
                f"{routes[PROFILE_RISK].temporal_provenance.get('buckets')}; the "
                f"profiles must share the same forecast")
        if tp.get("historical_demo_override") != bool(
                allow_historical_demo_override):
            raise ProfileError(
                f"profile {name} reports a different historical-demo override "
                f"than the comparison was asked for")

    risk_tp = routes[PROFILE_RISK].temporal_provenance
    fast, short = results[PROFILE_FASTEST], results[PROFILE_SHORTEST]
    provenance = dict(
        environment_dates=risk_tp.get("environment_dates"),
        chart_dates=risk_tp.get("chart_dates"),
        all_dates_aligned=risk_tp.get("all_dates_aligned"),
        historical_demo_override=bool(allow_historical_demo_override),
        override_is_never_the_default=True,
        provider_historical_demo_override=risk_tp.get(
            "provider_historical_demo_override"),
        polaris_ice_class=polaris_ice_class, polaris_riv_table=polaris_riv_table,
        polaris_selection=routes[PROFILE_RISK].polaris.get("selection"),
        iceberg_exposure_weight=risk_tp.get("iceberg_exposure_weight"),
        bucket_seconds=risk_tp.get("bucket_seconds"),
        buckets=risk_tp.get("buckets"),
        fastest_and_shortest_distance_coincided=(
            fast.path == short.path if fast.success and short.success else None),
        why_they_can_coincide=(
            "time_astar advances the clock by dist / vessel_speed_mps, so with a "
            "single constant vessel speed a path's travel time is a strictly "
            "increasing function of its length and the two objectives order "
            "every path identically. They separate once the speed stops being "
            "one constant."),
        shortest_distance_is_a_fuel_figure=False,
        this_is_not_a_ranking=True,
        profiles_are_different_objectives_in_different_units=True)
    return ProfileComparison(profiles=results, shared_inputs=shared,
                             provenance=provenance)

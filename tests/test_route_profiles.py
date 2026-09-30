"""
Tests for the three route profiles.

The ones that matter: all three ask the SAME question, fastest and
shortest_distance never see a POLARIS penalty or a weighted exposure while
choosing, and nothing here is a ranking or a fuel figure.

    python tests/test_route_profiles.py
"""
from __future__ import annotations

import ast
import math
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_end_to_end_route as E                                       # noqa: E402
from src.routing.constraints import NavigationConstraints               # noqa: E402
from src.routing.end_to_end_route import OUTCOME_DATE_MISMATCH          # noqa: E402
from src.routing.grid import RoutingGrid                                # noqa: E402
from src.routing.navigation_domain import NavigationDomain              # noqa: E402
from src.routing.route_profiles import (OBJECTIVES, PROFILE_FASTEST,    # noqa: E402
                                        PROFILE_RISK, PROFILE_SHORTEST,
                                        PROFILES, ProfileError,
                                        compare_profiles)

MODULE = ROOT / "src" / "routing" / "route_profiles.py"
PX = 10_000.0
DAY = E.DAY
DEPART = E.DEPART
_C: dict = {}


def corridor_comparison(weight=None, *, speed=PX / 3600.0, polaris=None,
                        chart=DAY, override=False, route_override=None,
                        start=(6, 0), goal=(6, 12)):
    """The asymmetric world: a short exposed shortcut, a longer clear detour.

    `override` is what the PROVIDERS were built with; `route_override` is what
    compare_profiles is asked for. They are separate on purpose: a provider
    carrying the override must not smuggle it into a comparison that did not
    ask for one.
    """
    route_override = override if route_override is None else route_override
    shape, blocked, exposure = E.corridor_world()
    envs = [np.ones(shape) for _ in range(30)]
    zero = np.zeros(shape, "float64")
    exps = [zero if b == 0 else exposure for b in range(30)]
    usnic = E.usnic_dir(shape, PX)
    base = E.base_provider(shape, PX, envs, chart=chart,
                           allow_historical_demo_mismatch=override)
    risk = E.iceberg_provider(base, shape, exps,
                              E.CFG.weight if weight is None else weight,
                              allow_historical_demo_mismatch=override)
    nc = NavigationConstraints(shape=shape, land_mask=blocked)
    return compare_profiles(
        lambda: E.make_grid(shape, PX), risk_provider=risk, usnic_dir=usnic,
        start=start, goal=goal, departure_time=DEPART, vessel_speed_mps=speed,
        constraints_factory=lambda g: nc, polaris_penalties=polaris,
        allow_historical_demo_override=route_override)


def cached_corridor():
    if "corridor" not in _C:
        _C["corridor"] = corridor_comparison()
    return _C["corridor"]


# ------------------------------------------------------- 1. all three run
def test_all_three_profiles_execute():
    cmp = cached_corridor()
    assert set(cmp.profiles) == set(PROFILES), sorted(cmp.profiles)
    for name in PROFILES:
        r = cmp.profiles[name]
        assert r.success, (name, r.outcome, r.reason)
        assert r.profile == name
        assert r.objective_units == OBJECTIVES[name]["units"]
        assert r.distance_m > 0 and r.travel_time_s > 0
        assert r.objective_value is not None and math.isfinite(r.objective_value)
        assert r.configured_cost is not None
    return ("; ".join(f"{n}={cmp.profiles[n].objective_value:,.0f} "
                      f"{cmp.profiles[n].objective_units}" for n in PROFILES))


# ---------------------------------------------- 2. identical inputs
def test_profiles_share_start_goal_departure_and_forecast():
    cmp = cached_corridor()
    starts = {cmp.profiles[n].start for n in PROFILES}
    goals = {cmp.profiles[n].goal for n in PROFILES}
    deps = {cmp.profiles[n].departure_time for n in PROFILES}
    buckets = {tuple(cmp.profiles[n].temporal_provenance["buckets"])
               for n in PROFILES}
    speeds = {cmp.shared_inputs["vessel_speed_mps"]}
    masks = {cmp.profiles[n].constraint_summary["excluded_cells"] for n in PROFILES}
    dates = {(tuple(cmp.profiles[n].temporal_provenance["environment_dates"]),
              tuple(cmp.profiles[n].temporal_provenance["chart_dates"]))
             for n in PROFILES}
    overrides = {cmp.profiles[n].temporal_provenance["historical_demo_override"]
                 for n in PROFILES}
    for label, s in (("start", starts), ("goal", goals), ("departure", deps),
                     ("buckets", buckets), ("speed", speeds), ("mask", masks),
                     ("dates", dates), ("override", overrides)):
        assert len(s) == 1, f"{label} differs across profiles: {s}"
    return (f"one start {starts.pop()}, one goal {goals.pop()}, one departure, "
            f"one speed, one mask of {masks.pop()} excluded cells and one set of "
            f"forecast buckets across all three")


# ------------------------------------------- 3/4. the minimisation holds
def test_fastest_has_the_least_travel_time_of_the_generated_routes():
    cmp = cached_corridor()
    times = {n: cmp.profiles[n].travel_time_s for n in PROFILES}
    assert times[PROFILE_FASTEST] == min(times.values()), times
    assert times[PROFILE_FASTEST] < times[PROFILE_RISK], times
    return (f"fastest {times[PROFILE_FASTEST] / 3600:.2f} h vs risk-oriented "
            f"{times[PROFILE_RISK] / 3600:.2f} h")


def test_shortest_distance_has_the_least_distance_of_the_generated_routes():
    cmp = cached_corridor()
    d = {n: cmp.profiles[n].distance_m for n in PROFILES}
    assert d[PROFILE_SHORTEST] == min(d.values()), d
    assert d[PROFILE_SHORTEST] < d[PROFILE_RISK], d
    return (f"shortest {d[PROFILE_SHORTEST] / 1000:.2f} km vs risk-oriented "
            f"{d[PROFILE_RISK] / 1000:.2f} km")


def test_risk_oriented_has_the_least_configured_cost_of_the_generated_routes():
    cmp = cached_corridor()
    c = {n: cmp.profiles[n].configured_cost for n in PROFILES}
    assert c[PROFILE_RISK] == min(c.values()), c
    r = cmp.profiles[PROFILE_RISK]
    assert r.objective_is_the_configured_cost is True
    assert abs(r.objective_value - r.configured_cost) < 1e-9
    for other in (PROFILE_FASTEST, PROFILE_SHORTEST):
        assert cmp.profiles[other].objective_is_the_configured_cost is False
    return (f"risk-oriented {c[PROFILE_RISK]:,.0f} vs {c[PROFILE_FASTEST]:,.0f} "
            f"for the other two; only it optimises that number")


# ------------------------------------- 5. the objectives really differ
def test_fastest_and_shortest_do_not_inherit_the_risk_cost_as_their_objective():
    cmp = cached_corridor()
    for name in (PROFILE_FASTEST, PROFILE_SHORTEST):
        r = cmp.profiles[name]
        #  the route they picked is NOT the one that minimises the configured
        #  cost, which is the observable proof they did not optimise it
        assert r.configured_cost > cmp.profiles[PROFILE_RISK].configured_cost
        assert r.objective_value != r.configured_cost
        assert OBJECTIVES[name]["polaris_penalties_applied"] is False
        assert OBJECTIVES[name]["iceberg_exposure_weight_applied"] == 0.0
        assert r.polaris.get("selected") is False
        assert r.iceberg_provenance["weight_used_while_choosing_this_route"] == 0.0
        #  ... and they still REPORT the risk metrics afterwards
        assert r.iceberg_exposure_contribution > 0
        assert r.max_iceberg_exposure > 0 and r.contributing_iceberg_ids
    assert cmp.profiles[PROFILE_RISK].iceberg_provenance[
        "weight_used_while_choosing_this_route"] == "the shipped weight"
    return ("fastest and shortest chose a route costing "
            f"{cmp.profiles[PROFILE_FASTEST].configured_cost:,.0f} against the "
            f"{cmp.profiles[PROFILE_RISK].configured_cost:,.0f} the risk "
            "objective found, while still reporting its exposure afterwards")


def test_fastest_route_differs_from_the_risk_oriented_route():
    cmp = cached_corridor()
    fast, risk = cmp.profiles[PROFILE_FASTEST], cmp.profiles[PROFILE_RISK]
    assert fast.path != risk.path
    shortcut = {(6, c) for c in (4, 5, 6, 7, 8)}
    assert set(fast.path) & shortcut, fast.path
    assert not (set(risk.path) & shortcut), risk.path
    assert fast.max_iceberg_exposure > risk.max_iceberg_exposure
    return ("the shortcut carries the cone and the fastest route takes it; the "
            f"risk objective detours, dropping max exposure "
            f"{fast.max_iceberg_exposure:.3f} -> {risk.max_iceberg_exposure:.3f}")


def test_the_shortest_route_carries_more_risk_than_the_risk_oriented_one():
    cmp = cached_corridor()
    short, risk = cmp.profiles[PROFILE_SHORTEST], cmp.profiles[PROFILE_RISK]
    assert short.distance_m < risk.distance_m
    assert short.iceberg_exposure_contribution > risk.iceberg_exposure_contribution
    assert short.configured_cost > risk.configured_cost
    return (f"{(risk.distance_m - short.distance_m) / 1000:.2f} km shorter, "
            f"{short.iceberg_exposure_contribution:,.0f} of exposure against "
            f"{risk.iceberg_exposure_contribution:,.0f}")


def test_fastest_and_shortest_distance_coincide_under_one_constant_speed():
    """Not a defect and not hidden: time_astar advances the clock by
    dist / vessel_speed_mps, so the two objectives order every path the same."""
    cmp = cached_corridor()
    fast, short = cmp.profiles[PROFILE_FASTEST], cmp.profiles[PROFILE_SHORTEST]
    assert fast.path == short.path
    assert cmp.provenance["fastest_and_shortest_distance_coincided"] is True
    assert "constant vessel speed" in cmp.provenance["why_they_can_coincide"]
    #  the objectives are still distinct: different units and different values
    assert fast.objective_units != short.objective_units
    assert fast.objective_value != short.objective_value
    speed = cmp.shared_inputs["vessel_speed_mps"]
    assert abs(short.objective_value / speed - fast.objective_value) < 1e-6
    return (f"same path, different objectives: {fast.objective_value:,.0f} s and "
            f"{short.objective_value:,.0f} m, related by the one constant speed "
            f"{speed:g} m/s. They separate as soon as the speed does not.")


# --------------------------------------- 6. arrival-time pricing is live
def test_temporal_arrival_pricing_is_still_active_in_every_profile():
    cmp = cached_corridor()
    for name in PROFILES:
        r = cmp.profiles[name]
        buckets = [c.bucket for c in r.cells]
        assert buckets == sorted(buckets) and max(buckets) > 0, (name, buckets)
        assert len(r.buckets_used) > 1, (name, r.buckets_used)
        for c, t in zip(r.cells, r.arrival_times):
            assert c.arrival_time == t
    #  the cone is absent in bucket 0 and present later, and the fastest route
    #  is charged for it only where it actually arrives inside it
    fast = cmp.profiles[PROFILE_FASTEST]
    early = [c for c in fast.cells if c.bucket == 0]
    assert early and all(c.iceberg_exposure == 0.0 for c in early)
    assert any(c.iceberg_exposure > 0 for c in fast.cells if c.bucket > 0)
    return ("every profile prices each cell at its own arrival bucket; the cone "
            "is 0 in bucket 0 and non-zero later on the same cells")


# ------------------------------------------- 7. constraints and no blocks
def test_every_profile_stays_inside_the_hard_constraints():
    shape, blocked, _ = E.corridor_world()
    cmp = cached_corridor()
    for name in PROFILES:
        r = cmp.profiles[name]
        assert not any(blocked[c.row, c.col] for c in r.cells), name
        assert r.constraint_summary["route_cells_inside_excluded"] == 0
        assert r.constraint_summary["masks_added_by_route_layer"] == 0
        assert r.constraint_summary["mask_unchanged_during_search"] is True
    return ("all three routes stay inside the same BedMachine-style exclusion; "
            "no profile added a mask")


def test_no_profile_creates_a_hard_block():
    """A cell the risk objective hates is still reachable by the others."""
    cmp = cached_corridor()
    tree = ast.parse(MODULE.read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "set_blocked_mask" not in attrs
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for banned in ("blocked_mask", "no_go", "no_go_mask", "navigable_mask"):
        assert banned not in names, f"{banned!r} exists in the module"
    fast = cmp.profiles[PROFILE_FASTEST]
    exposed = [c for c in fast.cells if (c.iceberg_exposure or 0) > 0]
    assert exposed, "the fastest route never entered the cone"
    for c in fast.cells:
        assert math.isfinite(c.final_cost) and c.final_cost > 0
    return (f"the fastest route crosses {len(exposed)} cells inside the cone; "
            f"every cost is finite and the module never writes a mask")


# --------------------------------------------- 8. historical mismatch
def test_historical_mismatch_is_still_refused_for_every_profile():
    #  providers built WITH the override, comparison asked WITHOUT it
    cmp = corridor_comparison(chart=date(2020, 12, 24), override=True,
                              route_override=False)
    for name in PROFILES:
        r = cmp.profiles[name]
        assert not r.success, (name, r.outcome)
        assert r.outcome == OUTCOME_DATE_MISMATCH, (name, r.outcome)
        assert r.path == () and r.objective_value is None
        assert r.temporal_provenance["chart_dates"] == ["2020-12-24"]
    assert cmp.provenance["historical_demo_override"] is False
    assert cmp.provenance["override_is_never_the_default"] is True
    assert cmp.succeeded == ()
    return ("a 2020 chart under 2025 fields fails all three profiles with "
            f"{OUTCOME_DATE_MISMATCH}; the override is never inherited")


def test_the_override_routes_all_three_and_is_stamped_in_the_comparison():
    shape, blocked, exposure = E.corridor_world()
    envs = [np.ones(shape) for _ in range(30)]
    zero = np.zeros(shape, "float64")
    exps = [zero if b == 0 else exposure for b in range(30)]
    usnic = E.usnic_dir(shape, PX)
    base = E.base_provider(shape, PX, envs, chart=date(2020, 12, 24),
                           allow_historical_demo_mismatch=True)
    risk = E.iceberg_provider(base, shape, exps, E.CFG.weight,
                              allow_historical_demo_mismatch=True)
    nc = NavigationConstraints(shape=shape, land_mask=blocked)
    cmp = compare_profiles(
        lambda: E.make_grid(shape, PX), risk_provider=risk, usnic_dir=usnic,
        start=(6, 0), goal=(6, 12), departure_time=DEPART,
        vessel_speed_mps=PX / 3600.0, constraints_factory=lambda g: nc,
        allow_historical_demo_override=True)
    assert cmp.succeeded == PROFILES, cmp.succeeded
    assert cmp.provenance["historical_demo_override"] is True
    assert cmp.provenance["all_dates_aligned"] is False
    assert cmp.provenance["chart_dates"] == ["2020-12-24"]
    assert "HISTORICAL DEMO OVERRIDE" in cmp.table()
    for name in PROFILES:
        assert cmp.profiles[name].temporal_provenance[
            "historical_demo_override"] is True
    return ("with the override asked for explicitly, all three profiles route "
            "and the comparison provenance carries the flag and both dates")


# ----------------------------------------------------- 9. determinism
def test_repeated_generation_is_deterministic():
    seen = []
    for _ in range(3):
        cmp = corridor_comparison()
        seen.append(tuple(
            (n, cmp.profiles[n].path, cmp.profiles[n].objective_value,
             cmp.profiles[n].configured_cost,
             cmp.profiles[n].iceberg_exposure_contribution)
            for n in PROFILES))
    assert seen[0] == seen[1] == seen[2], "identical inputs gave different profiles"
    return "3 rebuilds: identical paths, objective values and contributions"


# ------------------------------------------ 10. the comparison object
def test_the_comparison_is_not_a_ranking_and_names_no_fuel():
    cmp = cached_corridor()
    code = E.code_strings_and_names(MODULE)
    #  `fuel` is no longer forbidden here: fuel_efficient is a real objective
    #  built on src/routing/fuel_cost.py. `bunker` and `consumption` stay
    #  forbidden because this project cannot produce either quantity, and the
    #  ranking vocabulary stays forbidden for the reason it always was.
    for banned in ("best", "best_route", "recommended", "recommended_route",
                   "rank", "ranking", "score", "winner", "optimal_route",
                   "pareto", "dominates", "bunker", "consumption",
                   "safest", "litre", "tonne"):
        assert banned not in code, f"{banned!r} appears in the module's code"
    assert "safest" not in MODULE.read_text().lower()
    assert cmp.provenance["this_is_not_a_ranking"] is True
    #  distance is still not fuel -- now that a fuel figure exists, saying so
    #  matters more, not less
    assert cmp.provenance["shortest_distance_is_a_fuel_figure"] is False
    assert OBJECTIVES[PROFILE_SHORTEST]["is_a_fuel_or_consumption_figure"] is False
    t = cmp.table()
    assert "not a ranking" in t and "NOT a fuel" in t
    d = cmp.to_dict()
    #  this corridor comparison supplies no sea-ice field, so it is still the
    #  three original profiles and the fuel objective is absent by design
    assert set(d["profiles"]) == set(PROFILES)
    assert d["provenance"]["estimated_fuel_available"] is False
    assert set(d) == {"profiles", "shared_inputs", "provenance", "objectives"}
    return ("no ranking, recommendation, Pareto, bunker or consumption "
            "vocabulary in the code; the comparison states plainly that "
            "distance is not fuel, and offers no fuel figure without a field")


def test_the_comparison_carries_every_required_metric():
    cmp = cached_corridor()
    required = ("profile", "success", "path", "distance_m", "travel_time_s",
                "configured_cost", "environmental_contribution",
                "polaris_contribution", "coverage_uncertainty_contribution",
                "iceberg_exposure_contribution", "max_iceberg_exposure",
                "special_consideration_cells", "indeterminate_cells",
                "iceberg_provenance", "departure_time", "arrival_time",
                "temporal_provenance", "constraint_summary")
    for name in PROFILES:
        d = cmp.profiles[name].to_dict()
        missing = [k for k in required if k not in d]
        assert not missing, (name, missing)
        assert d["arrival_time"] and d["departure_time"]
    return f"all {len(required)} required fields present on all three profiles"


# --------------------------- 11. the re-pricing cannot drift silently
def test_the_reported_terms_reconcile_with_the_search_total():
    cmp = cached_corridor()
    for name in PROFILES:
        r = cmp.profiles[name]
        total = (r.environmental_contribution + r.polaris_contribution
                 + r.coverage_uncertainty_contribution
                 + r.iceberg_exposure_contribution)
        assert abs(total - r.configured_cost) < 1e-9 * max(abs(total), 1.0), name
    risk = cmp.profiles[PROFILE_RISK]
    assert abs(risk.configured_cost - risk.objective_value) < 1e-9
    return ("the four terms sum to the configured cost for every profile, and on "
            "the risk profile that number is plan_route's own search total")


def test_a_malformed_comparison_request_is_refused():
    shape, blocked, _ = E.corridor_world()
    usnic = E.usnic_dir(shape, PX)
    base = E.base_provider(shape, PX, [np.ones(shape) for _ in range(30)])
    risk = E.iceberg_provider(base, shape, [np.zeros(shape)] * 30, E.CFG.weight)
    nc = NavigationConstraints(shape=shape, land_mask=blocked)
    args = dict(risk_provider=risk, usnic_dir=usnic, start=(6, 0), goal=(6, 12),
                departure_time=DEPART, vessel_speed_mps=PX / 3600.0,
                constraints_factory=lambda g: nc)
    bad = 0
    for kw in (dict(departure_time=DAY), dict(vessel_speed_mps=0.0),
               dict(vessel_speed_mps=float("nan"))):
        a = dict(args)
        a.update(kw)
        try:
            compare_profiles(lambda: E.make_grid(shape, PX), **a)
        except ProfileError:
            bad += 1
        else:
            raise AssertionError(f"accepted a malformed request: {kw}")
    try:
        compare_profiles(E.make_grid(shape, PX), **args)     # not a factory
    except ProfileError:
        bad += 1
    else:
        raise AssertionError("accepted a grid instead of a grid factory")
    assert bad == 4
    return f"{bad} malformed comparison requests refused with ProfileError"


# -------------------------------------------------- real-data smoke
def test_real_data_smoke_2025_01_08_three_profiles():
    grid, dom, nc, excluded, comp, chart_date, pp = E._real_world()
    provider, _ = E._real_provider(chart_date, grid, comp, override=True)
    assert provider.config.weight == 5.0, provider.config.weight
    assert chart_date == date(2020, 12, 24), chart_date

    #  without the explicit override the historical chart refuses all three
    refused = compare_profiles(
        lambda: RoutingGrid.for_date(DAY), risk_provider=provider,
        usnic_dir=E.USNIC, start=E.REAL_START, goal=E.REAL_GOAL,
        departure_time=DEPART, vessel_speed_mps=5.0,
        constraints_factory=lambda g: NavigationDomain(g).add_geotiff(
            "land_mask", E.BEDMACHINE),
        polaris_ice_class=E.PC6, polaris_riv_table=E.TABLE_13,
        polaris_penalties=pp)
    assert refused.succeeded == (), refused.succeeded
    for name in PROFILES:
        assert refused.profiles[name].outcome == OUTCOME_DATE_MISMATCH

    cmp = compare_profiles(
        lambda: RoutingGrid.for_date(DAY), risk_provider=provider,
        usnic_dir=E.USNIC, start=E.REAL_START, goal=E.REAL_GOAL,
        departure_time=DEPART, vessel_speed_mps=5.0,
        constraints_factory=lambda g: NavigationDomain(g).add_geotiff(
            "land_mask", E.BEDMACHINE),
        polaris_ice_class=E.PC6, polaris_riv_table=E.TABLE_13,
        polaris_penalties=pp, allow_historical_demo_override=True)
    assert cmp.succeeded == PROFILES, cmp.succeeded
    for name in PROFILES:
        r = cmp.profiles[name]
        assert not any(excluded[c.row, c.col] for c in r.cells), name
        assert max(c.bucket for c in r.cells) > 0, name
        assert r.special_consideration_cells is not None
        assert r.indeterminate_cells is not None
    risk = cmp.profiles[PROFILE_RISK]
    assert risk.polaris["selection"]["ice_class"] == E.PC6
    assert risk.polaris["selection"]["riv_table_requested"] == E.TABLE_13
    assert risk.polaris_contribution > 0
    assert risk.configured_cost == min(cmp.profiles[n].configured_cost
                                       for n in PROFILES)
    assert cmp.profiles[PROFILE_FASTEST].travel_time_s == min(
        cmp.profiles[n].travel_time_s for n in PROFILES)
    assert cmp.profiles[PROFILE_SHORTEST].distance_m == min(
        cmp.profiles[n].distance_m for n in PROFILES)
    assert cmp.provenance["historical_demo_override"] is True
    assert cmp.provenance["iceberg_exposure_weight"] == 5.0
    return ("HISTORICAL DEMO (2020-12-24 chart over 2025-01-08 data, not an "
            "operational route) | " + " | ".join(
                f"{n}: {cmp.profiles[n].distance_km:.2f} km, "
                f"{cmp.profiles[n].travel_time_h:.2f} h, cost "
                f"{cmp.profiles[n].configured_cost:,.0f}, POLARIS "
                f"{cmp.profiles[n].polaris_contribution:,.0f}, iceberg "
                f"{cmp.profiles[n].iceberg_exposure_contribution:,.0f}"
                for n in PROFILES))


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78)
    print("route profiles: fastest / risk_oriented / shortest_distance")
    print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<62} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__)
            print(f"  FAIL  {fn.__name__:<62} {exc}")
        except Exception as exc:                        # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__:<62} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

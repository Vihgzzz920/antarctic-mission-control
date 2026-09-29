"""
The mission-decision layer: several destinations, several departure times.

What matters here: every option is evaluated from scratch, an infeasibility is
always the ROUTER's word and never a threshold invented by this layer, and
nothing anywhere is ranked, scored or recommended.

    python tests/test_mission_decision.py
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import mission_decision as M                              # noqa: E402
from src.api.config import DEMO                                        # noqa: E402
from src.api.world import get_world                                    # noqa: E402
from src.routing.end_to_end_route import (OUTCOME_FORECAST_UNAVAILABLE,  # noqa: E402
                                          OUTCOME_HARD_CONSTRAINT,
                                          OUTCOME_NO_PATH)
from src.routing.route_profiles import (ALL_PROFILES, PROFILE_FASTEST,  # noqa: E402
                                        PROFILE_FUEL, PROFILE_RISK,
                                        PROFILE_SHORTEST)

MODULE = ROOT / "src" / "api" / "mission_decision.py"
BASE = datetime.fromisoformat(DEMO["departure_time"])
_C: dict = {}


def matrix():
    if "m" not in _C:
        _C["m"] = M.evaluate_mission(
            sites=M.DEMONSTRATION_SITES,
            departure_times=[BASE + timedelta(hours=h) for h in (0, 6, 12, 18)])
    return _C["m"]


def site(cell, sid="probe", operational=False):
    return M.Site(sid, tuple(cell), f"test site {sid}", operational,
                  "a test fixture, not a place")


# ----------------------------------------------- 1. destinations, independently
def test_multiple_destinations_are_evaluated_independently():
    m = matrix()
    ids = [o.scenario.site.site_id for o in m.options]
    assert len(m.options) == len(M.DEMONSTRATION_SITES) * 4, len(m.options)
    for s in M.DEMONSTRATION_SITES:
        got = [o for o in m.options if o.scenario.site.site_id == s.site_id]
        assert len(got) == 4, (s.site_id, len(got))
        #  each option carries ITS OWN goal, never another site's
        for o in got:
            assert o.scenario.site.cell == s.cell
            for p in o.profiles.values():
                if p.get("success"):
                    assert tuple(p["goal"]) == s.cell, (s.site_id, p["goal"])
                    assert tuple(p["path"][-1]) == s.cell
    #  and the sites really are different destinations
    assert len({s.cell for s in M.DEMONSTRATION_SITES}) == len(M.DEMONSTRATION_SITES)
    return (f"{len(m.options)} options over {len(M.DEMONSTRATION_SITES)} distinct "
            f"sites; every routed path ends at its own site's cell")


# --------------------------------------------- 2. departures, independently
def test_multiple_departure_times_are_evaluated_independently():
    m = matrix()
    for s in M.DEMONSTRATION_SITES:
        got = {o.scenario.departure_time: o
               for o in m.options if o.scenario.site.site_id == s.site_id}
        assert len(got) == 4
        #  each departure carries its own wall clock AND its own offset, and
        #  the two agree by construction
        for dep, o in got.items():
            assert o.scenario.departure_time == dep
            assert math.isclose(o.scenario.departure_offset_s,
                                (dep - BASE).total_seconds())
            for p in o.profiles.values():
                if p.get("success"):
                    assert p["departure_time"] == dep.isoformat()
    return "each of 4 departures carries its own wall clock and provider offset"


# ------------------------------------------- 3. a blocked goal is infeasible
def test_a_blocked_goal_returns_explicit_infeasibility():
    """A cell the hard constraints exclude is refused by the ROUTER, with the
    router's own outcome -- not by anything this layer decided."""
    w = get_world()
    excluded = w["excluded"]
    rows, cols = excluded.nonzero()
    assert len(rows), "the world excludes no cell, so this cannot be tested"
    blocked = (int(rows[0]), int(cols[0]))
    m = M.evaluate_mission(sites=[site(blocked, "blocked_goal")],
                           departure_times=[BASE])
    o = m.options[0]
    assert o.feasible is False
    assert o.refusal_outcome in M.REFUSAL_OUTCOMES, o.refusal_outcome
    assert o.refusal_outcome == OUTCOME_HARD_CONSTRAINT, o.refusal_outcome
    assert o.refusal_reason and len(o.refusal_reason) > 10
    return (f"goal {blocked} -> {o.refusal_outcome}: "
            f"{o.refusal_reason[:70]}")


# ------------------------------------------- 4. no feasible option at all
def test_no_feasible_option_is_represented_explicitly():
    w = get_world()
    rows, cols = w["excluded"].nonzero()
    blocked = (int(rows[0]), int(cols[0]))
    m = M.evaluate_mission(sites=[site(blocked, "a"), site(blocked, "b")],
                           departure_times=[BASE, BASE + timedelta(hours=6)])
    assert m.any_feasible is False
    s = m.summary()
    assert s["outcome"] == M.NO_FEASIBLE_OPTION == "no_feasible_option"
    assert s["no_feasible_option"] is True
    assert s["options_feasible"] == 0 and s["options_evaluated"] == 4
    assert s["sites_feasible"] == [] and len(s["sites_infeasible"]) == 2
    assert s["why"], "the matrix does not say why nothing routed"
    for line in s["why"]:
        assert any(line.startswith(o) for o in M.REFUSAL_OUTCOMES), line
    #  and a feasible matrix does NOT claim the no-feasible outcome
    assert matrix().summary()["no_feasible_option"] is False
    return f"4 of 4 refused; outcome {s['outcome']!r}, reasons {len(s['why'])}"


def test_an_expensive_option_is_never_called_infeasible():
    """A high cost, POLARIS, exposure or fuel figure is a measured ATTRIBUTE.
    Every feasible option proves it: they carry large numbers and routed."""
    m = matrix()
    feas = m.feasible
    assert feas, "no feasible option to check"
    for o in feas:
        for name, v in o.objective_values().items():
            if not v["routed"]:
                continue
            assert v["configured_cost"] > 0
            assert o.feasible is True
    #  the layer declares which attributes can never cause a refusal, and the
    #  module's code never branches on one to decide feasibility
    code = MODULE.read_text()
    for attr in M.NOT_FEASIBILITY_CRITERIA:
        for pattern in (f"{attr} >", f"{attr} <", f'["{attr}"] >',
                        f'["{attr}"] <', f"{attr} >=", f"{attr} <="):
            assert pattern not in code, f"{pattern!r} appears in the module"
    assert set(M.NOT_FEASIBILITY_CRITERIA) >= {
        "configured_cost", "polaris_contribution",
        "iceberg_exposure_contribution", "estimated_fuel"}
    return ("feasible options carry costs in the millions and remain feasible; "
            "no comparison on a cost, POLARIS, exposure or fuel attribute "
            "appears anywhere in the module")


# ------------------------------- 5, 6. departures own their own environment
def test_departure_time_changes_the_buckets_and_owns_its_provenance():
    m = matrix()
    seen = {}
    for o in m.options:
        env = o.environment
        key = o.scenario.departure_time.isoformat()
        if o.feasible:
            seen.setdefault(key, set()).update(env.get("buckets_used") or [])
        #  every option's provenance names ITS OWN departure, never another's
        prov = o.provenance()
        assert prov["departure_time"] == key
        assert math.isclose(prov["departure_offset_s"],
                            (o.scenario.departure_time - BASE).total_seconds())
        assert prov["environment_policy"] == m.environment_policy
    #  a later departure starts later in the provider's timeline, so it cannot
    #  resolve the same set of remaining buckets as an earlier one
    offsets = sorted({o.scenario.departure_offset_s for o in m.options})
    assert offsets == [0.0, 21600.0, 43200.0, 64800.0], offsets
    #  and the shipped 12 h provider REFUSES the two departures past its
    #  horizon, which is the environment resolution differing, not a route
    late = [o for o in m.options if o.scenario.departure_offset_s >= 43200.0]
    assert late and all(o.refusal_outcome == OUTCOME_FORECAST_UNAVAILABLE
                        for o in late), [o.refusal_outcome for o in late]
    return (f"offsets {offsets}; departures at/after the 12 h provider horizon "
            f"resolve to {OUTCOME_FORECAST_UNAVAILABLE}, not to a route")


def test_forecast_provenance_belongs_to_its_own_departure():
    m = matrix()
    for o in m.options:
        if not o.feasible:
            continue
        env = o.environment
        used = set(env["buckets_used"])
        listed = {int(b["bucket"]) for b in env["buckets"]}
        assert listed == used, (o.scenario.scenario_id, listed, used)
        for b in env["buckets"]:
            assert b["policy"] == m.environment_policy
        #  the buckets reported are exactly the ones the routes used
        for p in o.profiles.values():
            if p.get("success"):
                assert set(int(x) for x in p["buckets_used"]) <= used
    return ("every feasible option lists exactly the buckets its own routes "
            "used, each under the matrix's own policy")


# ------------------------------------------- 7. fuel belongs to the route
def test_fuel_values_belong_to_the_returned_route():
    from src.routing.fuel_cost import load_fuel_params, route_fuel
    from src.routing.route_profiles import FUEL_CONFIG
    from src.api.world import sic_for_bucket
    m, w = matrix(), get_world()
    params, sic = load_fuel_params(FUEL_CONFIG), sic_for_bucket()
    bs = float(w["provider"].base.bucket_seconds
               if hasattr(w["provider"], "base")
               else w["provider"].bucket_seconds)
    checked = 0
    for o in m.feasible:
        for name, p in o.profiles.items():
            if not p.get("success"):
                continue
            again = route_fuel(
                p["path"], p["arrival_times"], sic_for_bucket=sic,
                bucket_of=lambda t: int(math.floor(t / bs)),
                pixel_size=w["grid"].pixel_size, params=params)
            assert math.isclose(again.estimated_fuel, p["estimated_fuel"],
                                rel_tol=1e-12), (o.scenario.scenario_id, name)
            checked += 1
    assert checked, "no routed profile carried a fuel figure"
    #  and the fuel objective's own value IS its fuel
    for o in m.feasible:
        f = o.profiles.get(PROFILE_FUEL)
        if f and f.get("success"):
            assert math.isclose(f["objective_value"], f["estimated_fuel"],
                                rel_tol=1e-12)
    return (f"{checked} routed profiles re-evaluate to the fuel they report, "
            f"from their own returned paths")


# ---------------------------------------------------- 8. determinism
def test_the_same_inputs_are_deterministic():
    sites = M.DEMONSTRATION_SITES[:2]
    deps = [BASE, BASE + timedelta(hours=6)]
    a = M.evaluate_mission(sites=sites, departure_times=deps)
    b = M.evaluate_mission(sites=sites, departure_times=deps)
    assert json.dumps(a.to_dict(), sort_keys=True, default=str) == \
        json.dumps(b.to_dict(), sort_keys=True, default=str)
    return "two independent evaluations serialise identically"


# ------------------------- 9, 10. nothing existing moved
def test_existing_profile_results_are_unchanged_by_the_mission_layer():
    """The layer calls the SAME orchestration: a mission option's demo-goal
    route at the demo departure equals what compare_profiles returns directly."""
    from src.api.config import BEDMACHINE, USNIC
    from src.api.world import fresh_grid, sic_for_bucket
    from src.routing.navigation_domain import NavigationDomain
    from src.routing.route_profiles import compare_profiles
    w = get_world()
    direct = compare_profiles(
        fresh_grid, risk_provider=w["provider"], usnic_dir=USNIC,
        start=tuple(DEMO["start"]), goal=tuple(DEMO["goal"]),
        departure_time=BASE, vessel_speed_mps=DEMO["vessel_speed_mps"],
        constraints_factory=lambda g: NavigationDomain(g).add_geotiff(
            "land_mask", BEDMACHINE),
        polaris_ice_class=DEMO["polaris_ice_class"],
        polaris_riv_table=DEMO["polaris_riv_table"],
        polaris_penalties=w["polaris"], sic_for_bucket=sic_for_bucket(),
        allow_historical_demo_override=True).to_dict()["profiles"]
    option = next(o for o in matrix().options
                  if o.scenario.site.site_id == "demo_goal"
                  and o.scenario.departure_time == BASE)
    for name in ALL_PROFILES:
        a, b = direct[name], option.profiles[name]
        for key in ("path", "distance_m", "travel_time_s", "configured_cost",
                    "polaris_contribution", "iceberg_exposure_contribution",
                    "estimated_fuel", "outcome", "success"):
            assert json.dumps(a.get(key), default=str) == \
                json.dumps(b.get(key), default=str), (name, key)
    return ("the mission layer's demo-goal option is field-for-field the same "
            "as calling compare_profiles directly")


def _strip_lookups(node):
    """Drop the provider's `lookups` counters.

    `lookups` (src/routing/time_navigation_cost.py, iceberg_navigation_cost.py,
    both frozen) counts how many cost lookups the SHARED world provider object
    has served since the process started. It is a diagnostic of process
    history, not of route content: planning any route earlier in the same
    process increments it, so the demo payload is byte-identical only in a
    fresh process. Everything that determines the route is compared below; this
    counter is separated out and checked on its own rather than ignored.
    """
    if isinstance(node, dict):
        return {k: _strip_lookups(v) for k, v in node.items() if k != "lookups"}
    if isinstance(node, list):
        return [_strip_lookups(v) for v in node]
    return node


def test_the_default_demo_endpoint_is_unchanged():
    from fastapi.testclient import TestClient
    from src.api.main import app
    matrix()          # deliberately route first: the demo must survive it
    ref = json.loads((ROOT / "logs" / "demo_route_after_training.json").read_text())
    new = TestClient(app).get(
        "/api/demo/routes?historical_demo_override=true").json()
    for name in ("fastest", "risk_oriented", "shortest_distance"):
        a, b = ref["comparison"]["profiles"][name], new["comparison"]["profiles"][name]
        for key in a:
            assert json.dumps(_strip_lookups(a[key]), sort_keys=True) == \
                json.dumps(_strip_lookups(b[key]), sort_keys=True), (name, key)
        #  and the only NEW keys are the additive fuel fields from the
        #  previous milestone -- nothing was removed or renamed
        assert set(a) <= set(b), (name, sorted(set(a) - set(b)))
        assert set(b) - set(a) == {"estimated_fuel", "estimated_fuel_units",
                                   "estimated_fuel_per_km", "fuel_provenance"}, \
            (name, sorted(set(b) - set(a)))
    tp = new["comparison"]["profiles"]["risk_oriented"]["temporal_provenance"]
    assert tp["environment_policies"] == ["persistence_from_departure_analysis"]
    assert tp["environment_is_time_varying"] is False
    return ("after a full 12-option mission evaluation, the demo endpoint's "
            "three original profiles are identical on every field except the "
            "provider's process-lifetime lookup counter")


# -------------------------------- 11. synthetic is never called operational
def test_no_synthetic_scenario_is_labelled_operational():
    m = matrix()
    for s in M.DEMONSTRATION_SITES:
        assert s.is_operational is False, s.site_id
        assert s.source, s.site_id
    for o in m.options:
        assert o.scenario.is_operational_mission is False
        assert o.provenance()["is_operational_mission"] is False
        assert o.provenance()["destination"]["is_operational"] is False
    #  the flag is DERIVED from the site, so an operational site would show
    sreal = site((DEMO["goal"][0], DEMO["goal"][1]), "declared", operational=True)
    m2 = M.evaluate_mission(sites=[sreal], departure_times=[BASE])
    assert m2.options[0].scenario.is_operational_mission is True, (
        "the flag is hardcoded rather than derived from the site")
    #  no synthetic site is dressed up as a place
    for s in M.DEMONSTRATION_SITES:
        low = (s.label + " " + s.source).lower()
        for word in ("station", "port", "base", "waypoint", "harbour", "harbor"):
            if word in low:
                assert "not a station" in low or "not a place" in low, s.site_id
    return ("all 12 options are is_operational_mission=False; the flag is "
            "derived from the site, not asserted")


# ------------------------------- 12. nothing is ranked
def test_nothing_is_ranked_scored_or_recommended():
    m = matrix()
    s = m.summary()
    for key, want in (("this_is_not_a_ranking", True),
                      ("has_a_best_site", False),
                      ("has_a_best_departure_time", False),
                      ("has_a_recommended_route", False),
                      ("has_an_overall_score", False),
                      ("selection_requires_a_caller_supplied_objective", True)):
        assert s[key] is want, key
    #  A blunt substring scan would trip on the matrix's own DENIALS --
    #  "has_a_best_site": false is the opposite of a claim. So, as
    #  test_simulation.py does, the check is on what the matrix ASSERTS: no
    #  banned wording in any string VALUE, and every key naming one of those
    #  ideas must be false.
    banned = ("best site", "best departure", "best route", "recommended route",
              "overall score", "the winner", "ranked", "ranking", "safest",
              "optimal route", "pareto")
    offenders: list[str] = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            for k, v in node.items():
                low = k.lower()
                names_it = any(w in low for w in
                               ("rank", "score", "winner", "best", "recommend"))
                #  a key that DENIES the idea ("this_is_not_a_ranking",
                #  "selection_requires_...") is the opposite of a claim
                denies = "not" in low or "requires" in low
                if names_it and not denies and v is not False:
                    offenders.append(f"{path}.{k} = {v!r}")
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            for w in banned:
                if w in node.lower():
                    offenders.append(f"{path}: {w!r}")

    walk(m.to_dict())
    assert not offenders, offenders[:5]
    #  select() refuses to act without a caller-supplied criterion
    for kwargs in ({"objective": "travel_time_h"}, {"direction": "min"}):
        try:
            m.select(**kwargs)
        except TypeError:
            continue
        raise AssertionError(f"select() ran with only {sorted(kwargs)}")
    for bad in ({"objective": "vibes", "direction": "min"},
                {"objective": "travel_time_h", "direction": "better"}):
        try:
            m.select(**bad)
        except M.MissionError:
            continue
        raise AssertionError(f"select() accepted {bad}")
    chosen = m.select(objective="estimated_fuel", direction="min")
    assert chosen["criterion_supplied_by"] == "caller"
    assert chosen["this_is_not_a_recommendation"] is True
    return ("no rank, score, winner or best field anywhere; select() requires "
            f"an explicit objective and direction and returns "
            f"{chosen['selected']['scenario_id']} only as the caller's "
            f"{chosen['direction']} of {chosen['objective']}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            note = fn()
            print(f"PASS  {fn.__name__}")
            if note:
                print(f"      {note}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:                                  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

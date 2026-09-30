"""
Tests for the inject-an-iceberg / replan interaction.

The ones that matter: nothing on disk changes, nothing in the cached world
changes, the replan goes through the existing search, and the explanation's
numbers are the two real results.

    python tests/test_simulation.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient                              # noqa: E402

import test_end_to_end_route as E                                      # noqa: E402
from src.api import simulate                                           # noqa: E402
from src.api.config import DEMO                                        # noqa: E402
from src.api.main import app                                           # noqa: E402
from src.api.world import get_world                                    # noqa: E402
from src.routing.constraints import NavigationConstraints              # noqa: E402
from src.routing.iceberg_risk import exposure_value                    # noqa: E402
from src.routing.route_profiles import (PROFILE_RISK, PROFILES,        # noqa: E402
                                        compare_profiles)

client = TestClient(app)
PX = 10_000.0
DATA = ROOT / "data"
_C: dict = {}

BODY = {
    "start": DEMO["start"], "goal": DEMO["goal"],
    "departure_time": DEMO["departure_time"],
    "vessel_speed_mps": DEMO["vessel_speed_mps"],
    "polaris_ice_class": DEMO["polaris_ice_class"],
    "polaris_riv_table": DEMO["polaris_riv_table"],
    "historical_demo_override": True,
}


# ───────────────────────────────────────────── controlled synthetic world
def synthetic():
    """The asymmetric corridor with NO pre-existing exposure, so the baseline
    takes the short way and a simulated berg is the only thing that can move
    it."""
    if "world" not in _C:
        shape, blocked, _cone = E.corridor_world()
        envs = [np.ones(shape) for _ in range(30)]
        zero = np.zeros(shape, "float64")
        usnic = E.usnic_dir(shape, PX)
        base = E.base_provider(shape, PX, envs)
        risk = E.iceberg_provider(base, shape, [zero] * 30, E.CFG.weight)
        _C["world"] = (shape, blocked, usnic, risk)
    return _C["world"]


def synthetic_raster(shape, centre, radius_cells=1.6) -> np.ndarray:
    """A stand-in exposure patch for the injection mechanics.

    This is TEST INPUT, not a second exposure model: the production path builds
    this array with iceberg_risk.build_exposure, which
    test_simulated_exposure_comes_from_the_existing_builder checks directly.
    """
    raster = np.zeros(shape, "float64")
    for r in range(shape[0]):
        for c in range(shape[1]):
            d = math.hypot(r - centre[0], c - centre[1])
            raster[r, c] = max(0.0, 1.0 - d / radius_cells)
    return raster


def synthetic_run(raster=None, from_bucket=0):
    shape, blocked, usnic, risk = synthetic()
    nc = NavigationConstraints(shape=shape, land_mask=blocked)
    provider = risk
    if raster is not None:
        fields = simulate.inject(
            list(risk.exposure.fields.values()), raster, from_bucket=from_bucket)
        provider = simulate.simulated_provider(
            risk, fields, shape, allow_mismatch=False)
    return compare_profiles(
        lambda: E.make_grid(shape, PX), risk_provider=provider, usnic_dir=usnic,
        start=(6, 0), goal=(6, 12), departure_time=E.DEPART,
        vessel_speed_mps=PX / 3600.0, constraints_factory=lambda g: nc)


def fingerprint(root: Path) -> str:
    """Size and modification time of every file under a tree."""
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            entries.append(f"{path.relative_to(ROOT)}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def demo_simulation():
    if "sim" not in _C:
        baseline = client.post("/api/mission/compare", json=BODY).json()
        route = baseline["comparison"]["profiles"][PROFILE_RISK]
        grid = get_world()["grid"]
        #  a cell the current route actually passes through
        cell = route["cells"][len(route["cells"]) // 3]
        x, y = grid.rowcol_to_xy(cell["row"], cell["col"])
        response = client.post("/api/mission/simulate", json={
            "request": BODY, "iceberg": {"x": x, "y": y}})
        assert response.status_code == 200, response.text
        _C["sim"] = (baseline, response.json())
    return _C["sim"]


# ─────────────────────────────────────────────────────────────── the tests
def test_the_original_route_works_before_any_injection():
    comparison = synthetic_run()
    for name in PROFILES:
        assert comparison.profiles[name].success, name
    route = comparison.profiles[PROFILE_RISK]
    shortcut = {(6, c) for c in (4, 5, 6, 7, 8)}
    assert set(route.path) & shortcut, route.path
    assert route.iceberg_exposure_contribution == 0.0
    return (f"with no exposure anywhere the baseline takes the short corridor: "
            f"{route.distance_km:.2f} km, {route.travel_time_h:.2f} h, "
            f"iceberg contribution 0")


def test_an_injected_iceberg_moves_the_route():
    shape, _blocked, _usnic, _risk = synthetic()
    before = synthetic_run().profiles[PROFILE_RISK]
    raster = synthetic_raster(shape, (6, 6))
    after = synthetic_run(raster).profiles[PROFILE_RISK]
    assert before.success and after.success
    assert after.path != before.path, "the simulated iceberg moved nothing"
    shortcut = {(6, c) for c in (4, 5, 6, 7, 8)}
    assert set(before.path) & shortcut and not (set(after.path) & shortcut)
    assert after.distance_m > before.distance_m
    assert after.travel_time_s > before.travel_time_s
    return (f"{before.distance_km:.2f} km -> {after.distance_km:.2f} km, "
            f"{before.travel_time_h:.2f} h -> {after.travel_time_h:.2f} h; the "
            f"replanned route leaves the corridor the simulated berg sits in")


def test_the_replanned_route_is_still_a_valid_route():
    shape, blocked, _usnic, _risk = synthetic()
    after = synthetic_run(synthetic_raster(shape, (6, 6))).profiles[PROFILE_RISK]
    assert after.success and after.outcome == "route_found"
    assert not any(blocked[c.row, c.col] for c in after.cells)
    assert after.constraint_summary["route_cells_inside_excluded"] == 0
    terms = (after.environmental_contribution + after.polaris_contribution
             + after.coverage_uncertainty_contribution
             + after.iceberg_exposure_contribution)
    assert abs(terms - after.configured_cost) < 1e-9 * max(abs(terms), 1.0)
    buckets = [c.bucket for c in after.cells]
    assert buckets == sorted(buckets)
    return (f"{len(after.cells)} cells, inside the hard constraints, terms "
            f"reconcile, arrival buckets still monotonic")


def test_no_hard_block_is_created_by_the_injection():
    shape, _blocked, _usnic, _risk = synthetic()
    #  the ONLY corridor, with the simulated berg sitting on it at full strength
    raster = synthetic_raster(shape, (6, 6), radius_cells=1.2)
    comparison = synthetic_run(raster)
    route = comparison.profiles[PROFILE_RISK]
    assert route.success, "an injected iceberg made the mission unroutable"
    for cell in route.cells:
        assert math.isfinite(cell.final_cost) and cell.final_cost > 0
    #  and a route forced through the berg is still priced, never refused
    exposed = [c for c in comparison.profiles["fastest"].cells
               if (c.iceberg_exposure or 0) > 0]
    assert exposed, "the fastest route avoided the berg entirely"
    assert all(math.isfinite(c.final_cost) for c in exposed)
    source = (ROOT / "src" / "api" / "simulate.py").read_text()
    tree = ast.parse(source)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "set_blocked_mask" not in attrs
    for banned in ("blocked_mask", "no_go", "navigable_mask"):
        assert banned not in {n.id for n in ast.walk(tree)
                              if isinstance(n, ast.Name)}
    return ("a berg on the only corridor is priced, not blocked: the fastest "
            f"route still crosses {len(exposed)} exposed cells at a finite cost")


def test_clearing_the_simulation_restores_the_original_route_exactly():
    shape, _blocked, _usnic, _risk = synthetic()
    first = synthetic_run().profiles[PROFILE_RISK]
    _moved = synthetic_run(synthetic_raster(shape, (6, 6)))
    restored = synthetic_run().profiles[PROFILE_RISK]
    assert restored.path == first.path
    assert restored.configured_cost == first.configured_cost
    assert restored.objective_value == first.objective_value
    assert restored.iceberg_exposure_contribution == \
        first.iceberg_exposure_contribution
    assert [c.bucket for c in restored.cells] == [c.bucket for c in first.cells]
    return ("after a replan, running the same mission again reproduces the "
            "original path, cost and per-cell buckets exactly")


def test_repeated_injection_is_deterministic():
    shape, _blocked, _usnic, _risk = synthetic()
    raster = synthetic_raster(shape, (6, 6))
    runs = [synthetic_run(raster) for _ in range(3)]
    seen = [
        tuple((n, c.profiles[n].path, c.profiles[n].configured_cost,
               c.profiles[n].iceberg_exposure_contribution) for n in PROFILES)
        for c in runs
    ]
    assert seen[0] == seen[1] == seen[2]
    return "3 identical injections give identical paths and contributions"


def test_the_injection_leaves_the_cached_world_untouched():
    world = get_world()
    before = [np.asarray(f.exposure).copy() for f in world["exposure_fields"]]
    before_ids = [f.forecasts for f in world["exposure_fields"]]
    demo_simulation()
    after = [np.asarray(f.exposure) for f in world["exposure_fields"]]
    for i, (was, now) in enumerate(zip(before, after)):
        assert np.array_equal(was, now), f"exposure field {i} was mutated"
    assert [f.forecasts for f in world["exposure_fields"]] == before_ids
    return (f"the {len(before)} cached exposure fields are bit-identical after a "
            f"simulate request; the scenario is built from copies")


def test_the_injection_writes_nothing_to_disk():
    before = fingerprint(DATA)
    before_configs = fingerprint(ROOT / "configs")
    demo_simulation()
    #  a second, differently-placed injection too
    grid = get_world()["grid"]
    x, y = grid.rowcol_to_xy(DEMO["start"][0] - 6, DEMO["start"][1])
    client.post("/api/mission/simulate",
                json={"request": BODY, "iceberg": {"x": x, "y": y}})
    assert fingerprint(DATA) == before, "a file under data/ changed"
    assert fingerprint(ROOT / "configs") == before_configs, "a config changed"
    return ("every file under data/ and configs/ has the same size and mtime "
            "after two simulate requests")


def test_the_simulation_is_labelled_as_one_everywhere():
    _baseline, response = demo_simulation()
    sim = response["simulation"]
    assert sim["label"] == simulate.SIMULATION_LABEL == "SIMULATED ICEBERG"
    assert sim["simulated"] is True and sim["temporary"] is True
    assert sim["written_to_any_dataset"] is False
    assert sim["iceberg_id"] == "simulated_iceberg"
    claims = response["change"]["claims"]
    assert claims["is_a_collision_probability"] is False
    assert claims["guarantees_avoidance"] is False
    assert claims["ranks_the_routes"] is False
    #  A blunt substring scan would trip on the response's own DENIALS --
    #  "is_a_collision_probability": false is the opposite of a claim. So the
    #  check is on what the response asserts: no banned wording in any string
    #  VALUE, and every key naming one of those ideas must be false.
    #  `fuel_efficient` is now a real objective (src/routing/fuel_cost.py), so
    #  the word itself is no longer forbidden -- what stays forbidden is any
    #  UNSUPPORTED fuel claim: a measured consumption, an operational
    #  prediction, a saving, or a mass or volume this project cannot produce.
    banned = ("safest", "collision probability", "guaranteed avoidance",
              "best route", "measured fuel", "actual fuel", "fuel saving",
              "saves fuel", "litres of fuel", "tonnes of fuel", "bunker")
    #  keys that ASSERT one of these ideas must be false. `fuel` alone is no
    #  longer such a key, so only the claim-shaped fuel keys are checked.
    claim_keys = ("safest", "collision", "guarantee",
                  "is_a_measured_fuel", "is_an_operational_fuel",
                  "fuel_is_measured", "fuel_is_an_operational",
                  "suitable_for_operational")
    offenders: list[str] = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            for key, value in node.items():
                low = key.lower()
                if any(word in low for word in claim_keys):
                    if value is not False:
                        offenders.append(f"{path}.{key} = {value!r}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
        elif isinstance(node, str):
            low = node.lower()
            for word in banned:
                if word in low:
                    offenders.append(f"{path}: {word!r}")

    walk(response)
    assert not offenders, offenders
    collection = response["simulated_exposure"]
    assert collection["features"], "the simulated berg exposed no cell"
    assert collection["properties"]["is_a_probability"] is False
    return (f"labelled {sim['label']!r}, temporary, {len(collection['features'])} "
            f"exposed cells, and no forbidden claim anywhere in the response")


def test_the_explanation_uses_the_two_real_results():
    _baseline, response = demo_simulation()
    change = response["change"]["profiles"]
    for name in PROFILES:
        was = response["baseline"]["comparison"]["profiles"][name]
        now = response["replanned"]["comparison"]["profiles"][name]
        entry = change[name]
        assert entry["path_changed"] == (was["path"] != now["path"])
        for key in ("distance_m", "travel_time_s",
                    "iceberg_exposure_contribution", "max_iceberg_exposure"):
            assert entry[key]["before"] == was[key], (name, key)
            assert entry[key]["after"] == now[key], (name, key)
            if entry[key]["delta"] is not None:
                assert abs(entry[key]["delta"]
                           - (now[key] - was[key])) < 1e-9, (name, key)
    risk = change[PROFILE_RISK]
    assert risk["cells_with_simulated_exposure"] > 0
    ib = risk["iceberg_exposure_contribution"]
    assert ib["after"] > ib["before"], (ib["before"], ib["after"])
    return (f"iceberg contribution {ib['before']:,.0f} -> {ib['after']:,.0f} on "
            f"{risk['cells_with_simulated_exposure']} route cells; every "
            f"before/after equals the corresponding route result")


def test_the_berg_is_placed_at_the_routes_own_arrival_time():
    _baseline, response = demo_simulation()
    sim = response["simulation"]
    route = response["baseline"]["comparison"]["profiles"][PROFILE_RISK]
    cell = next(c for c in route["cells"]
                if [c["row"], c["col"]] == sim["encounter_cell"])
    assert sim["from_bucket"] == cell["bucket"]
    assert sim["encounter_arrival_s"] == cell["arrival_time"]
    assert sim["encounter_datetime"] == cell["arrival_datetime"]
    #  before that bucket the injected field is the untouched original
    world = get_world()
    fields = {f.bucket: f for f in world["exposure_fields"]}
    raster = np.zeros(tuple(world["grid"].shape), "float64")
    raster[sim["row"], sim["col"]] = 1.0
    injected = {f.bucket: f for f in simulate.inject(
        list(world["exposure_fields"]), raster,
        from_bucket=int(sim["from_bucket"]))}
    for bucket, field in injected.items():
        if bucket < sim["from_bucket"]:
            assert field is fields[bucket], "an earlier bucket was rewritten"
        else:
            assert field is not fields[bucket]
    return (f"placed at bucket {sim['from_bucket']}, the route's own predicted "
            f"arrival {sim['encounter_datetime']} at cell "
            f"{sim['encounter_cell']}; earlier buckets are untouched")


def test_simulated_exposure_comes_from_the_existing_builder():
    """The cone is iceberg_risk's, not a second one written here."""
    source = (ROOT / "src" / "api" / "simulate.py").read_text()
    tree = ast.parse(source)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "build_exposure" in called, "the API builds its own exposure"
    for banned in ("exposure_value", "_exposure_array", "great_circle_m",
                   "_great_circle_grid"):
        assert banned not in source, f"{banned!r} is restated in the API"

    _baseline, response = demo_simulation()
    sim = response["simulation"]
    forecast = simulate.simulated_forecast(
        sim["latitude"], sim["longitude"], sim["radius_km"],
        sim["horizon_hours"], DEMO["environment_date"])
    raster = simulate.simulated_exposure_raster(forecast)
    assert raster[sim["row"], sim["col"]] > 0
    assert 0.0 <= raster.min() and raster.max() <= 1.0
    #  the centre value is what iceberg_risk's own scalar function gives
    assert abs(raster[sim["row"], sim["col"]]
               - exposure_value(0.0, sim["radius_km"] * 1000.0,
                                "linear")) < 0.06
    return (f"build_exposure() produced {int((raster > 0).sum())} exposed cells "
            f"for a {sim['radius_km']:.2f} km radius; no distance, cone or decay "
            f"rule is written in the API")


def test_a_second_injection_replaces_rather_than_accumulates():
    shape, _blocked, _usnic, risk = synthetic()
    one = simulate.inject(list(risk.exposure.fields.values()),
                          synthetic_raster(shape, (6, 6)), from_bucket=0)
    two = simulate.inject(list(risk.exposure.fields.values()),
                          synthetic_raster(shape, (0, 2)), from_bucket=0)
    #  each injection starts from the ORIGINAL fields, so the first berg is
    #  gone from the second scenario rather than piling up
    assert float(np.asarray(two[6].exposure)[6, 6]) == 0.0
    assert float(np.asarray(one[6].exposure)[6, 6]) > 0.0
    assert float(np.asarray(risk.exposure.fields[6].exposure).max()) == 0.0
    return ("a second injection is built from the original fields, so bergs do "
            "not accumulate and the world itself stays empty")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78)
    print("simulated iceberg injection and replanning")
    print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<58} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__)
            print(f"  FAIL  {fn.__name__:<58} {exc}")
        except Exception as exc:                        # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__:<58} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

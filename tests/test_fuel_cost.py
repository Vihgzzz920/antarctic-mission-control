"""
The estimated fuel proxy, and the fuel_efficient objective built on it.

What matters here: the number is an ESTIMATE and says so; it is reproducible
from the path that was actually returned; it is not a relabelled distance; and
adding it changed nothing about POLARIS, the iceberg layer or the three
objectives that already existed.

    python tests/test_fuel_cost.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_end_to_end_route as E                                      # noqa: E402
from src.routing.constraints import NavigationConstraints              # noqa: E402
from src.routing.fuel_cost import (UNITS, FuelError, FuelParams,       # noqa: E402
                                   load_fuel_params, resistance_factor,
                                   route_fuel)
from src.routing.route_profiles import (ALL_PROFILES, FUEL_CONFIG,     # noqa: E402
                                        OBJECTIVES, PROFILE_FASTEST,
                                        PROFILE_FUEL, PROFILE_RISK,
                                        PROFILE_SHORTEST, PROFILES,
                                        compare_profiles)

PX = 10_000.0
DAY, DEPART = E.DAY, E.DEPART
_C: dict = {}

#  ---------------------------------------------------------------------------
#  A TEST FIXTURE, NOT ANTARCTIC VESSEL DATA.
#  A 7x13 world with an open row 0 and a band of heavy ice across the direct
#  row-6 line. The concentrations below are invented for this test to make the
#  fuel objective separate from the distance objective; they are not observed
#  sea ice and are not used by anything outside this file.
#  ---------------------------------------------------------------------------
ICE_BAND_SHAPE = (7, 13)


def ice_band_sic() -> np.ndarray:
    sic = np.zeros(ICE_BAND_SHAPE, "float64")
    sic[4:7, 3:10] = 95.0            # heavy ice straddling the direct line
    sic[3, 3:10] = 60.0              # a softer margin above it
    return sic


def _comparison(sic=None, params=None, *, speed=PX / 3600.0, polaris=None,
                start=(6, 0), goal=(6, 12)):
    shape = ICE_BAND_SHAPE
    envs = [np.ones(shape) for _ in range(30)]
    zero = np.zeros(shape, "float64")
    usnic = E.usnic_dir(shape, PX)
    base = E.base_provider(shape, PX, envs, chart=DAY)
    risk = E.iceberg_provider(base, shape, [zero for _ in range(30)], E.CFG.weight)
    nc = NavigationConstraints(shape=shape, land_mask=np.zeros(shape, bool))
    return compare_profiles(
        lambda: E.make_grid(shape, PX), risk_provider=risk, usnic_dir=usnic,
        start=start, goal=goal, departure_time=DEPART, vessel_speed_mps=speed,
        constraints_factory=lambda g: nc, polaris_penalties=polaris,
        sic_for_bucket=(None if sic is None else (lambda b: sic)),
        fuel_params=params)


def ice_band():
    if "band" not in _C:
        _C["band"] = _comparison(ice_band_sic())
    return _C["band"]


def uniform_ice():
    """The same world with a CONSTANT concentration everywhere."""
    if "uniform" not in _C:
        _C["uniform"] = _comparison(np.full(ICE_BAND_SHAPE, 40.0))
    return _C["uniform"]


# ------------------------------------------------- 1. the model itself
def test_the_model_is_labelled_an_estimate_and_refuses_to_be_a_mass():
    p = load_fuel_params(FUEL_CONFIG)
    assert p.absolute_rate is None, "the shipped config must carry no absolute rate"
    #  the factor is 1.0 in open water by definition and rises with ice
    assert resistance_factor(0.0, p) == 1.0
    assert resistance_factor(100.0, p) == p.open_water_factor + p.ice_penalty
    assert resistance_factor(50.0, p) < resistance_factor(75.0, p)
    #  a missing concentration is never read as ice-free
    assert math.isnan(float(resistance_factor(float("nan"), p)))
    #  and a value off the percentage scale is refused, not guessed
    for bad in (-1.0, 120.0):
        try:
            resistance_factor(bad, p)
        except FuelError:
            continue
        raise AssertionError(f"sic {bad} was accepted")
    #  the degenerate case that would make fuel a distance is refused
    try:
        FuelParams(ice_penalty=0.0).validate()
    except FuelError as exc:
        assert "identical to travelled distance" in str(exc)
    else:
        raise AssertionError("ice_penalty = 0 was accepted")
    r = route_fuel([(6, 0), (6, 1)], [0.0, 2000.0],
                   sic_for_bucket=lambda b: np.zeros(ICE_BAND_SHAPE),
                   bucket_of=lambda t: 0, pixel_size=(PX, PX), params=p)
    prov = r.provenance()
    assert prov["status"] == "estimated"
    assert prov["is_a_measured_fuel_consumption"] is False
    assert prov["is_an_operational_fuel_prediction"] is False
    assert prov["units"] == UNITS
    assert prov["suitable_for_operational_fuel_prediction"] is False
    try:
        r.to_mass()
    except FuelError as exc:
        assert "not a mass or volume" in str(exc)
    else:
        raise AssertionError("a mass was produced without an absolute rate")
    return (f"factor 1.0 open water -> {resistance_factor(100.0, p):.1f} at 100% "
            f"ice; no absolute rate, so a mass is refused, not defaulted")


# ------------------------------------------- 2. the objectives are distinct
def test_each_objective_minimises_its_own_quantity():
    """1, 2 and 3: each objective wins on its OWN metric and no other."""
    cmp = ice_band()
    r = {n: cmp.profiles[n] for n in ALL_PROFILES}
    for n in ALL_PROFILES:
        assert r[n].success, (n, r[n].outcome, r[n].reason)
    best_time = min(x.travel_time_s for x in r.values())
    best_fuel = min(x.estimated_fuel for x in r.values())
    best_cost = min(x.configured_cost for x in r.values())
    best_dist = min(x.distance_m for x in r.values())
    assert math.isclose(r[PROFILE_FASTEST].travel_time_s, best_time, rel_tol=1e-12)
    assert math.isclose(r[PROFILE_FUEL].estimated_fuel, best_fuel, rel_tol=1e-12)
    assert math.isclose(r[PROFILE_RISK].configured_cost, best_cost, rel_tol=1e-12)
    assert math.isclose(r[PROFILE_SHORTEST].distance_m, best_dist, rel_tol=1e-12)
    return ("; ".join(f"{n} {r[n].objective_value:,.1f} {r[n].objective_units}"
                      for n in ALL_PROFILES))


def test_fuel_efficient_is_not_shortest_distance():
    """4: it takes a LONGER path to spend less fuel, which a distance
    objective can never do."""
    cmp = ice_band()
    fuel, short = cmp.profiles[PROFILE_FUEL], cmp.profiles[PROFILE_SHORTEST]
    assert fuel.path != short.path, "the two objectives returned the same path"
    assert fuel.distance_m > short.distance_m, (
        "the fuel route is not longer, so this does not yet prove it is not a "
        "distance objective")
    assert fuel.estimated_fuel < short.estimated_fuel
    assert cmp.provenance["fuel_efficient_is_shortest_distance"] is False
    assert cmp.provenance["fuel_efficient_and_shortest_distance_coincided"] is False
    assert OBJECTIVES[PROFILE_FUEL]["is_the_same_as_shortest_distance"] is False
    return (f"fuel route is {fuel.distance_m / short.distance_m - 1:+.1%} longer "
            f"and {fuel.estimated_fuel / short.estimated_fuel - 1:+.1%} in fuel")


def test_they_coincide_on_uniform_ice_and_the_reason_is_recorded():
    """The honest converse: where the concentration does not vary, the factor
    is a constant multiplier and the two objectives order paths identically.
    That is a fact about the field, not an aliasing of the objectives."""
    cmp = uniform_ice()
    fuel, short = cmp.profiles[PROFILE_FUEL], cmp.profiles[PROFILE_SHORTEST]
    assert fuel.path == short.path
    assert cmp.provenance["fuel_efficient_and_shortest_distance_coincided"] is True
    assert "uniform" in cmp.provenance["why_they_can_coincide_on_uniform_ice"]
    #  and the fuel is still the distance times the constant factor
    f = resistance_factor(40.0, load_fuel_params(FUEL_CONFIG))
    assert math.isclose(fuel.estimated_fuel, fuel.distance_m * f, rel_tol=1e-12)
    return ("on a constant 40% field the two coincide, and the comparison says "
            "why rather than claiming they are the same objective")


# ------------------------------------------------- 5. reproducibility
def test_route_fuel_is_reproducible_from_the_returned_path():
    """5: re-evaluating the returned path independently reproduces the number,
    and the fuel objective's own search cost IS that number."""
    cmp = ice_band()
    sic, params = ice_band_sic(), load_fuel_params(FUEL_CONFIG)
    bucket_seconds = cmp.provenance["bucket_seconds"]
    for name in ALL_PROFILES:
        r = cmp.profiles[name]
        again = route_fuel(
            r.path, r.arrival_times, sic_for_bucket=lambda b: sic,
            bucket_of=lambda t: int(math.floor(t / bucket_seconds)),
            pixel_size=(PX, PX), params=params)
        assert math.isclose(again.estimated_fuel, r.estimated_fuel, rel_tol=1e-12), \
            (name, again.estimated_fuel, r.estimated_fuel)
        assert math.isclose(
            again.fuel_per_km, r.estimated_fuel_per_km, rel_tol=1e-12)
    #  the fuel objective minimised exactly the quantity it reports
    f = cmp.profiles[PROFILE_FUEL]
    assert math.isclose(f.objective_value, f.estimated_fuel, rel_tol=1e-12), (
        "the fuel objective's search cost is not the fuel it reports")
    #  and the fuel of a DIFFERENT path is a different number, so the check
    #  above is not trivially satisfied by any path at all
    other = cmp.profiles[PROFILE_SHORTEST]
    assert not math.isclose(f.estimated_fuel, other.estimated_fuel, rel_tol=1e-9)
    return ("all four routes re-evaluate to their reported fuel; the fuel "
            "objective's search cost equals its reported fuel exactly")


# --------------------------------------- 6. a fuel parameter moves only fuel
def test_changing_a_fuel_parameter_moves_only_fuel():
    """6: a different ice_penalty changes the fuel numbers and can change the
    fuel route, and changes nothing about POLARIS or the iceberg layer."""
    base = ice_band()
    hotter = _comparison(ice_band_sic(), FuelParams(ice_penalty=8.0,
                                                    ice_exponent=1.5))
    for name in (PROFILE_FASTEST, PROFILE_RISK, PROFILE_SHORTEST):
        a, b = base.profiles[name], hotter.profiles[name]
        assert a.path == b.path, f"{name} moved when a FUEL parameter changed"
        assert math.isclose(a.configured_cost, b.configured_cost, rel_tol=1e-12)
        assert math.isclose(a.polaris_contribution, b.polaris_contribution,
                            rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(a.iceberg_exposure_contribution,
                            b.iceberg_exposure_contribution,
                            rel_tol=1e-12, abs_tol=1e-12)
        assert a.max_iceberg_exposure == b.max_iceberg_exposure
        #  their fuel figures DO move, because the metric itself changed
        assert not math.isclose(a.estimated_fuel, b.estimated_fuel, rel_tol=1e-9)
    assert hotter.provenance["estimated_fuel_parameters"]["ice_penalty"] == 8.0
    assert base.provenance["estimated_fuel_parameters"]["ice_penalty"] == 2.0
    #  POLARIS and iceberg provenance are untouched by the fuel parameter
    for key in ("iceberg_exposure_weight", "polaris_ice_class",
                "polaris_riv_table", "buckets", "bucket_seconds"):
        assert base.provenance[key] == hotter.provenance[key], key
    return ("ice_penalty 2.0 -> 8.0 moves every fuel figure and no POLARIS, "
            "iceberg, cost or path of the other three objectives")


def test_the_fuel_parameter_break_even_is_measured_not_asserted():
    """The shipped ice_penalty is an assumption, so the value at which the fuel
    route stops agreeing with the distance route is MEASURED and reported
    rather than left as an unexamined magic number."""
    lo, hi = 0.01, 64.0
    short = _comparison(ice_band_sic(),
                        FuelParams(ice_penalty=lo)).profiles[PROFILE_SHORTEST]

    def agrees(penalty: float) -> bool:
        c = _comparison(ice_band_sic(), FuelParams(ice_penalty=penalty))
        return c.profiles[PROFILE_FUEL].path == short.path

    assert agrees(lo), "even a vanishing penalty already diverts the route"
    assert not agrees(hi), "even a huge penalty never diverts the route"
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if agrees(mid):
            lo = mid
        else:
            hi = mid
    shipped = load_fuel_params(FUEL_CONFIG).ice_penalty
    assert lo < shipped, (
        f"the shipped ice_penalty {shipped} sits below the break-even {lo:.4f}, "
        f"so the fuel objective is a distance objective in this fixture")
    return (f"break-even ice_penalty on this fixture is {lo:.4f}-{hi:.4f}; the "
            f"shipped {shipped} is above it, so the objective genuinely diverts")


# ------------------------------------- 7, 8. the existing objectives are intact
def test_the_existing_three_objectives_are_unchanged_by_fuel_being_absent():
    """7 and 8: with no sea-ice field the comparison is exactly the old one --
    three profiles, no fuel, and the absence is stated rather than filled in."""
    cmp = _comparison(sic=None)
    assert set(cmp.profiles) == set(PROFILES), sorted(cmp.profiles)
    assert PROFILE_FUEL not in cmp.profiles
    assert cmp.provenance["estimated_fuel_available"] is False
    assert "will not substitute one" in cmp.provenance["why_no_estimated_fuel"]
    for name in PROFILES:
        r = cmp.profiles[name]
        assert r.estimated_fuel is None and r.fuel_provenance == {}
        assert r.success
    #  and against the run WITH fuel, the other three chose the same routes
    withfuel = ice_band()
    for name in PROFILES:
        a, b = cmp.profiles[name], withfuel.profiles[name]
        assert a.path == b.path, f"{name} moved when the fuel profile was added"
        assert math.isclose(a.objective_value, b.objective_value, rel_tol=1e-12)
        assert math.isclose(a.configured_cost, b.configured_cost, rel_tol=1e-12)
    return ("without a sea-ice field: three profiles, fuel absent and declared "
            "absent; with one: the same three routes, plus a fourth")


# -------------------------------------------- the claims the module may make
def test_no_unsupported_fuel_claim_is_made_anywhere():
    """The fuel vocabulary is now allowed, but only as an ESTIMATE. No output
    may claim a measured consumption, an operational prediction, a saving, or
    any mass or volume unit."""
    cmp = ice_band()
    d = json.loads(json.dumps(cmp.to_dict(), default=str)).__str__().lower()
    banned = ("litre", "liter", "tonne", "gallon", "kilogram", " kg ",
              "fuel saving", "saves fuel", "measured fuel", "actual fuel",
              "fuel consumption of", "bunker", "best route", "safest",
              "recommended route", "optimal route")
    for word in banned:
        assert word not in d, f"{word!r} appears in the serialised comparison"
    for key, want in (("estimated_fuel_is_measured_consumption", False),
                      ("estimated_fuel_is_an_operational_prediction", False),
                      ("estimated_fuel_suitable_for_operational_prediction", False),
                      ("estimated_fuel_suitable_for_relative_comparison", True),
                      ("this_is_not_a_ranking", True)):
        assert cmp.provenance[key] is want, key
    obj = OBJECTIVES[PROFILE_FUEL]
    assert obj["is_a_fuel_or_consumption_figure"] is True
    assert obj["is_a_measured_fuel_consumption"] is False
    assert obj["is_an_operational_fuel_prediction"] is False
    assert cmp.profiles[PROFILE_FUEL].estimated_fuel_units == UNITS
    return ("the comparison names no mass, volume, saving or measured "
            "consumption; every fuel claim key is explicitly False")


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

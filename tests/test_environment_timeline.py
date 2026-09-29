"""
Which environmental field each routing bucket is priced with, and proof that
the choice reaches the route.

Two halves:

  * the planner itself (src/api/environment_timeline.py) -- pure, no archive,
    no rasters: window arithmetic, the persistence and dated-observation
    policies, what happens when a required date is absent.

  * integration over the UNMODIFIED src/routing/time_astar.py: the same
    mission, the same grid, the same solver, run once with a time-varying
    environment and once with the persistence timeline. The routes differ, and
    the only thing that differs between the two runs is which environmental
    field each bucket was built from.

Every composed field here comes from the real navigation_cost.compose_grid over
a small synthetic USNIC raster set, so no cost formula is restated in the test.
The synthetic values are a TEST FIXTURE and are never read by production code.

    python tests/test_environment_timeline.py
"""
from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.environment_timeline import (ON_MISSING_ERROR,                   # noqa: E402
                                          ON_MISSING_PERSIST,
                                          POLICY_DATED_OBSERVATION,
                                          POLICY_PERSISTENCE,
                                          EnvironmentTimelineError,
                                          EnvironmentUnavailable,
                                          plan_environment_timeline,
                                          timeline_summary)
from src.data.rasterize_sigrid3 import (AMBIGUITY_TIF, COVERAGE_TIF,          # noqa: E402
                                        DOMINANT_FRACTION_TIF, FRACTION_NODATA,
                                        POLYGON_ID_NODATA, POLYGON_ID_TIF,
                                        WATER_TIF)
from src.routing.navigation_cost import (compose_grid,                        # noqa: E402
                                         load_composition_config)
from src.routing.time_astar import time_astar                                 # noqa: E402
from src.routing.time_navigation_cost import (ComposedField,                  # noqa: E402
                                              TimeIndexedNavigationCost)

CONFIG = ROOT / "configs" / "navigation_cost_composition.json"

EPSG = 3976
PX = 10_000.0                     # 10 km cells -> tidy hours
HOUR = 3600.0
SPEED = PX / HOUR                 # exactly one cell per hour orthogonally
TF = (PX, 0.0, 0.0, 0.0, -PX, 0.0)
BUCKET_S = 6 * HOUR

DEPARTURE = datetime(2025, 1, 8, 0, 0)
DAY = DEPARTURE.date()
NEXT_DAY = DAY + timedelta(days=1)
CHART = date(2025, 1, 8)


# =================================================== the planner, on its own
def test_bucket_windows_are_departure_plus_elapsed_time():
    """window(b) = [b*bucket, (b+1)*bucket) after departure, and the cost
    adapter's own bucket_for maps every instant in that window back to b."""
    timeline = plan_environment_timeline(
        buckets=[0, 1, 2], bucket_seconds=BUCKET_S, departure_time=DEPARTURE,
        available_dates=[DAY])
    seen = []
    for entry in timeline:
        assert entry.valid_from_s == entry.bucket * BUCKET_S
        assert entry.valid_to_s == (entry.bucket + 1) * BUCKET_S
        assert entry.valid_from == DEPARTURE + timedelta(seconds=entry.valid_from_s)
        assert entry.valid_to == DEPARTURE + timedelta(seconds=entry.valid_to_s)
        assert entry.horizon_hours == entry.bucket * 6.0
        seen.append(f"b{entry.bucket}@+{entry.horizon_hours:g}h")

    #  the two layers must agree about the clock, not merely look similar
    probe = TimeIndexedNavigationCost(fields={}, bucket_seconds=BUCKET_S)
    for entry in timeline:
        for t in (entry.valid_from_s, entry.valid_from_s + 1.0,
                  entry.valid_to_s - 1e-6):
            assert probe.bucket_for(t) == entry.bucket, (entry.bucket, t)
    return "windows " + " ".join(seen) + "; bucket_for agrees at every edge"


def test_persistence_carries_the_analysis_forward_and_says_so():
    """A bucket whose window opens on a later date is served the departure
    analysis -- and is marked as carried forward rather than left silent."""
    timeline = plan_environment_timeline(
        buckets=[0, 1, 4, 5], bucket_seconds=BUCKET_S, departure_time=DEPARTURE,
        policy=POLICY_PERSISTENCE, available_dates=[DAY, NEXT_DAY])
    inside = [e for e in timeline if e.bucket in (0, 1)]
    beyond = [e for e in timeline if e.bucket in (4, 5)]

    for e in inside:
        assert e.environment_date == DAY
        assert e.fallback is None and e.persisted is False
    for e in beyond:
        assert e.valid_from.date() == NEXT_DAY
        assert e.environment_date == DAY, "persistence must not jump forward"
        assert e.persisted is True
        assert e.fallback == f"persisted_from_{DAY.isoformat()}"
        #  the later observation exists in the archive and was still not used:
        #  that is the policy, stated, not an accident
        assert e.uses_future_observation is False
    return (f"buckets 0,1 inside {DAY} carry nothing forward; buckets 4,5 open "
            f"on {NEXT_DAY} and are stamped {beyond[0].fallback}")


def test_the_shipped_demo_window_persists_nothing():
    """Two six-hour buckets from midnight close inside the analysis day, so the
    shipped persistence policy is exact there and records no fallback."""
    timeline = plan_environment_timeline(
        buckets=[0, 1], bucket_seconds=BUCKET_S, departure_time=DEPARTURE,
        available_dates=[DAY])
    assert [e.fallback for e in timeline] == [None, None]
    assert not any(e.persisted for e in timeline)
    summary = timeline_summary(timeline)
    assert summary["environment_is_time_varying"] is False
    assert summary["environment_persisted_buckets"] == []
    return (f"{summary['policy']}: one field, {summary['environment_dates']}, "
            f"declared rather than assumed")


def test_dated_policy_gives_each_bucket_its_own_observation():
    timeline = plan_environment_timeline(
        buckets=[0, 4], bucket_seconds=BUCKET_S, departure_time=DEPARTURE,
        policy=POLICY_DATED_OBSERVATION, available_dates=[DAY, NEXT_DAY])
    first, later = timeline
    assert first.environment_date == DAY
    assert first.uses_future_observation is False
    assert later.environment_date == NEXT_DAY
    assert later.fallback is None and later.persisted is False
    #  an observation from after the analysis is perfect foresight, and the
    #  entry says so rather than calling itself a forecast
    assert later.uses_future_observation is True
    summary = timeline_summary(timeline)
    assert summary["environment_is_time_varying"] is True
    assert summary["uses_future_observations"] is True
    assert "forecast" not in " ".join(e.source for e in timeline)
    return (f"bucket 0 -> {first.source}, bucket 4 -> {later.source} "
            f"[flagged as an observation from after the analysis]")


def test_a_missing_environment_is_refused_not_substituted():
    try:
        plan_environment_timeline(
            buckets=[0, 4], bucket_seconds=BUCKET_S, departure_time=DEPARTURE,
            policy=POLICY_DATED_OBSERVATION, available_dates=[DAY],
            on_missing=ON_MISSING_ERROR)
    except EnvironmentUnavailable as exc:
        assert NEXT_DAY.isoformat() in str(exc)
        assert "Nothing is substituted" in str(exc)
        return f"refused: {str(exc).splitlines()[0][:96]}"
    raise AssertionError("a missing environment date was substituted silently")


def test_the_fallback_is_available_but_must_be_asked_for():
    timeline = plan_environment_timeline(
        buckets=[0, 4], bucket_seconds=BUCKET_S, departure_time=DEPARTURE,
        policy=POLICY_DATED_OBSERVATION, available_dates=[DAY],
        on_missing=ON_MISSING_PERSIST)
    later = timeline[-1]
    assert later.environment_date == DAY
    assert later.persisted is True
    assert later.fallback == f"persisted_from_{DAY.isoformat()}"
    assert timeline_summary(timeline)["environment_persisted_buckets"] == [4]
    return ("the documented fallback is opt-in and is recorded on the bucket "
            "that used it")


def test_invalid_specifications_are_refused():
    bad = [
        dict(policy="whatever_seems_reasonable"),
        dict(on_missing="guess"),
        dict(bucket_seconds=0.0),
        dict(bucket_seconds=float("nan")),
        dict(buckets=[]),
        dict(buckets=[-1, 0]),
        dict(departure_time="2025-01-08"),
    ]
    for over in bad:
        kw = dict(buckets=[0, 1], bucket_seconds=BUCKET_S,
                  departure_time=DEPARTURE, available_dates=[DAY])
        kw.update(over)
        try:
            plan_environment_timeline(**kw)
        except EnvironmentTimelineError:
            continue
        raise AssertionError(f"should have refused {over}")
    return f"{len(bad)} malformed specifications refused"


# ====================================== integration: does the route follow it?
#  A 7 x 13 grid with a wall that leaves two gaps. Corridor A is cheap in the
#  environment of the departure day and dear in the next day's; corridor B is
#  the reverse. The gaps sit far enough down-track that the vessel cannot
#  reach either of them before the second bucket opens, so a router that
#  prices each cell at its ARRIVAL time takes B, and a router frozen on the
#  departure analysis takes A.
BIG = (7, 13)
START, GOAL = (3, 0), (3, 12)
GAP_COL = 8
WALL = [(r, GAP_COL) for r in range(BIG[0]) if r not in (1, 5)]
A_CELLS = [(1, GAP_COL)]            # the northern gap
B_CELLS = [(5, GAP_COL)]            # the southern gap

_DIRS: dict = {}


def _usnic_dir(key: str, height: int, width: int, layout: dict) -> Path:
    if key in _DIRS:
        return _DIRS[key]
    d = Path(tempfile.mkdtemp(prefix=f"usnic_{key}_"))
    prof = dict(driver="GTiff", height=height, width=width, count=1,
                crs=CRS.from_epsg(EPSG), transform=Affine(*TF))
    for name, k, dt, nd in ((COVERAGE_TIF, "coverage", "uint8", None),
                            (POLYGON_ID_TIF, "polygon_id", "int32", POLYGON_ID_NODATA),
                            (WATER_TIF, "water", "uint8", None),
                            (AMBIGUITY_TIF, "ambiguity", "uint8", None),
                            (DOMINANT_FRACTION_TIF, "fraction", "float32",
                             FRACTION_NODATA)):
        with rasterio.open(d / name, "w", dtype=dt, nodata=nd, **prof) as dst:
            dst.write(np.asarray(layout[k], dtype=dt), 1)
    _DIRS[key] = d
    return d


#  entirely uncharted: no POLARIS polygon, so the composed cost IS the
#  environmental term and this test cannot accidentally be about POLARIS
BLANK = dict(coverage=np.zeros(BIG, "uint8"),
             polygon_id=np.full(BIG, POLYGON_ID_NODATA, "int32"),
             water=np.zeros(BIG, "uint8"),
             ambiguity=np.zeros(BIG, "uint8"),
             fraction=np.full(BIG, FRACTION_NODATA, "float32"))

#  TEST FIXTURE VALUES. Not sea-ice concentrations, not read by production code.
CHEAP, DEAR = 1.0, 60.0


def _environment(cheap_corridor) -> np.ndarray:
    env = np.full(BIG, CHEAP, dtype="float64")
    for cells, value in ((A_CELLS, CHEAP if cheap_corridor == "A" else DEAR),
                         (B_CELLS, CHEAP if cheap_corridor == "B" else DEAR)):
        for (r, c) in cells:
            env[r, c] = value
    return env


#  the two environmental states this fixture switches between
ENVIRONMENTS = {DAY: _environment("A"), NEXT_DAY: _environment("B")}


def _fields_from(timeline) -> list[ComposedField]:
    """One composed field per bucket, built from the environment the TIMELINE
    resolved for that bucket. This is the production wiring in miniature."""
    usnic = _usnic_dir("corridor", BIG[0], BIG[1], BLANK)
    cfg = load_composition_config(CONFIG)
    out = []
    for entry in timeline:
        comp = compose_grid(ENVIRONMENTS[entry.environment_date], {}, cfg, usnic)
        out.append(ComposedField(
            bucket=entry.bucket, composition=comp, environment_date=DAY,
            chart_date=CHART, epsg=EPSG, shape=BIG, transform=TF,
            label=f"bucket {entry.bucket}", environment=entry.to_dict()))
    return out


def _provider(timeline) -> TimeIndexedNavigationCost:
    return TimeIndexedNavigationCost.from_fields(
        _fields_from(timeline), bucket_seconds=BUCKET_S)


def _grid():
    from src.routing.grid import RoutingGrid
    blocked = np.zeros(BIG, bool)
    for (r, c) in WALL:
        blocked[r, c] = True
    z = np.zeros(BIG, "float32")
    return RoutingGrid(transform=Affine(*TF), crs=CRS.from_epsg(EPSG), shape=BIG,
                       sic=z.copy(), current_u=z.copy(), current_v=z.copy(),
                       blocked_mask=blocked, day=DAY)


#  every mission input, fixed in one place so the two runs cannot differ in
#  anything except the environment timeline
MISSION = dict(start=START, goal=GOAL, vessel_speed_mps=SPEED, time_step_s=HOUR)


def _run(provider):
    return time_astar(_grid(), MISSION["start"], MISSION["goal"],
                      cost_fn=provider.cost_fn("conservative"),
                      min_cost=provider.min_cost("conservative"),
                      vessel_speed_mps=MISSION["vessel_speed_mps"],
                      time_step_s=MISSION["time_step_s"],
                      max_horizon_s=provider.horizon_s)


#  An 18:00 departure puts bucket 0 inside the analysis day and buckets 1 and 2
#  on the next one, which is what lets a DAILY archive change under a route.
EVENING = datetime(2025, 1, 8, 18, 0)


def _timelines():
    kw = dict(buckets=[0, 1, 2], bucket_seconds=BUCKET_S,
              departure_time=EVENING, analysis_date=DAY,
              available_dates=[DAY, NEXT_DAY])
    return (plan_environment_timeline(policy=POLICY_DATED_OBSERVATION, **kw),
            plan_environment_timeline(policy=POLICY_PERSISTENCE, **kw))


def test_a_different_future_environment_gives_a_different_route():
    """THE REGRESSION. Same mission, same grid, same solver: only the
    environment each bucket resolves to differs, and the route differs with it.

    This fails if the router ever reverts to pricing every arrival time against
    one static environmental field.
    """
    varying, static = _timelines()
    #  the two timelines really do differ, and only in the environment
    assert [e.environment_date for e in varying] == [DAY, NEXT_DAY, NEXT_DAY]
    assert [e.environment_date for e in static] == [DAY, DAY, DAY]
    assert timeline_summary(varying)["environment_is_time_varying"] is True
    assert timeline_summary(static)["environment_is_time_varying"] is False

    hot = _run(_provider(varying))
    cold = _run(_provider(static))
    assert hot.success and cold.success, (hot.reason, cold.reason)

    assert set(cold.path) & set(A_CELLS), f"static route left corridor A: {cold.path}"
    assert not (set(cold.path) & set(B_CELLS))
    assert set(hot.path) & set(B_CELLS), f"time-varying route missed B: {hot.path}"
    assert not (set(hot.path) & set(A_CELLS))
    assert hot.path != cold.path

    #  and the difference is the environment, not the mission: every input the
    #  search was given is identical between the two runs
    assert hot.start == cold.start and hot.goal == cold.goal
    assert hot.start_time == cold.start_time
    assert len(hot.path) == len(cold.path), "same geometry length, other corridor"

    #  what the persistence plan would ACTUALLY have cost, priced cell by cell
    #  at the times the ship reaches them under the time-varying environment.
    #  The two corridors are symmetric, so the planned totals tie -- the cost
    #  landscape they were chosen from is what differs.
    fn = _provider(varying).cost_fn("conservative")
    repriced = 0.0
    for (a, ta), (b, tb) in zip(zip(cold.path, cold.arrival_times),
                                zip(cold.path[1:], cold.arrival_times[1:])):
        d = float(np.hypot((b[1] - a[1]) * PX, (b[0] - a[0]) * PX))
        repriced += d * 0.5 * (fn(a[0], a[1], ta) + fn(b[0], b[1], tb))
    assert repriced > hot.total_cost, (repriced, hot.total_cost)
    return (f"time-varying -> corridor B (cost {hot.total_cost:,.0f}), "
            f"persistence -> corridor A (planned {cold.total_cost:,.0f}, but "
            f"{repriced:,.0f} once each cell is priced on arrival); "
            f"identical mission inputs")


def test_each_cell_is_priced_from_the_field_its_bucket_resolved_to():
    varying, _ = _timelines()
    provider = _provider(varying)
    res = _run(provider)
    assert res.success, res.reason
    by_bucket = {e.bucket: e for e in varying}

    seen = []
    for (r, c), t in zip(res.path, res.arrival_times):
        bucket = provider.bucket_for(t)
        entry = by_bucket[bucket]
        paid = float(provider.field_for(t).cost("conservative")[r, c])
        expected = float(ENVIRONMENTS[entry.environment_date][r, c])
        assert paid == expected, ((r, c), t, paid, expected)
        if (r, c) in B_CELLS:
            #  the corridor cell was entered in the later bucket and paid the
            #  later field's price, not the departure analysis's
            assert bucket >= 1 and entry.environment_date == NEXT_DAY
            assert paid == CHEAP
            assert float(ENVIRONMENTS[DAY][r, c]) == DEAR
            seen.append(f"{(r, c)}@{t / HOUR:.2f}h -> bucket {bucket} "
                        f"({entry.source}) = {paid:g}")
    assert seen, "no corridor cell on the path"
    return ("arrival time selects the field: " + "; ".join(seen)
            + f"  (the departure analysis would have charged {DEAR:g})")


def test_the_fields_carry_their_resolution_into_the_provenance():
    varying, static = _timelines()
    for timeline, varying_expected in ((varying, True), (static, False)):
        prov = _provider(timeline).provenance()
        recorded = {int(b): v["environment"] for b, v in prov["fields"].items()}
        assert all(recorded.values()), "a field recorded no environment"
        dates = {e["environment_date"] for e in recorded.values()}
        assert (len(dates) > 1) is varying_expected
        for bucket, entry in recorded.items():
            assert entry["bucket"] == bucket
            assert entry["source"].startswith("observation:")
            assert entry["policy"] in (POLICY_PERSISTENCE, POLICY_DATED_OBSERVATION)
    return "every bucket's field reports the environment it was built from"


def test_an_unpriceable_bucket_is_still_refused():
    """The timeline decides WHICH field a bucket gets; it never invents one.
    A bucket with no field at all is still the cost adapter's refusal."""
    varying, _ = _timelines()
    one_bucket = TimeIndexedNavigationCost.from_fields(
        [_fields_from(varying)[0]], bucket_seconds=BUCKET_S)
    res = _run(one_bucket)
    assert not res.success, "a route was returned past the last priced bucket"
    assert "horizon" in res.reason.lower() or "unreachable" in res.reason.lower()
    return f"refused rather than extrapolated: {res.reason[:88]}"


# ------------------------------------------------------------------ runner
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

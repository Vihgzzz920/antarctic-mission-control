"""
The temporal contract between forecast artifacts and routing.

Three things are proved here:

  * the LEAD REGISTRY says what the archive can honestly support, and the
    numbers behind it are recounted from the archive rather than trusted;
  * the RESOLVER answers an arrival time with exactly one of
    observed_analysis / model_forecast / explicit_persistence / unavailable,
    never with a field stretched to a window it does not cover;
  * the ROUTING TIMELINE consumes that answer, and a bucket served by a
    legitimate forecast produces different time-dependent edge costs from the
    same bucket served by persistence.

The corridor grid and its cost values are a TEST FIXTURE. They are never read
by production code. The decision about WHICH source each bucket gets is made by
the real resolver against the real artifact.

    python tests/test_forecast_temporal_architecture.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.environment_forecast import (LEAD_SUPPORT, MODEL_NAME,           # noqa: E402
                                          RESOLVE_FORECAST_ELSE_PERSISTENCE,
                                          RESOLVE_FORECAST_ONLY,
                                          SOURCE_FORECAST, SOURCE_OBSERVED,
                                          SOURCE_PERSISTENCE,
                                          SOURCE_UNAVAILABLE,
                                          STATUS_DATA_ONLY, STATUS_MODELLED,
                                          STATUS_NO_DATA, STATUS_OBSERVATION,
                                          SUPPORTED_LEAD_HOURS,
                                          VALIDITY_EXACT, VALIDITY_RULES,
                                          VALIDITY_TARGET_DAY,
                                          EnvironmentForecastError,
                                          lead_support, lead_support_table,
                                          read_environment,
                                          resolve_environment,
                                          validity_interval)
from src.api.environment_timeline import (ON_MISSING_ERROR,                   # noqa: E402
                                          ON_MISSING_PERSIST,
                                          POLICY_MODEL_FORECAST,
                                          POLICY_PERSISTENCE,
                                          EnvironmentTimelineError,
                                          EnvironmentUnavailable,
                                          plan_environment_timeline,
                                          timeline_summary)
from src.data.preprocess import RAW_DIR                                       # noqa: E402
from src.models.build_forecast_dataset import CURRENTS_DIR                    # noqa: E402
from src.models.forecast_sic_raster import (LEAD_HOURS, predict_sic,          # noqa: E402
                                            raster_path,
                                            write_forecast_raster)
from src.routing.navigation_cost import (compose_grid,                        # noqa: E402
                                         load_composition_config)
from src.routing.time_astar import time_astar                                 # noqa: E402
from src.routing.time_navigation_cost import (ComposedField,                  # noqa: E402
                                              TimeIndexedNavigationCost)

ORIGIN_DAY = date(2025, 1, 8)
ORIGIN = datetime(2025, 1, 8)
BUCKET_S = 6 * 3600.0


def artifact() -> Path:
    path = raster_path(ORIGIN_DAY)
    if not path.exists():
        write_forecast_raster(predict_sic(ORIGIN_DAY))
    return path


def ask(hours_from, hours_to=None, **kw):
    return resolve_environment(
        ORIGIN + timedelta(hours=hours_from),
        None if hours_to is None else ORIGIN + timedelta(hours=hours_to),
        origin_time=ORIGIN, **kw)


# =================================================== 1. the lead registry
def test_the_supported_leads_are_explicit_and_match_the_archive():
    """The registry's claims are recounted from the files on disk."""
    sic = {p.name[4:12] for p in RAW_DIR.glob("sic_????????.tif")}
    cur = {p.name[9:17] for p in CURRENTS_DIR.glob("currents_????????.tif")}
    D = lambda s: datetime.strptime(s, "%Y%m%d").date()          # noqa: E731
    sic_days, cur_days = set(map(D, sic)), set(map(D, cur))
    pairs = {days: sum(1 for d in sic_days & cur_days
                       if d + timedelta(days=days) in sic_days)
             for days in (1, 2)}

    table = {row["lead_hours"]: row for row in lead_support_table()}
    assert table[0]["status"] == STATUS_OBSERVATION
    #  no sub-daily sea-ice observation exists, so no 6 h or 12 h target can
    for lead in (6, 12):
        assert table[lead]["status"] == STATUS_NO_DATA
        assert table[lead]["data_supported"] is False
        assert table[lead]["model"] is None
        assert table[lead]["routable"] is False
    assert table[24]["status"] == STATUS_MODELLED
    assert table[24]["model"] == MODEL_NAME and table[24]["routable"] is True
    assert pairs[1] > 0, "the registry claims 24 h pairs exist; none were found"
    #  48 h: the data supports it AND a model has now been trained for it
    assert table[48]["status"] == STATUS_MODELLED
    assert table[48]["data_supported"] is True and table[48]["model"] == MODEL_NAME
    assert table[48]["routable"] is True
    assert pairs[2] > 0, "the registry claims 48 h pairs exist; none were found"
    #  and STATUS_DATA_ONLY still exists as a distinct verdict for any future
    #  lead whose targets exist before its model does
    assert STATUS_DATA_ONLY != STATUS_MODELLED

    assert tuple(SUPPORTED_LEAD_HOURS) == (24, 48)
    assert LEAD_HOURS == 24, "the default lead moved"
    return (f"archive: {pairs[1]} true +24 h pairs, {pairs[2]} true +48 h "
            f"pairs, 0 sub-daily observations -> modelled {sorted(SUPPORTED_LEAD_HOURS)}")


def test_an_unregistered_lead_is_not_quietly_assumed():
    assert lead_support(9) is None and lead_support(72) is None
    assert lead_support(24) is not None
    return "a lead the project never considered returns None, not a default"


def test_the_dataset_builder_supports_whole_day_leads_and_nothing_finer():
    """What it would take to add +48 h: a dataset at that lead, which the
    builder can now produce because the targets genuinely exist."""
    from src.models.build_forecast_dataset import (SOURCE_CADENCE_DAYS,
                                                   list_pairs)
    assert SOURCE_CADENCE_DAYS == 1
    counts = {}
    for lead in (1, 2):
        pairs = list_pairs(RAW_DIR, CURRENTS_DIR, 0, lead)
        assert pairs, f"no {lead}-day pairs in the archive"
        assert all((t - b).days == lead for b, t, _ in pairs)
        counts[lead] = len(pairs)
    #  the default is byte-for-byte the shipped behaviour
    assert list_pairs(RAW_DIR, CURRENTS_DIR, 0) == list_pairs(RAW_DIR,
                                                              CURRENTS_DIR, 0, 1)
    #  and a sub-daily lead cannot even be expressed
    for bad in (0, -1, 0.25, "6h"):
        try:
            list_pairs(RAW_DIR, CURRENTS_DIR, 0, bad)
        except ValueError as exc:
            assert "whole number of days" in str(exc)
            continue
        raise AssertionError(f"accepted lead_days={bad!r}")
    return (f"+24 h: {counts[1]} pairs, +48 h: {counts[2]} pairs; sub-daily "
            f"leads cannot be requested at all")


# ======================================== 2. validity intervals, not guesses
def test_validity_intervals_are_derived_and_named():
    valid = ORIGIN + timedelta(hours=24)
    point = validity_interval(valid, VALIDITY_EXACT)
    assert point == (valid, valid), point
    day = validity_interval(valid, VALIDITY_TARGET_DAY)
    assert day == (datetime(2025, 1, 9), datetime(2025, 1, 10)), day
    try:
        validity_interval(valid, "whatever_covers_it")
    except EnvironmentForecastError as exc:
        assert "No interval is invented" in str(exc)
        return (f"{VALIDITY_EXACT} -> a point; {VALIDITY_TARGET_DAY} -> "
                f"[{day[0].date()}, {day[1].date()}); unknown rules are refused")
    raise AssertionError("an unknown validity rule produced an interval")


def test_a_point_valid_forecast_cannot_serve_a_six_hour_bucket():
    """THE IMPORTANT QUESTION, under the shipped default rule."""
    artifact()
    exact = ask(24)
    assert exact.source_type == SOURCE_FORECAST, exact.reason
    for window in ((18, 24), (24, 30), (0, 6), (6, 12)):
        answer = ask(*window)
        assert answer.source_type == SOURCE_UNAVAILABLE, (window, answer.reason)
        assert answer.path is None
        assert "never stretched" in answer.reason or "registered as" in answer.reason
    return ("+24h resolves at the instant only; buckets 0-6, 6-12, 18-24 and "
            "24-30 h are all refused under exact_valid_time")


def test_the_composite_day_reading_covers_only_buckets_inside_the_target_day():
    """The second, documented reading: a bucket must lie ENTIRELY inside the
    target composite day of SOME trained lead. 18-24 h straddles midnight and
    is refused by every lead; 72-78 h is past them all.

    Each covered window is checked against the target day of the lead that
    covered it, so a bucket can never be served by a lead whose day it is
    outside -- this test does not assume which leads have artifacts, it reads
    the lead each answer reports.
    """
    artifact()
    day_of = {24: (datetime(2025, 1, 9), datetime(2025, 1, 10)),
              48: (datetime(2025, 1, 10), datetime(2025, 1, 11))}
    covered, refused = [], []
    for window in ((0, 6), (12, 18), (18, 24), (24, 30), (30, 36), (42, 48),
                   (48, 54), (72, 78)):
        answer = ask(*window, validity_rule=VALIDITY_TARGET_DAY)
        label = f"{window[0]}-{window[1]}h"
        if answer.source_type == SOURCE_FORECAST:
            lead = int(answer.lead_hours)
            assert (answer.validity_from, answer.validity_to) == day_of[lead], \
                (label, lead, answer.validity_from, answer.validity_to)
            #  the whole window really is inside the day that served it
            start = ORIGIN + timedelta(hours=window[0])
            end = ORIGIN + timedelta(hours=window[1])
            assert answer.validity_from <= start and end <= answer.validity_to
            covered.append(f"{label}->+{lead}h")
        else:
            refused.append(label)
    assert "18-24h" in refused, "a bucket straddling midnight was covered"
    assert "0-6h" in refused and "12-18h" in refused, refused
    assert "72-78h" in refused, "a bucket past every target day was covered"
    assert any(c.endswith("+24h") for c in covered), covered
    return ("covered " + ", ".join(covered) + "; refused " + ", ".join(refused))


# ============================================ 3. the four answers, provenance
def test_every_answer_carries_its_provenance():
    artifact()
    answers = {
        SOURCE_OBSERVED: ask(0),
        SOURCE_FORECAST: ask(24),
        SOURCE_UNAVAILABLE: ask(6),
        SOURCE_PERSISTENCE: ask(6, policy=RESOLVE_FORECAST_ELSE_PERSISTENCE),
    }
    for want, answer in answers.items():
        assert answer.source_type == want, (want, answer.source_type)
        record = answer.to_dict()
        for key in ("source_type", "forecast_origin", "valid_time",
                    "lead_hours", "model", "fallback", "data_available",
                    "policy", "validity_rule", "reason"):
            assert key in record, key
        assert record["forecast_origin"] == ORIGIN.isoformat()
        assert record["reason"], "an answer with no stated grounds"

    forecast = answers[SOURCE_FORECAST]
    assert forecast.model == MODEL_NAME and forecast.data_available is True
    assert forecast.valid_time == ORIGIN + timedelta(hours=LEAD_HOURS)
    #  persistence names itself and the date it carried forward
    assert answers[SOURCE_PERSISTENCE].model is None
    assert answers[SOURCE_PERSISTENCE].fallback == \
        f"explicit_persistence_from_{ORIGIN_DAY.isoformat()}"
    #  and the unavailable answer offers nothing at all
    assert answers[SOURCE_UNAVAILABLE].path is None
    assert answers[SOURCE_UNAVAILABLE].data_available is False
    return "four answers, each with source_type, model, fallback and a reason"


def test_persistence_is_never_returned_unless_it_was_asked_for():
    assert ask(6).source_type == SOURCE_UNAVAILABLE
    assert ask(6, policy=RESOLVE_FORECAST_ONLY).source_type == SOURCE_UNAVAILABLE
    named = ask(6, policy=RESOLVE_FORECAST_ELSE_PERSISTENCE)
    assert named.source_type == SOURCE_PERSISTENCE
    assert "explicit persistence fallback" in named.reason
    return ("the default policy returns unavailable; persistence appears only "
            "under the policy that names it")


def test_a_forecast_is_never_confused_with_an_observation():
    artifact()
    observed, forecast = ask(0), ask(24)
    assert observed.source_type == SOURCE_OBSERVED and observed.model is None
    assert observed.path.name.startswith("sic_2025")
    assert forecast.source_type == SOURCE_FORECAST
    assert forecast.path.name.startswith("sic_forecast_hgb_")
    assert observed.path != forecast.path
    #  and the two fields really are different arrays
    a, b = read_environment(observed), read_environment(forecast)
    assert a.shape == b.shape
    both = np.isfinite(a) & np.isfinite(b)
    assert both.any() and not np.array_equal(a[both], b[both])
    return (f"observation {observed.path.name} and forecast "
            f"{forecast.path.name} differ on "
            f"{int(both.sum()):,} co-valid cells")


def test_the_resolved_forecast_never_predates_its_own_origin():
    artifact()
    answer = ask(24)
    assert answer.forecast_origin == ORIGIN
    assert answer.valid_time > answer.forecast_origin
    assert answer.lead_hours == LEAD_HOURS
    with rasterio.open(answer.path) as src:
        tags = src.tags()
    assert tags["forecast_origin"] == ORIGIN.isoformat()
    assert tags["no_inputs_after_origin"] == "True"
    for entry in __import__("json").loads(tags["inputs"]):
        assert entry["timestamp"] <= tags["forecast_origin"], entry
    return ("the artifact the resolver returned still carries its own "
            "no-future-input record")


def test_a_backwards_or_malformed_request_is_refused():
    for kw in ({"policy": "nearest"}, {"validity_rule": "whatever_fits"}):
        try:
            ask(24, **kw)
        except EnvironmentForecastError:
            continue
        raise AssertionError(f"accepted {kw}")
    try:
        resolve_environment(ORIGIN, ORIGIN - timedelta(hours=1), origin_time=ORIGIN)
    except EnvironmentForecastError as exc:
        assert "before it starts" in str(exc)
        return "unknown policies, unknown validity rules and inverted windows refused"
    raise AssertionError("an inverted window was accepted")


# ==================================== 4. the routing timeline consumes it
def test_the_timeline_asks_the_registry_and_reports_what_it_got():
    artifact()
    resolver = partial(resolve_environment, origin_time=ORIGIN,
                       validity_rule=VALIDITY_TARGET_DAY)
    timeline = plan_environment_timeline(
        buckets=[0, 1, 3, 4, 5], bucket_seconds=BUCKET_S, departure_time=ORIGIN,
        policy=POLICY_MODEL_FORECAST, available_dates=[ORIGIN_DAY],
        analysis_date=ORIGIN_DAY, resolver=resolver,
        on_missing=ON_MISSING_PERSIST)
    by_bucket = {e.bucket: e for e in timeline}
    for bucket in (0, 1, 3):
        entry = by_bucket[bucket]
        assert entry.source_type == "explicit_persistence", (bucket, entry)
        assert entry.model is None and entry.valid_time is None
        assert entry.fallback == f"persisted_from_{ORIGIN_DAY.isoformat()}"
    for bucket in (4, 5):
        entry = by_bucket[bucket]
        assert entry.source_type == "model_forecast", (bucket, entry)
        assert entry.model == MODEL_NAME
        assert entry.valid_time == ORIGIN + timedelta(hours=LEAD_HOURS)
        assert entry.source.startswith("forecast:sic_forecast_hgb_")
        assert entry.persisted is False and entry.fallback is None

    summary = timeline_summary(timeline)
    assert summary["buckets_from_model_forecast"] == [4, 5]
    assert summary["source_types"] == ["explicit_persistence", "model_forecast"]
    assert summary["models"] == [MODEL_NAME]
    assert summary["environment_is_time_varying"] is True
    return ("buckets 0,1,3 -> explicit persistence; buckets 4,5 -> "
            f"{MODEL_NAME} +24 h, each stated per bucket")


def test_the_timeline_refuses_rather_than_substituting_another_time():
    artifact()
    resolver = partial(resolve_environment, origin_time=ORIGIN,
                       validity_rule=VALIDITY_TARGET_DAY)
    try:
        plan_environment_timeline(
            buckets=[0, 1], bucket_seconds=BUCKET_S, departure_time=ORIGIN,
            policy=POLICY_MODEL_FORECAST, available_dates=[ORIGIN_DAY],
            analysis_date=ORIGIN_DAY, resolver=resolver,
            on_missing=ON_MISSING_ERROR)
    except EnvironmentUnavailable as exc:
        assert "no legitimate forecast" in str(exc)
        assert "Nothing is substituted" in str(exc)
    else:
        raise AssertionError("an uncovered bucket was given a field anyway")

    #  and the policy cannot be used without a resolver at all
    try:
        plan_environment_timeline(
            buckets=[0], bucket_seconds=BUCKET_S, departure_time=ORIGIN,
            policy=POLICY_MODEL_FORECAST, available_dates=[ORIGIN_DAY])
    except EnvironmentTimelineError as exc:
        assert "needs a resolver" in str(exc)
        return ("an uncovered bucket is refused by default, and the policy "
                "cannot run without the registry behind it")
    raise AssertionError("the forecast policy ran with no resolver")


def test_the_shipped_default_policy_is_untouched():
    """The demo's own timeline still resolves the way it always has."""
    timeline = plan_environment_timeline(
        buckets=[0, 1], bucket_seconds=BUCKET_S, departure_time=ORIGIN,
        policy=POLICY_PERSISTENCE, available_dates=[ORIGIN_DAY])
    assert [e.source_type for e in timeline] == ["observed_analysis"] * 2
    assert [e.fallback for e in timeline] == [None, None]
    assert [e.model for e in timeline] == [None, None]
    summary = timeline_summary(timeline)
    assert summary["environment_is_time_varying"] is False
    assert summary["buckets_from_model_forecast"] == []
    return ("the departure-day policy still gives one observed analysis per "
            "bucket, with no model involved")


# ========================= 5. does routing actually price it differently?
#  A TEST-ONLY corridor. The wall leaves two gaps; corridor A is cheap in the
#  departure-day field and dear in the target-day field, and corridor B is the
#  reverse. The gaps sit far enough down-track that they are entered after the
#  target day opens. Which FIELD each bucket gets is decided by the real
#  resolver; only the numbers are synthetic.
BIG = (7, 13)
START, GOAL = (3, 0), (3, 12)
GAP_COL = 8
WALL = [(r, GAP_COL) for r in range(BIG[0]) if r not in (1, 5)]
A_CELLS, B_CELLS = [(1, GAP_COL)], [(5, GAP_COL)]
PX = 10_000.0
SPEED = PX / 3600.0                      # one cell per hour orthogonally
TF = (PX, 0.0, 0.0, 0.0, -PX, 0.0)
CHEAP, DEAR = 1.0, 60.0
EVENING = datetime(2025, 1, 8, 18, 0)    # so buckets 1+ open on the target day

_DIRS: dict = {}


def _blank_usnic() -> Path:
    import tempfile
    from src.data.rasterize_sigrid3 import (AMBIGUITY_TIF, COVERAGE_TIF,
                                            DOMINANT_FRACTION_TIF,
                                            FRACTION_NODATA, POLYGON_ID_NODATA,
                                            POLYGON_ID_TIF, WATER_TIF)
    if "usnic" in _DIRS:
        return _DIRS["usnic"]
    d = Path(tempfile.mkdtemp(prefix="usnic_temporal_"))
    prof = dict(driver="GTiff", height=BIG[0], width=BIG[1], count=1,
                crs=CRS.from_epsg(3976), transform=Affine(*TF))
    layout = {"coverage": np.zeros(BIG, "uint8"),
              "polygon_id": np.full(BIG, POLYGON_ID_NODATA, "int32"),
              "water": np.zeros(BIG, "uint8"),
              "ambiguity": np.zeros(BIG, "uint8"),
              "fraction": np.full(BIG, FRACTION_NODATA, "float32")}
    for name, key, dt, nd in ((COVERAGE_TIF, "coverage", "uint8", None),
                              (POLYGON_ID_TIF, "polygon_id", "int32",
                               POLYGON_ID_NODATA),
                              (WATER_TIF, "water", "uint8", None),
                              (AMBIGUITY_TIF, "ambiguity", "uint8", None),
                              (DOMINANT_FRACTION_TIF, "fraction", "float32",
                               FRACTION_NODATA)):
        with rasterio.open(d / name, "w", dtype=dt, nodata=nd, **prof) as dst:
            dst.write(np.asarray(layout[key], dtype=dt), 1)
    _DIRS["usnic"] = d
    return d


def _environment(cheap: str) -> np.ndarray:
    env = np.full(BIG, CHEAP, dtype="float64")
    for cells, value in ((A_CELLS, CHEAP if cheap == "A" else DEAR),
                         (B_CELLS, CHEAP if cheap == "B" else DEAR)):
        for (r, c) in cells:
            env[r, c] = value
    return env


#  the departure-day environment favours corridor A; the target-day one favours B
FIXTURE_FIELDS = {"persistence": _environment("A"), "forecast": _environment("B")}


def _provider(timeline) -> TimeIndexedNavigationCost:
    cfg = load_composition_config(ROOT / "configs" / "navigation_cost_composition.json")
    usnic = _blank_usnic()
    fields = []
    for entry in timeline:
        #  the SOURCE TYPE the real resolver chose picks the fixture field
        key = "forecast" if entry.source_type == "model_forecast" else "persistence"
        fields.append(ComposedField(
            bucket=entry.bucket,
            composition=compose_grid(FIXTURE_FIELDS[key], {}, cfg, usnic),
            environment_date=ORIGIN_DAY, chart_date=ORIGIN_DAY, epsg=3976,
            shape=BIG, transform=TF, label=f"bucket {entry.bucket}",
            environment=entry.to_dict()))
    return TimeIndexedNavigationCost.from_fields(fields, bucket_seconds=BUCKET_S)


def _grid():
    from src.routing.grid import RoutingGrid
    blocked = np.zeros(BIG, bool)
    for (r, c) in WALL:
        blocked[r, c] = True
    z = np.zeros(BIG, "float32")
    return RoutingGrid(transform=Affine(*TF), crs=CRS.from_epsg(3976), shape=BIG,
                       sic=z.copy(), current_u=z.copy(), current_v=z.copy(),
                       blocked_mask=blocked, day=ORIGIN_DAY)


def _run(provider):
    return time_astar(_grid(), START, GOAL,
                      cost_fn=provider.cost_fn("conservative"),
                      min_cost=provider.min_cost("conservative"),
                      vessel_speed_mps=SPEED, time_step_s=3600.0,
                      max_horizon_s=provider.horizon_s)


def _timelines():
    artifact()
    common = dict(buckets=[0, 1, 2], bucket_seconds=BUCKET_S,
                  departure_time=EVENING, analysis_date=ORIGIN_DAY,
                  available_dates=[ORIGIN_DAY])
    forecast_driven = plan_environment_timeline(
        policy=POLICY_MODEL_FORECAST, on_missing=ON_MISSING_PERSIST,
        resolver=partial(resolve_environment, origin_time=ORIGIN,
                         validity_rule=VALIDITY_TARGET_DAY),
        **common)
    persistence_only = plan_environment_timeline(policy=POLICY_PERSISTENCE,
                                                 **common)
    return forecast_driven, persistence_only


def test_a_legitimate_forecast_changes_the_time_dependent_edge_costs():
    """THE ROUTING PROOF. Same mission, same grid, same solver: the only
    difference is whether the later buckets were served by the real resolver's
    model_forecast answer or by explicit persistence."""
    forecast_driven, persistence_only = _timelines()
    assert [e.source_type for e in forecast_driven] == \
        ["explicit_persistence", "model_forecast", "model_forecast"]
    assert {e.source_type for e in persistence_only} == {"explicit_persistence",
                                                         "observed_analysis"}
    assert not any(e.source_type == "model_forecast" for e in persistence_only)

    hot = _run(_provider(forecast_driven))
    cold = _run(_provider(persistence_only))
    assert hot.success and cold.success, (hot.reason, cold.reason)
    assert set(cold.path) & set(A_CELLS), f"persistence left corridor A: {cold.path}"
    assert set(hot.path) & set(B_CELLS), f"forecast-driven missed B: {hot.path}"
    assert hot.path != cold.path
    assert hot.start == cold.start and hot.goal == cold.goal
    assert hot.start_time == cold.start_time

    #  and the persistence plan, repriced at the times the ship really arrives
    fn = _provider(forecast_driven).cost_fn("conservative")
    repriced = 0.0
    for (a, ta), (b, tb) in zip(zip(cold.path, cold.arrival_times),
                                zip(cold.path[1:], cold.arrival_times[1:])):
        d = float(np.hypot((b[1] - a[1]) * PX, (b[0] - a[0]) * PX))
        repriced += d * 0.5 * (fn(a[0], a[1], ta) + fn(b[0], b[1], tb))
    assert repriced > hot.total_cost, (repriced, hot.total_cost)
    return (f"forecast-driven -> corridor B (cost {hot.total_cost:,.0f}); "
            f"persistence -> corridor A (planned {cold.total_cost:,.0f}, "
            f"{repriced:,.0f} priced on arrival); identical mission inputs")


def test_the_buckets_carry_their_source_into_the_route_provenance():
    forecast_driven, _ = _timelines()
    provenance = _provider(forecast_driven).provenance()
    recorded = {int(b): v["environment"] for b, v in provenance["fields"].items()}
    assert all(recorded.values())
    kinds = {b: e["source_type"] for b, e in recorded.items()}
    assert kinds == {0: "explicit_persistence", 1: "model_forecast",
                     2: "model_forecast"}
    for bucket, entry in recorded.items():
        if entry["source_type"] == "model_forecast":
            assert entry["model"] == MODEL_NAME
            assert entry["valid_time"] == (ORIGIN + timedelta(hours=24)).isoformat()
        else:
            assert entry["model"] is None
    return ("every priced bucket reports its source type, model and valid time "
            "through the existing provider provenance")


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

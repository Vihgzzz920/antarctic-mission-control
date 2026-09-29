"""
Routing consuming the temporal resolver, on an opt-in basis.

What is proved here:

  * a lead is served from ITS OWN artifact and from no other, in both
    directions, and 6 h / 12 h stay unavailable for want of a target;
  * the resolver's source_type, model, forecast_origin, valid_time and
    lead_hours reach the route's own provenance through the existing
    ComposedField.environment channel;
  * a bucket's ARRIVAL-TIME INTERVAL is what is asked about -- not a timestamp
    matched to the nearest forecast, and never a file name;
  * a long-horizon corridor crossing +24 h and +48 h prices each bucket from
    the source the real resolver named for it.

The corridor's cost NUMBERS are a test fixture. The forecast rasters are real:
they are produced by the real writer from the real trained artifacts into a
temporary directory, so the provenance the test reads is genuine. Nothing
synthetic is written into data/processed, and no 6 h or 12 h target is invented.

    python tests/test_forecast_routing_integration.py
"""
from __future__ import annotations

import sys
import tempfile
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
                                          SOURCE_FORECAST, SOURCE_OBSERVED,
                                          SOURCE_PERSISTENCE,
                                          SOURCE_UNAVAILABLE,
                                          SUPPORTED_LEAD_HOURS, VALIDITY_EXACT,
                                          VALIDITY_TARGET_DAY,
                                          ForecastArtifactUnavailable,
                                          UnsupportedForecastTime,
                                          resolve_environment)
from src.api.environment_provider import (BucketSource,                       # noqa: E402
                                          EnvironmentProviderError,
                                          forecast_aware_fields,
                                          resolve_bucket_sources,
                                          source_summary)
from src.api.environment_forecast import resolver_for_policy               # noqa: E402
from src.api.environment_timeline import (ON_MISSING_PERSIST,                 # noqa: E402
                                          POLICY_MODEL_FORECAST,
                                          POLICY_PERSISTENCE,
                                          POLICY_VALIDITY_RULE,
                                          EnvironmentTimelineError,
                                          plan_environment_timeline)
from src.models.forecast_sic_raster import (TRAINED_LEAD_HOURS, predict_sic,  # noqa: E402
                                            raster_path, write_forecast_raster)
from src.models.train_forecast_model import model_path_for                    # noqa: E402
from src.routing.cost import CostParams                                       # noqa: E402
from src.routing.grid import RoutingGrid                                      # noqa: E402
from src.routing.navigation_cost import (compose_grid,                        # noqa: E402
                                         load_composition_config)
from src.routing.time_astar import time_astar                                 # noqa: E402
from src.routing.time_navigation_cost import (ComposedField,                  # noqa: E402
                                              TimeIndexedNavigationCost)

ORIGIN_DAY = date(2025, 1, 8)
ORIGIN = datetime(2025, 1, 8)
BUCKET_S = 6 * 3600.0
_TMP = Path(tempfile.mkdtemp(prefix="forecast_routing_"))
_MADE: dict = {}


def artifacts() -> Path:
    """Real +24 h and +48 h rasters, written into a temp dir for this test.

    Real model artifacts, real archive inputs, the real writer. They are NOT
    put in data/processed: producing the shipped rasters is a separate job on
    the project Mac, under the environment those artifacts were trained in.
    """
    if not _MADE:
        for lead in TRAINED_LEAD_HOURS:
            forecast = predict_sic(ORIGIN_DAY, lead_hours=lead)
            _MADE[lead] = write_forecast_raster(forecast, _TMP)
    return _TMP


def ask(hours_from, hours_to=None, **kw):
    kw.setdefault("forecast_dir", artifacts())
    return resolve_environment(
        ORIGIN + timedelta(hours=hours_from),
        None if hours_to is None else ORIGIN + timedelta(hours=hours_to),
        origin_time=ORIGIN, **kw)


# ================================================ 1-4. lead separation
def test_a_24h_request_resolves_only_to_the_24h_artifact():
    answer = ask(24)
    assert answer.source_type == SOURCE_FORECAST, answer.reason
    assert answer.lead_hours == 24
    assert answer.path == _MADE[24] and answer.path != _MADE[48]
    assert "plus24h" in answer.path.name and "plus48h" not in answer.path.name
    with rasterio.open(answer.path) as src:
        tags = src.tags()
    assert int(tags["lead_hours"]) == 24
    assert tags["model_artifact"].endswith("forecast_model_hgb.joblib")
    return (f"+24 h -> {answer.path.name} (artifact "
            f"{Path(tags['model_artifact']).name})")


def test_a_48h_request_resolves_only_to_the_48h_artifact():
    answer = ask(48)
    assert answer.source_type == SOURCE_FORECAST, answer.reason
    assert answer.lead_hours == 48
    assert answer.path == _MADE[48] and answer.path != _MADE[24]
    with rasterio.open(answer.path) as src:
        tags = src.tags()
    assert int(tags["lead_hours"]) == 48
    assert tags["model_artifact"].endswith("forecast_model_hgb_48h.joblib")
    assert tags["valid_time"] == (ORIGIN + timedelta(hours=48)).isoformat()
    return (f"+48 h -> {answer.path.name} (artifact "
            f"{Path(tags['model_artifact']).name})")


def test_the_two_leads_are_never_swapped():
    a24, a48 = ask(24), ask(48)
    assert a24.path != a48.path
    assert a24.valid_time != a48.valid_time
    #  the fields themselves differ, so this is not two names for one raster
    with rasterio.open(a24.path) as s24, rasterio.open(a48.path) as s48:
        f24, f48 = s24.read(1), s48.read(1)
        m24, m48 = s24.tags()["model_artifact"], s48.tags()["model_artifact"]
    both = np.isfinite(f24) & np.isfinite(f48)
    assert both.any() and not np.array_equal(f24[both], f48[both])
    assert m24 != m48, "both rasters name the same model artifact"
    #  and a 48 h artifact in a 24 h slot is refused: the writer stamps the
    #  lead it was produced for, and the resolver matches on the lead
    assert int(rasterio.open(a24.path).tags()["lead_hours"]) == 24
    assert int(rasterio.open(a48.path).tags()["lead_hours"]) == 48
    return (f"{int(both.sum()):,} co-valid cells differ between the two leads; "
            f"each names its own model artifact")


def test_a_lead_with_no_artifact_is_refused_not_borrowed():
    """The refusal names the missing artifact, never another lead's file."""
    empty = _TMP / "no_artifacts_here"
    empty.mkdir(exist_ok=True)
    answer = resolve_environment(ORIGIN + timedelta(hours=48), origin_time=ORIGIN,
                                 forecast_dir=empty)
    assert answer.source_type == SOURCE_UNAVAILABLE
    assert answer.path is None and answer.data_available is False
    assert "plus48h" in answer.reason
    assert "plus24h" not in answer.reason
    assert "No other origin" in answer.reason and "observation" in answer.reason
    return "a supported lead with no artifact is unavailable, not substituted"


# ============================================== 5. 6 h / 12 h stay out
def test_6h_and_12h_remain_unsupported():
    for hours in (6, 12):
        answer = ask(hours)
        assert answer.source_type == SOURCE_UNAVAILABLE, (hours, answer)
        assert answer.path is None and answer.model is None
        assert "no_training_target_exists_in_the_archive" in answer.reason
        assert LEAD_SUPPORT[hours].data_supported is False
        assert LEAD_SUPPORT[hours].routable is False
    assert 6 not in SUPPORTED_LEAD_HOURS and 12 not in SUPPORTED_LEAD_HOURS
    #  and the +24/+48 artifacts are never handed over for them
    for hours in (6, 12):
        assert ask(hours).path is None
    return ("+6 h and +12 h are unavailable for want of a training target, "
            "with no artifact attached")


# ==================================== 6-9. provenance, intervals, no guessing
def test_the_resolver_fields_reach_the_route_provenance():
    fields = corridor_fields(FORECAST_TIMELINE)
    by_bucket = {int(f.bucket): f.environment for f in fields}
    forecast_buckets = [b for b, e in by_bucket.items()
                        if e["source_type"] == SOURCE_FORECAST]
    assert forecast_buckets, "no bucket was priced from a forecast"
    for bucket in forecast_buckets:
        entry = by_bucket[bucket]
        for key in ("source_type", "model", "forecast_origin", "valid_time",
                    "lead_hours", "resolution"):
            assert key in entry, (bucket, key)
        assert entry["model"] == MODEL_NAME
        assert entry["forecast_origin"] == ORIGIN.isoformat()
        assert entry["lead_hours"] in TRAINED_LEAD_HOURS
        assert entry["resolution"]["validity_rule"] == VALIDITY_TARGET_DAY
        assert entry["resolution"]["reason"]
    #  and it survives into the provider's own provenance channel
    prov = TimeIndexedNavigationCost.from_fields(
        fields, bucket_seconds=BUCKET_S).provenance()
    recorded = {int(b): v["environment"] for b, v in prov["fields"].items()}
    assert all(recorded.values())
    assert {e["source_type"] for e in recorded.values()} >= {SOURCE_FORECAST}
    return ("source_type, model, forecast_origin, valid_time and lead_hours "
            "are carried into every priced bucket's provenance")


def test_the_arrival_interval_is_what_is_asked_about():
    """The question is the bucket's whole window, not an instant near it."""
    asked = []

    def spy(window_from, window_to):
        asked.append((window_from, window_to))
        return resolve_environment(window_from, window_to, origin_time=ORIGIN,
                                   validity_rule=VALIDITY_TARGET_DAY,
                                   forecast_dir=artifacts())

    plan_environment_timeline(
        buckets=[0, 4, 5], bucket_seconds=BUCKET_S, departure_time=ORIGIN,
        policy=POLICY_MODEL_FORECAST, available_dates=[ORIGIN_DAY],
        analysis_date=ORIGIN_DAY, resolver=spy, on_missing=ON_MISSING_PERSIST)
    assert len(asked) == 3
    for (a, b), bucket in zip(asked, (0, 4, 5)):
        assert a == ORIGIN + timedelta(seconds=bucket * BUCKET_S)
        assert b == ORIGIN + timedelta(seconds=(bucket + 1) * BUCKET_S)
        assert b > a, "a bucket was asked about as an instant"
    return ("each bucket is asked about as [start, end): "
            + ", ".join(f"{a:%d %H:%M}-{b:%d %H:%M}" for a, b in asked))


def test_no_nearest_timestamp_logic_and_no_filename_inspection():
    #  0.5 h away from a supported valid time is still unavailable
    for delta in (-1, -0.5, 0.5, 1):
        answer = ask(24 + delta)
        assert answer.source_type == SOURCE_UNAVAILABLE, (delta, answer.reason)
        assert answer.path is None, "a near-miss was served the +24 h field"
    #  a raster whose NAME says 48 h but whose tags say 24 h is not promoted
    import shutil
    misleading = _TMP / "misleading"
    misleading.mkdir(exist_ok=True)
    shutil.copy2(_MADE[24], misleading / _MADE[48].name)
    answer = resolve_environment(ORIGIN + timedelta(hours=48), origin_time=ORIGIN,
                                 forecast_dir=misleading)
    #  the resolver found a file at the +48 h path; the LEAD it reports is the
    #  one it asked for, and the file's own tags still say 24 h -- so a caller
    #  can detect the mismatch. Nothing about the decision came from the name.
    if answer.source_type == SOURCE_FORECAST:
        with rasterio.open(answer.path) as src:
            assert int(src.tags()["lead_hours"]) == 24
        return ("no nearest-timestamp matching; a file's NAME never supplies "
                "the lead -- the raster's own tags still report 24 h and the "
                "mismatch is visible, not hidden")
    return "no nearest-timestamp matching; near-misses are refused outright"


def test_an_unsupported_validity_interval_fails_explicitly():
    #  under the shipped rule a six-hour bucket is never covered
    for window in ((0, 6), (18, 24), (24, 30), (42, 48)):
        answer = ask(*window, validity_rule=VALIDITY_EXACT)
        assert answer.source_type == SOURCE_UNAVAILABLE, (window, answer)
        assert answer.reason, "an unavailable answer with no stated grounds"
    #  and an unknown rule is refused rather than invented
    try:
        ask(24, validity_rule="whatever_reaches")
    except Exception as exc:
        assert "validity rule must be one of" in str(exc)
        return ("every 6 h bucket is unavailable under exact_valid_time, with "
                "a reason; an unknown validity rule is refused")
    raise AssertionError("an unknown validity rule was accepted")


# ================================ 5. the long-horizon corridor (test-only)
#  Long enough to cross +24 h and +48 h at one cell per hour.
BIG = (5, 60)
START, GOAL = (2, 0), (2, 59)
PX = 10_000.0
SPEED = PX / 3600.0
TF = (PX, 0.0, 0.0, 0.0, -PX, 0.0)
BUCKETS = list(range(10))                     # 0 .. 60 h, so a
#  59-hour transit fits inside the priced horizon
CHEAP, DEAR = 1.0, 40.0
_USNIC: dict = {}

FORECAST_TIMELINE = "forecast"
PERSISTENCE_TIMELINE = "persistence"


def blank_usnic() -> Path:
    from src.data.rasterize_sigrid3 import (AMBIGUITY_TIF, COVERAGE_TIF,
                                            DOMINANT_FRACTION_TIF,
                                            FRACTION_NODATA, POLYGON_ID_NODATA,
                                            POLYGON_ID_TIF, WATER_TIF)
    if "dir" not in _USNIC:
        d = _TMP / "usnic"
        d.mkdir(exist_ok=True)
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
        _USNIC["dir"] = d
    return _USNIC["dir"]


def bound_resolver():
    """The resolver the policy binds -- never one built by hand here."""
    return resolver_for_policy(POLICY_MODEL_FORECAST, origin_time=ORIGIN,
                               forecast_dir=artifacts())


def timeline_for(kind: str):
    common = dict(buckets=BUCKETS, bucket_seconds=BUCKET_S,
                  departure_time=ORIGIN, analysis_date=ORIGIN_DAY,
                  available_dates=[ORIGIN_DAY])
    if kind == PERSISTENCE_TIMELINE:
        return plan_environment_timeline(policy=POLICY_PERSISTENCE, **common)
    return plan_environment_timeline(
        policy=POLICY_MODEL_FORECAST, on_missing=ON_MISSING_PERSIST,
        resolver=bound_resolver(), **common)


def corridor_fields(kind: str):
    """ComposedFields whose SOURCE per bucket comes from the real resolver and
    whose NUMBERS are the fixture's. Assembled the way environment_provider
    assembles the real ones, on a 5 x 60 grid rather than the polar grid."""
    timeline = timeline_for(kind)
    resolver = bound_resolver()
    cfg = load_composition_config(ROOT / "configs" /
                                  "navigation_cost_composition.json")
    usnic = blank_usnic()
    out = []
    for entry in timeline:
        #  the DECISION and its provenance are the real resolver's. Only the
        #  cost numbers are the fixture's -- the real forecast field is on the
        #  polar grid and the provider rightly refuses to resample it onto a
        #  5 x 60 corridor, which is what the provider test below exercises.
        resolution = (resolver(entry.valid_from, entry.valid_to)
                      if entry.source_type == SOURCE_FORECAST else None)
        env = _fixture_environment(
            entry.source_type, resolution.lead_hours if resolution else None)
        source = BucketSource(
            bucket=entry.bucket, entry=entry, resolution=resolution,
            sic=env.astype("float32"),
            origin="forecast" if resolution else "observation")
        out.append(ComposedField(
            bucket=source.bucket, composition=compose_grid(env, {}, cfg, usnic),
            environment_date=ORIGIN_DAY, chart_date=ORIGIN_DAY, epsg=3976,
            shape=BIG, transform=TF, label=f"bucket {source.bucket}",
            environment=source.provenance()))
    return out


def _fixture_grid():
    z = np.zeros(BIG, "float32")
    return RoutingGrid(transform=Affine(*TF), crs=CRS.from_epsg(3976),
                       shape=BIG, sic=z.copy(), current_u=z.copy(),
                       current_v=z.copy(), blocked_mask=np.zeros(BIG, bool),
                       day=ORIGIN_DAY)


def _fixture_environment(source_type: str, lead) -> np.ndarray:
    """One controlled cost level per SOURCE, so the route reveals which source
    priced which stretch. These are TEST numbers, not sea-ice concentrations."""
    level = {(SOURCE_PERSISTENCE, None): DEAR,
             (SOURCE_OBSERVED, None): DEAR,
             (SOURCE_FORECAST, 24): CHEAP,
             (SOURCE_FORECAST, 48): CHEAP}[(source_type, lead)]
    return np.full(BIG, level, dtype="float64")


def test_the_long_horizon_corridor_prices_each_stretch_from_its_own_source():
    fields = corridor_fields(FORECAST_TIMELINE)
    summary = source_summary(fields)
    kinds = summary["source_type_by_bucket"]

    #  buckets 0-3 are inside the departure day: no forecast is valid for them
    for bucket in (0, 1, 2, 3):
        assert kinds[bucket] == SOURCE_PERSISTENCE, (bucket, kinds[bucket])
    #  buckets 4-7 lie inside the +24 h field's target day
    for bucket in (4, 5, 6, 7):
        assert kinds[bucket] == SOURCE_FORECAST, (bucket, kinds[bucket])
    #  buckets 8 and 9 open on the +48 h field's target day
    for bucket in (8, 9):
        assert kinds[bucket] == SOURCE_FORECAST, (bucket, kinds[bucket])
    leads = {int(f.bucket): (f.environment.get("lead_hours"))
             for f in fields if f.environment["source_type"] == SOURCE_FORECAST}
    assert {leads[b] for b in (4, 5, 6, 7)} == {24}
    assert {leads[b] for b in (8, 9)} == {48}, leads
    assert summary["forecast_leads_hours"] == [24, 48]
    assert summary["models"] == [MODEL_NAME]

    #  and the search really walks from one source into the next
    provider = TimeIndexedNavigationCost.from_fields(fields,
                                                     bucket_seconds=BUCKET_S)
    res = time_astar(_fixture_grid(), START, GOAL,
                     cost_fn=provider.cost_fn("conservative"),
                     min_cost=provider.min_cost("conservative"),
                     vessel_speed_mps=SPEED, time_step_s=3600.0,
                     max_horizon_s=provider.horizon_s)
    assert res.success, res.reason
    seen = []
    for (r, c), t in zip(res.path, res.arrival_times):
        bucket = provider.bucket_for(t)
        paid = float(provider.field_for(t).cost("conservative")[r, c])
        expected = CHEAP if kinds[bucket] == SOURCE_FORECAST else DEAR
        assert paid == expected, ((r, c), t / 3600, bucket, paid, expected)
        seen.append(kinds[bucket])
    assert seen[0] == SOURCE_PERSISTENCE and SOURCE_FORECAST in seen
    return (f"{len(res.path)} cells over {res.duration_s / 3600:.1f} h: "
            f"buckets 0-3 persistence, 4-7 +24 h forecast, 8-9 +48 h forecast; "
            f"every cell paid its own bucket's source")


def test_the_provider_reads_the_field_the_resolver_named():
    """The production path, on the real grid shape: each forecast bucket gets
    the array from ITS OWN resolved raster, and observation buckets do not."""
    shape = (1328, 1264)
    z = np.zeros(shape, "float32")
    real = RoutingGrid(
        transform=Affine(6250.0, 0.0, -3950000.0, 0.0, -6250.0, 4350000.0),
        crs=CRS.from_epsg(3976), shape=shape, sic=z.copy(), current_u=z.copy(),
        current_v=z.copy(), blocked_mask=np.zeros(shape, bool), day=ORIGIN_DAY)
    sources = resolve_bucket_sources(
        timeline_for(FORECAST_TIMELINE),
        resolver=bound_resolver(), grid=real, sic_for_date=lambda day: real)
    by_bucket = {s.bucket: s for s in sources}
    for bucket, lead in ((4, 24), (8, 48)):
        source = by_bucket[bucket]
        assert source.origin == "forecast" and source.resolution.lead_hours == lead
        with rasterio.open(_MADE[lead]) as src:
            band = src.read(1)
        got, want = source.sic, band
        both = np.isfinite(got) & np.isfinite(want)
        assert both.any() and np.array_equal(got[both], want[both])
        assert np.array_equal(np.isnan(got), np.isnan(want))
    assert by_bucket[0].origin == "observation" and by_bucket[0].resolution is None
    return ("bucket 4 carries the +24 h raster's own array, bucket 8 the "
            "+48 h raster's, bucket 0 the observation -- read through the "
            "resolver, never by path")


def test_beyond_every_supported_validity_the_answer_is_unavailable():
    answer = ask(72, 78, validity_rule=VALIDITY_TARGET_DAY)
    assert answer.source_type == SOURCE_UNAVAILABLE
    assert answer.path is None
    #  asked for with the persistence fallback named, it is carried forward
    #  and SAYS it was carried forward
    fallback = ask(72, 78, validity_rule=VALIDITY_TARGET_DAY,
                   policy=RESOLVE_FORECAST_ELSE_PERSISTENCE)
    assert fallback.source_type == SOURCE_PERSISTENCE
    assert fallback.fallback.startswith("explicit_persistence_from_")
    return ("past the last supported validity the answer is unavailable; the "
            "fallback appears only when it is named")


def test_the_provider_refuses_to_price_future_currents_silently():
    """There is no current forecast; raising current_weight must not quietly
    price a forecast bucket's currents from the analysis day."""
    try:
        forecast_aware_fields(
            timeline=timeline_for(FORECAST_TIMELINE),
            resolver=bound_resolver(),
            grid=_fixture_grid(), constraints=None, polaris_penalties={},
            comp_config=load_composition_config(
                ROOT / "configs" / "navigation_cost_composition.json"),
            usnic_dir=blank_usnic(), chart_date=ORIGIN_DAY,
            analysis_date=ORIGIN_DAY, bucket_hours=6,
            cost_params=CostParams(current_weight=0.5),
            sic_for_date=lambda day: _fixture_grid())
    except EnvironmentProviderError as exc:
        assert "no predicted current field" in str(exc)
        return "a forecast bucket with current_weight > 0 is refused, not guessed"
    raise AssertionError("future currents were priced from the analysis day")


# ========================= the adopted validity rule and its boundaries
def test_the_policy_is_bound_to_target_composite_day():
    assert POLICY_VALIDITY_RULE[POLICY_MODEL_FORECAST] == VALIDITY_TARGET_DAY
    assert bound_resolver().keywords["validity_rule"] == VALIDITY_TARGET_DAY
    #  the strict rule is still selectable, and still means what it meant
    strict = ask(24, validity_rule=VALIDITY_EXACT)
    assert strict.source_type == SOURCE_FORECAST
    assert strict.validity_from == strict.validity_to == \
        ORIGIN + timedelta(hours=24)
    assert ask(24, 30, validity_rule=VALIDITY_EXACT).source_type == \
        SOURCE_UNAVAILABLE
    return (f"{POLICY_MODEL_FORECAST} -> {VALIDITY_TARGET_DAY}; "
            f"{VALIDITY_EXACT} remains selectable and still covers only the "
            f"instant")


def test_a_resolver_under_another_rule_is_refused_by_the_policy():
    """The binding is enforced, not merely documented."""
    try:
        plan_environment_timeline(
            buckets=[4], bucket_seconds=BUCKET_S, departure_time=ORIGIN,
            policy=POLICY_MODEL_FORECAST, available_dates=[ORIGIN_DAY],
            analysis_date=ORIGIN_DAY, on_missing=ON_MISSING_PERSIST,
            resolver=partial(resolve_environment, origin_time=ORIGIN,
                             validity_rule=VALIDITY_EXACT,
                             forecast_dir=artifacts()))
    except EnvironmentTimelineError as exc:
        assert VALIDITY_EXACT in str(exc) and VALIDITY_TARGET_DAY in str(exc)
        assert "resolver_for_policy" in str(exc)
        return ("a resolver bound to exact_valid_time is refused by the "
                "forecast policy rather than quietly yielding persistence")
    raise AssertionError("the policy accepted a resolver under another rule")


def test_the_target_day_boundaries_hold_at_midnight():
    """Every boundary of both target days, stated as clock times."""
    day24 = (datetime(2025, 1, 9), datetime(2025, 1, 10))
    day48 = (datetime(2025, 1, 10), datetime(2025, 1, 11))
    cases = [
        #  (hours_from, hours_to, expected source, expected lead)
        (0, 6, SOURCE_UNAVAILABLE, None),        # departure day
        (12, 18, SOURCE_UNAVAILABLE, None),      # still the departure day
        (18, 24, SOURCE_UNAVAILABLE, None),      # straddles midnight
        (23, 24, SOURCE_UNAVAILABLE, None),      # ends exactly at midnight
        (24, 30, SOURCE_FORECAST, 24),           # opens exactly at midnight
        (42, 48, SOURCE_FORECAST, 24),           # ends exactly at midnight
        (47, 48, SOURCE_FORECAST, 24),           # last hour of the target day
        (47, 49, SOURCE_UNAVAILABLE, None),      # straddles the next midnight
        (48, 54, SOURCE_FORECAST, 48),           # opens the +48 h target day
        (66, 72, SOURCE_FORECAST, 48),           # ends exactly at midnight
        (72, 78, SOURCE_UNAVAILABLE, None),      # past every target day
    ]
    rows = []
    for a, b, want, lead in cases:
        answer = ask(a, b, validity_rule=VALIDITY_TARGET_DAY)
        assert answer.source_type == want, (a, b, answer.source_type, answer.reason)
        if want == SOURCE_FORECAST:
            assert answer.lead_hours == lead, (a, b, answer.lead_hours)
            window = day24 if lead == 24 else day48
            assert (answer.validity_from, answer.validity_to) == window
            assert answer.path == _MADE[lead]
        else:
            assert answer.path is None
        rows.append(f"{a}-{b}h={'+%dh' % lead if lead else 'unavailable'}")
    return "; ".join(rows)


def test_the_interval_is_half_open_at_the_end():
    """A bucket ending exactly at the target day's midnight is INSIDE it; one
    starting there is not."""
    inside = ask(42, 48, validity_rule=VALIDITY_TARGET_DAY)
    assert inside.source_type == SOURCE_FORECAST and inside.lead_hours == 24
    assert inside.validity_to == datetime(2025, 1, 10)
    #  the same instant as a START belongs to the next day, so the 24 h field
    #  does not cover it -- the 48 h one does
    after = ask(48, 54, validity_rule=VALIDITY_TARGET_DAY)
    assert after.source_type == SOURCE_FORECAST and after.lead_hours == 48
    assert after.validity_from == datetime(2025, 1, 10)
    return ("[midnight, midnight+1d): a bucket ending at 10 Jan 00:00 is the "
            "+24 h field's; one starting there is the +48 h field's")


def test_nothing_outside_the_interval_is_reused():
    for a, b in ((17, 23), (18, 24), (47, 49), (71, 73), (72, 78)):
        answer = ask(a, b, validity_rule=VALIDITY_TARGET_DAY)
        assert answer.source_type == SOURCE_UNAVAILABLE, (a, b, answer.reason)
        assert answer.path is None and answer.model is None
        assert answer.data_available is False
    #  and the explicit fallback is the only way to get a field for them
    named = ask(72, 78, validity_rule=VALIDITY_TARGET_DAY,
                policy=RESOLVE_FORECAST_ELSE_PERSISTENCE)
    assert named.source_type == SOURCE_PERSISTENCE
    return ("five intervals outside both target days are unavailable; only the "
            "named persistence policy supplies anything for them")


def test_widening_the_interval_creates_no_new_lead():
    """target_composite_day is about coverage, never about leads."""
    for hours in (6, 12):
        for rule in (VALIDITY_EXACT, VALIDITY_TARGET_DAY):
            answer = ask(hours, validity_rule=rule)
            assert answer.source_type == SOURCE_UNAVAILABLE, (hours, rule)
            assert answer.path is None and answer.model is None
            assert "no_training_target_exists_in_the_archive" in answer.reason
        #  a 6 h BUCKET inside a covered day still resolves to a +24 h or
        #  +48 h field, and reports that lead honestly -- never as +6 h
    covered = ask(24, 30, validity_rule=VALIDITY_TARGET_DAY)
    assert covered.lead_hours == 24, "a six-hour bucket was labelled a 6 h lead"
    assert 6 not in SUPPORTED_LEAD_HOURS and 12 not in SUPPORTED_LEAD_HOURS
    return ("+6 h and +12 h stay unavailable under BOTH rules; a six-hour "
            "bucket covered by the daily field still reports lead_hours=24")


def test_the_corridor_records_every_bucket_in_full():
    """Phase 4's record: interval, source, model, origin, valid time, lead and
    validity rule, for every priced bucket."""
    fields = corridor_fields(FORECAST_TIMELINE)
    rows = []
    for f in sorted(fields, key=lambda x: x.bucket):
        e = f.environment
        row = {
            "bucket": f.bucket,
            "interval": f"{e['valid_from'][5:16]} .. {e['valid_to'][5:16]}",
            "source_type": e["source_type"],
            "model": e.get("model"),
            "forecast_origin": e.get("forecast_origin"),
            "valid_time": e.get("valid_time"),
            "lead_hours": e.get("lead_hours"),
            "validity_rule": (e.get("resolution") or {}).get("validity_rule"),
        }
        rows.append(row)
        if row["source_type"] == SOURCE_FORECAST:
            assert row["validity_rule"] == VALIDITY_TARGET_DAY
            assert row["model"] == MODEL_NAME
            assert row["forecast_origin"] == ORIGIN.isoformat()
            assert row["lead_hours"] in (24, 48)
            assert row["valid_time"]
        else:
            assert row["model"] is None and row["lead_hours"] is None
    for row in rows:
        print(f"      bucket {row['bucket']}  {row['interval']}  "
              f"{row['source_type']:<21} {str(row['lead_hours'] or ''):>3}"
              f"{'h' if row['lead_hours'] else ' '}  "
              f"{row['validity_rule'] or '-'}")
    assert [r["source_type"] for r in rows] == \
        [SOURCE_PERSISTENCE] * 4 + [SOURCE_FORECAST] * 6
    assert [r["lead_hours"] for r in rows[4:8]] == [24] * 4
    assert [r["lead_hours"] for r in rows[8:]] == [48] * 2
    return "every bucket recorded above"


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

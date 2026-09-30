"""
The SIC forecast artifact pipeline: trained model -> predicted field -> routing
grid -> validated GeoTIFF -> a resolver routing may ask, and be refused by.

These tests run against the REAL project data and the REAL trained artifact.
They skip, loudly, when an input is absent rather than inventing one.

    python tests/test_sic_forecast_raster.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.environment_forecast import (KIND_OBSERVED,                     # noqa: E402
                                          KIND_PREDICTED,
                                          POLICY_EXACT,
                                          POLICY_OBSERVED_ANALYSIS,
                                          SUPPORTED_LEAD_HOURS,
                                          EnvironmentForecastError,
                                          ForecastArtifactUnavailable,
                                          UnsupportedForecastTime,
                                          read_environment,
                                          resolve_environment,
                                          resolve_environment_forecast,
                                          supported_valid_times)
from src.data.preprocess import OBSERVED, RAW_DIR, load_sic                  # noqa: E402
from src.models.build_forecast_dataset import (CURRENTS_DIR,                 # noqa: E402
                                               FEATURE_NAMES)
from src.models.forecast_sic_raster import (LEAD_HOURS, OUT_DIR,             # noqa: E402
                                            TRAINED_LEAD_HOURS, model_label,
                                            SIC_HI, SIC_LO,
                                            ForecastRasterError,
                                            FutureInputError,
                                            assert_no_future_inputs,
                                            hindcast_validation, load_model,
                                            predict_sic, provenance,
                                            raster_path,
                                            validate_forecast_raster,
                                            write_forecast_raster)
from src.routing.grid import RoutingGrid                                     # noqa: E402

#  the demonstration's own environment date, and a date from the model's
#  held-out test window. Both are archive days with an observed +24 h field.
ORIGIN = date(2025, 1, 8)
HELD_OUT = date(2025, 12, 15)

_C: dict = {}


def _have(day: date) -> bool:
    return ((RAW_DIR / f"sic_{day:%Y%m%d}.tif").exists()
            and (CURRENTS_DIR / f"currents_{day:%Y%m%d}.tif").exists())


def forecast(origin: date = ORIGIN):
    """One prediction, computed once and reused across the tests."""
    if origin not in _C:
        if not _have(origin):
            raise AssertionError(f"archive inputs for {origin} are missing")
        _C[origin] = predict_sic(origin, model=_model())
    return _C[origin]


def _model():
    if "model" not in _C:
        _C["model"] = load_model()
    return _C["model"]


def written(origin: date = ORIGIN) -> Path:
    path = raster_path(origin)
    if not path.exists():
        write_forecast_raster(forecast(origin))
    return path


# ------------------------------------------------------------ 1. it exists
def test_a_24h_forecast_raster_can_be_generated():
    path = written()
    assert path.exists(), path
    with rasterio.open(path) as src:
        band = src.read(1)
        assert src.count == 1 and src.dtypes[0] == "float32"
    finite = np.isfinite(band)
    assert finite.any(), "the raster has no predicted cell at all"
    return (f"{path.name}: {int(finite.sum()):,} predicted cells, "
            f"{band[finite].min():.2f}..{band[finite].max():.2f} %")


def test_valid_time_is_origin_plus_the_models_own_lead():
    f = forecast()
    assert f.lead_hours == LEAD_HOURS == 24
    assert f.valid_time == f.origin_time + timedelta(hours=LEAD_HOURS)
    with rasterio.open(written()) as src:
        tags = src.tags()
    assert tags["forecast_origin"] == f.origin_time.isoformat()
    assert tags["valid_time"] == f.valid_time.isoformat()
    assert int(tags["lead_hours"]) == LEAD_HOURS
    return (f"origin {tags['forecast_origin']} -> valid {tags['valid_time']} "
            f"(+{tags['lead_hours']} h), in the raster's own tags")


# ------------------------------------------------------------- 2. leakage
def test_no_input_comes_from_after_the_forecast_origin():
    """THE LEAKAGE TEST. Every file the prediction opened, against the origin."""
    f = forecast()
    assert f.inputs, "the forecast recorded no inputs"
    for item in f.inputs:
        assert item.timestamp <= f.origin_time, (
            f"{item.role} {item.path.name} is dated {item.timestamp}, after "
            f"the origin {f.origin_time}")
        #  the file the input names is the origin day's, not the valid day's
        assert f"{f.origin.strftime('%Y%m%d')}" in item.path.name, item.path.name
        assert f"{f.valid_time.date():%Y%m%d}" not in item.path.name, (
            f"{item.role} reads the valid-time field: that is the target")
    #  and the guard itself bites when something later is handed to it
    from src.models.forecast_sic_raster import InputFile
    tomorrow = InputFile("sic_t1", RAW_DIR / "sic_x.tif",
                         f.origin_time + timedelta(hours=1))
    try:
        assert_no_future_inputs((*f.inputs, tomorrow), f.origin_time)
    except FutureInputError as exc:
        assert "postdate" in str(exc)
        return ("inputs " + ", ".join(i.path.name for i in f.inputs)
                + f" all <= {f.origin_time.isoformat()}; a later one is refused")
    raise AssertionError("a future input was accepted")


def test_the_target_field_is_never_a_feature():
    f = forecast()
    assert list(f.feature_names) == list(FEATURE_NAMES)
    assert all(name.endswith("_t") for name in f.feature_names), f.feature_names
    assert not any("t1" in name for name in f.feature_names), f.feature_names
    #  the valid-time observation is opened only by the validation path
    opened = {i.path.name for i in f.inputs}
    assert f"sic_{f.valid_time.date():%Y%m%d}.tif" not in opened
    return f"features {list(f.feature_names)}; the t+1 field is the target only"


# -------------------------------------------------------- 3. the grid
def test_the_raster_is_on_the_routing_grid():
    """CRS, dimensions, transform and pixel alignment, against the grid the
    router itself loads -- not against a constant typed in here."""
    grid = RoutingGrid.for_date(ORIGIN)
    report = validate_forecast_raster(written(), transform=tuple(grid.transform)[:6])
    assert report["epsg"] == OBSERVED.epsg == grid.crs.to_epsg()
    assert report["shape"] == [OBSERVED.height, OBSERVED.width]
    assert report["shape"] == [int(grid.shape[0]), int(grid.shape[1])]
    for got, want in zip(report["transform"], tuple(grid.transform)[:6]):
        assert abs(got - want) <= 1e-9, (got, want)
    return (f"EPSG:{report['epsg']} {report['shape']} transform "
            f"{report['transform'][:3]}... identical to RoutingGrid")


def test_a_mismatched_raster_is_rejected_not_reprojected():
    import tempfile
    from rasterio.transform import Affine
    bad = Path(tempfile.mkdtemp()) / "wrong_grid.tif"
    with rasterio.open(bad, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="float32", crs=rasterio.crs.CRS.from_epsg(4326),
                       transform=Affine(1, 0, 0, 0, -1, 0),
                       nodata=float("nan")) as dst:
        dst.write(np.full((4, 4), 50.0, "float32"), 1)
    try:
        validate_forecast_raster(bad)
    except ForecastRasterError as exc:
        assert "EPSG" in str(exc) and "shape" in str(exc)
        assert "reprojected at runtime" in str(exc)
        return "a 4x4 EPSG:4326 raster is refused, with every mismatch named"
    raise AssertionError("a mismatched raster passed validation")


# ------------------------------------------------- 4. values and masking
def test_predicted_values_stay_inside_the_physical_sic_bounds():
    f = forecast()
    values = f.sic[np.isfinite(f.sic)]
    assert values.size, "nothing was predicted"
    assert float(values.min()) >= SIC_LO - 1e-6
    assert float(values.max()) <= SIC_HI + 1e-6
    return (f"{values.size:,} cells within [{SIC_LO:g}, {SIC_HI:g}]: "
            f"{values.min():.2f}..{values.max():.2f}  "
            f"(clipped {f.clipped_cells:,})")


def test_invalid_input_cells_stay_invalid():
    """A cell without all its features is NaN in the output. It is never
    filled, interpolated or promoted to a concentration."""
    f = forecast()
    observed = np.asarray(load_sic(RAW_DIR / f"sic_{ORIGIN:%Y%m%d}.tif").sic,
                          dtype="float64")
    with rasterio.open(CURRENTS_DIR / f"currents_{ORIGIN:%Y%m%d}.tif") as src:
        u, v = src.read(1).astype("float64"), src.read(2).astype("float64")
    expected = np.isfinite(observed) & np.isfinite(u) & np.isfinite(v)

    assert np.array_equal(f.valid_mask, expected), "the valid mask was widened"
    assert np.array_equal(np.isfinite(f.sic), expected)
    assert np.isnan(f.sic[~expected]).all(), "an invalid cell carries a value"
    invalid = int((~expected).sum())
    return (f"{int(expected.sum()):,} predicted, {invalid:,} left NaN "
            f"(exactly the cells whose inputs were not all finite)")


def test_the_nodata_policy_is_explicit_in_the_file():
    with rasterio.open(written()) as src:
        assert src.nodata is not None and np.isnan(src.nodata)
        tags = src.tags()
    assert "never promoted" in tags["nodata_policy"]
    return f"nodata=nan, policy: {tags['nodata_policy'][:64]}..."


# --------------------------------------------------------- 5. provenance
def test_the_forecast_records_its_own_provenance():
    meta = provenance(forecast())
    for key in ("kind", "forecast_origin", "valid_time", "lead_hours",
                "model_name", "model_artifact", "model_artifact_sha256",
                "feature_names", "training_period_base_dates", "inputs",
                "target_crs", "source_grid", "transform", "grid_shape",
                "prediction_policy", "nodata_policy", "clipping"):
        assert key in meta, f"missing provenance key {key!r}"
    assert meta["kind"] == "FORECAST / HINDCAST VALIDATION"
    assert meta["is_operational_realtime_forecast"] is False
    assert len(meta["model_artifact_sha256"]) == 64
    assert meta["target_crs"] == f"EPSG:{OBSERVED.epsg}"
    assert meta["no_inputs_after_origin"] is True

    #  and no absolute path from this machine reaches the public metadata
    blob = json.dumps(meta)
    assert str(ROOT) not in blob, "an absolute local path leaked into metadata"
    for entry in meta["inputs"]:
        assert not entry["path"].startswith("/"), entry
        assert entry["path"].startswith("data/"), entry
    sidecar = written().with_suffix(".json")
    assert sidecar.exists() and json.loads(sidecar.read_text())["lead_hours"] == 24
    return (f"{len(meta)} provenance fields, artifact sha "
            f"{meta['model_artifact_sha256'][:12]}..., repo-relative paths only")


def test_the_forecast_is_deterministic():
    again = predict_sic(ORIGIN, model=_model())
    first = forecast()
    assert np.array_equal(np.nan_to_num(first.sic, nan=-1.0),
                          np.nan_to_num(again.sic, nan=-1.0))
    assert again.model_sha256 == first.model_sha256
    assert again.valid_time == first.valid_time
    return "same origin, same artifact -> bit-identical field"


# ---------------------------------------------------- 6. the resolver
def test_the_resolver_serves_the_exact_supported_valid_time():
    written()
    origin = datetime(ORIGIN.year, ORIGIN.month, ORIGIN.day)
    valid = origin + timedelta(hours=LEAD_HOURS)
    #  one entry per TRAINED lead; the default lead is among them
    assert supported_valid_times(origin) == [
        origin + timedelta(hours=h) for h in SUPPORTED_LEAD_HOURS]
    assert valid in supported_valid_times(origin)
    resolved = resolve_environment_forecast(origin, valid)
    assert resolved.kind == KIND_PREDICTED and resolved.is_predicted
    assert resolved.lead_hours == LEAD_HOURS
    assert resolved.path == raster_path(ORIGIN)
    field = read_environment(resolved)
    assert field.shape == (OBSERVED.height, OBSERVED.width)
    return (f"+{LEAD_HOURS} h resolves to {resolved.path.name} "
            f"({int(np.isfinite(field).sum()):,} predicted cells)")


def test_the_resolver_cannot_be_asked_for_6h_and_handed_the_24h_field():
    """THE ONE THAT MATTERS. A router asking for its 6 h or 12 h bucket must
    not receive the 24 h forecast under another name."""
    path = written()
    origin = datetime(ORIGIN.year, ORIGIN.month, ORIGIN.day)
    refused = []
    for hours in (6, 12, 18, 23, 25, 0.5):
        try:
            got = resolve_environment_forecast(
                origin, origin + timedelta(hours=hours))
        except UnsupportedForecastTime as exc:
            message = str(exc)
            assert path.name not in message, "the refusal handed back the file"
            assert "not returned for another lead" in message.lower()
            refused.append(f"+{hours:g}h")
            continue
        raise AssertionError(
            f"+{hours} h was served {got.path.name} -- the 24 h field was "
            f"relabelled as another lead")
    assert len(refused) == 6
    #  +48 h is a supported lead in its own right; it is answered from the
    #  +48 h artifact or refused for the want of one -- never from this file
    other = resolve_environment(origin, origin + timedelta(hours=48),
                                origin_time=origin)
    assert getattr(other, "path", None) != path, \
        "the +24 h artifact was handed over for a +48 h request"
    return ("refused " + ", ".join(refused)
            + "; +48h is served only from its own artifact")


def test_the_resolver_does_not_substitute_persistence_or_an_observation():
    origin = datetime(ORIGIN.year, ORIGIN.month, ORIGIN.day)
    #  a supported lead with no artifact: refused, not answered from elsewhere
    missing_origin = datetime(2025, 3, 3)
    try:
        resolve_environment_forecast(missing_origin,
                                     missing_origin + timedelta(hours=LEAD_HOURS),
                                     forecast_dir=Path("/nonexistent"))
    except ForecastArtifactUnavailable as exc:
        assert "No other origin" in str(exc) and "observation" in str(exc)
    else:
        raise AssertionError("a missing artifact was answered from somewhere")

    #  there is no persistence / latest / nearest policy to ask for
    for bad in ("persistence", "latest", "nearest", "whatever_exists"):
        try:
            resolve_environment_forecast(origin, origin, policy=bad)
        except EnvironmentForecastError as exc:
            assert "environment_timeline" in str(exc)
            continue
        raise AssertionError(f"policy {bad!r} was accepted")

    #  and the observation policy serves the analysis time only
    resolved = resolve_environment_forecast(origin, origin,
                                            policy=POLICY_OBSERVED_ANALYSIS)
    assert resolved.kind == KIND_OBSERVED and resolved.lead_hours == 0.0
    try:
        resolve_environment_forecast(origin, origin + timedelta(hours=6),
                                     policy=POLICY_OBSERVED_ANALYSIS)
    except UnsupportedForecastTime as exc:
        assert "persistence" in str(exc)
        return ("no artifact -> refused; no persistence/latest/nearest policy "
                "exists; the observation is served for its own time only")
    raise AssertionError("the observation was served for a later time")


def test_a_valid_time_before_the_origin_is_refused():
    origin = datetime(ORIGIN.year, ORIGIN.month, ORIGIN.day)
    try:
        resolve_environment_forecast(origin, origin - timedelta(hours=24))
    except UnsupportedForecastTime as exc:
        assert "before origin_time" in str(exc)
        return "a backwards request is refused rather than mirrored"
    raise AssertionError("a valid time before the origin was accepted")


# ------------------------------------------- the reported lead is the real one
def test_a_hindcast_is_labelled_with_the_lead_it_actually_ran():
    """A +48 h validation can never be reported as +24 h.

    The label comes from the lead the forecast CARRIES, so a run of one lead
    cannot borrow another's name -- in the report dict, in the metrics file, or
    in the line the CLI prints.
    """
    assert model_label(24) == "HGB +24h"
    assert model_label(48) == "HGB +48h"

    labels = {}
    for lead in TRAINED_LEAD_HOURS:
        f = predict_sic(ORIGIN, lead_hours=lead)
        assert f.lead_hours == lead
        report = hindcast_validation(f)
        assert report["lead_hours"] == lead
        assert report["model_label"] == f"HGB +{lead}h", report["model_label"]
        labels[lead] = report["model_label"]
        #  and no OTHER lead's name appears anywhere in the report
        blob = json.dumps(report)
        for other in TRAINED_LEAD_HOURS:
            if other != lead:
                assert f"HGB +{other}h" not in blob, (lead, other)
    assert labels[24] != labels[48]

    #  the label is derived, not typed: no fixed lead label exists in the source
    source = (ROOT / "src" / "models" / "forecast_sic_raster.py").read_text()
    for lead in TRAINED_LEAD_HOURS:
        assert f'"HGB +{lead}h"' not in source and f"'HGB +{lead}h'" not in source, \
            f"a fixed 'HGB +{lead}h' label is written into the source"
    #  and the lead never comes from a path
    assert "model_label(forecast.lead_hours)" in source
    return (f"{labels[24]} and {labels[48]}, each from its own run's validated "
            f"lead; no fixed label in the source")


def test_the_label_refuses_an_unknown_lead():
    for bad in (None, "plus48h", 0, -24, "24h"):
        try:
            model_label(bad)
        except ForecastRasterError:
            continue
        raise AssertionError(f"model_label accepted {bad!r}")
    return "a missing or non-numeric lead is refused rather than defaulted"


# -------------------------------------------------- 7. hindcast scoring
def test_the_hindcast_is_scored_against_the_real_observation():
    report = hindcast_validation(forecast())
    assert report["kind"] == "FORECAST / HINDCAST VALIDATION"
    assert report["is_operational_realtime_forecast"] is False
    hgb, per = report["metrics"]["hgb"], report["metrics"]["persistence"]
    for block in (hgb, per):
        for key in ("n", "mae", "rmse", "bias", "mae_miz", "mae_open",
                    "mae_miz", "mae_ice"):
            assert key in block, key
        assert block["n"] == report["scored_cells"]
        assert np.isfinite(block["mae"]) and np.isfinite(block["rmse"])
    #  persistence here is the project's published baseline, SIC(t+1) = SIC(t)
    observed_t = np.asarray(load_sic(RAW_DIR / f"sic_{ORIGIN:%Y%m%d}.tif").sic,
                            dtype="float64")
    truth = np.asarray(
        load_sic(RAW_DIR / f"sic_{ORIGIN + timedelta(days=1):%Y%m%d}.tif").sic,
        dtype="float64")
    pred = np.asarray(forecast().sic, dtype="float64")
    scored = np.isfinite(pred) & np.isfinite(observed_t) & np.isfinite(truth)
    assert int(scored.sum()) == report["scored_cells"]
    expected = float(np.abs(observed_t[scored] - truth[scored]).mean())
    assert abs(per["mae"] - expected) < 1e-9, (per["mae"], expected)
    return (f"{report['scored_cells']:,} cells: HGB MAE {hgb['mae']:.3f} / "
            f"RMSE {hgb['rmse']:.3f}, persistence MAE {per['mae']:.3f} / "
            f"RMSE {per['rmse']:.3f}"
            + ("   [origin inside the TRAINING period: in-sample]"
               if report["origin_is_inside_the_training_period"] else ""))


def test_a_held_out_origin_is_scored_the_same_way():
    """The same pipeline on a date from the model's own test window, so the
    numbers reported are not only in-sample ones."""
    if not _have(HELD_OUT):
        raise AssertionError(f"archive inputs for {HELD_OUT} are missing")
    f = forecast(HELD_OUT)
    report = hindcast_validation(f)
    assert report["origin_is_inside_the_training_period"] is False
    hgb, per = report["metrics"]["hgb"], report["metrics"]["persistence"]
    return (f"{HELD_OUT} (held out): HGB MAE {hgb['mae']:.3f} / RMSE "
            f"{hgb['rmse']:.3f} vs persistence MAE {per['mae']:.3f} / RMSE "
            f"{per['rmse']:.3f} on {report['scored_cells']:,} cells")


def test_the_observed_archive_is_untouched():
    """The pipeline reads the archive and writes only under data/processed."""
    before = {p.name: p.stat().st_mtime for p in RAW_DIR.glob("sic_*.tif")}
    predict_sic(ORIGIN, model=_model())
    after = {p.name: p.stat().st_mtime for p in RAW_DIR.glob("sic_*.tif")}
    assert before == after, "the prediction modified the observation archive"
    assert OUT_DIR.is_relative_to(ROOT / "data" / "processed")
    return f"{len(before)} observed rasters unchanged; output under {OUT_DIR.name}/"


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

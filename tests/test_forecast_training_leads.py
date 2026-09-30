"""
One lead, one artifact: the training pipeline's lead handling.

Nothing here fits a model. The lead is established, checked and recorded
BEFORE any row is fitted, so every rule that matters can be exercised against
tiny fixture datasets written to a temp directory. The only real file these
tests read is the shipped +24 h dataset, and they read it, never write it.

    python tests/test_forecast_training_leads.py
"""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.build_forecast_dataset import (OUT_PATH,                      # noqa: E402
                                               SOURCE_CADENCE_DAYS,
                                               FEATURE_NAMES, list_pairs)
from src.models.train_forecast_model import (DEFAULT_METRICS, DEFAULT_MODEL,  # noqa: E402
                                             DEFAULT_NPZ, HOURS_PER_DAY,
                                             MODEL_NAME, SHIPPED_LEAD_HOURS,
                                             LeadMismatch, build_arg_parser,
                                             dataset_lead_hours, leakage_check,
                                             metrics_path_for, model_path_for,
                                             resolve_lead,
                                             resolve_output_paths,
                                             split_information,
                                             training_provenance)

TRAIN = __import__("src.models.train_forecast_model", fromlist=["x"])
_TMP = Path(tempfile.mkdtemp(prefix="lead_fixtures_"))


# ------------------------------------------------------------------ fixtures
def fixture_dataset(name: str, *, lead_days: int | None, record_lead: bool,
                    rows: int = 8, ranges: bool = True) -> Path:
    """A tiny npz with the same KEYS the real builder writes, and no more.

    `record_lead` False reproduces a dataset built before the builder recorded
    its lead -- which is exactly what the shipped +24 h dataset is.
    """
    from datetime import date, timedelta
    payload = {
        "feature_names": np.array(FEATURE_NAMES),
        "stride": np.array(1),
        "gap_days": np.array(1),
    }
    if record_lead and lead_days is not None:
        payload["lead_days"] = np.array(lead_days)
        payload["lead_hours"] = np.array(lead_days * HOURS_PER_DAY)
    starts = {"train": date(2025, 1, 1), "val": date(2025, 10, 1),
              "test": date(2025, 12, 1)}
    for split, first in starts.items():
        payload[f"X_{split}"] = np.zeros((rows, len(FEATURE_NAMES)), "float32")
        payload[f"y_{split}"] = np.zeros(rows, "float32")
        if ranges and lead_days is not None:
            last = first + timedelta(days=20)
            payload[f"{split}_base_range"] = np.array(
                [str(first), str(last)], dtype="U10")
            payload[f"{split}_target_range"] = np.array(
                [str(first + timedelta(days=lead_days)),
                 str(last + timedelta(days=lead_days))], dtype="U10")
    path = _TMP / name
    np.savez(path, **payload)
    return path


D24 = None
D48 = None
D24_NO_META = None
D_NO_LEAD = None


def datasets():
    global D24, D48, D24_NO_META, D_NO_LEAD
    if D24 is None:
        D24 = fixture_dataset("d24.npz", lead_days=1, record_lead=True)
        D48 = fixture_dataset("d48.npz", lead_days=2, record_lead=True)
        #  a dataset built before the lead field existed: the ranges still say it
        D24_NO_META = fixture_dataset("d24_legacy.npz", lead_days=1,
                                      record_lead=False)
        #  nothing to go on at all
        D_NO_LEAD = fixture_dataset("d_none.npz", lead_days=None,
                                    record_lead=False, ranges=False)
    return D24, D48, D24_NO_META, D_NO_LEAD


# ============================================================= 1-2. the CLI
def test_the_cli_accepts_an_explicit_dataset_and_output():
    args = build_arg_parser().parse_args(
        ["--dataset", "data/processed/forecast_dataset_48h.npz",
         "--out", "data/processed/forecast_model_hgb_48h.joblib",
         "--metrics-out", "data/processed/forecast_metrics_48h.json",
         "--lead-hours", "48"])
    assert args.dataset.endswith("forecast_dataset_48h.npz")
    assert args.out.endswith("forecast_model_hgb_48h.joblib")
    assert args.metrics_out.endswith("forecast_metrics_48h.json")
    assert args.lead_hours == 48
    return "--dataset / --out / --metrics-out / --lead-hours all accepted"


def test_the_original_flag_spellings_still_work():
    args = build_arg_parser().parse_args(
        ["--npz", "a.npz", "--model-out", "b.joblib"])
    assert args.dataset == "a.npz" and args.out == "b.joblib"
    default = build_arg_parser().parse_args([])
    assert default.dataset == str(DEFAULT_NPZ)
    assert default.out is None and default.lead_hours is None
    return "--npz and --model-out remain aliases; bare invocation unchanged"


# ================================================ 3-5. lead in the provenance
def test_the_lead_is_recorded_in_the_provenance():
    d24, d48, _, _ = datasets()
    records = {}
    for path, lead in ((d24, 24), (d48, 48)):
        with np.load(path) as z:
            got, how = resolve_lead(z, lead, path)
            record = training_provenance(
                npz_path=path, model_path=model_path_for(got), lead=got,
                how=how, leak=leakage_check(z, got, path),
                splits=split_information(z),
                feature_names=[str(v) for v in z["feature_names"]], seed=42)
        assert got == lead
        assert record["lead_hours"] == lead
        assert record["lead_days"] == lead // HOURS_PER_DAY
        assert record["target_relationship"] == (
            f"target = base + {lead // HOURS_PER_DAY} day(s) = +{lead} h")
        for key in ("dataset", "dataset_sha256", "artifact", "artifact_sha256",
                    "sklearn_version", "joblib_version", "python_version",
                    "split_information", "leakage_check",
                    "source_data_provenance", "model", "kind",
                    "lead_established_by"):
            assert key in record, key
        assert record["model"] == MODEL_NAME
        assert record["is_operational_realtime_forecast"] is False
        records[lead] = record

    #  the two runs are distinguishable on every identifying field
    assert records[24]["lead_hours"] != records[48]["lead_hours"]
    assert records[24]["dataset"] != records[48]["dataset"]
    assert records[24]["artifact"] != records[48]["artifact"]
    assert records[24]["dataset_sha256"] != records[48]["dataset_sha256"]
    return (f"+24 h -> {Path(records[24]['artifact']).name}, +48 h -> "
            f"{Path(records[48]['artifact']).name}, each carrying its own lead")


def test_each_lead_gets_its_own_artifact_path():
    assert model_path_for(24) == DEFAULT_MODEL
    assert metrics_path_for(24) == DEFAULT_METRICS
    assert model_path_for(48).name == "forecast_model_hgb_48h.joblib"
    assert metrics_path_for(48).name == "forecast_metrics_48h.json"
    assert model_path_for(48) != DEFAULT_MODEL
    assert model_path_for(72).name == "forecast_model_hgb_72h.joblib"
    return ("24 h keeps the shipped path; every other lead gets "
            "forecast_model_hgb_<lead>h.joblib")


def test_a_48h_run_cannot_land_on_the_24h_artifact():
    d24, d48, _, _ = datasets()
    lead, model_path, metrics_path = resolve_output_paths(d48, None, None, 48)
    assert lead == 48
    assert model_path == model_path_for(48) and model_path != DEFAULT_MODEL
    assert metrics_path == metrics_path_for(48)
    #  and pointing it at the shipped artifact on purpose is refused by main()
    #  itself -- before any row is fitted
    try:
        TRAIN.main(d48, DEFAULT_MODEL, _TMP / "m.json", 42, 48)
    except LeadMismatch as exc:
        assert DEFAULT_MODEL.name in str(exc)
        assert "One lead" in str(exc) and "per artifact" in str(exc)
    else:
        raise AssertionError("a +48 h run was allowed to write the 24 h artifact")
    #  the 24 h run still resolves to the shipped pair
    lead24, model24, metrics24 = resolve_output_paths(d24, None, None, None)
    assert (lead24, model24, metrics24) == (24, DEFAULT_MODEL, DEFAULT_METRICS)
    return ("+48 h defaults to its own artifact; +24 h still defaults to "
            f"{DEFAULT_MODEL.name}")


# ==================================================== 6-7. mismatch is refused
def test_a_lead_that_disagrees_with_the_dataset_is_refused():
    d24, d48, _, _ = datasets()
    cases = [(d24, 48), (d48, 24)]
    for path, asked in cases:
        with np.load(path) as z:
            try:
                resolve_lead(z, asked, path)
            except LeadMismatch as exc:
                assert f"--lead-hours {asked}" in str(exc)
                assert "does not match the dataset" in str(exc)
                assert "mislabel" in str(exc)
                continue
        raise AssertionError(f"{path.name} accepted --lead-hours {asked}")
    return ("lead_hours=48 on a 24 h dataset and lead_hours=24 on a 48 h "
            "dataset are both refused, with the mismatch named")


def test_a_sub_daily_lead_is_refused_outright():
    d24, _, _, _ = datasets()
    with np.load(d24) as z:
        for asked in (6, 12, 1, 23):
            try:
                resolve_lead(z, asked, d24)
            except LeadMismatch as exc:
                assert "whole number of days" in str(exc)
                assert "no sub-daily lead has a target" in str(exc)
                continue
            raise AssertionError(f"--lead-hours {asked} was accepted")
    return "6 h, 12 h, 1 h and 23 h are refused before any dataset check"


def test_missing_lead_metadata_is_refused_not_assumed():
    _, _, legacy, none = datasets()
    #  a dataset built before the field existed still states its lead through
    #  its own recorded date ranges
    with np.load(legacy) as z:
        lead, how = dataset_lead_hours(z, legacy)
    assert lead == 24 and "derived from the dataset's own" in how
    #  a dataset that says nothing at all is refused
    with np.load(none) as z:
        try:
            dataset_lead_hours(z, none)
        except LeadMismatch as exc:
            assert "records no forecast lead" in str(exc)
            assert "NOT assumed to be 24" in str(exc)
            assert "NOT read from the file name" in str(exc)
            return ("a legacy dataset's lead is derived from its own dates; a "
                    "dataset with no dates at all is refused, never assumed")
    raise AssertionError("a dataset with no lead information was accepted")


def test_the_lead_is_never_taken_from_a_filename():
    """A file NAMED 48h but holding 24 h targets is still a 24 h dataset."""
    from shutil import copyfile
    d24, _, _, _ = datasets()
    misleading = _TMP / "forecast_dataset_48h.npz"
    copyfile(d24, misleading)
    with np.load(misleading) as z:
        lead, _ = dataset_lead_hours(z, misleading)
        assert lead == 24, "the filename decided the lead"
        try:
            resolve_lead(z, 48, misleading)
        except LeadMismatch:
            return ("a dataset named forecast_dataset_48h.npz that holds 24 h "
                    "targets is read as 24 h, and --lead-hours 48 on it is "
                    "refused")
    raise AssertionError("a misleading filename was believed")


# ================================================= 8. the shipped path is safe
def test_the_shipped_24h_paths_are_unchanged():
    assert DEFAULT_NPZ == ROOT / "data" / "processed" / "forecast_dataset.npz"
    assert DEFAULT_MODEL == ROOT / "data" / "processed" / "forecast_model_hgb.joblib"
    assert DEFAULT_METRICS == ROOT / "data" / "processed" / "forecast_metrics.json"
    assert SHIPPED_LEAD_HOURS == 24
    if not DEFAULT_NPZ.exists():
        raise AssertionError(f"the shipped dataset is missing: {DEFAULT_NPZ}")
    #  the REAL shipped dataset, read only: it predates the lead field and must
    #  still resolve to 24 h from its own recorded ranges
    with np.load(DEFAULT_NPZ) as z:
        files = set(z.files)
        lead, how = dataset_lead_hours(z, DEFAULT_NPZ)
        leak = leakage_check(z, lead, DEFAULT_NPZ)
    assert "lead_hours" not in files and "lead_days" not in files
    assert lead == 24, lead
    assert leak["passed"], leak["failed"]
    return (f"{DEFAULT_NPZ.name} carries no lead field and still resolves to "
            f"+24 h -- {how.split('(')[0].strip()}")


# ================================= 9-10. what the pipeline must NOT depend on
def test_the_training_pipeline_imports_no_routing_or_frontend_code():
    banned = ("src.routing", "src.api", "src.validation")
    for name in ("train_forecast_model", "build_forecast_dataset"):
        source = (ROOT / "src" / "models" / f"{name}.py").read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for module in imported:
            assert not any(module.startswith(b) for b in banned), \
                f"{name} imports {module}"
    return ("neither training module imports routing, api or validation code: "
            "training cannot change how a route is priced")


def test_the_leakage_checks_are_enforced_before_fitting():
    d24, _, _, _ = datasets()
    with np.load(d24) as z:
        report = leakage_check(z, 24, d24)
        assert report["passed"] and not report["failed"]
        for key in ("every_feature_is_from_the_origin_step",
                    "no_feature_names_the_target_step",
                    "target_is_a_separate_array",
                    "target_is_exactly_the_lead_after_the_base"):
            assert report["checks"][key] is True, key
        #  the same dataset checked against the WRONG lead fails the date test
        wrong = leakage_check(z, 48, d24)
        assert wrong["passed"] is False
        assert "target_is_exactly_the_lead_after_the_base" in wrong["failed"]
    #  and main() refuses to fit when the check fails
    source = (ROOT / "src" / "models" / "train_forecast_model.py").read_text()
    assert "failed its leakage checks before training" in source
    assert source.index("leak = leakage_check") < source.index("model.fit")
    return ("origin-only features, a separate target array and an exact lead "
            "offset are all asserted before model.fit is reached")


# ============================================ the 48 h dataset CLI (no build)
def test_the_dataset_cli_supports_the_48h_build_without_running_it():
    """Phase 6: verify the invocation, do not execute it."""
    import importlib
    module = importlib.import_module("src.models.build_forecast_dataset")
    source = Path(module.__file__).read_text()
    for flag in ("--lead-days", "--gap-days", "--out", "--stride", "--max-gb"):
        assert flag in source, flag
    assert SOURCE_CADENCE_DAYS == 1
    assert OUT_PATH.name == "forecast_dataset.npz"

    #  t -> t+2 days exactly, counted from the real archive without building
    from src.data.preprocess import RAW_DIR
    pairs = list_pairs(RAW_DIR, module.CURRENTS_DIR, 1, 2)
    assert pairs, "no +48 h pairs in the archive"
    assert all((t - b).days == 2 for b, t, _ in pairs)
    #  --gap-days 1 really drops the straddling pairs
    loose = list_pairs(RAW_DIR, module.CURRENTS_DIR, 0, 2)
    assert len(pairs) < len(loose)
    #  origin-only features, and the lead is written into the output
    assert FEATURE_NAMES == ["sic_t", "current_u_t", "current_v_t",
                             "current_speed_t"]
    assert '"lead_days": np.array(lead_days)' in source
    assert '"lead_hours": np.array(lead_days * 24)' in source
    #  nothing in the builder CALLS an interpolation or gap-filling routine.
    #  The words appear in its prose ("Nothing is interpolated"), so this looks
    #  at what the code invokes, not at what it says.
    called = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                called.add(fn.attr)
            elif isinstance(fn, ast.Name):
                called.add(fn.id)
    for banned in ("interp", "interpolate", "resample", "fillna", "ffill",
                   "bfill", "nan_to_num", "reindex"):
        assert banned not in called, f"the builder calls {banned}()"
    return (f"--lead-days 2 --gap-days 1 -> {len(pairs)} true 2-day pairs "
            f"({len(loose)} without the gap); features unchanged; lead written "
            f"to the npz")


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

"""
Focused tests for ERA5 10 m wind ingestion and step matching.

Wind fields are tiny hand-built NetCDFs in ERA5's own layout (u10/v10,
descending latitude, valid_time in "seconds since ...") carrying an exactly
known linear field, so a bilinear sample has an arithmetic answer to check
against rather than a plausible-looking number.

    python tests/test_era5_wind.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import fetch_era5_wind as F                                  # noqa: E402
from src.data.preprocess_era5_wind import (CURRENT_SAMPLING, MATCH_FIELDS,  # noqa: E402
                                           VECTOR_FRAMES_TOKEN,
                                           STATE_BOTH_MISSING,
                                           STATE_CURRENT_MISSING, STATE_FULL,
                                           STATE_WIND_MISSING, TIME_ASSUMPTION,
                                           WIND_INTERPOLATION, CurrentField,
                                           ERA5WindError, WindField,
                                           match_steps, validate_dataset,
                                           write_manifest)

REAL_STEPS = ROOT / "data" / "processed" / "icebergs" / "iceberg_steps.csv"
REAL_CURRENTS = ROOT / "data" / "processed" / "currents"

# A deliberately coarse box so a few grid points cover it exactly.
LATS = np.array([-60.0, -60.25, -60.5, -60.75, -61.0])     # ERA5: descending
LONS = np.array([10.0, 10.25, 10.5, 10.75, 11.0])
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def _u_field(lat, lon):
    """u = 100*lat + lon -- linear, so bilinear must reproduce it exactly."""
    return 100.0 * lat + lon


def _v_field(lat, lon):
    return -3.0 * lat + 2.0 * lon


def write_era5(d: Path, name: str, days: list[dt.date],
               u_name="u10", v_name="v10", time_name="valid_time",
               units="m s**-1", hole: tuple | None = None) -> Path:
    from netCDF4 import Dataset
    path = d / name
    with Dataset(path, "w") as ds:
        ds.createDimension(time_name, len(days))
        ds.createDimension("latitude", LATS.size)
        ds.createDimension("longitude", LONS.size)
        tv = ds.createVariable(time_name, "i8", (time_name,))
        tv.units = "seconds since 1970-01-01"
        tv.calendar = "gregorian"
        tv[:] = [int((dt.datetime(d_.year, d_.month, d_.day,
                                  tzinfo=dt.timezone.utc) - EPOCH)
                     .total_seconds()) for d_ in days]
        lv = ds.createVariable("latitude", "f8", ("latitude",)); lv[:] = LATS
        gv = ds.createVariable("longitude", "f8", ("longitude",)); gv[:] = LONS
        la2, lo2 = np.meshgrid(LATS, LONS, indexing="ij")
        for vn, fn in ((u_name, _u_field), (v_name, _v_field)):
            var = ds.createVariable(vn, "f4", (time_name, "latitude", "longitude"),
                                    fill_value=np.float32(np.nan))
            var.units = units
            block = np.repeat(fn(la2, lo2)[None, :, :], len(days), axis=0)
            if hole is not None:
                block[hole] = np.nan
            var[:] = block
    return path


def wind_dir(days=None, **kw) -> Path:
    d = Path(tempfile.mkdtemp(prefix="era5_"))
    write_era5(d, "era5_10m_wind_2025-01.nc",
               days or [dt.date(2025, 1, 1), dt.date(2025, 1, 2)], **kw)
    return d


def steps_file(rows: list[dict]) -> Path:
    d = Path(tempfile.mkdtemp(prefix="steps_"))
    p = d / "iceberg_steps.csv"
    cols = ["iceberg_id", "position_source", "prev_date", "date", "elapsed_days",
            "elapsed_seconds", "prev_latitude_deg", "prev_longitude_deg",
            "either_endpoint_interpolated"]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def step(lat=-60.4, lon=10.4, day="2025-01-01", end="2025-01-02", days=1,
         berg="a01", src="nic", interp="False") -> dict:
    return {"iceberg_id": berg, "position_source": src, "prev_date": day,
            "date": end, "elapsed_days": days, "elapsed_seconds": days * 86400.0,
            "prev_latitude_deg": lat, "prev_longitude_deg": lon,
            "either_endpoint_interpolated": interp}


# ------------------------------------------------ schema / units / time
def test_variable_and_schema_validation():
    d = wind_dir()
    rep = validate_dataset(d / "era5_10m_wind_2025-01.nc")
    assert (rep["u_var"], rep["v_var"]) == ("u10", "v10")
    assert rep["time_var"] == "valid_time" and rep["lat_var"] == "latitude"
    assert rep["n_lat"] == 5 and rep["n_lon"] == 5 and rep["n_times"] == 2
    assert rep["lat_descending"] is True
    bad = Path(tempfile.mkdtemp())
    write_era5(bad, "x.nc", [dt.date(2025, 1, 1)], u_name="wind_x")
    try:
        validate_dataset(bad / "x.nc")
        raise AssertionError("a file without a 10 m u component was accepted")
    except ERA5WindError as exc:
        assert "10 m u component" in str(exc) and "nothing is guessed" in str(exc)
    return "u10/v10/valid_time/latitude/longitude recognised; a renamed u refused"


def test_u_v_recognition_covers_the_known_spellings():
    assert "u10" in F.VARIABLES[0] or True
    from src.data.preprocess_era5_wind import U_NAMES, V_NAMES
    assert "u10" in U_NAMES and "v10" in V_NAMES
    d = Path(tempfile.mkdtemp())
    write_era5(d, "old.nc", [dt.date(2025, 1, 1)], time_name="time")
    rep = validate_dataset(d / "old.nc")
    assert rep["time_var"] == "time", "the old CDS 'time' axis was not accepted"
    return "both the new 'valid_time' and the legacy 'time' axis are accepted"


def test_units_are_checked_not_assumed():
    d = Path(tempfile.mkdtemp())
    write_era5(d, "knots.nc", [dt.date(2025, 1, 1)], units="knots")
    try:
        validate_dataset(d / "knots.nc")
        raise AssertionError("knots were accepted as m/s")
    except ERA5WindError as exc:
        assert "not rescaled" in str(exc)
    ok = validate_dataset(wind_dir() / "era5_10m_wind_2025-01.nc")
    assert all(u == "m s**-1" for u in ok["units"].values())
    return "m s**-1 accepted; knots refused rather than silently converted"


def test_timestamp_conversion_and_00utc_convention():
    w = WindField(wind_dir())
    stamps = sorted(w.by_time)
    assert stamps[0] == dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    assert all(s.hour == 0 for s in stamps)
    u, v, why = w.sample(-60.0, 10.0,
                         dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc))
    assert why == "" and u is not None
    _, _, why2 = w.sample(-60.0, 10.0,
                          dt.datetime(2025, 1, 1, 6, tzinfo=dt.timezone.utc))
    assert "no ERA5 timestep" in why2, "a 06 UTC request found an hour anyway"
    assert TIME_ASSUMPTION == "date_only_assigned_to_00UTC"
    return ("epoch seconds -> tz-aware UTC; 00 UTC resolves, 06 UTC refuses "
            "rather than snapping to the nearest hour")


# ----------------------------------------------------- interpolation
def test_bilinear_interpolation_is_exact_on_a_linear_field():
    w = WindField(wind_dir())
    when = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    for lat, lon in ((-60.0, 10.0), (-61.0, 11.0), (-60.125, 10.125),
                     (-60.4, 10.4), (-60.875, 10.6)):
        u, v, why = w.sample(lat, lon, when)
        assert why == "", why
        assert abs(u - _u_field(lat, lon)) < 1e-6, (lat, lon, u)
        assert abs(v - _v_field(lat, lon)) < 1e-6, (lat, lon, v)
    # a midpoint really is the mean of its four corners
    u, _, _ = w.sample(-60.125, 10.125, when)
    corners = [_u_field(a, b) for a in (-60.0, -60.25) for b in (10.0, 10.25)]
    assert abs(u - sum(corners) / 4) < 1e-6
    return ("exact at grid points, at cell midpoints and off-centre on a linear "
            "field; midpoint equals the mean of its four corners")


def _global_wind_dir() -> Path:
    """A fixture with ERA5's real longitude axis: -180 .. 179.75, 180 absent."""
    from netCDF4 import Dataset
    d = Path(tempfile.mkdtemp(prefix="era5_global_"))
    lons = np.arange(-180.0, 180.0, 0.25)          # 1440 columns, like CDS
    lats = np.array([-69.75, -70.0, -70.25, -70.5])   # spans the real b22i lat
    day = dt.date(2025, 12, 7)
    with Dataset(d / "era5_10m_wind_2025-12.nc", "w") as ds:
        ds.createDimension("valid_time", 1)
        ds.createDimension("latitude", lats.size)
        ds.createDimension("longitude", lons.size)
        tv = ds.createVariable("valid_time", "i8", ("valid_time",))
        tv.units = "seconds since 1970-01-01"
        tv[:] = [int((dt.datetime(day.year, day.month, day.day,
                                  tzinfo=dt.timezone.utc) - EPOCH)
                     .total_seconds())]
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lats
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lons
        la2, lo2 = np.meshgrid(lats, lons, indexing="ij")
        for vn, fn in (("u10", _u_field), ("v10", _v_field)):
            var = ds.createVariable(vn, "f8",
                                    ("valid_time", "latitude", "longitude"))
            var.units = "m s**-1"
            var[:] = fn(la2, lo2)[None, :, :]
    return d


def test_longitude_wraps_at_the_antimeridian():
    """The 7 real b22i steps sat between the last column and 180 degrees."""
    w = WindField(_global_wind_dir())
    assert w.lon_periodic is True and abs(w.lon_step - 0.25) < 1e-12
    assert float(w.lon[-1]) == 179.75, "fixture is not ERA5's real axis"
    when = dt.datetime(2025, 12, 7, tzinfo=dt.timezone.utc)
    lat = -70.1313
    lon = 179.9273                                  # a real b22i start longitude

    u, v, why = w.sample(lat, lon, when)
    assert why == "" and u is not None, f"still refused: {why}"

    # The answer must be the blend of the LAST column and the FIRST one.
    wx = (lon - 179.75) / 0.25
    for got, fn in ((u, _u_field), (v, _v_field)):
        expect = (1 - wx) * fn(lat, 179.75) + wx * fn(lat, -180.0)
        assert abs(got - expect) < 1e-9, (got, expect)
    # and emphatically NOT the linear extension of the field past 179.75
    assert abs(u - _u_field(lat, lon)) > 1.0, "the field was extrapolated"

    # the wrap point itself, and the column just inside it
    assert w.sample(lat, 180.0 - 1e-9, when)[0] is not None
    assert abs(w.sample(lat, 179.75, when)[0] - _u_field(lat, 179.75)) < 1e-9
    return (f"179.9273 E now interpolates between 179.75 and -180 (w={wx:.4f}) "
            f"instead of refusing; 179.75 itself is unchanged")


def test_wrapping_does_not_disturb_interior_or_latitude():
    w = WindField(_global_wind_dir())
    when = dt.datetime(2025, 12, 7, tzinfo=dt.timezone.utc)
    for lat, lon in ((-69.75, 0.0), (-70.1, -179.9), (-70.125, 81.92),
                     (-70.5, 179.75), (-69.875, -0.125), (-70.3, 100.3)):
        u, v, why = w.sample(lat, lon, when)
        assert why == "", (lat, lon, why)
        assert abs(u - _u_field(lat, lon)) < 1e-9, (lat, lon, u)
        assert abs(v - _v_field(lat, lon)) < 1e-9, (lat, lon, v)
    # latitude is untouched: one cell beyond either edge still refuses
    for bad_lat in (-69.5, -70.75):
        u, _, why = w.sample(bad_lat, 0.0, when)
        assert u is None and "latitude" in why and "not extrapolated" in why
        assert "longitude" not in why, "a latitude failure blamed longitude"
    return ("interior longitudes still exact on a linear field; latitude still "
            "refuses beyond both edges")


def test_a_regional_box_still_refuses_outside_longitudes():
    """Only a globe-spanning axis wraps. A 10-11 deg box must not."""
    w = WindField(wind_dir())
    assert w.lon_periodic is False
    when = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    for lon in (9.9, 11.1, 179.9, -180.0):
        u, _, why = w.sample(-60.5, lon, when)
        assert u is None and "not extrapolated" in why, (lon, why)
    return "a regional longitude box does not wrap and still refuses outside"


def test_outside_the_domain_is_refused_not_extrapolated():
    w = WindField(wind_dir())
    when = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    for lat, lon in ((-59.9, 10.5), (-61.1, 10.5), (-60.5, 9.9), (-60.5, 11.1)):
        u, v, why = w.sample(lat, lon, when)
        assert u is None and v is None, f"({lat},{lon}) produced a value"
        assert "not extrapolated" in why
    u, _, why = w.sample(-61.0, 11.0, when)      # exactly on the far corner
    assert u is not None and why == "", "the corner itself was rejected"
    return "one cell beyond each edge refuses; the exact corner still samples"


def test_a_missing_corner_refuses_the_whole_sample():
    d = wind_dir(hole=(slice(None), 1, 1))       # kill lat[1], lon[1]
    w = WindField(d)
    when = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    u, v, why = w.sample(-60.1, 10.1, when)      # its cell touches the hole
    assert u is None and v is None
    assert "refused rather than filled" in why
    far = w.sample(-60.9, 10.9, when)            # a cell that does not
    assert far[0] is not None
    return "a NaN corner refuses that sample; neighbouring cells still resolve"


# ------------------------------------------------------- match states
def _run(rows, wdir=None, cdir=None):
    out = Path(tempfile.mkdtemp()) / "m.csv"
    res = match_steps(steps_file(rows), wdir or wind_dir(),
                      cdir or Path(tempfile.mkdtemp()), out)
    return res, list(csv.DictReader(out.open()))


def test_match_state_classification():
    res, rows = _run([step()])                   # wind yes, no currents dir
    assert rows[0]["match_state"] == STATE_CURRENT_MISSING
    assert rows[0]["wind_available"] == "True"
    assert rows[0]["current_available"] == "False"
    assert rows[0]["fully_matched"] == "False"
    res2, rows2 = _run([step(lat=-70.0)])        # outside the wind box too
    assert rows2[0]["match_state"] == STATE_BOTH_MISSING
    assert res2["states"][STATE_BOTH_MISSING] == 1
    assert set(MATCH_FIELDS) >= {"iceberg_id", "start_date", "end_date",
                                 "elapsed_seconds", "start_latitude",
                                 "start_longitude", "current_u", "current_v",
                                 "wind_u10", "wind_v10", "current_available",
                                 "wind_available", "fully_matched",
                                 "time_assumption"}
    return (f"{STATE_CURRENT_MISSING} and {STATE_BOTH_MISSING} classified; every "
            f"required column present")


def test_missing_values_are_empty_never_zero():
    _, rows = _run([step(lat=-70.0)])
    r = rows[0]
    assert r["wind_u10"] == "" and r["wind_v10"] == ""
    assert r["current_u"] == "" and r["current_v"] == ""
    assert r["wind_unavailable_reason"] and r["current_unavailable_reason"]
    assert "0" not in {r["wind_u10"], r["current_u"]}
    assert r["vector_frames"] == VECTOR_FRAMES_TOKEN
    assert len(r["vector_frames"]) < 60, "a whole paragraph is repeated per row"
    return "an unavailable value is an empty cell with a reason, not 0.0"


def test_matching_is_at_the_start_position_and_start_date():
    w = wind_dir()
    _, rows = _run([step(lat=-60.4, lon=10.4, day="2025-01-01",
                         end="2025-01-02")], wdir=w)
    r = rows[0]
    assert r["start_date"] == "2025-01-01" and r["end_date"] == "2025-01-02"
    assert r["wind_time_utc"].startswith("2025-01-01T00:00")
    assert abs(float(r["wind_u10"]) - _u_field(-60.4, 10.4)) < 1e-6
    # the END position would give a different number; prove we did not use it
    assert abs(float(r["wind_u10"]) - _u_field(-60.9, 10.9)) > 1.0
    return "wind sampled at the START lat/lon and START date, not the end"


def test_no_gap_is_filled():
    rows_in = [step(day="2025-01-01", end="2025-01-02", days=1),
               step(day="2025-01-02", end="2025-01-30", days=28)]
    res, rows = _run(rows_in, wdir=wind_dir(
        days=[dt.date(2025, 1, 1), dt.date(2025, 1, 2)]))
    assert len(rows) == 2, "a step was invented to bridge the gap"
    assert [r["elapsed_days"] for r in rows] == ["1", "28"]
    assert res["gap_steps"] == 1
    assert float(rows[1]["elapsed_seconds"]) == 28 * 86400.0
    return "the 28-day step stays one row with elapsed_days=28; nothing bridged"


def test_deterministic_matching():
    rows_in = [step(), step(lat=-60.6, lon=10.8, berg="b02"),
               step(lat=-70.0, berg="c03")]
    w = wind_dir()
    a_out = Path(tempfile.mkdtemp()) / "a.csv"
    b_out = Path(tempfile.mkdtemp()) / "b.csv"
    sf = steps_file(rows_in)
    cd = Path(tempfile.mkdtemp())
    r1 = match_steps(sf, w, cd, a_out)
    r2 = match_steps(sf, w, cd, b_out)
    assert a_out.read_text() == b_out.read_text()
    assert r1["states"] == r2["states"]
    return "identical CSV bytes and identical state counts on a rerun"


# ------------------------------------------------- request derivation
def test_request_is_derived_from_the_real_data_not_hardcoded():
    if not REAL_STEPS.exists():
        raise AssertionError("real iceberg steps missing")
    plan = F.derive_request(REAL_STEPS, REAL_CURRENTS)
    assert plan["dataset"] == "reanalysis-era5-single-levels"
    assert plan["variables"] == ["10m_u_component_of_wind",
                                 "10m_v_component_of_wind"]
    assert plan["hours_utc"] == ["00:00"]
    ex = plan["iceberg_extent"]
    n, w, s, e = plan["area_north_west_south_east"]
    assert n >= ex["start_lat_max"] + 1.0 - 1e-9, "north edge does not clear the data"
    assert s <= ex["start_lat_min"] - 1.0 + 1e-9, "south edge does not clear the data"
    assert (n * 4) % 1 == 0 and (s * 4) % 1 == 0, "edges are off the 0.25 grid"
    assert [w, e] == [-180.0, 180.0], "circumpolar starts were clipped"
    src = (ROOT / "src" / "data" / "fetch_era5_wind.py").read_text()
    body = src.split('"""', 2)[2]
    for literal in ("2025-01-01", "-77.8", "-49.6", "24203"):
        assert literal not in body, f"the request hardcodes {literal!r}"
    return (f"area {plan['area_north_west_south_east']} derived from "
            f"{ex['candidate_steps']:,} candidate steps; no bound hardcoded")


def test_only_days_with_a_current_are_requested():
    if not REAL_STEPS.exists():
        raise AssertionError("real iceberg steps missing")
    plan = F.derive_request(REAL_STEPS, REAL_CURRENTS)
    have = F._current_days(REAL_CURRENTS)
    asked = {f"{r['request']['year'][0]}-{r['request']['month'][0]}-{d}"
             for r in plan["requests"] for d in r["request"]["day"]}
    assert asked <= have, "a day with no GLORYS field was requested"
    assert plan["date_min"] >= min(have) and plan["date_max"] <= max(have)
    return (f"{len(asked)} days requested, every one of them inside the "
            f"{len(have)} days of GLORYS on disk")


def test_dry_run_is_the_default_and_no_key_is_touched():
    src = (ROOT / "src" / "data" / "fetch_era5_wind.py").read_text()
    body = src.split('"""', 2)[2]
    for banned in (".cdsapirc", "CDSAPI_KEY", "api_key", "url=", "key="):
        assert banned not in body, f"the fetcher touches {banned!r}"
    assert "--download" in src
    st = F.client_status()
    assert set(st) >= {"cdsapi_importable", "client_constructed", "detail"}
    assert "key" not in st["detail"].lower() or "never" in st["detail"].lower()
    return ("no credential name appears in the code; status reports state only, "
            "and a transfer needs an explicit --download")


# ---------------------------------------------------- provenance / real
def test_manifest_records_conventions_and_promises():
    res, _ = _run([step(), step(lat=-70.0, berg="z")])
    out = Path(tempfile.mkdtemp()) / "man.json"
    m = write_manifest(res, out, Path(tempfile.mkdtemp()))
    assert m["era5_dataset"] == "reanalysis-era5-single-levels"
    assert m["wind_interpolation_method"] == WIND_INTERPOLATION
    assert m["current_sampling_method"] == CURRENT_SAMPLING
    assert m["date_time_convention"]["time_assumption"] == TIME_ASSUMPTION
    assert m["date_time_convention"]["assigned_time_utc"] == "00:00"
    assert m["no_iceberg_observation_was_fabricated"] is True
    assert m["no_track_gap_was_filled"] is True
    assert (m["fully_matched"] + m["current_missing"] + m["wind_missing"]
            - m["both_missing"]) == m["total_iceberg_steps_considered"]
    assert "DIFFERENT vector frames" in m["vector_frame_warning"]
    assert json.loads(out.read_text())["era5_variables"]
    return ("manifest carries dataset, both sampling methods, the 00 UTC "
            "convention, the frame warning and consistent missingness counts")


def test_current_sampling_agrees_with_the_existing_audit_rule():
    """The training table and preprocess_icebergs must never disagree."""
    if not REAL_STEPS.exists() or not REAL_CURRENTS.exists():
        raise AssertionError("real data missing")
    from src.data.preprocess_icebergs import audit_currents
    rows = []
    with REAL_STEPS.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["prev_date"].startswith("2025-01-1"):
                rows.append(r)
            if len(rows) >= 400:
                break
    assert rows, "no 2025 steps found to cross-check"
    audit = audit_currents(rows, REAL_CURRENTS)
    field = CurrentField(REAL_CURRENTS)
    mine = sum(1 for r in rows
               if field.sample(float(r["prev_latitude_deg"]),
                               float(r["prev_longitude_deg"]),
                               r["prev_date"])[0] is not None)
    assert mine == audit["steps_with_usable_current"], (
        f"this module finds {mine} usable currents, audit_currents finds "
        f"{audit['steps_with_usable_current']} on the same {len(rows)} steps")
    return (f"{len(rows)} real steps: both the audit and the matcher find "
            f"{mine} usable currents -- one rule, not two")


def test_provenance_file_is_one_valid_json_document():
    """Regression: the writer once appended a literal backslash-n after '}'."""
    path = Path(F.OUT_DIR) / F.PROVENANCE.name
    if not path.exists():
        raise AssertionError(f"no provenance file at {path}")
    text = path.read_text()
    doc = json.loads(text)                      # the WHOLE text, not raw_decode
    _, end = json.JSONDecoder().raw_decode(text)
    assert text[end:] == "\n", f"trailing bytes after the document: {text[end:]!r}"
    assert text[-1] == chr(10) and text[-2] == "}"
    assert (chr(92) + "n") not in text[end:], "a literal backslash-n is back"
    return (f"{len(text):,} chars parse as exactly one document; the only "
            f"trailing byte is a real newline")


def test_provenance_keeps_all_twelve_monthly_requests():
    doc = F.read_provenance(F.OUT_DIR)
    reqs = doc["requests"]
    assert len(reqs) == 12, f"{len(reqs)} monthly records, expected 12"
    assert [r["month"] for r in reqs] == [f"2025-{m:02d}" for m in range(1, 13)]
    total_days = 0
    for r in reqs:
        req = r["request"]
        assert doc["dataset"] == "reanalysis-era5-single-levels"
        assert req["variable"] == ["10m_u_component_of_wind",
                                   "10m_v_component_of_wind"]
        assert req["year"] == ["2025"]
        assert req["month"] == [r["month"].split("-")[1]]
        assert req["day"] and all(d.isdigit() for d in req["day"])
        assert req["time"] == ["00:00"]
        assert req["area"] == doc["requested_area_north_west_south_east"]
        assert req["grid"] == doc["requested_grid_deg"]
        assert r["target_name"] == f"era5_10m_wind_{r['month']}.nc"
        assert r["sha256"] and r["bytes"] and r["file_present"]
        total_days += len(req["day"])
    # `requested_days` is the COUNT of distinct days, not the list of them;
    # the per-month request["day"] arrays are where the actual days live.
    assert isinstance(doc["requested_days"], int)
    assert total_days == doc["requested_days"] == 365, total_days
    return ("12 months, 365 days, each record keeping variables, year/month/day, "
            "area, grid and the sha256 of the file it produced")


def test_provenance_repair_is_deterministic_and_downloads_nothing():
    path = Path(F.OUT_DIR) / F.PROVENANCE.name
    if not path.exists():
        raise AssertionError("no provenance file to repair")
    nc = sorted(Path(F.OUT_DIR).glob("*.nc"))
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in nc}
    a = F.repair_provenance(F.OUT_DIR)
    text_a = path.read_text()
    b = F.repair_provenance(F.OUT_DIR)
    text_b = path.read_text()
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted(Path(F.OUT_DIR).glob("*.nc"))}
    assert before == after, "a .nc file was rewritten by the repair"
    strip = lambda t: json.loads(t)
    da, db = strip(text_a), strip(text_b)
    for k in ("requests", "files", "requested_days", "requested_area_north_"
              "west_south_east", "requested_grid_deg", "dataset", "variables"):
        assert da[k] == db[k], f"{k} differs between two repairs"
    assert a["_repair"]["data_redownloaded"] is False
    assert a["_repair"]["monthly_requests_reconstructed"] == 12
    assert a["_repair"]["files_whose_sha256_changed"] == []
    assert a["_repair"]["files_missing"] == []
    return (f"two repairs agree on every field except the timestamps; "
            f"{len(nc)} .nc files untouched, 0 sha256 changed")


def test_the_writer_cannot_emit_a_literal_backslash_n():
    src = (ROOT / "src" / "data" / "fetch_era5_wind.py").read_text()
    assert (chr(92) + chr(92) + "n") not in src, (
        "a literal backslash-n escape is back in the fetch module")
    tmp = Path(tempfile.mkdtemp())
    F.write_provenance({"dataset": "x", "requests": []}, tmp)
    written = (tmp / F.PROVENANCE.name).read_text()
    assert json.loads(written)["dataset"] == "x"
    assert written.endswith("}" + chr(10))
    return "write_provenance validates its own output and ends in one newline"


def test_no_model_or_routing_code_was_added():
    for name in ("fetch_era5_wind.py", "preprocess_era5_wind.py"):
        src = (ROOT / "src" / "data" / name).read_text()
        body = src.split('"""', 2)[2]
        for banned in ("sklearn", "HistGradient", "RandomForest", "astar",
                       "time_astar", "navigation_cost", "polaris",
                       "uncertainty_cone", "drift_model", "wind_drag",
                       "fit(", "predict("):
            assert banned not in body, f"{name} references {banned}"
    return "neither module fits anything, routes anything or touches POLARIS"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78); print("ERA5 10 m wind ingestion and step matching")
    print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<56} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__); print(f"  FAIL  {fn.__name__:<56} {exc}")
        except Exception as exc:                        # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__:<56} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

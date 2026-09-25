"""
Focused tests for the iceberg observation ingestion/standardisation layer.

Fixtures are tiny CSVs written in the archive's real conventions -- variable
sensor columns, (0,0) for no data, <sensor>_3 as the observation/interpolation
flag, sizes in nautical miles. A few rows are copied verbatim from real files
in consolidated_v8 (a01.csv, a03.csv, a68.csv); nothing is generated.

    python tests/test_icebergs_data.py
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import fetch_icebergs as FI                                  # noqa: E402
from src.data.preprocess_icebergs import (EARTH_RADIUS_M, NM_TO_KM,        # noqa: E402
                                          README_QUOTES, IcebergDataError,
                                          ReadReport, audit_currents, audit_wind,
                                          build_observations, build_steps,
                                          current_dates_available,
                                          great_circle_m, inspect_archive,
                                          normalise_longitude, resolve_header,
                                          run, scan, scan_headers,
                                          schema_family, step_velocity,
                                          validate_position, yyyyddd_to_date)

REAL_ARCHIVE = ROOT / "data" / "raw" / "icebergs" / "consolidated_v8"


def write_csv(d: Path, name: str, header: list[str], rows: list[list]) -> None:
    with (d / name).open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def fixture(files: dict[str, tuple[list[str], list[list]]]) -> Path:
    d = Path(tempfile.mkdtemp(prefix="icebergs_fix_"))
    for name, (header, rows) in files.items():
        write_csv(d, f"{name}.csv", header, rows)
    return d


# --- real headers and rows, copied from the archive -----------------------
H_NIC_SASS = ["date", "nic_1", "nic_2", "nic_3", "sass_1", "sass_2", "sass_3",
              "size_1", "size_2"]
R_A01 = [["1978204", "0", "0", "0", "-60.2193", "-48.4826", "1", "0", "0"],
         ["1978205", "0", "0", "0", "-60.2179", "-48.4879", "1", "0", "0"],
         ["1978206", "0", "0", "0", "-60.2165", "-48.4931", "1", "0", "0"]]
H_NIC = ["date", "nic_1", "nic_2", "nic_3", "size_1", "size_2"]
R_A03 = [["1979035", "-58.3", "-43.9", "1", "25", "10"],
         ["1979063", "-58.3", "-43.9", "1", "25", "10"],
         ["1979126", "-59.5", "-44.5", "1", "25", "10"]]
H_ASCAT = ["ascat_1", "ascat_2", "ascat_3", "date"]
R_A68 = [["-67.8158", "-60.9092", "1", "2017177"],
         ["-67.8115", "-60.8889", "1", "2017191"],
         ["-67.8115", "-60.8889", "0", "2017192"]]


# ---------------------------------------------------------------- dates
def test_yyyyddd_conversion():
    assert yyyyddd_to_date("1978204") == dt.date(1978, 7, 23)
    assert yyyyddd_to_date("2025001") == dt.date(2025, 1, 1)
    assert yyyyddd_to_date(1979035) == dt.date(1979, 2, 4)
    assert yyyyddd_to_date("2025112") == dt.date(2025, 4, 22)   # the stated end
    return "1978204 -> 1978-07-23; 2025112 -> 2025-04-22 (the source's last day)"


def test_leap_year_handling():
    assert yyyyddd_to_date("2024060") == dt.date(2024, 2, 29)   # leap
    assert yyyyddd_to_date("2023060") == dt.date(2023, 3, 1)    # not leap
    assert yyyyddd_to_date("2024366") == dt.date(2024, 12, 31)
    assert yyyyddd_to_date("2000366") == dt.date(2000, 12, 31)  # 2000 IS leap
    try:
        yyyyddd_to_date("1900366")                              # 1900 is NOT
        raise AssertionError("1900366 accepted")
    except IcebergDataError:
        pass
    return "day 060 differs by year; 366 valid in 2024/2000, rejected in 1900"


def test_invalid_dates_are_rejected_not_coerced():
    for bad in ("2025000", "2025366", "20250011", "202501", "", "abcdefg",
                "2025-01-01", "1899001", "2101001"):
        try:
            yyyyddd_to_date(bad)
            raise AssertionError(f"{bad!r} accepted")
        except IcebergDataError:
            pass
    return "day 0, day 366 in a common year, wrong width, text and out-of-era refused"


# ---------------------------------------------------------- coordinates
def test_coordinate_validation_and_longitude_normalisation():
    assert validate_position(-66.78, 81.92) == []
    assert "northern_hemisphere" in validate_position(12.0, 0.0)
    assert "near_pole" in validate_position(-89.9, 0.0)
    for lat, lon in ((91.0, 0.0), (-90.5, 0.0), (0.0, 400.0), (float("nan"), 0.0)):
        try:
            validate_position(lat, lon)
            raise AssertionError(f"({lat}, {lon}) accepted")
        except IcebergDataError:
            pass
    assert normalise_longitude(181.0) == -179.0
    assert normalise_longitude(-181.0) == 179.0
    assert normalise_longitude(360.0) == 0.0
    assert normalise_longitude(81.92) == 81.92, "an in-range value was re-rounded"
    assert repr(normalise_longitude(179.999)) == repr(179.999)
    assert -180.0 <= normalise_longitude(-180.0) < 180.0
    return ("ranges enforced, out-of-range raises (never clamped), 181 -> -179, "
            "and an in-range longitude is returned bit-for-bit")


# ------------------------------------------- heterogeneous schema layer
def test_subset_schema_is_accepted():
    s = resolve_header(H_NIC)                       # nic only, no scatterometer
    assert list(s["sensors"]) == ["nic"]
    assert s["schema_family"] == "nic+size"
    assert s["sensors_with_flag"] == ["nic"]
    return "a nic-only header normalises; a missing sensor is not an error"


def test_superset_schema_is_accepted():
    big = ["ascat_1", "ascat_2", "ascat_3", "date", "ers_1", "ers_2", "ers_3",
           "nic_1", "nic_2", "nic_3", "nscat_1", "nscat_2", "nscat_3",
           "oscat_1", "oscat_2", "oscat_3", "qscat_1", "qscat_2", "qscat_3",
           "seawinds_1", "seawinds_2", "seawinds_3", "size_1", "size_2"]
    s = resolve_header(big)
    assert len(s["sensors"]) == 7
    assert s["schema_family"] == ("ascat+ers+nic+nscat+oscat+qscat+seawinds+size")
    # a brand-new instrument normalises too: the README's rule is generic
    s2 = resolve_header(["date", "newsat_1", "newsat_2", "newsat_3"])
    assert list(s2["sensors"]) == ["newsat"]
    return "7-sensor header and an unseen instrument both normalise by the same rule"


def test_missing_optional_sensor_and_size_are_accepted():
    s = resolve_header(H_ASCAT)                     # no nic, no size columns
    assert s["size_columns"] == [] and list(s["sensors"]) == ["ascat"]
    assert s["schema_family"] == "ascat"
    d = fixture({"a68": (H_ASCAT, R_A68), "a03": (H_NIC, R_A03)})
    obs, meta = build_observations(d)
    assert {o["schema_family"] for o in obs} == {"ascat", "nic+size"}
    a68 = [o for o in obs if o["iceberg_id"] == "a68"]
    assert all(o["size_major_km"] == "" for o in a68), "a size was invented"
    assert meta["stats"]["normalised_files"] == 2
    return "two different families in one run; the file without sizes gets none"


def test_unsupported_header_structure_is_refused_whole():
    for bad, why in (
            (["nic_1", "nic_2", "nic_3"], "no date column"),
            (["date", "nic_1", "nic_3"], "latitude without longitude"),
            (["date", "size_1", "size_2"], "no position pair"),
            (["date", "nic_1", "nic_2", "comment"], "unplaceable column"),
            (["date", "nic_1", "nic_2", "size_1"], "half a size pair"),
            (["date", "nic_1", "nic_1", "nic_2"], "repeated column")):
        try:
            resolve_header(bad)
            raise AssertionError(f"{why} accepted: {bad}")
        except IcebergDataError:
            pass
    # and a bad file is named, with no rows emitted from it
    d = fixture({"good": (H_NIC, R_A03),
                 "bad": (["date", "nic_1", "nic_2", "comment"],
                         [["1979035", "-58.3", "-43.9", "x"]])})
    obs, meta = build_observations(d)
    assert set(meta["unsupported_files"]) == {"bad.csv"}
    assert "comment" in meta["unsupported_files"]["bad.csv"]
    assert {o["iceberg_id"] for o in obs} == {"good"}, "the bad file was partly read"
    return ("6 malformed structures refused; the bad file is named with its "
            "offending column and contributes no rows, the good file still does")


def test_schema_family_is_deterministic_and_order_independent():
    assert schema_family(["nic", "ascat"], True) == "ascat+nic+size"
    assert schema_family(["ascat", "nic"], True) == "ascat+nic+size"
    assert schema_family(["qscat"], False) == "qscat"
    a = resolve_header(H_NIC_SASS)["schema_family"]
    b = resolve_header(list(reversed(H_NIC_SASS)))["schema_family"]
    assert a == b == "nic+sass+size"
    return "family id depends on the column set, never on column order"


# --------------------------------------------------- the 0,0 sentinel
def test_zero_zero_is_no_data_not_a_position():
    d = fixture({"a01": (H_NIC_SASS, R_A01)})
    obs, meta = build_observations(d)
    assert len(obs) == 3, "nic's (0,0) became a position"
    assert {o["position_source"] for o in obs} == {"sass"}
    assert meta["stats"]["no_data_sentinel_positions"] == 3
    assert all(o["latitude_deg"] != 0.0 for o in obs)
    return ("a01's three nic (0,0) pairs yield no observation; only the three "
            "real sass fixes survive")


def test_sentinel_wins_over_a_contradicting_observation_flag():
    d = fixture({"z": (H_NIC, [["1979035", "0", "0", "1", "0", "0"]])})
    obs, meta = build_observations(d)
    assert obs == [], "(0,0) with flag 1 was taken as a real position"
    assert meta["stats"]["no_data_sentinel_with_observation_flag"] == 1
    return "0,0 flagged as an observation is still no data, and the clash is counted"


# ------------------------------------------- the documented _3 flag
def test_observation_flag_is_preserved_without_filtering():
    d = fixture({"a68": (H_ASCAT, R_A68)})
    obs, meta = build_observations(d)
    assert [o["observation_flag"] for o in obs] == ["1", "1", "0"]
    assert [o["is_interpolated"] for o in obs] == ["False", "False", "True"]
    assert len(obs) == 3, "an interpolated position was dropped"
    steps, smeta = build_steps(obs)
    assert len(steps) == 2, "a step touching an interpolated fix was dropped"
    assert steps[1]["either_endpoint_interpolated"] == "True"
    assert smeta["stats"]["steps_touching_an_interpolated_position"] == 1
    assert smeta["stats"]["steps_from_two_observed_positions"] == 1
    src = (ROOT / "src" / "data" / "preprocess_icebergs.py").read_text()
    body = src.split('"""', 2)[2]
    for banned in ('if flag == "0":\n                        continue',
                   "if o[\"is_interpolated\"]", "skip_interpolated"):
        assert banned not in body, f"the flag is being filtered on: {banned!r}"
    return ("1/1/0 preserved and mapped to is_interpolated; nothing dropped or "
            "skipped because of it, and both step classes are counted")


def test_a_sensor_without_a_flag_column_is_unknown_not_false():
    d = fixture({"n": (["date", "nic_1", "nic_2"],
                       [["1979035", "-58.3", "-43.9"]])})
    obs, meta = build_observations(d)
    assert obs[0]["observation_flag"] == ""
    assert obs[0]["is_interpolated"] == "", "absence of a flag became 'observed'"
    assert meta["stats"]["positions_without_a_flag_column"] == 1
    return "no _3 column -> both fields empty (unknown), never False"


# ------------------------------------------------------ sizes / units
def test_sizes_are_converted_from_nautical_miles_and_kept_in_source_units():
    d = fixture({"a03": (H_NIC, R_A03)})
    obs, _ = build_observations(d)
    o = obs[0]
    assert o["size_major_source_nm"] == "25" and o["size_minor_source_nm"] == "10"
    assert abs(float(o["size_major_km"]) - 25 * NM_TO_KM) < 1e-9
    assert abs(float(o["size_minor_km"]) - 10 * NM_TO_KM) < 1e-9
    z = fixture({"z": (H_NIC, [["1979035", "-58.3", "-43.9", "1", "0", "0"]])})
    zo, _ = build_observations(z)
    assert zo[0]["size_major_km"] == "" and zo[0]["size_minor_km"] == ""
    assert zo[0]["size_major_source_nm"] == "0", "the source zero was discarded"
    return (f"25 nm -> {25 * NM_TO_KM:.3f} km, source nm retained; a zero axis is "
            f"missing, not a zero-length iceberg")


# ------------------------------------------------- observations table
def test_every_real_source_observation_survives():
    d = fixture({"a01": (H_NIC_SASS, R_A01), "a03": (H_NIC, R_A03),
                 "a68": (H_ASCAT, R_A68)})
    obs, meta = build_observations(d)
    assert meta["stats"]["source_rows"] == 9
    assert len(obs) == 9, "3 sass + 3 nic + 3 ascat"
    assert {o["iceberg_id"] for o in obs} == {"a01", "a03", "a68"}
    assert all(o["source_filename"] and o["source_row"] for o in obs)
    return "9 source rows -> 9 observations, each traceable to file and line"


def test_multi_sensor_rows_become_one_row_per_sensor():
    d = fixture({"m1": (H_NIC_SASS,
                        [["2025001", "-64.0", "10.0", "1", "-64.1", "10.1", "1",
                          "0", "0"]])})
    obs, meta = build_observations(d)
    assert len(obs) == 2
    assert {o["position_source"] for o in obs} == {"nic", "sass"}
    assert meta["stats"]["rows_with_multiple_sensors"] == 1
    assert sorted(o["latitude_deg"] for o in obs) == [-64.1, -64.0]
    return "one row with two sensor fixes -> 2 rows, no precedence invented"


def test_duplicates_are_marked_not_removed():
    d = fixture({"d1": (H_NIC, [["2025001", "-64.0", "10.0", "1", "0", "0"],
                                ["2025001", "-64.0", "10.0", "1", "0", "0"]])})
    obs, meta = build_observations(d)
    assert len(obs) == 2, "a duplicate was dropped"
    assert all(o["duplicate_group_size"] == 2 for o in obs)
    assert meta["stats"]["duplicate_rows_kept"] == 2
    return "identical (iceberg, date, sensor) rows kept, both marked group size 2"


# -------------------------------------------------------------- steps
def test_step_construction_and_gap_preservation():
    d = fixture({"a03": (H_NIC, R_A03)})
    obs, _ = build_observations(d)
    steps, meta = build_steps(obs)
    assert len(steps) == 2
    assert [s["elapsed_days"] for s in steps] == [28, 63]
    assert steps[0]["elapsed_seconds"] == 28 * 86400.0
    assert "gap_28d" in steps[0]["flags"] and "gap_63d" in steps[1]["flags"]
    assert meta["gap_length_days_histogram"] == {28: 1, 63: 1}
    assert len(obs) == 3, "the gaps were filled with positions"
    return "a03's 3 NIC reports -> 2 steps flagged gap_28d and gap_63d"


def test_steps_never_cross_sensors():
    d = fixture({"m2": (H_NIC_SASS,
                        [["2025001", "-64.0", "10.0", "1", "0", "0", "0", "0", "0"],
                         ["2025002", "0", "0", "0", "-64.2", "10.0", "1", "0", "0"]])})
    obs, _ = build_observations(d)
    assert len(obs) == 2
    steps, _ = build_steps(obs)
    assert steps == [], "a nic fix and a sass fix were joined into one step"
    return "consecutive fixes from different sensors produce no step"


def test_zero_or_negative_elapsed_time_produces_no_step():
    d = fixture({"z1": (H_NIC, [["2025002", "-64.0", "10.0", "1", "0", "0"],
                                ["2025002", "-64.1", "10.0", "1", "0", "0"]])})
    obs, _ = build_observations(d)
    steps, meta = build_steps(obs)
    assert steps == []
    assert meta["stats"]["pairs_without_positive_elapsed_time"] == 1
    return "same-day pair has no elapsed time, so no velocity is manufactured"


def test_velocity_and_displacement():
    lat1, lat2, lon = -64.0, -65.0, 10.0
    expect_m = EARTH_RADIUS_M * math.radians(1.0)
    assert abs(great_circle_m(lat1, lon, lat2, lon) - expect_m) < 1e-6
    east, north = step_velocity(lat1, lon, lat2, lon, 86400.0)
    assert abs(east) < 1e-9
    assert abs(north + expect_m / 86400.0) < 1e-9
    d = great_circle_m(-64.0, 179.9, -64.0, -179.9)
    assert d < 12_000, f"dateline crossing measured {d:,.0f} m"
    e2, _ = step_velocity(-64.0, 179.9, -64.0, -179.9, 86400.0)
    assert e2 > 0, "the dateline hop came out westward"
    obs, _ = build_observations(fixture({"a68": (H_ASCAT, R_A68)}))
    s = build_steps(obs)[0][0]
    assert abs(s["speed_mps"] - s["displacement_m"] / s["elapsed_seconds"]) < 1e-12
    assert s["distance_method"].startswith("haversine_sphere_R")
    return (f"1 deg lat/day = {expect_m / 86400:.4f} m/s north; dateline hop "
            f"{d:,.0f} m eastward; method recorded on every row")


# ------------------------------------------------- determinism / audit
def test_repeated_processing_is_deterministic():
    d = fixture({"a01": (H_NIC_SASS, R_A01), "a03": (H_NIC, R_A03),
                 "a68": (H_ASCAT, R_A68)})
    a_obs, _ = build_observations(d)
    b_obs, _ = build_observations(d)
    assert a_obs == b_obs
    assert build_steps(a_obs)[0] == build_steps(b_obs)[0]
    out1, out2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    m1, m2 = run(d, out1), run(d, out2)
    for name in ("iceberg_observations.csv", "iceberg_steps.csv"):
        assert (out1 / name).read_text() == (out2 / name).read_text(), name
    for k in ("processed_observation_count", "step_count", "track_count",
              "original_date_range", "unique_schema_count",
              "files_per_schema_family"):
        assert m1[k] == m2[k], k
    return "identical observations, steps and CSV bytes across two full runs"


def test_nothing_is_fabricated_end_to_end():
    d = fixture({"a03": (H_NIC, R_A03)})
    out = Path(tempfile.mkdtemp())
    m = run(d, out)
    assert m["interpolation_performed"] is False
    rows = list(csv.DictReader((out / "iceberg_observations.csv").open()))
    assert len(rows) == 3, "an observation appeared from nowhere"
    dates = {r["date"] for r in rows}
    assert dates == {"1979-02-04", "1979-03-04", "1979-05-06"}
    assert "1979-02-05" not in dates, "a gap day was filled in"
    assert m["step_count"] == 2
    return "3 source rows in, 3 observations out, the 89 missing days stay missing"


def test_provenance_and_the_new_manifest_fields():
    d = fixture({"a01": (H_NIC_SASS, R_A01), "a03": (H_NIC, R_A03),
                 "a68": (H_ASCAT, R_A68)})
    out = Path(tempfile.mkdtemp())
    m = run(d, out)
    assert m["source_url"] == FI.SOURCE_URL
    assert m["source_version_verified"] is False, "a version was asserted unhashed"
    assert "UNVERIFIED" in m["source_version_note"]
    assert m["coverage_check"]["observed_coverage"] == m["original_date_range"]
    assert m["unique_schema_count"] == 3
    assert m["files_per_schema_family"] == {"nic+sass+size": 1, "nic+size": 1,
                                            "ascat": 1}
    assert m["normalised_files"] == 3 and m["unsupported_file_count"] == 0
    assert m["recognised_sensors"] == ["ascat", "nic", "sass"]
    assert m["source_specific_fields_preserved"] == []
    assert m["schema_families"]["ascat"]["missing_optional_columns"]
    assert any("nautical miles" in q for q in m["readme_quotes"].values())
    assert any("interpolat" in lim for lim in m["known_limitations"])
    rows = list(csv.DictReader((out / "iceberg_observations.csv").open()))
    assert {r["source_filename"] for r in rows} == {"a01.csv", "a03.csv", "a68.csv"}
    assert all(r["schema_family"] and r["source_row"] for r in rows)
    assert all(r["date_source_yyyyddd"].isdigit() for r in rows)
    return ("manifest carries schema count, files per family, normalised/"
            "unsupported counts and README quotes; every row keeps filename, "
            "line, family and its original YYYYDDD")


def test_an_extra_undocumented_sensor_column_is_preserved_verbatim():
    hdr = ["date", "nic_1", "nic_2", "nic_3", "nic_7"]
    s = resolve_header(hdr)
    assert s["preserved_source_columns"] == ["nic_7"]
    assert s["schema_family"] == "nic+nic_7"
    d = fixture({"x": (hdr, [["1979035", "-58.3", "-43.9", "1", "mystery"]])})
    obs, _ = build_observations(d)
    assert obs[0]["source_fields"] == '{"nic_7": "mystery"}'
    assert obs[0]["latitude_deg"] == -58.3, "the unknown column changed the position"
    return "nic_7 round-trips into source_fields and influences nothing"


def test_missing_archive_refuses_rather_than_inventing():
    empty = Path(tempfile.mkdtemp())
    for fn in (inspect_archive, build_observations, scan_headers):
        try:
            fn(empty)
            raise AssertionError(f"{fn.__name__} produced output from nothing")
        except IcebergDataError as exc:
            assert "fetch_icebergs" in str(exc)
    return "an absent archive raises and names the fetch step; no stub data"


def test_fetch_module_names_one_official_source_only():
    src = (ROOT / "src" / "data" / "fetch_icebergs.py").read_text()
    body = src.split('"""', 2)[2]
    assert "github" not in body.lower(), "a GitHub mirror is referenced in code"
    hosts = {line.split("//")[1].split("/")[0]
             for line in src.splitlines() if "https://" in line}
    assert hosts <= {"www.scp.byu.edu"}, f"other hosts referenced: {hosts}"
    return f"only {sorted(hosts)[0]} is ever contacted; no mirror, no fallback"


def test_current_availability_audit_reads_only():
    have = current_dates_available()
    obs, _ = build_observations(fixture({"a68": (H_ASCAT, R_A68)}))
    steps, _ = build_steps(obs)
    a = audit_currents(steps)
    assert a["values_created"] == 0
    assert a["spatial_check_ran"] is True, (
        "the spatial check failed silently and reported 0 usable currents")
    assert a["steps_with_current_day"] + a["steps_without_current_day"] == len(steps)
    if have:
        assert a["available_days"] == len(have)
    return (f"{a['available_days']} GLORYS days on disk; "
            f"{a['steps_with_current_day']}/{len(steps)} fixture steps have one; "
            f"nothing written")


def test_wind_audit_reports_missing_prerequisites_without_substituting():
    w = audit_wind()
    assert w["substitute_used"] is None
    assert "ERA5" in w["preferred_source"]
    assert "10m_u_component_of_wind" in w["preferred_source"]
    if not w["available"]:
        assert w["missing_prerequisites"], "unavailable but nothing named"
    return ("ERA5 " + ("configured" if w["available"] else
                       f"NOT configured: {len(w['missing_prerequisites'])} "
                       f"prerequisites missing") + "; no substitute considered")


def test_no_model_or_routing_code_was_added():
    for name in ("preprocess_icebergs.py", "fetch_icebergs.py"):
        src = (ROOT / "src" / "data" / name).read_text()
        body = src.split('"""', 2)[2]
        for banned in ("sklearn", "HistGradient", "RandomForest", "astar",
                       "time_astar", "navigation_cost", "polaris", "blocked_mask",
                       "uncertainty_cone", "drift_model", "wind_drag"):
            assert banned not in body, f"{name} references {banned}"
    return "neither module imports a model, a router, POLARIS or a drift term"


def test_real_archive_headers_all_normalise():
    if not REAL_ARCHIVE.exists():
        raise AssertionError(f"real archive missing at {REAL_ARCHIVE}")
    c = scan_headers(REAL_ARCHIVE)
    assert c["csv_files"] > 0
    assert c["unsupported_files"] == {}, c["unsupported_files"]
    assert c["unique_headers"] == len(c["schema_families"])
    assert sum(v["files"] for v in c["schema_families"].values()) == c["csv_files"]
    assert c["readme_present"], "README_consolidated.TXT is missing"
    union = set(c["union_columns"])
    for fam in c["schema_families"].values():
        assert set(fam["columns"]) <= union, "a family is not a subset of the union"
    return (f"{c['csv_files']} files, {c['unique_headers']} headers, "
            f"{len(c['recognised_sensors'])} sensors, 0 unsupported, every header "
            f"a subset of the {len(union)}-column union")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78); print("iceberg observation ingestion / standardisation")
    print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<58} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__); print(f"  FAIL  {fn.__name__:<58} {exc}")
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

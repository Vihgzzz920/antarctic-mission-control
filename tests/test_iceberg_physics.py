"""
Focused tests for the first-order current + windage iceberg drift baseline.

The vector-frame tests are the important ones: they check this module's local
basis against the cached grid geometry the GLORYS rasters were actually rotated
with, so the two conventions cannot silently diverge.

    python tests/test_iceberg_physics.py
"""
from __future__ import annotations

import ast
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess_currents import rotate                            # noqa: E402
from src.data.preprocess_icebergs import EARTH_RADIUS_M, great_circle_m    # noqa: E402
from src.models.iceberg_physics import (CONFIG, QUALITY_CLEAN,             # noqa: E402
                                        PhysicsConfig, PhysicsError,
                                        Prediction, StepInputs, advect,
                                        calibrate, drift_velocity, load_config,
                                        load_rows, local_basis,
                                        predict_persistence, predict_physics,
                                        score, to_geographic, to_projected,
                                        turn_wind)

GEOM_NPZ = ROOT / "data" / "processed" / "currents" / "_grid_geometry_3976_1328x1264.npz"
CFG = load_config(CONFIG)
HOUR, DAY = 3600.0, 86400.0


def cfg(coeff=0.0, turn=0.0) -> PhysicsConfig:
    return CFG.replace(windage_coefficient=coeff, wind_turn_deg=turn)


def step(lat=-65.0, lon=10.0, cu=0.0, cv=0.0, we=0.0, wn=0.0, dt=DAY):
    return StepInputs(lat, lon, cu, cv, we, wn, dt)


# ------------------------------------------------------- the vector frame
def test_local_basis_matches_the_convention_the_currents_were_rotated_with():
    """This module's basis must equal the cached one, cell for cell."""
    if not GEOM_NPZ.exists():
        raise AssertionError(f"cached grid geometry missing: {GEOM_NPZ}")
    with np.load(GEOM_NPZ) as z:
        lat, lon = z["lat"], z["lon"]
        ex, ey, nx, ny = z["ex"], z["ey"], z["nx"], z["ny"]
    worst = 0.0
    for r, c in ((300, 400), (700, 700), (1000, 200), (150, 1100), (664, 632)):
        got = local_basis(float(lat[r, c]), float(lon[r, c]))
        want = (float(ex[r, c]), float(ey[r, c]),
                float(nx[r, c]), float(ny[r, c]))
        worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
    assert worst < 1e-6, f"basis differs from the cached one by {worst:g}"
    return (f"5 grid cells: this module's east/north basis matches the cached "
            f"_grid_geometry to {worst:.2e} -- one convention, not two")


def test_basis_is_orthonormal_so_the_transpose_really_is_the_inverse():
    worst_dot = worst_norm = 0.0
    for lat in (-50.0, -65.0, -77.0):
        for lon in (-170.0, -90.0, 0.0, 90.0, 179.0):
            ex, ey, nx, ny = local_basis(lat, lon)
            worst_dot = max(worst_dot, abs(ex * nx + ey * ny))
            worst_norm = max(worst_norm, abs(math.hypot(ex, ey) - 1.0),
                             abs(math.hypot(nx, ny) - 1.0))
    assert worst_dot < 1e-6 and worst_norm < 1e-9, (worst_dot, worst_norm)
    return (f"east.north <= {worst_dot:.1e}, |unit| - 1 <= {worst_norm:.1e} over "
            f"15 Antarctic positions")


def test_pure_eastward_wind_points_the_expected_way_in_epsg3976():
    """A known location, checked against the projection's own geometry.

    EPSG:3976 is south polar stereographic with lat_0=-90, lon_0=0, so

        x = rho * sin(lambda),   y = rho * cos(lambda)

    and moving east (lambda increasing) moves the point by

        d/dlambda (x, y) = (rho * cos(lambda), -rho * sin(lambda)).

    So due east is +x at 0 E, -y at 90 E, -x at 180 and +y at 90 W. Checked
    against the projection's own algebra, not against intuition: the first
    version of this test asserted +y at 90 E and was wrong.
    """
    checks = {0.0: (1.0, 0.0), 90.0: (0.0, -1.0), 180.0: (-1.0, 0.0),
              -90.0: (0.0, 1.0)}
    got = {}
    for lon, (want_x, want_y) in checks.items():
        basis = local_basis(-70.0, lon)
        ux, uy = to_projected(1.0, 0.0, basis)       # 1 m/s due east
        got[lon] = (round(ux, 6), round(uy, 6))
        assert abs(ux - want_x) < 1e-5 and abs(uy - want_y) < 1e-5, (lon, ux, uy)
        assert abs(math.hypot(ux, uy) - 1.0) < 1e-9, "the rotation changed speed"
    # and due north at 0 E points toward the pole, i.e. +y on this map
    nx_, ny_ = to_projected(0.0, 1.0, local_basis(-70.0, 0.0))
    assert abs(nx_) < 1e-5 and abs(ny_ - 1.0) < 1e-5, (nx_, ny_)
    return (f"1 m/s due east at 70S -> {got[0.0]} at 0E, {got[90.0]} at 90E, "
            f"{got[180.0]} at 180; due north at 0E -> (0, 1)")


def test_wind_and_current_are_in_the_same_basis_before_they_are_added():
    """Round-trip, and the sum computed both ways."""
    basis = local_basis(-68.0, 143.0)
    for ue, un in ((1.0, 0.0), (0.0, 1.0), (-0.3, 0.7), (2.5, -1.25)):
        ux, uy = to_projected(ue, un, basis)
        back = to_geographic(ux, uy, basis)
        assert abs(back[0] - ue) < 1e-9 and abs(back[1] - un) < 1e-9
        assert abs(math.hypot(ux, uy) - math.hypot(ue, un)) < 1e-9
    # summing in 3976 then unrotating == unrotating then summing
    cur_x, cur_y = 0.11, -0.04                       # already in 3976
    w_e, w_n = 8.0, -3.0                             # ERA5 geographic
    k = 0.02
    wx, wy = to_projected(w_e, w_n, basis)
    a = to_geographic(cur_x + k * wx, cur_y + k * wy, basis)
    ce, cn = to_geographic(cur_x, cur_y, basis)
    b = (ce + k * w_e, cn + k * w_n)
    assert abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12
    return ("round-trip exact and speed-preserving; summing in EPSG:3976 equals "
            "summing in the geographic basis, so one frame is used throughout")


def test_raw_component_addition_would_have_been_wrong():
    """Guards the whole point: the naive sum differs, and by how much."""
    basis = local_basis(-70.0, 90.0)                 # east here is +y, not +x
    cur_x, cur_y = 0.20, 0.00                        # 0.2 m/s along +x in 3976
    w_e, w_n, k = 10.0, 0.0, 0.03                    # 10 m/s due east
    ve, vn, _, _ = drift_velocity(
        StepInputs(-70.0, 90.0, cur_x, cur_y, w_e, w_n, DAY), cfg(coeff=k))
    naive = (cur_x + k * w_e, cur_y + k * w_n)       # what NOT converting gives
    proper = (ve, vn)
    sep = math.hypot(proper[0] - naive[0], proper[1] - naive[1])
    assert sep > 0.1, f"the two agree ({sep:g}); the test location is not diagnostic"
    return (f"at 70S 90E the frame-correct drift is {proper[0]:.3f}E/"
            f"{proper[1]:.3f}N m/s; adding raw components gives "
            f"{naive[0]:.3f}/{naive[1]:.3f} -- {sep:.3f} m/s apart")


# ------------------------------------------------------------- the model
def test_zero_wind_and_zero_current():
    ve, vn, _, _ = drift_velocity(step(cu=0.0, cv=0.0, we=0.0, wn=0.0),
                                  cfg(coeff=0.05))
    assert abs(ve) < 1e-12 and abs(vn) < 1e-12
    p = predict_physics(step(), cfg(coeff=0.05))
    assert abs(p.latitude - (-65.0)) < 1e-12 and abs(p.longitude - 10.0) < 1e-12
    return "no forcing -> the berg does not move"


def test_pure_current_is_passed_through_untouched():
    basis = local_basis(-65.0, 10.0)
    cx, cy = to_projected(0.25, -0.1, basis)         # 0.25 E, -0.1 N
    ve, vn, _, _ = drift_velocity(step(cu=cx, cv=cy), cfg(coeff=0.05))
    assert abs(ve - 0.25) < 1e-9 and abs(vn + 0.1) < 1e-9
    return "with zero wind the drift is exactly the current, in east/north"


def test_pure_wind_scales_by_the_windage_coefficient_only():
    for k in (0.0, 0.01, 0.05):
        ve, vn, _, _ = drift_velocity(step(we=10.0, wn=0.0), cfg(coeff=k))
        assert abs(ve - k * 10.0) < 1e-9, (k, ve)
        assert abs(vn) < 1e-9
    return "10 m/s east, zero current -> drift east = k*10 for k in 0, 0.01, 0.05"


def test_wind_turning_angle():
    e, n = turn_wind(1.0, 0.0, 90.0)
    assert abs(e) < 1e-12 and abs(n - 1.0) < 1e-12, "+90 must take east to north"
    e, n = turn_wind(1.0, 0.0, -90.0)
    assert abs(e) < 1e-12 and abs(n + 1.0) < 1e-12
    for turn in (-45.0, 0.0, 30.0, 90.0):
        e, n = turn_wind(3.0, -4.0, turn)
        assert abs(math.hypot(e, n) - 5.0) < 1e-12, "turning changed the speed"
    ve, vn, _, _ = drift_velocity(step(we=10.0, wn=0.0), cfg(coeff=0.02, turn=90.0))
    assert abs(ve) < 1e-9 and abs(vn - 0.2) < 1e-9
    assert "counter-clockwise" in CFG.raw["turn_sign_convention"]
    return ("+90 deg rotates east to north (CCW), speed preserved; the sign "
            "convention is stated in the config")


def test_displacement_integration_and_elapsed_time():
    #  1 m/s north for one day
    lat, lon = advect(-65.0, 10.0, 0.0, 1.0, DAY)
    expect = math.degrees(DAY / EARTH_RADIUS_M)
    assert abs(lat - (-65.0 + expect)) < 1e-9 and abs(lon - 10.0) < 1e-12
    # distance actually travelled matches speed * time
    d = great_circle_m(-65.0, 10.0, lat, lon)
    assert abs(d - DAY) < 1.0, d
    # halving the time halves the displacement
    lat_h, _ = advect(-65.0, 10.0, 0.0, 1.0, DAY / 2)
    assert abs((lat_h - (-65.0)) * 2 - (lat - (-65.0))) < 1e-9
    # 1 m/s east for one day, near the pole, moves further in longitude
    _, lon_a = advect(-50.0, 0.0, 1.0, 0.0, DAY)
    _, lon_b = advect(-75.0, 0.0, 1.0, 0.0, DAY)
    assert lon_b > lon_a > 0, "eastward degrees must grow toward the pole"
    for bad in (0.0, -DAY, float("inf"), float("nan")):
        try:
            advect(-65.0, 10.0, 0.0, 1.0, bad)
            raise AssertionError(f"elapsed_seconds={bad} accepted")
        except PhysicsError:
            pass
    return ("1 m/s x 86400 s = 86.4 km travelled; halving dt halves it; a "
            "zero, negative or non-finite elapsed time is refused")


def test_longitude_wrapping_and_a_dateline_crossing_trajectory():
    """A berg drifting east over 180 E must reappear at -180, not run off.

    Distance is checked at a realistic iceberg speed. advect() lays the
    displacement along the PARALLEL while great_circle_m measures along the
    great circle, and for a pure east-west leg those differ: 0.4 m over a day
    at 0.5 m/s, but 381 m at an unphysical 5 m/s. So the fast case checks only
    that the wrap happened and the slow one checks the distance.
    """
    fast_lat, fast_lon = advect(-65.0, 179.9, 5.0, 0.0, DAY)
    assert fast_lon < 0, f"the trajectory did not wrap: {fast_lon}"
    assert -180.0 <= fast_lon < 180.0
    lat, lon = advect(-65.0, 179.9, 0.5, 0.0, DAY)    # ~43 km, a fast berg
    assert lon < 0, f"a 43 km eastward step did not cross 180: {lon}"
    d = great_circle_m(-65.0, 179.9, lat, lon)
    assert abs(d - 0.5 * DAY) < 1.0, f"wrapping distorted the distance: {d}"
    lat2, lon2 = advect(-65.0, -179.9, -0.5, 0.0, DAY)    # westward back
    assert lon2 > 0
    assert abs(great_circle_m(-65.0, -179.9, lat2, lon2) - 0.5 * DAY) < 1.0
    return (f"0.5 m/s east from 179.9E lands at {lon:.4f} with the distance "
            f"preserved to <1 m; the westward crossing mirrors it")


def test_repeated_prediction_is_deterministic():
    s = step(lat=-68.3, lon=-143.2, cu=0.07, cv=-0.02, we=6.1, wn=-2.4)
    c = cfg(coeff=0.018, turn=-20.0)
    first = predict_physics(s, c)
    for _ in range(5):
        again = predict_physics(s, c)
        assert (again.latitude, again.longitude) == (first.latitude,
                                                     first.longitude)
        assert again.drift_east_mps == first.drift_east_mps
    return "6 identical calls, bit-for-bit"


# ---------------------------------------------------------- data handling
def test_missing_environmental_input_is_refused_not_filled():
    from src.models.iceberg_physics import inputs_for
    row = {"iceberg_id": "x", "start_date": "2025-01-01", "has_environment": False}
    try:
        inputs_for(row)
        raise AssertionError("a row without both inputs produced a StepInputs")
    except PhysicsError as exc:
        assert "never fabricated" in str(exc) or "ever fabricated" in str(exc)
    fields = set(StepInputs.__dataclass_fields__)
    assert all(f in fields for f in ("current_u_3976", "wind_u10_east"))
    return "a row lacking a current or a wind value is refused, never zero-filled"


def test_the_target_endpoint_cannot_reach_the_predictor():
    fields = set(StepInputs.__dataclass_fields__)
    for leak in ("observed_end_latitude", "observed_end_longitude", "end_lat",
                 "end_longitude", "observed_end"):
        assert leak not in fields, f"StepInputs exposes {leak}"
    assert fields == {"start_latitude", "start_longitude", "current_u_3976",
                      "current_v_3976", "wind_u10_east", "wind_v10_north",
                      "elapsed_seconds"}, fields
    src = (ROOT / "src" / "models" / "iceberg_physics.py").read_text()
    tree = ast.parse(src)
    for fn in ("predict_physics", "drift_velocity", "advect",
               "predict_persistence"):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == fn)
        names = {m.attr for m in ast.walk(node) if isinstance(m, ast.Attribute)}
        names |= {m.value for m in ast.walk(node)
                  if isinstance(m, ast.Constant) and isinstance(m.value, str)}
        bad = {s for s in names if isinstance(s, str) and "observed_end" in s}
        assert not bad, f"{fn} references {bad}"
    return ("StepInputs has exactly the 7 permitted fields; none of the four "
            "prediction functions mentions an observed endpoint")


def test_persistence_uses_the_previous_step_not_the_target():
    #  a known previous velocity carried forward
    lat, lon = predict_persistence(-65.0, 10.0, 0.0, 1.0, DAY)
    assert abs(lat - advect(-65.0, 10.0, 0.0, 1.0, DAY)[0]) < 1e-12
    src = (ROOT / "src" / "models" / "iceberg_physics.py").read_text()
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_previous_step_velocity")
    keys = {m.value for m in ast.walk(node)
            if isinstance(m, ast.Constant) and isinstance(m.value, str)}
    assert "date" in keys and "prev_date" in keys
    assert "latitude_deg" not in keys, "the predecessor's END POSITION is read"
    return ("persistence carries the preceding step's observed velocity; it "
            "reads velocities and dates, never an end position")


def test_configured_parameter_validation():
    good = load_config(CONFIG)
    assert good.windage_coefficient == 0.0 and good.wind_turn_deg == 0.0
    assert good.raw["defaults_rationale"], "the zeros are unexplained"
    base = json.loads(CONFIG.read_text())
    for key, bad in (("windage_coefficient", 0.5),
                     ("windage_coefficient", -0.01),
                     ("windage_coefficient", "0.02"),
                     ("wind_turn_deg", 180.0),
                     ("windage_coefficient", float("inf"))):
        raw = dict(base); raw[key] = bad
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(raw, f, default=str); p = f.name
        try:
            load_config(p)
            raise AssertionError(f"{key}={bad!r} accepted")
        except PhysicsError:
            pass
    raw = dict(base); raw.pop("windage_coefficient")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(raw, f); p = f.name
    try:
        load_config(p)
        raise AssertionError("a config with no windage_coefficient was accepted")
    except PhysicsError:
        pass
    try:
        load_config(Path(tempfile.mkdtemp()) / "nope.json")
        raise AssertionError("a missing config was accepted")
    except PhysicsError as exc:
        assert "not chosen in code" in str(exc)
    return ("out-of-limit, negative, non-numeric, infinite, absent and missing-"
            "file configurations all refused; the shipped defaults are 0.0/0.0")


def test_no_scientific_constant_is_hardcoded():
    src = (ROOT / "src" / "models" / "iceberg_physics.py").read_text()
    body = src.split('"""', 2)[2]
    tree = ast.parse(src)
    nums = {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    # the windage figures that appear in the literature must not be inlined
    for literature in (0.018, 0.02, 0.017, 0.015, 0.012):
        assert literature not in nums, f"a windage-like constant {literature} is inlined"
    for token in ("windage_coefficient =", "wind_turn_deg ="):
        assert token not in body, f"{token} is assigned in code"
    return "no literature windage value or turning angle is written into the code"


def test_calibration_declines_when_the_sample_is_too_small():
    rows = [{"elapsed_days": 1, "interpolation_quality": QUALITY_CLEAN,
             "has_environment": True, "start_date": "2025-01-01",
             "iceberg_id": "a01", "position_source": "nic"} for _ in range(5)]
    out = calibrate(rows, CFG, 1, min_rows=200)
    assert out["performed"] is False and out["n_usable"] == 5
    assert "below the 200" in out["reason"]
    return "with 5 usable rows the fit is declined and the configured values stand"


def test_both_baselines_are_scored_on_identical_rows():
    """Regression: the first version compared 4,645 rows against 433,591."""
    from src.models.iceberg_physics import MATCHES_CSV, STEPS_CSV
    from src.models.iceberg_physics import _previous_step_velocity
    if not MATCHES_CSV.exists():
        raise AssertionError("the real matched dataset is missing")
    rows, _ = load_rows(MATCHES_CSV, STEPS_CSV)
    prev = _previous_step_velocity(STEPS_CSV)
    out = score(rows, CFG, prev, 1, QUALITY_CLEAN)
    a = out["physics_current_plus_windage"]["n"]
    b = out["constant_velocity_persistence"]["n"]
    assert a == b == out["evaluated_on_common_rows"], (a, b)
    assert out["both_baselines_scored_on_identical_rows"] is True
    assert a > 0
    return (f"both baselines scored on the same {a:,} rows; "
            f"{out['steps_with_zero_observed_displacement']:,} of them have "
            f"zero observed displacement")


def test_no_ml_cone_risk_or_routing_code():
    src = (ROOT / "src" / "models" / "iceberg_physics.py").read_text()
    body = src.split('"""', 2)[2]
    for banned in ("sklearn", "HistGradient", "RandomForest", "torch",
                   "uncertainty_cone", "risk_raster", "astar", "time_astar",
                   "navigation_cost", "polaris", "RoutingGrid", ".fit(",
                   "residual_model"):
        assert banned not in body, f"the module references {banned}"
    return "no estimator, cone, risk raster or routing symbol anywhere in the code"


def test_real_dataset_loads_and_the_clean_windows_are_what_they_should_be():
    from src.models.iceberg_physics import MATCHES_CSV, STEPS_CSV
    if not MATCHES_CSV.exists():
        raise AssertionError("the real matched dataset is missing")
    rows, stats = load_rows(MATCHES_CSV, STEPS_CSV)
    assert stats["elapsed_days_1"] > 0 and stats["elapsed_days_gt3"] > 0
    clean24 = [r for r in rows if r["elapsed_days"] == 1
               and r["interpolation_quality"] == QUALITY_CLEAN
               and r["has_environment"]]
    clean48 = [r for r in rows if r["elapsed_days"] == 2
               and r["interpolation_quality"] == QUALITY_CLEAN
               and r["has_environment"]]
    assert all(r["observed_end_latitude"] is not None for r in clean24)
    assert len(clean24) > len(clean48), "the 24h window should dominate"
    return (f"{len(rows):,} rows joined; {len(clean24):,} clean 24h and "
            f"{len(clean48):,} clean 48h rows with both environmental inputs")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78); print("iceberg drift physics baseline"); print("=" * 78)
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

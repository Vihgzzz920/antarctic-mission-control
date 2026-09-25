"""
Focused tests for the iceberg forecast uncertainty cone.

The ones that matter: the radius depends on the horizon and the calibration
alone, the test split never touches the calibration, and anything past 24 h is
labelled extrapolated.

    python tests/test_iceberg_uncertainty.py
"""
from __future__ import annotations

import ast
import csv
import inspect
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.iceberg_ml_residual import (CONFIG as SPLIT_CONFIG,          # noqa: E402
                                            build_population,
                                            chronological_split,
                                            load_config as load_split_config)
from src.models.iceberg_uncertainty import (CONFIG, LAW_LINEAR, LAW_POWER,   # noqa: E402
                                            PRED_FIELDS, SECONDS_PER_HOUR,
                                            Calibration, UncertaintyError,
                                            _quantile, calibrate, coverage,
                                            growth_factor, load_config,
                                            uncertainty_at, write_predictions)

CFG = load_config(CONFIG)
MODULE = ROOT / "src" / "models" / "iceberg_uncertainty.py"
_C: dict = {}


def pop():
    if "pop" not in _C:
        p, _ = build_population()
        chronological_split(p, load_split_config(SPLIT_CONFIG))
        _C["pop"] = p
        _C["cal"] = calibrate(p, CFG)
    return _C["pop"]


def cal() -> Calibration:
    pop()
    return _C["cal"]


def moved(split):
    return [r for r in pop() if r["split"] == split and r["moved"]]


# ----------------------------------------------------------- the horizon
def test_zero_horizon_gives_zero_radius():
    u = uncertainty_at(0.0, cal(), CFG)
    assert u.radius_km == 0.0
    assert u.horizon_hours == 0.0
    assert u.extrapolated_uncertainty is False
    assert growth_factor(0.0, CFG) == 0.0
    assert growth_factor(0.0, CFG.with_growth(law=LAW_LINEAR)) == 0.0
    return "t=0 gives exactly 0 km under both growth laws, and is not extrapolated"


def test_negative_horizon_is_rejected():
    for bad in (-1.0, -SECONDS_PER_HOUR, -1e-9):
        try:
            uncertainty_at(bad, cal(), CFG)
            raise AssertionError(f"horizon {bad} was accepted")
        except UncertaintyError as exc:
            assert "negative forecast horizon" in str(exc)
    for bad in (float("nan"), float("inf"), "24h", None, True):
        try:
            growth_factor(bad, CFG)
            raise AssertionError(f"horizon {bad!r} was accepted")
        except UncertaintyError:
            pass
    return "negative, non-finite, string, None and bool horizons all refused"


def test_the_24h_radius_is_the_calibrated_quantile():
    c = cal()
    u = uncertainty_at(24.0 * SECONDS_PER_HOUR, c, CFG)
    assert abs(u.radius_km - c.radius_km) < 1e-12, "24 h is not the calibrated point"
    assert abs(growth_factor(24.0 * SECONDS_PER_HOUR, CFG) - 1.0) < 1e-12
    assert u.extrapolated_uncertainty is False
    assert u.calibration_quantile == CFG.quantile == 0.9
    assert u.calibration_n == c.n
    #  the radius really is the empirical quantile of the calibration errors
    from src.models.iceberg_uncertainty import _physics_error_km
    errs = sorted(_physics_error_km(r) for r in pop()
                  if r["split"] in CFG.splits_used and r["moved"])
    assert abs(c.radius_km - _quantile(errs, 0.9)) < 1e-12
    inside = sum(1 for e in errs if e <= c.radius_km) / len(errs)
    assert 0.88 <= inside <= 0.92, f"in-sample coverage {inside:.3f} is not ~0.90"
    return (f"radius(24h) = {c.radius_km:.4f} km = the q{int(CFG.quantile*100)} "
            f"of {c.n} calibration errors; in-sample coverage {inside:.4f}")


def test_square_root_growth():
    c = cal()
    r24 = uncertainty_at(24 * SECONDS_PER_HOUR, c, CFG).radius_km
    r48 = uncertainty_at(48 * SECONDS_PER_HOUR, c, CFG).radius_km
    r96 = uncertainty_at(96 * SECONDS_PER_HOUR, c, CFG).radius_km
    assert abs(r48 / r24 - math.sqrt(2.0)) < 1e-12, r48 / r24
    assert abs(r96 / r24 - 2.0) < 1e-12, "four times the time should double it"
    r6 = uncertainty_at(6 * SECONDS_PER_HOUR, c, CFG).radius_km
    assert abs(r6 / r24 - 0.5) < 1e-12
    assert CFG.exponent == 0.5
    assert CFG.raw["growth"]["is_this_measured"] is False
    assert "ASSUMED here, not fitted" in CFG.raw["growth"]["assumption_statement"]
    return (f"48 h / 24 h = sqrt(2) exactly; 96 h doubles, 6 h halves; the "
            f"config states the exponent is assumed, not measured")


def test_linear_growth_is_available_for_sensitivity():
    c = cal()
    lin = CFG.with_growth(law=LAW_LINEAR)
    r24 = uncertainty_at(24 * SECONDS_PER_HOUR, c, lin).radius_km
    r48 = uncertainty_at(48 * SECONDS_PER_HOUR, c, lin).radius_km
    assert abs(r24 - c.radius_km) < 1e-12
    assert abs(r48 / r24 - 2.0) < 1e-12
    sqrt48 = uncertainty_at(48 * SECONDS_PER_HOUR, c, CFG).radius_km
    assert r48 > sqrt48, "linear must be wider than sqrt beyond the calibration"
    assert LAW_LINEAR in CFG.raw["growth"]["available_laws"]
    assert "48 h" in CFG.raw["growth"]["law_selection_forbidden_on"] or \
        "48 h rows" in CFG.raw["growth"]["law_selection_forbidden_on"]
    #  the shipped default is NOT linear
    assert CFG.law == LAW_POWER
    return (f"linear gives {r48:.2f} km at 48 h against sqrt's {sqrt48:.2f} km; "
            f"the shipped law stays {LAW_POWER} and the config forbids choosing "
            f"between them on the 48 h rows")


def test_radius_is_always_finite_and_non_negative():
    c = cal()
    for hours in (0.0, 0.5, 1.0, 6.0, 24.0, 25.0, 48.0, 72.0, 240.0):
        for law in (CFG, CFG.with_growth(law=LAW_LINEAR)):
            u = uncertainty_at(hours * SECONDS_PER_HOUR, c, law)
            assert math.isfinite(u.radius_km) and u.radius_km >= 0.0, (hours, u)
    #  the configured cap is a backstop on the RADIUS, not a limit on the
    #  horizon: under sqrt growth it only bites at absurd times, which is the
    #  point -- it catches a misconfigured exponent or a linear runaway.
    ok_far = uncertainty_at(1e9, c, CFG)          # ~277,778 h -> ~1,948 km
    assert math.isfinite(ok_far.radius_km) and ok_far.extrapolated_uncertainty
    try:
        uncertainty_at(1e12, c, CFG)
        raise AssertionError("a radius past the configured cap was produced")
    except UncertaintyError as exc:
        assert "exceeds the configured maximum" in str(exc)
    return (f"9 horizons x 2 laws all finite and >= 0; the {CFG.max_radius_km:g} km "
            f"cap refuses a runaway radius")


def test_24h_versus_extrapolated_48h_labelling():
    c = cal()
    u24 = uncertainty_at(24 * SECONDS_PER_HOUR, c, CFG)
    u48 = uncertainty_at(48 * SECONDS_PER_HOUR, c, CFG)
    u23 = uncertainty_at(23.9 * SECONDS_PER_HOUR, c, CFG)
    assert u24.extrapolated_uncertainty is False
    assert u48.extrapolated_uncertainty is True
    assert u23.extrapolated_uncertainty is False
    for u in (u24, u48):
        assert u.empirically_calibrated_to_hours == 24.0
    d = u48.as_dict()
    for key in ("radius_km", "horizon_seconds", "extrapolated_uncertainty",
                "empirically_calibrated_to_hours", "calibration_quantile",
                "calibration_n", "growth_law", "growth_exponent"):
        assert key in d, key
    return ("24 h is calibrated, 48 h carries extrapolated_uncertainty=True, and "
            "both state empirically_calibrated_to_hours=24.0")


# ------------------------------------------------------------- no leakage
def test_the_test_split_is_never_used_to_calibrate():
    c = cal()
    assert "test" not in c.splits_used
    assert set(c.splits_used) == {"train", "validation"}
    assert c.n == len(moved("train")) + len(moved("validation"))
    assert c.date_max < min(r["start_date"] for r in moved("test"))
    #  a config naming the test split is refused outright
    raw = json.loads(CONFIG.read_text())
    raw["calibration"]["splits_used"] = ["train", "validation", "test"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(raw, fh); p = fh.name
    try:
        load_config(p)
        raise AssertionError("a config calibrating on the test split was accepted")
    except UncertaintyError as exc:
        assert "held-out rows" in str(exc)
    #  and the calibrated radius is unchanged if the test rows are removed
    without = calibrate([r for r in pop() if r["split"] != "test"], CFG)
    assert without.radius_km == c.radius_km and without.n == c.n
    return (f"calibration uses {c.n} train+validation rows ending "
            f"{c.date_max}; deleting every test row changes nothing, and a "
            f"config naming 'test' is refused")


def test_no_observed_endpoint_can_reach_the_radius():
    sig = set(inspect.signature(uncertainty_at).parameters)
    for leak in ("observed_end_latitude", "observed_end_longitude",
                 "observed", "truth", "actual"):
        assert leak not in sig, f"uncertainty_at accepts {leak}"
    assert "observed" not in "".join(sig)
    tree = ast.parse(MODULE.read_text())
    for fn in ("uncertainty_at", "growth_factor"):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == fn)
        body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                 and isinstance(node.body[0].value, ast.Constant)
                                 ) else node.body
        names = {m.attr for stmt in body for m in ast.walk(stmt)
                 if isinstance(m, ast.Attribute)}
        names |= {m.value for stmt in body for m in ast.walk(stmt)
                  if isinstance(m, ast.Constant) and isinstance(m.value, str)}
        bad = {s for s in names if isinstance(s, str) and "observed" in s}
        assert not bad, f"{fn} references {bad}"
    #  two different icebergs at the same horizon get the SAME radius
    c = cal()
    a, b = moved("test")[0], moved("test")[5]
    ra = uncertainty_at(48 * SECONDS_PER_HOUR, c, CFG, iceberg_id=a["iceberg_id"])
    rb = uncertainty_at(48 * SECONDS_PER_HOUR, c, CFG, iceberg_id=b["iceberg_id"])
    assert ra.radius_km == rb.radius_km
    return ("uncertainty_at takes no observed field and names none; the radius "
            "is identical for two different icebergs at one horizon")


# ----------------------------------------------------------- determinism
def test_output_is_deterministic():
    c1 = calibrate(pop(), CFG)
    c2 = calibrate(pop(), CFG)
    assert c1.radius_km == c2.radius_km and c1.n == c2.n
    assert c1.quantile_radii_km == c2.quantile_radii_km
    for hours in (0.0, 24.0, 48.0, 72.0):
        a = uncertainty_at(hours * SECONDS_PER_HOUR, c1, CFG).as_dict()
        b = uncertainty_at(hours * SECONDS_PER_HOUR, c2, CFG).as_dict()
        assert a == b
    d1, d2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    rows = [r for r in pop() if r["moved"]][:200]
    write_predictions(rows, c1, CFG, d1 / "p.csv")
    write_predictions(rows, c2, CFG, d2 / "p.csv")
    assert (d1 / "p.csv").read_text() == (d2 / "p.csv").read_text()
    return "identical calibration, radii and CSV bytes across two runs"


def test_coverage_counts_are_what_they_claim():
    c = cal()
    cov = coverage(moved("test"), c, CFG)
    from src.models.iceberg_uncertainty import _physics_error_km
    errs = [_physics_error_km(r) for r in moved("test")]
    assert cov["n"] == len(errs)
    for q in CFG.reported_quantiles:
        key = f"fraction_inside_q{int(q * 100)}_radius"
        r = c.quantile_radii_km[f"q{int(q * 100)}"]
        want = round(sum(1 for e in errs if e <= r) / len(errs), 4)
        assert cov[key] == want, (key, cov[key], want)
    assert cov["selected_radius_km"] == cov["mean_radius_km"], (
        "a circular cone at one horizon has a single radius")
    return (f"{cov['n']} held-out rows; every reported fraction recomputed from "
            f"the raw errors and matches")


# -------------------------------------------------------------- the rest
def test_configuration_validation():
    for mutate, why in (
            (lambda r: r["calibration"].update(quantile=1.0), "quantile 1.0"),
            (lambda r: r["calibration"].update(quantile=0.0), "quantile 0.0"),
            (lambda r: r["calibration"].update(quantile="0.9"), "a string quantile"),
            (lambda r: r["calibration"].update(horizon_hours=0), "zero horizon"),
            (lambda r: r["growth"].update(law="magic"), "an unknown law"),
            (lambda r: r["growth"].update(exponent=9.0), "an out-of-limit exponent"),
            (lambda r: r["growth"].update(exponent=float("nan")), "a NaN exponent"),
            (lambda r: r["growth"].update(is_this_measured=True),
             "claiming the growth law is measured"),
            (lambda r: r["calibration"].update(splits_used=[]), "no splits")):
        raw = json.loads(CONFIG.read_text())
        mutate(raw)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(raw, fh, default=str); p = fh.name
        try:
            load_config(p)
            raise AssertionError(f"{why} was accepted")
        except UncertaintyError:
            pass
    try:
        load_config(Path(tempfile.mkdtemp()) / "nope.json")
        raise AssertionError("a missing config was accepted")
    except UncertaintyError as exc:
        assert "not chosen in code" in str(exc)
    return "nine malformed configurations and a missing file all refused"


def test_prediction_table_carries_full_provenance():
    for col in ("iceberg_id", "start_date", "horizon_hours",
                "physics_pred_latitude", "physics_pred_longitude",
                "uncertainty_radius_km", "extrapolated_uncertainty",
                "calibration_quantile", "calibration_n", "growth_law",
                "interpolation_quality"):
        assert col in PRED_FIELDS, col
    d = Path(tempfile.mkdtemp())
    rows = [r for r in pop() if r["moved"]][:50]
    n = write_predictions(rows, cal(), CFG, d / "p.csv")
    got = list(csv.DictReader((d / "p.csv").open()))
    assert n == len(got) == len(rows) * len(CFG.raw["horizons_hours_for_output"])
    by_h = {}
    for g in got:
        by_h.setdefault(float(g["horizon_hours"]), set()).add(
            g["uncertainty_radius_km"])
    assert all(len(v) == 1 for v in by_h.values()), "the radius varied per row"
    assert {g["extrapolated_uncertainty"] for g in got
            if float(g["horizon_hours"]) > 24.0} == {"True"}
    assert {g["extrapolated_uncertainty"] for g in got
            if float(g["horizon_hours"]) == 24.0} == {"False"}
    return (f"{n} rows over {len(by_h)} horizons; one radius per horizon; "
            f"everything above 24 h flagged extrapolated")


def test_no_risk_raster_route_cost_probability_or_ml():
    body = MODULE.read_text().split('"""', 2)[2]
    for banned in ("sklearn", "HistGradient", "torch", "risk_raster",
                   "collision_prob", "route_cost", "astar", "time_astar",
                   "navigation_cost", "polaris", "RoutingGrid", "replan"):
        assert banned not in body, f"the module references {banned}"
    #  and it does not redefine physics: windage may be RECORDED as provenance,
    #  but must never be computed with here.
    assert "def predict_physics" not in body
    assert "def drift_velocity" not in body and "def advect" not in body
    assert "from src.models.iceberg_physics import" in MODULE.read_text()
    for line in body.splitlines():
        if "windage" in line:
            assert line.strip().startswith('"windage_coefficient": 0.0'), line
    return ("no estimator, risk raster, collision probability, route cost or "
            "router; advect and the physics model are imported, and windage "
            "appears only as a recorded constant")


def test_secondary_component_statistics_stay_secondary():
    c = cal()
    s = c.component_statistics_km
    assert s["status"].startswith("SECONDARY")
    body = MODULE.read_text().split('"""', 2)[2]
    #  nothing in the radius path consults them
    tree = ast.parse(MODULE.read_text())
    for fn in ("uncertainty_at", "growth_factor"):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == fn)
        names = {m.attr for m in ast.walk(node) if isinstance(m, ast.Attribute)}
        assert "component_statistics_km" not in names, fn
    return (f"east/north sd {s['stdev_east']:.2f}/{s['stdev_north']:.2f} km "
            f"(ratio {s['anisotropy_stdev_ratio']:.3f}, r={s['correlation_east_north']:+.3f}) "
            f"reported but never read by the radius")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78); print("iceberg forecast uncertainty cone"); print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<54} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__); print(f"  FAIL  {fn.__name__:<54} {exc}")
        except Exception as exc:                        # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__:<54} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

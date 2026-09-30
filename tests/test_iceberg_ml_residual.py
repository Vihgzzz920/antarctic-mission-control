"""
Focused tests for the ML residual correction on the physics drift baseline.

The leakage tests are the ones that matter: the observed endpoint builds the
target and scores the result, and must reach nothing else.

    python tests/test_iceberg_ml_residual.py
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

from src.data.preprocess_icebergs import great_circle_m                    # noqa: E402
from src.models.iceberg_ml_residual import (CONFIG, HORIZON_S, PRED_FIELDS,  # noqa: E402
                                            SPLITS, ResidualError,
                                            _fit_pair, _leakage_checks,
                                            build_population,
                                            chronological_split,
                                            displacement_components,
                                            feature_matrix, load_config,
                                            offset_position,
                                            predict_corrected)

CFG = load_config(CONFIG)
MODULE = ROOT / "src" / "models" / "iceberg_ml_residual.py"
_CACHE: dict = {}


def population():
    if "pop" not in _CACHE:
        pop, stats = build_population()
        _CACHE["pop"] = pop
        _CACHE["stats"] = stats
        _CACHE["split"] = chronological_split(pop, CFG)
    return _CACHE["pop"]


def moved_by_split():
    pop = population()
    return {s: [r for r in pop if r["split"] == s and r["moved"]] for s in SPLITS}


# ------------------------------------------------------------- no leakage
def test_the_target_is_not_among_the_features():
    feats = CFG.features
    banned = CFG.raw["forbidden_feature_substrings"]
    for f in feats:
        for b in banned:
            assert b not in f, f"feature {f!r} contains {b!r}"
    for name in ("residual_east_m", "residual_north_m", "observed_end_latitude",
                 "observed_end_longitude", "observed_m"):
        assert name not in feats, f"{name} is a feature"
    # and the loader refuses a config that tries
    raw = dict(CFG.raw); raw["features"] = feats + ["observed_end_latitude"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(raw, fh); p = fh.name
    try:
        load_config(p)
        raise AssertionError("a config listing the observed endpoint was accepted")
    except ResidualError as exc:
        assert "forbidden substring" in str(exc)
    return (f"{len(feats)} features, none matching {banned}; a config that "
            f"lists the endpoint is refused at load time")


def test_the_observed_endpoint_cannot_enter_the_prediction_api():
    tree = ast.parse(MODULE.read_text())
    for fn in ("predict_corrected", "feature_matrix"):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == fn)
        # Scan the CODE, not the prose: a docstring that says "no target is
        # read" is not a reference to the target.
        body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                 and isinstance(node.body[0].value, ast.Constant)
                                 and isinstance(node.body[0].value.value, str)
                                 ) else node.body
        nodes = [n for stmt in body for n in ast.walk(stmt)]
        strings = {m.value for m in nodes
                   if isinstance(m, ast.Constant) and isinstance(m.value, str)}
        attrs = {m.attr for m in nodes if isinstance(m, ast.Attribute)}
        bad = {s for s in strings | attrs
               if isinstance(s, str) and ("observed" in s or "residual_" in s)}
        assert not bad, f"{fn} references {bad}"
    #  a row stripped of every target field still predicts
    rows = moved_by_split()["train"][:20]
    models = _fit_pair(rows, CFG.features, CFG.grid[0], CFG.random_state)
    keep = set(CFG.features) | {"start_latitude", "start_longitude",
                                "physics_disp_east_m", "physics_disp_north_m",
                                "elapsed_seconds"}
    bare = [{k: r[k] for k in keep} for r in rows]
    for b in bare:
        for k in ("observed_end_latitude", "residual_east_m"):
            assert k not in b
    out = predict_corrected(models, bare, CFG.features)
    assert len(out) == len(rows) and all(math.isfinite(o[0]) for o in out)
    return ("neither predict_corrected nor feature_matrix names an observed or "
            "residual field, and prediction works on rows with none present")


def test_no_future_date_leaks_across_the_splits():
    pop = population()
    by = {s: [r for r in pop if r["split"] == s] for s in SPLITS}
    checks = _leakage_checks(CFG, by, moved_by_split(), CFG.features)
    assert checks["train_strictly_before_validation"] is True
    assert checks["validation_strictly_before_test"] is True
    assert checks["splits_are_disjoint"] is True
    assert checks["no_forbidden_substring_in_any_feature"] is True
    assert checks["observed_endpoint_in_feature_list"] == []
    assert checks["max_train_date"] < checks["min_validation_date"]
    assert checks["max_validation_date"] < checks["min_test_date"]
    return (f"train <= {checks['max_train_date']} < "
            f"{checks['min_validation_date']} <= validation <= "
            f"{checks['max_validation_date']} < {checks['min_test_date']} "
            f"<= test; the three sets are disjoint")


def test_no_row_from_validation_or_test_is_ever_fitted():
    src = MODULE.read_text()
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "run")
    fits = [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "_fit_pair"]
    assert fits, "no _fit_pair call found in run()"
    for call in fits:
        first = call.args[0]
        assert isinstance(first, ast.Subscript), ast.dump(first)
        assert isinstance(first.slice, ast.Constant) and first.slice.value == "train", (
            f"_fit_pair is called on {ast.dump(first.slice)}, not moved['train']")
    return f"all {len(fits)} _fit_pair calls in run() take moved['train'] only"


# --------------------------------------------------------------- the split
def test_chronological_split_is_never_randomised():
    pop = population()
    s = _CACHE["split"]
    for name in SPLITS:
        assert s["counts"][name]["rows_moved"] > 0
    a = chronological_split(list(pop), CFG)
    b = chronological_split(list(reversed(pop)), CFG)
    assert a["train_before"] == b["train_before"] == s["train_before"]
    assert a["validation_before"] == b["validation_before"]
    src = MODULE.read_text().split('"""', 2)[2]
    for banned in ("train_test_split", "shuffle", "random.sample",
                   "np.random", "KFold"):
        assert banned not in src, f"the module uses {banned}"
    #  a whole date never straddles a boundary
    by_date: dict[str, set] = {}
    for r in pop:
        by_date.setdefault(r["start_date"], set()).add(r["split"])
    straddling = {d: v for d, v in by_date.items() if len(v) > 1}
    assert not straddling, f"dates in two splits: {list(straddling)[:3]}"
    return (f"boundaries {s['train_before']} / {s['validation_before']} are "
            f"reproduced from reversed input; no date sits in two splits; no "
            f"shuffling primitive appears")


def test_training_population_is_moved_rows_only():
    mv = moved_by_split()
    assert all(r["moved"] for r in mv["train"])
    assert all(r["observed_m"] > 0.0 for r in mv["train"])
    pop = population()
    zero_train = [r for r in pop if r["split"] == "train" and not r["moved"]]
    assert zero_train, "the fixture has no zero-displacement training rows to exclude"
    assert not any(id(r) in {id(x) for x in mv["train"]} for r in zero_train)
    cfg_pop = CFG.raw["population"]
    assert cfg_pop["moved_only_for_training"] is True
    assert "great_circle_m" in cfg_pop["moved_definition"]
    return (f"{len(mv['train'])} moved training rows; the "
            f"{len(zero_train)} zero-displacement rows in the same window are "
            f"excluded from the fit")


# --------------------------------------------------------------- mechanics
def test_corrected_displacement_is_physics_plus_predicted_residual():
    rows = moved_by_split()["train"][:40]
    models = _fit_pair(rows, CFG.features, CFG.grid[0], CFG.random_state)
    preds = predict_corrected(models, rows, CFG.features)
    worst = 0.0
    for r, (lat, lon, de, dn) in zip(rows, preds):
        want_e = r["physics_disp_east_m"] + de
        want_n = r["physics_disp_north_m"] + dn
        got_e, got_n = displacement_components(
            r["start_latitude"], r["start_longitude"], lat, lon)
        worst = max(worst, abs(got_e - want_e), abs(got_n - want_n))
    assert worst < 1e-3, f"the corrected endpoint is off by {worst:g} m"
    return (f"over 40 rows the endpoint decomposes back to physics + residual "
            f"to within {worst:.2e} m")


def test_displacement_and_offset_are_inverses():
    for lat, lon in ((-65.0, 10.0), (-70.1, 179.9), (-55.0, -179.95)):
        for de, dn in ((0.0, 0.0), (5000.0, -3000.0), (-42000.0, 8000.0)):
            lat1, lon1 = offset_position(lat, lon, de, dn, HORIZON_S)
            back = displacement_components(lat, lon, lat1, lon1)
            assert abs(back[0] - de) < 1e-4 and abs(back[1] - dn) < 1e-4, (
                lat, lon, de, dn, back)
            assert -180.0 <= lon1 < 180.0
    return "offset then decompose returns the same east/north, dateline included"


def test_feature_matrix_ordering_is_the_configured_order():
    rows = moved_by_split()["train"][:5]
    X = feature_matrix(rows, CFG.features)
    assert X.shape == (5, len(CFG.features))
    for j, f in enumerate(CFG.features):
        for i, r in enumerate(rows):
            assert X[i, j] == float(r[f]), (i, j, f)
    swapped = list(reversed(CFG.features))
    Y = feature_matrix(rows, swapped)
    assert not np.array_equal(X, Y), "column order made no difference"
    assert np.array_equal(X, Y[:, ::-1])
    return f"{len(CFG.features)} columns land in the configured order, not sorted"


def test_missing_or_non_finite_input_is_refused_not_imputed():
    rows = [dict(r) for r in moved_by_split()["train"][:3]]
    for bad in (None, float("nan"), float("inf")):
        broken = [dict(r) for r in rows]
        broken[1][CFG.features[0]] = bad
        try:
            feature_matrix(broken, CFG.features)
            raise AssertionError(f"{bad!r} was accepted as a feature value")
        except ResidualError as exc:
            assert "nothing is imputed" in str(exc)
    missing = [dict(r) for r in rows]
    del missing[0][CFG.features[2]]
    try:
        feature_matrix(missing, CFG.features)
        raise AssertionError("an absent feature was accepted")
    except ResidualError:
        pass
    return "None, NaN, inf and an absent column are all refused, never filled"


def test_repeated_fitting_is_deterministic():
    rows = moved_by_split()["train"]
    a = _fit_pair(rows, CFG.features, CFG.grid[0], CFG.random_state)
    b = _fit_pair(rows, CFG.features, CFG.grid[0], CFG.random_state)
    X = feature_matrix(rows[:60], CFG.features)
    for comp in ("east", "north"):
        assert np.array_equal(a[comp].predict(X), b[comp].predict(X)), comp
    pa = predict_corrected(a, rows[:60], CFG.features)
    pb = predict_corrected(b, rows[:60], CFG.features)
    assert pa == pb
    assert isinstance(CFG.random_state, int)
    return f"two fits with random_state={CFG.random_state} predict identically"


def test_moved_definition_uses_the_project_distance_function():
    pop = population()
    for r in pop[:200]:
        d = great_circle_m(r["start_latitude"], r["start_longitude"],
                           r["observed_end_latitude"], r["observed_end_longitude"])
        assert r["moved"] == (d > 0.0)
        assert abs(r["observed_m"] - d) < 1e-9
    return "moved == great_circle_m(start, observed_end) > 0 on 200 real rows"


def test_configuration_validation():
    for mutate, why in (
            (lambda raw: raw.pop("features"), "no feature list"),
            (lambda raw: raw.update(features=["a", "a"]), "a repeated feature"),
            (lambda raw: raw.update(
                physics_displacement_features=["not_a_feature"]),
             "a physics feature outside the list"),
            (lambda raw: raw["estimator"].update(grid=[]), "an empty grid")):
        raw = json.loads(CONFIG.read_text())
        mutate(raw)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(raw, fh); p = fh.name
        try:
            load_config(p)
            raise AssertionError(f"{why} was accepted")
        except ResidualError:
            pass
    try:
        load_config(Path(tempfile.mkdtemp()) / "nope.json")
        raise AssertionError("a missing config was accepted")
    except ResidualError:
        pass
    return "four malformed configurations and a missing file all refused"


def test_the_ablation_reports_its_own_degeneracy():
    """Under windage 0.0 the physics features are 86400x the current features."""
    from src.models.iceberg_ml_residual import _ablation_is_degenerate
    rows = moved_by_split()["train"]
    chk = _ablation_is_degenerate(rows, CFG)
    for f, v in chk["per_feature"].items():
        assert v["most_correlated_surviving_feature"] is not None, f
        if v["is_a_rescale_of_it"]:
            assert v["abs_pearson_r"] > 1 - 1e-9
            assert abs(v["median_ratio"] - HORIZON_S) < 1.0, v
    if chk["physics_features_are_redundant"]:
        assert "cannot test" in chk["consequence"]
    #  and the claim is true of the fitted models, not just the columns
    feats = CFG.features
    nop = [f for f in feats if f not in CFG.physics_features]
    a = _fit_pair(rows, feats, CFG.grid[0], CFG.random_state)
    b = _fit_pair(rows, nop, CFG.grid[0], CFG.random_state)
    test_rows = moved_by_split()["test"]
    same = predict_corrected(a, test_rows, feats) == predict_corrected(
        b, test_rows, nop)
    assert same == chk["physics_features_are_redundant"], (
        "the degeneracy flag disagrees with what the two models actually do")
    return (f"physics features are a x{HORIZON_S:.0f} rescale of the current "
            f"features (r=1); the flag matches the two models predicting "
            f"identically on all {len(test_rows)} test rows")


def test_no_deep_learning_cone_risk_or_routing_code():
    body = MODULE.read_text().split('"""', 2)[2]
    for banned in ("torch", "tensorflow", "keras", "MLPRegressor",
                   "uncertainty_cone", "risk_raster", "astar", "time_astar",
                   "navigation_cost", "polaris", "RoutingGrid"):
        assert banned not in body, f"the module references {banned}"
    assert "HistGradientBoostingRegressor" in body
    return "one sklearn estimator; no network, cone, risk raster or router"


def test_prediction_table_schema_covers_all_three_baselines():
    for col in ("physics_pred_latitude", "ml_pred_latitude",
                "constant_velocity_pred_latitude", "physics_error_km",
                "ml_error_km", "constant_velocity_error_km", "split", "moved",
                "observed_end_latitude"):
        assert col in PRED_FIELDS, col
    return f"{len(PRED_FIELDS)} columns carry all three baselines and the split"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78); print("iceberg ML residual correction"); print("=" * 78)
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

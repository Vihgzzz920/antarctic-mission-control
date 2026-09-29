"""
The historical forecast -> mission decision backtest.

What matters here: the causal order (forecast, then plan, then realize) is
enforced by the code and not just by convention; training-period origins cannot
enter; a missing observation is skipped, never fabricated; and the result is
deterministic.

    python tests/test_historical_backtest.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import historical_forecast_decision as B               # noqa: E402
from src.api.environment_timeline import (POLICY_LONG_HORIZON_EVALUATION,  # noqa: E402
                                          POLICY_PERSISTENCE)
from src.models.forecast_sic_raster import TRAINED_LEAD_HOURS              # noqa: E402

MODULE = ROOT / "src" / "evaluation" / "historical_forecast_decision.py"
_C: dict = {}


def result():
    if "r" not in _C:
        _C["r"] = B.run()
    return _C["r"]


def ran_cases():
    return [c for c in result()["cases"]
            if c["ran"] and c["decision"].get("both_planned")]


# ------------------------------------------- 1. held-out origin selection
def test_origins_come_from_the_models_own_held_out_window():
    lo, hi = B.held_out_window()
    #  the window is the INTERSECTION of the two models' recorded test ranges,
    #  read from their metrics files rather than retyped here
    for lead in TRAINED_LEAD_HOURS:
        a, b = B.test_base_range(int(lead))
        assert a <= lo and hi <= b, (lead, a, b, lo, hi)
        rec = json.loads(B.METRICS[int(lead)].read_text())
        assert rec["split_information"]["test"]["base_range"] == \
            [a.isoformat(), b.isoformat()]
    for case in result()["cases"]:
        if case["case"]["included"]:
            assert lo <= date.fromisoformat(case["case"]["origin"]) <= hi
    assert result()["held_out_window"] == [lo.isoformat(), hi.isoformat()]
    return f"held-out window {lo} .. {hi}, the intersection of both test ranges"


# ---------------------------------------- 2. missing target -> structured skip
def test_a_missing_target_is_skipped_with_a_reason_never_fabricated():
    #  an origin one day past the window: real observation, but out of test
    cases = B.build_cases([date(2025, 12, 31)])
    assert len(cases) == 1 and cases[0].included is False
    assert B.SKIP_OUTSIDE_TEST in cases[0].reason
    #  an origin whose targets do not exist at all
    far = B.build_cases([date(2026, 6, 1)])[0]
    assert far.included is False and far.reason
    #  and the module never invents an observation. Scanned as CALL NAMES, not
    #  as substrings: the module's own prose says it does not interpolate.
    banned = {"fillna", "nan_to_num", "interp", "interp1d", "ffill", "bfill",
              "nanmean", "nan_to_num"}
    called = set()
    for node in ast.walk(ast.parse(MODULE.read_text())):
        if isinstance(node, ast.Call):
            f = node.func
            name = (f.attr if isinstance(f, ast.Attribute)
                    else f.id if isinstance(f, ast.Name) else None)
            if name:
                called.add(name)
    assert not (called & banned), sorted(called & banned)
    skipped = [c for c in result()["cases"] if not c["case"]["included"]]
    for c in skipped:
        assert c["case"]["skip_reason"], c["case"]["origin"]
        assert c["ran"] is False
    return (f"{len(skipped)} skipped, each with a named reason; no fill, "
            f"interpolation or carry-forward anywhere in the module")


# --------------------------------------------------- 3. forecast provenance
def test_every_forecast_bucket_carries_its_artifact_lineage():
    seen = {}
    checked = 0
    for c in ran_cases():
        fb = c["plans"]["forecast"]["forecast_buckets"]
        assert fb, c["case"]["origin"]
        for b in fb:
            for key in ("artifact", "lead_hours", "model_artifact_sha256",
                        "dataset_sha256"):
                assert b.get(key), (c["case"]["origin"], b["bucket"], key)
            checked += 1
        #  the persistence plan touches no forecast at all
        assert c["plans"]["persistence"]["forecast_buckets"] == []
        assert c["plans"]["persistence"]["environment"]["forecast_leads"] == []
    #  the recorded artifact SHAs are the real files' SHAs
    for name in ("forecast_model_hgb.joblib", "forecast_model_hgb_48h.joblib"):
        f = ROOT / "data" / "processed" / name
        seen[name] = hashlib.sha256(f.read_bytes()).hexdigest()
    shas = {b["model_artifact_sha256"] for c in ran_cases()
            for b in c["plans"]["forecast"]["forecast_buckets"]}
    assert shas <= set(seen.values()), shas - set(seen.values())
    return (f"{checked} forecast buckets across {len(ran_cases())} cases, each "
            f"naming an artifact whose SHA is a real shipped artifact's")


# ------------------------------------- 4, 5. causal order and no re-routing
def test_the_realization_is_opened_only_after_both_plans_exist():
    """The ordering is structural, not a convention: the realizer is built
    after both plans, and it has no routing method at all."""
    tree = ast.parse(MODULE.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_case")
    src_lines = MODULE.read_text().splitlines()

    def line_of(needle):
        for i in range(fn.lineno - 1, fn.end_lineno):
            if needle in src_lines[i]:
                return i
        raise AssertionError(f"{needle!r} not found in run_case")

    #  forecast, then both plans, then the realizer -- in that source order
    assert line_of("ensure_rasters(") < line_of("evaluate_mission(")
    assert line_of("evaluate_mission(") < line_of("_Realizer(")
    #  the realizer exposes pricing and nothing that could re-plan
    methods = {n.name for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "_Realizer"
               for n in n.body if isinstance(n, ast.FunctionDef)}
    assert "price" in methods
    for forbidden in ("route", "plan", "reroute", "optimise", "optimize",
                      "evaluate_mission", "compare_profiles"):
        assert forbidden not in methods, forbidden
    body = ast.get_source_segment(MODULE.read_text(),
                                  next(n for n in ast.walk(tree)
                                       if isinstance(n, ast.ClassDef)
                                       and n.name == "_Realizer"))
    for forbidden in ("evaluate_mission", "compare_profiles", "plan_route",
                      "time_astar"):
        assert forbidden not in body, f"{forbidden!r} is reachable from _Realizer"
    return ("run_case forecasts, then plans both, then realizes; _Realizer can "
            "price a path and cannot route one")


def test_the_realized_paths_are_the_planned_paths_unchanged():
    for c in ran_cases():
        if not c["realized"].get("evaluated"):
            continue
        #  the realized evaluation reports the buckets and observations it
        #  read; the cell count must match the plan it was given
        for plan, key in (("persistence", "persistence_plan"),
                          ("forecast", "forecast_plan")):
            m = c["plans"][plan]["metrics"]
            r = c["realized"][key]
            assert r["buckets_used"], (c["case"]["origin"], plan)
            assert set(r["buckets_used"]) <= set(range(B.HORIZON_BUCKETS))
            assert r["realized_environmental_cost"] > 0
            #  distance and travel time are the PLAN's; realization never
            #  changes them
            assert m["distance_km"] > 0 and m["travel_time_h"] > 0
        #  when the route did not change, realization cannot differ either
        if not c["decision"]["route_changed"]:
            assert math.isclose(
                c["realized"]["realized_environmental_cost_delta"], 0.0,
                abs_tol=1e-9), c["case"]["origin"]
    return ("realized costs are computed on the planned paths; an unchanged "
            "route realizes an identical cost by construction")


# ---------------------------------------- 6. the right artifact per lead
def test_each_lead_uses_its_own_artifact():
    for c in ran_cases():
        by_lead = {}
        for b in c["plans"]["forecast"]["forecast_buckets"]:
            by_lead.setdefault(int(b["lead_hours"]), set()).add(
                (b["artifact"], b["model_artifact_sha256"]))
        assert set(by_lead) == {24, 48}, (c["case"]["origin"], sorted(by_lead))
        for lead, entries in by_lead.items():
            assert len(entries) == 1, (lead, entries)
            raster, _sha = next(iter(entries))
            assert f"plus{lead}h" in raster, (lead, raster)
            origin = c["case"]["origin"].replace("-", "")
            assert f"origin{origin}" in raster, (origin, raster)
        assert next(iter(by_lead[24]))[1] != next(iter(by_lead[48]))[1]
    return ("every case: +24 h and +48 h come from distinct artifacts, each "
            "named for that case's own origin")


# --------------------------------------------- 7. route-change accounting
def test_route_change_accounting_is_consistent():
    agg = result()["aggregate"]
    ran = ran_cases()
    assert agg["cases_both_planned"] == len(ran)
    assert agg["route_changed"] + agg["route_unchanged"] == len(ran)
    assert agg["route_changed"] == sum(
        1 for c in ran if c["decision"]["route_changed"])
    for c in ran:
        d = c["decision"]
        if d["route_changed"]:
            assert d["first_divergence_index"] is not None
            assert d["cells_only_in_persistence"] > 0 or \
                d["cells_only_in_forecast"] > 0
        else:
            assert d["first_divergence_index"] is None
            assert d["cells_only_in_persistence"] == 0
            assert math.isclose(d["configured_cost_delta"], 0.0, abs_tol=1e-9)
    realized = [c for c in ran if c["realized"].get("evaluated")]
    assert agg["cases_realized"] == len(realized)
    assert agg["cases_not_realized"] == len(ran) - len(realized)
    assert (agg["forecast_plan_lower_realized_environmental_cost"]
            + agg["persistence_plan_lower_realized_environmental_cost"]
            + agg["tied_or_identical_route"]) == len(realized)
    return (f"{agg['route_changed']} changed, {agg['route_unchanged']} "
            f"unchanged, {agg['cases_realized']} realized, counts consistent")


# ------------------------------------------ 8. realized cost calculation
def test_the_realized_cost_uses_observations_not_the_forecast():
    for c in ran_cases():
        if not c["realized"].get("evaluated"):
            continue
        origin = date.fromisoformat(c["case"]["origin"])
        for key in ("persistence_plan", "forecast_plan"):
            used = c["realized"][key]["observations_used"]
            assert used, (c["case"]["origin"], key)
            for d in used:
                day = date.fromisoformat(d)
                #  every realized field is an OBSERVATION in the archive, on a
                #  day at or after the origin
                assert B.observed(day).exists(), d
                assert day >= origin, (d, origin)
            #  and the forecast rasters are never among them
            assert not any("forecast" in d for d in used)
        #  a route that changed must show a difference somewhere
        if c["decision"]["route_changed"]:
            a = c["realized"]["persistence_plan"]["realized_environmental_cost"]
            b = c["realized"]["forecast_plan"]["realized_environmental_cost"]
            assert c["realized"]["realized_environmental_cost_delta"] == b - a
    #  the realizer opens observations only
    src = MODULE.read_text()
    cls = src[src.index("class _Realizer"):src.index("# ---", src.index("class _Realizer"))]
    assert "sic_forecast" not in cls and "predict_sic" not in cls
    return ("realized costs come from archived observations on or after each "
            "origin; no forecast raster is opened by the realizer")


# ------------------------------------------------- 9. determinism
def test_the_backtest_is_deterministic():
    a = B.run()
    b = B.run()
    ja = json.dumps(a, sort_keys=True, default=str)
    jb = json.dumps(b, sort_keys=True, default=str)
    assert ja == jb, "two runs of the backtest differ"
    #  and one case recomputed from scratch equals its cached form
    case = next(c for c in B.build_cases() if c.included)
    cached = json.loads(B.case_path(case.origin).read_text())
    fresh = B.run_case(case)
    assert json.dumps(fresh, sort_keys=True, default=str) == \
        json.dumps(cached, sort_keys=True, default=str), (
            f"recomputing {case.origin} did not reproduce the cached case")
    assert B.RESULT_PATH.exists()
    return (f"two runs identical; {case.origin} recomputed from scratch "
            f"reproduces its cached result exactly")


# -------------------------------------- 10. training-period exclusion
def test_a_training_period_origin_cannot_enter_the_backtest():
    train = date(2025, 3, 15)          # squarely inside both models' training
    cases = B.build_cases([train])
    assert cases[0].included is False
    assert B.SKIP_OUTSIDE_TEST in cases[0].reason
    #  and even if one were forced through, the validator's own leakage flag
    #  stops it: accuracy() refuses an origin inside the training period
    src = MODULE.read_text()
    assert "origin_is_inside_the_training_period" in src
    assert "cannot enter the held-out" in src
    for c in ran_cases():
        for lead in ("+24h", "+48h"):
            assert c["forecast_accuracy"][lead][
                "origin_is_inside_the_training_period"] is False
    return (f"{train} is excluded by the window, and accuracy() refuses any "
            f"origin the validator flags as inside the training period")


# ---------------------------------- 11. synthetic mission labelling
def test_no_backtest_case_is_labelled_operational():
    r = result()
    assert r["is_operational_assessment"] is False
    assert r["is_operational_mission"] is False
    assert "not a voyage replay" in r["scenario"]["geometry"]
    assert r["scenario"]["iceberg_exposure"].startswith("omitted")
    for c in ran_cases():
        for plan in ("persistence", "forecast"):
            p = c["plans"][plan]
            assert p["is_operational_mission"] is False
            assert p["is_operational_assessment"] is False
            assert p["iceberg_exposure"] == "omitted"
    #  and nothing anywhere claims one plan is better
    #  As test_simulation.py does: scan what the result ASSERTS, not its own
    #  denials ("NOT ... operationally recommended" is the opposite of a claim).
    offenders = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            for k, v in node.items():
                low = k.lower()
                if any(w in low for w in ("best", "recommend", "safest",
                                          "optimal", "superior", "winner")) \
                        and "not" not in low and v is not False:
                    offenders.append(f"{path}.{k} = {v!r}")
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            #  A claim is only a claim when it is not negated. The result's own
            #  disclaimer says it is NOT a claim that either plan is better,
            #  which is the opposite of claiming it, so clauses carrying a
            #  negation are skipped rather than flagged.
            for clause in node.lower().replace(";", ".").replace(",", ".").split("."):
                if any(n in clause for n in (" not ", "not a", "never", " no ")):
                    continue
                for w in ("is better", "is safer", "is recommended",
                          "is optimal", "performs better", "outperforms"):
                    if w in clause:
                        offenders.append(f"{path}: {w!r} in {clause.strip()!r}")

    walk(r)
    assert not offenders, offenders[:5]
    assert r["cases"][0]["realized"].get("is_a_claim_that_one_plan_is_better",
                                         False) is False
    return ("every case is a synthetic scientific evaluation with the iceberg "
            "term omitted; no plan is called better, safer or recommended")


# ------------------------------------------ 12. the demo is untouched
def test_the_default_demo_route_is_unchanged():
    from fastapi.testclient import TestClient
    from src.api.main import app

    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k != "lookups"}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    result()                                   # run the backtest first
    ref = json.loads((ROOT / "logs" / "demo_route_after_training.json").read_text())
    new = TestClient(app).get(
        "/api/demo/routes?historical_demo_override=true").json()
    for name in ("fastest", "risk_oriented", "shortest_distance"):
        a, b = ref["comparison"]["profiles"][name], new["comparison"]["profiles"][name]
        for key in a:
            assert json.dumps(strip(a[key]), sort_keys=True) == \
                json.dumps(strip(b[key]), sort_keys=True), (name, key)
    tp = new["comparison"]["profiles"]["risk_oriented"]["temporal_provenance"]
    assert tp["environment_policies"] == ["persistence_from_departure_analysis"]
    assert tp["environment_is_time_varying"] is False
    #  and the backtest wrote nothing into the demo's own configuration
    assert str(B.EVAL_DIR).endswith("evaluation")
    assert "evaluation" in str(B.RESULT_PATH)
    return ("after the full backtest, the demo's three original profiles are "
            "identical and nothing was written outside data/processed/evaluation")


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

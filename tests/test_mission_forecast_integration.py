"""
The mission layer reaching the forecast-aware routing, explicitly and only on
request.

What matters here: the default is untouched, the forecast path is reached only
by naming it, every forecast bucket names the artifact and dataset behind it,
the iceberg omission travels with the result, and nothing is ranked.

    python tests/test_mission_forecast_integration.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import mission_decision as M                             # noqa: E402
from src.api.config import DEMO                                       # noqa: E402
from src.api.environment_forecast import (LEAD_SUPPORT,               # noqa: E402
                                          SUPPORTED_LEAD_HOURS,
                                          VALIDITY_TARGET_DAY)
from src.api.environment_timeline import (POLICY_LONG_HORIZON_EVALUATION,  # noqa: E402
                                          POLICY_PERSISTENCE)
from src.routing.route_profiles import PROFILE_RISK                   # noqa: E402

BASE = datetime.fromisoformat(DEMO["departure_time"])
PROF = ("risk_oriented",)
_C: dict = {}


def run(policy, *, sites=None, departures=None, buckets=16, profiles=PROF):
    key = (policy, tuple(s.site_id for s in (sites or [M.LONG_HORIZON_SITE])),
           tuple(d.isoformat() for d in (departures or [BASE])), buckets, profiles)
    if key not in _C:
        _C[key] = M.evaluate_mission(
            sites=sites or [M.LONG_HORIZON_SITE],
            departure_times=departures or [BASE],
            start=M.LONG_HORIZON_START, environment_policy=policy,
            profiles=profiles, horizon_buckets=buckets)
    return _C[key]


def forecast_buckets(option):
    return [b for b in option.environment["buckets"]
            if b.get("source_type") == "model_forecast"]


# ---------------------------------------- 1. the default is untouched
def test_the_default_mission_evaluation_remains_persistence_based():
    m = M.evaluate_mission(sites=[M.DEMONSTRATION_SITES[0]],
                           departure_times=[BASE], profiles=PROF)
    assert m.environment_policy == M.DEFAULT_POLICY == POLICY_PERSISTENCE
    assert m.evaluation_mode == M.MODE_OPERATIONAL_DEMO
    assert m.is_operational_assessment is True
    assert m.iceberg_exposure == "priced"
    assert m.limitations == ()
    o = m.options[0]
    assert o.scenario.horizon_buckets is None
    assert o.environment["models"] == []
    assert o.environment["source_types"] == ["observed_analysis"]
    assert o.provenance()["iceberg_exposure"]["status"] == "priced"
    return ("no policy named -> persistence, operational_demo, iceberg priced, "
            "no forecast model anywhere")


# ---------------------------------- 2. the explicit policy reaches forecast
def test_the_explicit_forecast_policy_reaches_the_forecast_aware_path():
    o = run(POLICY_LONG_HORIZON_EVALUATION).options[0]
    assert o.feasible, (o.refusal_outcome, o.refusal_reason)
    fc = forecast_buckets(o)
    assert fc, "no bucket was priced from a model forecast"
    assert o.environment["models"] == ["HistGradientBoostingRegressor"]
    assert set(o.environment["validity_rules"]) == {VALIDITY_TARGET_DAY}
    assert o.scenario.environment_policy == POLICY_LONG_HORIZON_EVALUATION
    assert o.scenario.evaluation_mode == M.MODE_LONG_HORIZON_EVALUATION
    return (f"{len(fc)} of {len(o.environment['buckets'])} used buckets priced "
            f"from a model forecast under {VALIDITY_TARGET_DAY}")


# ------------------------------- 3. the two policies are separate runs
def test_persistence_and_forecast_are_separate_runs():
    a = run(POLICY_PERSISTENCE).options[0]
    b = run(POLICY_LONG_HORIZON_EVALUATION).options[0]
    assert a.feasible and b.feasible
    assert not forecast_buckets(a), "the baseline touched a forecast"
    assert forecast_buckets(b)
    va = a.objective_values()[PROFILE_RISK]
    vb = b.objective_values()[PROFILE_RISK]
    #  the same geometry, priced differently
    assert math.isclose(va["distance_km"], vb["distance_km"], rel_tol=1e-12)
    assert not math.isclose(va["configured_cost"], vb["configured_cost"],
                            rel_tol=1e-9)
    #  and the difference sits entirely in the environmental term
    assert math.isclose(va["polaris_contribution"], vb["polaris_contribution"],
                        rel_tol=1e-12)
    dcost = vb["configured_cost"] - va["configured_cost"]
    denv = vb["environmental_contribution"] - va["environmental_contribution"]
    assert math.isclose(dcost, denv, rel_tol=1e-9), (dcost, denv)
    pa = tuple(tuple(c) for c in a.profiles[PROFILE_RISK]["path"])
    pb = tuple(tuple(c) for c in b.profiles[PROFILE_RISK]["path"])
    changed = pa != pb
    return (f"ROUTE {'CHANGED' if changed else 'UNCHANGED'}; cost delta "
            f"{dcost:+,.3f}, all of it environmental, POLARIS identical")


# -------------------------- 4, 5. provenance at mission-option level
def test_forecast_provenance_appears_at_mission_option_level():
    o = run(POLICY_LONG_HORIZON_EVALUATION).options[0]
    prov = o.provenance()
    #  not buried: the option's own provenance carries the environment
    for key in ("scenario_id", "start", "destination", "departure_time",
                "environment_policy", "evaluation_mode",
                "is_operational_assessment", "iceberg_exposure",
                "estimated_fuel_provenance", "environment"):
        assert key in prov, key
    env = prov["environment"]
    for key in ("source_types", "models", "forecast_origins", "forecast_leads",
                "validity_rules", "artifacts", "buckets_used", "buckets"):
        assert key in env, key
    assert env["forecast_origins"] == ["2025-01-08T00:00:00"]
    return (f"option provenance names policy, mode, iceberg status, fuel "
            f"provenance and {len(env['buckets'])} per-bucket records")


def test_every_forecast_bucket_points_to_its_own_artifact_and_dataset():
    o = run(POLICY_LONG_HORIZON_EVALUATION).options[0]
    seen = {}
    for b in forecast_buckets(o):
        for key in ("model", "model_artifact", "model_artifact_sha256",
                    "dataset", "dataset_sha256", "dataset_link"):
            assert b.get(key), (b["bucket"], key)
        for key in ("path", "lead_hours", "validity_rule", "validity_from",
                    "validity_to", "forecast_origin"):
            assert M._res(b, key) is not None, (b["bucket"], key)
        assert "artifact_sha256" in b["dataset_link"], b["dataset_link"]
        #  the recorded SHAs are the files' real SHAs
        for path_key, sha_key in (("model_artifact", "model_artifact_sha256"),
                                  ("dataset", "dataset_sha256")):
            f = ROOT / b[path_key]
            assert f.exists(), b[path_key]
            if f not in seen:
                seen[f] = hashlib.sha256(f.read_bytes()).hexdigest()
            assert seen[f] == b[sha_key], (b["bucket"], path_key)
    return (f"{len(forecast_buckets(o))} forecast buckets name their artifact "
            f"and dataset; every recorded SHA equals the file's real SHA")


# ----------------------------------------- 6, 7. the leads are not mixed
def test_each_lead_is_served_by_its_own_artifact():
    o = run(POLICY_LONG_HORIZON_EVALUATION).options[0]
    by_lead = {}
    for b in forecast_buckets(o):
        lead = int(M._res(b, "lead_hours"))
        by_lead.setdefault(lead, set()).add(
            (M._res(b, "path"), b["model_artifact_sha256"]))
    assert set(by_lead) == {24, 48}, sorted(by_lead)
    for lead, entries in by_lead.items():
        assert len(entries) == 1, (lead, entries)
        raster, sha = next(iter(entries))
        assert f"plus{lead}h" in raster, (lead, raster)
    #  the two leads never share a fitted artifact
    shas = {lead: next(iter(e))[1] for lead, e in by_lead.items()}
    assert shas[24] != shas[48], "both leads came from the same artifact"
    a24 = next(iter(by_lead[24]))[0]
    a48 = next(iter(by_lead[48]))[0]
    assert "plus24h" in a24 and "plus48h" not in a24
    assert "plus48h" in a48 and "plus24h" not in a48
    return (f"+24 h -> {a24[:44]}...; +48 h -> {a48[:44]}...; distinct artifacts")


def test_six_and_twelve_hour_leads_remain_unsupported():
    for lead in (6, 12):
        assert lead not in SUPPORTED_LEAD_HOURS, lead
        assert LEAD_SUPPORT[lead].status == \
            "no_training_target_exists_in_the_archive", lead
        assert LEAD_SUPPORT[lead].routable is False, lead
    assert set(SUPPORTED_LEAD_HOURS) == {24, 48}
    #  and no mission bucket was ever served by one
    o = run(POLICY_LONG_HORIZON_EVALUATION).options[0]
    leads = {int(M._res(b, "lead_hours")) for b in forecast_buckets(o)}
    assert leads <= {24, 48}, leads
    assert not (leads & {6, 12})
    return ("6 h and 12 h remain no_training_target_exists_in_the_archive; no "
            f"mission bucket used one (leads used: {sorted(leads)})")


# ---------------------------------------- 9. the iceberg omission travels
def test_the_iceberg_omission_is_explicit_on_every_evaluation_result():
    m = run(POLICY_LONG_HORIZON_EVALUATION)
    assert m.iceberg_exposure == M.ICEBERG_OMITTED == "omitted"
    assert m.is_operational_assessment is False
    assert m.limitations and any("12 h" in x for x in m.limitations)
    assert m.summary()["iceberg_exposure"] == "omitted"
    assert m.summary()["is_operational_assessment"] is False
    for o in m.options:
        assert o.scenario.iceberg_exposure == "omitted"
        assert "12 h horizon" in o.scenario.iceberg_omission_reason
        st = o.provenance()["iceberg_exposure"]
        assert st["status"] == "omitted"
        assert st["reinterpreted_24h_or_48h_rasters"] is False
        assert st["zero_filled"] is False and st["persisted"] is False
        assert st["is_an_operational_assessment"] is False
        for p in o.profiles.values():
            if p.get("success"):
                assert p["max_iceberg_exposure"] is None, \
                    "an omitted layer reported an exposure"
    #  the harness's own reason string is reused, not restated
    from src.api.long_horizon_evaluation import ICEBERG_OMISSION_REASON
    assert m.options[0].scenario.iceberg_omission_reason == ICEBERG_OMISSION_REASON
    return ("omitted at matrix, scenario and option level; exposure reported as "
            "absent rather than zero, and the reason is the harness's own")


# ------------------------------------- 10. synthetic stays synthetic
def test_a_forecast_mission_cannot_be_labelled_operational():
    m = run(POLICY_LONG_HORIZON_EVALUATION)
    assert M.LONG_HORIZON_SITE.is_operational is False
    for o in m.options:
        assert o.scenario.is_operational_mission is False
        assert o.scenario.is_operational_assessment is False
        assert o.provenance()["is_operational_mission"] is False
        assert o.provenance()["is_operational_assessment"] is False
    low = (M.LONG_HORIZON_SITE.label + " " + M.LONG_HORIZON_SITE.source).lower()
    assert "synthetic" in low
    for word in ("station", "port", "base", "waypoint"):
        assert word not in low or "not a" in low, word
    #  even a caller DECLARING the site operational cannot make an evaluation
    #  run an operational assessment
    declared = M.Site("declared", M.LONG_HORIZON_SITE.cell, "declared", True,
                      "a test fixture")
    m2 = M.evaluate_mission(sites=[declared], departure_times=[BASE],
                            start=M.LONG_HORIZON_START,
                            environment_policy=POLICY_LONG_HORIZON_EVALUATION,
                            profiles=PROF, horizon_buckets=4)
    assert m2.options[0].scenario.is_operational_mission is True
    assert m2.options[0].scenario.is_operational_assessment is False, (
        "an evaluation run claimed to be an operational assessment")
    return ("the site is synthetic and named as such; an evaluation run is "
            "never an operational assessment even for a declared real site")


# ------------------------------------- 11, 12. independence holds
def test_multiple_sites_keep_independent_forecast_results():
    other = M.Site("second", (M.LONG_HORIZON_SITE.cell[0],
                              M.LONG_HORIZON_SITE.cell[1] + 6),
                   "SYNTHETIC second evaluation destination", False,
                   "grid arithmetic on the long-horizon goal; not a place")
    m = run(POLICY_LONG_HORIZON_EVALUATION,
            sites=[M.LONG_HORIZON_SITE, other], buckets=16)
    assert len(m.options) == 2
    by = {o.scenario.site.site_id: o for o in m.options}
    assert set(by) == {"long_horizon_demo", "second"}
    for sid, o in by.items():
        for p in o.profiles.values():
            if p.get("success"):
                assert tuple(p["goal"]) == o.scenario.site.cell, sid
                assert tuple(p["path"][-1]) == o.scenario.site.cell, sid
    routed = [o for o in m.options if o.feasible]
    if len(routed) == 2:
        pa = tuple(tuple(c) for c in routed[0].profiles[PROFILE_RISK]["path"])
        pb = tuple(tuple(c) for c in routed[1].profiles[PROFILE_RISK]["path"])
        assert pa != pb, "two different destinations returned one path"
    return (f"{len(m.options)} sites, each routing to its own cell; "
            f"{len(routed)} routed")


def test_multiple_departures_keep_independent_forecast_provenance():
    deps = [BASE, BASE + timedelta(hours=6)]
    m = run(POLICY_LONG_HORIZON_EVALUATION, departures=deps, buckets=16)
    assert len(m.options) == 2
    for o in m.options:
        want = o.scenario.departure_time
        assert o.provenance()["departure_time"] == want.isoformat()
        assert math.isclose(o.scenario.departure_offset_s,
                            (want - BASE).total_seconds())
        for b in forecast_buckets(o):
            #  every forecast bucket's origin is THIS departure's own origin
            assert M._res(b, "forecast_origin").startswith(
                want.date().isoformat()), (o.scenario.scenario_id, b["bucket"])
    origins = {o.scenario.departure_time.isoformat(): sorted(
        {M._res(b, "forecast_origin") for b in forecast_buckets(o)})
        for o in m.options}
    return f"each departure resolves its own forecast origin: {origins}"


# -------------------------------------- 13. nothing is ranked
def test_no_forecast_option_receives_a_winner_or_ranking():
    m = run(POLICY_LONG_HORIZON_EVALUATION,
            sites=[M.LONG_HORIZON_SITE], departures=[BASE], buckets=16)
    s = m.summary()
    for key, want in (("this_is_not_a_ranking", True),
                      ("has_a_best_site", False),
                      ("has_a_best_departure_time", False),
                      ("has_a_recommended_route", False),
                      ("has_an_overall_score", False)):
        assert s[key] is want, key
    offenders = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            for k, v in node.items():
                low = k.lower()
                if any(w in low for w in ("rank", "score", "winner", "best",
                                          "recommend")) and \
                        "not" not in low and "requires" not in low and \
                        v is not False:
                    offenders.append(f"{path}.{k} = {v!r}")
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            for w in ("best site", "best departure", "best route", "safest",
                      "recommended route", "optimal route"):
                if w in node.lower():
                    offenders.append(f"{path}: {w!r}")

    walk(m.to_dict())
    assert not offenders, offenders[:5]
    return "no rank, score, winner or recommendation in a forecast matrix"


# -------------------------------------- 14. the demo is untouched
def test_the_default_demo_route_is_unchanged():
    from fastapi.testclient import TestClient
    from src.api.main import app

    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k != "lookups"}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    run(POLICY_LONG_HORIZON_EVALUATION)      # route a forecast mission first
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
    return ("after a forecast-driven mission, the demo's three original "
            "profiles are identical and still persistence-based")


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

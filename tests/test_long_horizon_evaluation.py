"""
The long-horizon forecast-vs-persistence evaluation mode.

These tests run the REAL scenario against the REAL Mac-generated forecast
rasters. They assert what the evaluation is allowed to claim -- which buckets
came from which source, that the iceberg omission is recorded on every one of
them, and that the comparison is deterministic -- not that either route is
better.

    python tests/test_long_horizon_evaluation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import long_horizon_evaluation as ev                            # noqa: E402
from src.api.environment_forecast import (SOURCE_FORECAST, SOURCE_PERSISTENCE,
                                          VALIDITY_TARGET_DAY)               # noqa: E402
from src.api.environment_timeline import (POLICIES,                          # noqa: E402
                                          POLICY_LONG_HORIZON_EVALUATION,
                                          POLICY_MODEL_FORECAST,
                                          POLICY_PERSISTENCE,
                                          POLICY_VALIDITY_RULE)
from src.routing.iceberg_navigation_cost import IcebergTimeNavigationCost     # noqa: E402

_RUNS: dict = {}


def run(policy: str):
    if policy not in _RUNS:
        _RUNS[policy] = ev.run(policy, max_expansions=400_000)
    return _RUNS[policy]


def test_the_evaluation_policy_is_separate_from_the_production_one():
    assert POLICY_LONG_HORIZON_EVALUATION in POLICIES
    assert POLICY_LONG_HORIZON_EVALUATION != POLICY_MODEL_FORECAST
    assert POLICY_LONG_HORIZON_EVALUATION != POLICY_PERSISTENCE
    #  it is bound to the same validity rule, and says so
    assert POLICY_VALIDITY_RULE[POLICY_LONG_HORIZON_EVALUATION] == \
        VALIDITY_TARGET_DAY
    #  and nothing in routing or the shipped world imports the harness
    for path in ("src/api/world.py", "src/api/main.py", "src/api/simulate.py"):
        assert "long_horizon_evaluation" not in (ROOT / path).read_text(), path
    for path in (ROOT / "src" / "routing").glob("*.py"):
        assert "long_horizon_evaluation" not in path.read_text(), path.name
    return ("the evaluation policy is its own name, bound to "
            f"{VALIDITY_TARGET_DAY}, and unreachable from production code")


def test_the_iceberg_omission_is_recorded_on_every_bucket():
    """The evaluation runs without the iceberg term; that is stated, per
    bucket, and the provider really carries no iceberg layer."""
    for policy in (POLICY_PERSISTENCE, POLICY_LONG_HORIZON_EVALUATION):
        r = run(policy)
        assert r.iceberg_exposure == "omitted"
        assert "12 h horizon" in r.iceberg_omission_reason
        assert "reinterpreting" in r.iceberg_omission_reason
        for entry in r.bucket_provenance():
            assert entry["iceberg_exposure"] == "omitted", entry["bucket"]
    prov = ev.provenance([run(POLICY_PERSISTENCE),
                          run(POLICY_LONG_HORIZON_EVALUATION)])
    assert prov["iceberg_exposure"] == "omitted"
    assert prov["is_an_operational_assessment"] is False
    assert "iceberg" not in " ".join(prov["priced_terms"]).lower()
    #  the 24 h / 48 h exposure rasters are never opened by the harness
    source = (ROOT / "src" / "api" / "long_horizon_evaluation.py").read_text()
    assert "iceberg_exposure_24h" not in source
    assert "iceberg_exposure_48h" not in source
    assert IcebergTimeNavigationCost.__name__ not in source
    return ("every bucket in both runs records iceberg_exposure=omitted; the "
            "harness never opens a 24 h or 48 h exposure raster")


def test_each_bucket_is_priced_from_the_source_its_arrival_day_allows():
    r = run(POLICY_LONG_HORIZON_EVALUATION)
    by_bucket = {e["bucket"]: e for e in r.bucket_provenance()}
    #  the departure day has no forecast valid for it
    for bucket in (0, 1, 2, 3):
        assert by_bucket[bucket]["source_type"] == SOURCE_PERSISTENCE, bucket
        assert by_bucket[bucket]["model"] is None
    #  the +24 h field's target day
    for bucket in (4, 5, 6, 7):
        e = by_bucket[bucket]
        assert e["source_type"] == SOURCE_FORECAST and e["lead_hours"] == 24
        assert "plus24h" in e["artifact"] and "plus48h" not in e["artifact"]
        assert e["validity_rule"] == VALIDITY_TARGET_DAY
    #  the +48 h field's target day
    for bucket in (8, 9, 10, 11):
        e = by_bucket[bucket]
        assert e["source_type"] == SOURCE_FORECAST and e["lead_hours"] == 48
        assert "plus48h" in e["artifact"]
    #  past every supported validity: the named fallback, never a forecast
    for bucket in (12, 13, 14, 15):
        e = by_bucket[bucket]
        assert e["source_type"] == SOURCE_PERSISTENCE, bucket
        assert e["artifact"] is None and e["model"] is None
        assert e["fallback"].startswith("persisted_from_")
    return ("buckets 0-3 persistence, 4-7 +24 h, 8-11 +48 h, 12-15 the named "
            "persistence fallback -- each from its own artifact")


def test_every_forecast_bucket_can_answer_what_forecast_it_used():
    r = run(POLICY_LONG_HORIZON_EVALUATION)
    forecast = [e for e in r.bucket_provenance()
                if e["source_type"] == SOURCE_FORECAST]
    assert forecast, "no bucket was priced from a forecast"
    for e in forecast:
        for key in ("model", "lead_hours", "forecast_origin", "valid_time",
                    "validity_rule", "validity_from", "validity_to",
                    "artifact", "iceberg_exposure"):
            assert e[key] is not None, (e["bucket"], key)
        assert e["forecast_origin"] == ev.SCENARIO["departure_time"]
        assert e["arrival_interval"][0] and e["arrival_interval"][1]
    return (f"{len(forecast)} forecast-priced buckets, each naming its model, "
            f"lead, origin, valid time, validity window and artifact")


def test_the_persistence_run_uses_no_forecast_at_all():
    r = run(POLICY_PERSISTENCE)
    for e in r.bucket_provenance():
        assert e["source_type"] != SOURCE_FORECAST, e["bucket"]
        assert e["artifact"] is None and e["model"] is None
    return "the baseline run touches no forecast artifact"


def test_the_comparison_is_deterministic_and_states_only_what_it_measured():
    a, b = run(POLICY_PERSISTENCE), run(POLICY_LONG_HORIZON_EVALUATION)
    assert a.success and b.success, (a.route.outcome, b.route.outcome)
    first = ev.compare(a, b)
    #  a second, independent pair of runs gives the same answer
    again = ev.compare(ev.run(POLICY_PERSISTENCE, max_expansions=400_000),
                       ev.run(POLICY_LONG_HORIZON_EVALUATION,
                              max_expansions=400_000))
    assert first["path_changed"] == again["path_changed"]
    assert first["cells"] == again["cells"]
    assert first["configured_cost"] == again["configured_cost"]
    #  the comparison reports measurements, not a verdict
    blob = " ".join(str(k) for k in first).lower()
    for banned in ("better", "worse", "improved", "optimal", "recommended",
                   "safer"):
        assert banned not in blob, banned
    #  the mission inputs really were identical; only the policy differed
    assert a.route.start == b.route.start and a.route.goal == b.route.goal
    assert a.route.departure_time == b.route.departure_time
    verdict = "CHANGED" if first["path_changed"] else "UNCHANGED"
    return (f"deterministic across repeated runs; routing decision {verdict} "
            f"({first['cells_only_in_first']} cells only in the persistence "
            f"route)")


def test_a_changed_route_is_attributed_to_forecast_priced_buckets():
    """If the decision changed, every divergence must BEGIN in a bucket the
    two runs priced from different fields.

    A divergence is a contiguous run of indices where the two paths differ.
    It is attributed to the bucket in which it begins, because that is where
    the two runs first disagreed about the cost of the next step. A segment
    may then persist past a bucket boundary into a bucket both runs priced
    identically -- that is the vessel finishing a manoeuvre it has already
    committed to, not a second, independently caused change -- so the
    continuation is reported, not attributed.
    """
    a, b = run(POLICY_PERSISTENCE), run(POLICY_LONG_HORIZON_EVALUATION)
    cmp = ev.compare(a, b)
    forecast_buckets = set(cmp["forecast_priced_buckets"][b.policy])
    pa = [tuple(c) for c in a.route.path]
    pb = [tuple(c) for c in b.route.path]
    if not cmp["path_changed"]:
        return "ROUTE UNCHANGED: nothing to attribute"
    #  cells are index-aligned with path, so the bucket is read by position
    assert len(a.route.cells) == len(pa), "cells are not aligned with the path"
    bucket_at = [c.bucket for c in a.route.cells]
    differing = [i for i in range(min(len(pa), len(pb))) if pa[i] != pb[i]]
    differing += list(range(min(len(pa), len(pb)), len(pa)))

    segments, run_start = [], None
    for i in differing + [None]:
        if run_start is None:
            run_start, prev = i, i
            continue
        if i is not None and i == prev + 1:
            prev = i
            continue
        segments.append((run_start, prev))
        run_start, prev = i, i

    origins = {bucket_at[s] for s, _ in segments}
    spans = {bucket_at[i] for s, e in segments for i in range(s, e + 1)}
    assert origins <= forecast_buckets, (
        f"a divergence began in buckets {sorted(origins - forecast_buckets)}, "
        f"which both runs priced from the same field")
    carried = sorted(spans - forecast_buckets)
    note = (f"ROUTE CHANGED in {len(segments)} segments "
            f"({len(differing)} indices); every segment begins in a "
            f"forecast-priced bucket {sorted(origins)}")
    if carried:
        note += (f"; {carried} is entered only as the continuation of a "
                 f"segment that began earlier")
    return note


def test_every_forecast_bucket_names_the_artifact_and_dataset_behind_it():
    """Phase 7's question -- 'what forecast actually affected this route?' --
    must be answerable down to the fitted artifact and the dataset it was
    fitted on, and the link must be a SHA match, not a filename guess."""
    import hashlib
    r = run(POLICY_LONG_HORIZON_EVALUATION)
    fore = [e for e in r.bucket_provenance()
            if e["source_type"] == SOURCE_FORECAST]
    assert fore, "no forecast-priced bucket to check"
    seen = {}
    for e in fore:
        for key in ("model_artifact", "model_artifact_sha256", "dataset",
                    "dataset_sha256", "dataset_link",
                    "sklearn_version_at_prediction"):
            assert e.get(key), f"bucket {e['bucket']} does not name {key}"
        assert "artifact_sha256" in e["dataset_link"], e["dataset_link"]
        #  the recorded SHAs are the files' real SHAs, not copied strings
        for path_key, sha_key in (("model_artifact", "model_artifact_sha256"),
                                  ("dataset", "dataset_sha256")):
            f = ROOT / e[path_key]
            assert f.exists(), f"{e[path_key]} is missing"
            if f not in seen:
                seen[f] = hashlib.sha256(f.read_bytes()).hexdigest()
            assert seen[f] == e[sha_key], (
                f"{e[path_key]} hashes to {seen[f][:16]}..., but bucket "
                f"{e['bucket']} claims {e[sha_key][:16]}...")
        #  a lead is never served by the other lead's artifact
        assert f"_{int(e['lead_hours'])}h" in e["model_artifact"] or \
            int(e["lead_hours"]) == 24, e["model_artifact"]
    leads = {int(e["lead_hours"]): e["model_artifact_sha256"] for e in fore}
    assert len(set(leads.values())) == len(leads), \
        "two leads were served by the same fitted artifact"
    return (f"{len(fore)} forecast buckets name their artifact and dataset; "
            f"every recorded SHA equals the file's real SHA, and leads "
            f"{sorted(leads)} come from distinct artifacts")


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

"""
Focused tests for the POLARIS navigation-risk adapter, on REAL chart data.

    python tests/test_polaris_risk.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sigrid3_interpret import load_policy, read_dbf_records      # noqa: E402
from src.routing.polaris import load_riv_table                            # noqa: E402
from src.routing.polaris_rio import (CLASSIFICATION_INDETERMINATE,        # noqa: E402
                                     compute_chart)
from src.routing.polaris_risk import (DEFAULT_RISK_CONFIG, RiskConfigError,  # noqa: E402
                                      assess, assess_cells, load_risk_config)

DBF = ROOT / "data" / "raw" / "usnic" / "ANTARC20201224" / "ANTARC20201224.dbf"
_S: dict = {}


def setup():
    if not _S:
        if not DBF.exists():
            raise SystemExit(f"missing test chart: {DBF}")
        pol = load_policy()
        recs = read_dbf_records(DBF)
        t13 = load_riv_table(ROOT / pol.raw["riv_tables"]["default"])
        _S["config"] = load_risk_config()
        _S["cells"] = {}
        for cls in ("PC6", "IA", "Not Ice Strengthened"):
            _S["cells"][cls] = compute_chart(recs, cls, t13, pol, "1.3")
    return _S


def risks(cls="PC6"):
    s = setup()
    return assess_cells(s["cells"][cls], s["config"])


def first(pred, cls="PC6"):
    for r in risks(cls):
        if pred(r):
            return r
    raise AssertionError(f"no cell matches (ice class {cls})")


def configured(name: str) -> float:
    return setup()["config"].penalty_for(name)


# ------------------------------------------------------------------ tests
def test_normal_operation_gets_the_configured_normal_penalty():
    r = first(lambda x: x.conservative and x.conservative.classification == "normal_operation")
    assert r.conservative.penalty == configured("normal_operation")
    assert r.classification_used_conservative == "normal_operation"
    return f"#{r.index} normal_operation -> penalty {r.conservative.penalty:g}"


def test_elevated_risk_gets_the_configured_elevated_penalty():
    r = first(lambda x: x.conservative
              and x.conservative.classification == "elevated_operational_risk")
    assert r.conservative.penalty == configured("elevated_operational_risk")
    return f"#{r.index} (PC6) elevated_operational_risk -> {r.conservative.penalty:g}"


def test_special_consideration_gets_the_configured_special_penalty():
    r = first(lambda x: x.conservative and x.conservative.classification ==
              "operation_subject_to_special_consideration")
    assert r.conservative.penalty == configured(
        "operation_subject_to_special_consideration")
    return f"#{r.index} special consideration -> {r.conservative.penalty:g}"


def test_indeterminate_gets_the_configured_indeterminate_penalty():
    r = first(lambda x: x.conservative and x.conservative.indeterminate)
    assert r.conservative.classification == CLASSIFICATION_INDETERMINATE
    assert r.conservative.penalty == configured("indeterminate")
    assert r.indeterminate is True
    return f"#{r.index} indeterminate -> {r.conservative.penalty:g}"


def test_unknown_gets_the_configured_unknown_penalty():
    r = first(lambda x: not x.has_rio)
    assert r.conservative is None and r.optimistic is None
    assert r.unknown_penalty == configured("unknown")
    assert r.conservative_penalty == configured("unknown")
    assert r.optimistic_penalty == configured("unknown")
    assert r.classification_used_conservative == "unknown"
    assert any("NOT an exclusion" in w for w in r.warnings)
    return f"#{r.index} no RIO -> penalty {r.unknown_penalty:g}, still not an exclusion"


def test_conservative_and_optimistic_penalties_stay_distinct():
    r = first(lambda x: x.conservative and x.optimistic
              and x.conservative.penalty != x.optimistic.penalty)
    assert r.conservative.penalty != r.optimistic.penalty
    assert r.conservative_penalty != r.optimistic_penalty
    assert any("both are reported" in w for w in r.warnings)
    n = sum(1 for x in risks() if x.conservative and x.optimistic
            and x.conservative.penalty != x.optimistic.penalty)
    return (f"#{r.index} conservative {r.conservative.penalty:g} vs optimistic "
            f"{r.optimistic.penalty:g}; {n} such cells in the chart")


def test_band_straddling_stays_indeterminate_with_both_endpoints():
    r = first(lambda x: x.conservative and x.conservative.indeterminate)
    b = r.conservative
    assert b.classification == CLASSIFICATION_INDETERMINATE
    assert b.classification_worst != b.classification_best, "endpoints must differ"
    assert b.penalty_worst is not None and b.penalty_best is not None
    assert b.penalty_worst != b.penalty_best
    assert b.rio_worst < b.rio_best
    return (f"#{r.index} RIO {b.rio_worst:g}..{b.rio_best:g}  "
            f"{b.classification_worst}(p={b.penalty_worst:g}) .. "
            f"{b.classification_best}(p={b.penalty_best:g})")


def test_mapping_sensitive_preserves_both_branches():
    r = first(lambda x: x.mapping_classification_difference and x.has_rio)
    assert r.mapping_sensitive is True
    assert r.conservative is not None and r.optimistic is not None, \
        "the optimistic branch must never be dropped"
    assert r.classification_used_conservative != r.classification_used_optimistic
    return (f"#{r.index} conservative={r.classification_used_conservative} "
            f"optimistic={r.classification_used_optimistic}")


def test_changing_the_configuration_changes_the_penalties():
    s = setup()
    base = json.loads(DEFAULT_RISK_CONFIG.read_text())
    tweaked = json.loads(json.dumps(base))
    tweaked["penalties"] = {k: (v + 7.0) for k, v in base["penalties"].items()}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "tweaked.json"
        p.write_text(json.dumps(tweaked))
        alt = load_risk_config(p)
        i = first(lambda x: x.has_rio).index
        a = assess(s["cells"]["PC6"][i], s["config"])
        b = assess(s["cells"]["PC6"][i], alt)
        assert b.conservative.penalty == a.conservative.penalty + 7.0
        assert b.config_path != a.config_path
    return (f"#{i} default penalty {a.conservative.penalty:g} -> "
            f"{b.conservative.penalty:g} with an alternative config")


def test_no_output_can_create_a_hard_block():
    import math
    forbidden = {"blocked", "no_go", "nogo", "navigable", "passable", "forbidden",
                 "excluded", "mask"}
    for r in risks()[:200]:
        names = {n.lower() for n in dir(r) if not n.startswith("_")}
        assert not (names & forbidden), f"blocking vocabulary on the result: {names & forbidden}"
        assert math.isfinite(r.conservative_penalty)
        assert math.isfinite(r.optimistic_penalty)
    # and the loader refuses a penalty that would be a block by another name
    with tempfile.TemporaryDirectory() as tmp:
        bad = json.loads(DEFAULT_RISK_CONFIG.read_text())
        bad["penalties"]["unknown"] = float("inf")
        p = Path(tmp) / "inf.json"
        p.write_text(json.dumps(bad).replace('"unknown": Infinity', '"unknown": 1e999'))
        try:
            load_risk_config(p)
            raise AssertionError("an infinite penalty must be rejected")
        except RiskConfigError as exc:
            assert "hard navigation block" in str(exc)
    maxp = max(r.conservative_penalty for r in risks())
    return (f"no blocking attribute on any result; all penalties finite "
            f"(max seen {maxp:g}); infinite penalty rejected by the loader")


def test_penalties_are_deterministic():
    s = setup()
    for i in (0, 5, 101, 186, 334, 360):
        a = assess(s["cells"]["PC6"][i], s["config"])
        b = assess(s["cells"]["PC6"][i], s["config"])
        assert a == b, f"cell {i} assessed differently on a second call"
    return "cells 0, 5, 101, 186, 334, 360 identical on repeat assessment"


def test_every_classification_has_a_configured_penalty():
    seen = set()
    for cls in ("PC6", "IA", "Not Ice Strengthened"):
        for r in risks(cls):
            seen.add(r.classification_used_conservative)
            seen.add(r.classification_used_optimistic)
    cfg = set(setup()["config"].penalties)
    assert seen <= cfg, f"classification with no configured penalty: {seen - cfg}"
    return f"classifications seen: {sorted(seen)}; all configured"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    setup()
    print("=" * 78); print(f"POLARIS risk adapter -- {DBF.name}"); print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<54} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__); print(f"  FAIL  {fn.__name__:<54} {exc}")
        except Exception as exc:                       # noqa: BLE001
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

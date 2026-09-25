"""
Focused tests for the SIGRID-3 interpretation layer, against REAL chart data.

Every case is located by scanning ANTARC20201224 and the chosen record index is
printed, so a reviewer can open the same polygon in the DBF and check the claim
by hand. One case (an unlisted stage code) does not occur in this chart; it is
built by altering one field of a real record, and is labelled as such.

    python tests/test_sigrid3_interpret.py
    pytest tests/test_sigrid3_interpret.py      # also works if pytest is present
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sigrid3_interpret import (          # noqa: E402
    load_policy, interpret_polygon, read_dbf_records, Policy)

DBF = ROOT / "data" / "raw" / "usnic" / "ANTARC20201224" / "ANTARC20201224.dbf"

_POLICY: Policy | None = None
_RECORDS: list[dict] | None = None


def setup():
    global _POLICY, _RECORDS
    if _POLICY is None:
        _POLICY = load_policy()
    if _RECORDS is None:
        if not DBF.exists():
            raise SystemExit(f"missing test chart: {DBF}")
        _RECORDS = read_dbf_records(DBF)
    return _POLICY, _RECORDS


def find(pred) -> int:
    _, recs = setup()
    for i, r in enumerate(recs):
        if pred({k: ("" if v is None else str(v).strip()) for k, v in r.items()}):
            return i
    raise AssertionError("no record matches the predicate")


def interp(i: int):
    pol, recs = setup()
    return interpret_polygon(recs[i], i, pol)


# ------------------------------------------------------------------ tests
def test_water_polygon_is_not_sea_ice():
    i = find(lambda r: r["POLY_TYPE"] == "W")
    o = interp(i)
    assert o.is_water is True
    assert o.state == "resolved"
    assert o.components == (), "a water polygon must produce no ice components"
    assert o.antarctic_convention_applied is False
    assert o.total_concentration.low_tenths == 0 and o.total_concentration.exact
    return f"#{i}  CT={o.raw['CT']}  state={o.state}  components={len(o.components)}"


def test_unexpected_poly_type_fails_closed():
    pol, recs = setup()
    i = find(lambda r: r["POLY_TYPE"] == "I")
    rec = dict(recs[i]); rec["POLY_TYPE"] = "L"       # MODIFIED real record
    o = interpret_polygon(rec, i, pol)
    assert o.state == "unknown"
    assert o.components == ()
    assert any("neither" in w for w in o.warnings)
    return f"#{i} with POLY_TYPE forced to 'L'  ->  state={o.state}"


def test_normal_ice_polygon_reads_stage_from_SA():
    pol, _ = setup()
    i = find(lambda r: r["POLY_TYPE"] == "I" and r["SA"] not in ("98",) and r["SA"] != "-9")
    o = interp(i)
    assert o.antarctic_convention_applied is False
    assert "SA" in o.stage_source_fields
    sea = o.sea_ice_components
    assert sea, "expected at least one sea-ice component"
    first = sea[0]
    assert first.slot_stage_field == "SA"
    # the mapping must come from the policy, not from the module
    expect = pol.stage_mapping[first.stage_code]
    assert first.polaris_conservative == expect["conservative"]
    assert first.polaris_optimistic == expect["optimistic"]
    return (f"#{i}  SA={first.stage_code} -> {first.polaris_conservative!r} "
            f"(optimistic {first.polaris_optimistic!r})")


def test_sa98_applies_convention_and_reads_SB():
    i = find(lambda r: r["SA"] == "98" and r["SB"] not in ("-9", ""))
    o = interp(i)
    assert o.antarctic_convention_applied is True
    assert o.stage_source_fields and o.stage_source_fields[0] == "SB"
    assert all(c.slot_stage_field != "SA" for c in o.sea_ice_components), \
        "SA must never contribute a sea-ice component when it holds the iceberg code"
    ice = [c for c in o.components if c.role == "iceberg"]
    assert len(ice) == 1 and ice[0].slot_stage_field == "SA"
    assert ice[0].polaris_conservative is None, "icebergs must have no POLARIS type"
    assert o.iceberg_concentration is not None
    return (f"#{i}  SA=98 CA={o.raw['CA']} -> iceberg "
            f"{o.iceberg_concentration.low_tenths}/10; sea ice from SB={o.raw['SB']}")


def test_sa98_with_sb_missing_is_unknown():
    i = find(lambda r: r["SA"] == "98" and r["SB"] == "-9")
    o = interp(i)
    assert o.state == "unknown", "no sea-ice stage is defensible here"
    assert o.sea_ice_components == ()
    assert o.iceberg_concentration is not None
    return f"#{i}  SA=98 SB=-9 CT={o.raw['CT']}  ->  state={o.state}"


def test_partials_present_are_preserved_exactly():
    i = find(lambda r: r["POLY_TYPE"] == "I" and r["CA"] != "-9" and r["CB"] != "-9")
    o = interp(i)
    assert o.partials_present is True
    named = [c for c in o.components if c.concentration.usable]
    assert named, "expected decoded partial concentrations"
    for c in named:
        lo, hi = c.concentration.low_tenths, c.concentration.high_tenths
        assert lo == hi, "a partial concentration code must decode to an exact value"
        assert 0 <= lo <= 10
    shown = ", ".join(f"{c.slot_concentration_field}={c.raw_concentration}"
                      f"->{c.concentration.low_tenths}/10" for c in named)
    return f"#{i}  {shown}"


def test_partials_absent_sets_state_and_warns():
    i = find(lambda r: r["POLY_TYPE"] == "I" and r["CA"] == "-9"
             and r["CB"] == "-9" and r["CC"] == "-9" and r["SA"] not in ("-9", "98"))
    o = interp(i)
    assert o.partials_present is False
    assert o.state == "partials_absent"
    assert any("INTERPRETATION" in w for w in o.warnings), \
        "the partials-absent reading must be flagged as an interpretation"
    assert len(o.sea_ice_components) == 1, "must not invent extra ice stages"
    return f"#{i}  CA/CB/CC all -9, SA={o.raw['SA']}  ->  state={o.state}"


def test_ct_interval_is_not_collapsed():
    i = find(lambda r: r["CT"] in ("13", "24", "46", "68", "81"))
    o = interp(i)
    t = o.total_concentration
    assert t.exact is False
    assert t.low_tenths is not None and t.high_tenths is not None
    assert t.low_tenths < t.high_tenths
    pol, _ = setup()
    assert [t.low_tenths, t.high_tenths] == pol.ct_codes[t.raw], \
        "the interval must come from the policy table"
    return f"#{i}  CT={t.raw} -> {t.low_tenths}..{t.high_tenths}/10 (exact={t.exact})"


def test_unlisted_stage_code_becomes_unknown():
    pol, recs = setup()
    i = find(lambda r: r["POLY_TYPE"] == "I" and r["SA"] not in ("98", "-9"))
    rec = dict(recs[i]); rec["SA"] = "91"     # MODIFIED: 91 is absent from this chart
    assert "91" not in pol.stage_mapping and "91" not in pol.non_sea_ice
    o = interpret_polygon(rec, i, pol)
    assert o.state == "unknown"
    assert any("not listed in the policy" in w for w in o.warnings)
    assert all(c.polaris_conservative is None for c in o.components
               if c.stage_code == "91"), "an unlisted code must never be nearest-matched"
    return f"#{i} with SA forced to '91' (unlisted)  ->  state={o.state}"


def test_so98_sets_the_trace_iceberg_flag():
    i = find(lambda r: r["SO"] == "98")
    o = interp(i)
    assert o.icebergs_trace_present is True
    j = find(lambda r: r["SO"] == "-9")
    assert interp(j).icebergs_trace_present is False
    return f"#{i} SO=98 -> trace flag True;  #{j} SO=-9 -> False"


def test_minus9_is_unused_not_zero():
    i = find(lambda r: r["POLY_TYPE"] == "I" and r["CC"] == "-9")
    o = interp(i)
    cc = [c for c in o.components if c.slot_concentration_field == "CC"]
    for c in cc:
        assert c.concentration.low_tenths is None, "-9 must not decode to a number"
        assert c.concentration.kind == "unused"
    # and a -9 stage slot must not appear as a component at all
    k = find(lambda r: r["POLY_TYPE"] == "I" and r["SC"] == "-9")
    o2 = interp(k)
    assert all(c.slot_stage_field != "SC" for c in o2.components)
    return f"#{i} CC=-9 -> kind='unused' (not 0);  #{k} SC=-9 -> slot omitted"


def test_interpretation_is_deterministic():
    pol, recs = setup()
    for i in (0, 10, 186, 311, 334):
        a, b = interpret_polygon(recs[i], i, pol), interpret_polygon(recs[i], i, pol)
        assert a == b, f"record {i} interpreted differently on a second call"
    return "records 0, 10, 186, 311, 334 identical on repeat interpretation"


def test_whole_chart_states_are_accounted_for():
    pol, recs = setup()
    from collections import Counter
    states = Counter(interpret_polygon(r, i, pol).state for i, r in enumerate(recs))
    assert sum(states.values()) == len(recs)
    assert set(states) <= set(pol.states), f"state outside the policy vocabulary: {set(states)}"
    assert states["unknown"] >= 44, "the 44 SA=98/SB=-9 polygons must be unknown"
    return "  ".join(f"{k}={v}" for k, v in sorted(states.items()))


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    setup()
    print("=" * 78)
    print(f"SIGRID-3 interpretation layer -- {DBF.name}")
    print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            detail = fn()
            print(f"  PASS  {fn.__name__:<46} {detail or ''}")
        except AssertionError as exc:
            failures.append((fn.__name__, str(exc)))
            print(f"  FAIL  {fn.__name__:<46} {exc}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((fn.__name__, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {fn.__name__:<46} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

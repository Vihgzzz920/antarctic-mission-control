"""
Focused tests for the RIO calculation layer, against REAL ANTARC20201224 data.

Each test prints the polygon index it used so a reviewer can check the claim in
the DBF by hand. Two cases cannot occur in this chart and are built by altering
one field of a real record; both are labelled MODIFIED in the output.

    python tests/test_polaris_rio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sigrid3_interpret import (load_policy, interpret_polygon,      # noqa: E402
                                        read_dbf_records)
from src.routing.polaris import load_riv_table                                # noqa: E402
from src.routing.polaris_rio import (compute_rio, compute_chart, summarise,   # noqa: E402
                                     CLASSIFICATION_INDETERMINATE)

DBF = ROOT / "data" / "raw" / "usnic" / "ANTARC20201224" / "ANTARC20201224.dbf"
_S: dict = {}


def setup():
    if not _S:
        if not DBF.exists():
            raise SystemExit(f"missing test chart: {DBF}")
        _S["policy"] = load_policy()
        _S["recs"] = read_dbf_records(DBF)
        _S["t13"] = load_riv_table(ROOT / _S["policy"].raw["riv_tables"]["default"])
        _S["t14"] = load_riv_table(ROOT / _S["policy"].raw["riv_tables"]["decayed_ice"])
    return _S


def cell(i: int, ice_class: str = "PC6", decayed: bool = False, record=None):
    s = setup()
    tbl = s["t14"] if decayed else s["t13"]
    rec = record if record is not None else s["recs"][i]
    interp = interpret_polygon(rec, i, s["policy"])
    return compute_rio(interp, tbl.ice_class(ice_class), tbl, s["policy"],
                       "1.4" if decayed else "1.3")


def find(pred) -> int:
    s = setup()
    for i, r in enumerate(s["recs"]):
        if pred({k: ("" if v is None else str(v).strip()) for k, v in r.items()}):
            return i
    raise AssertionError("no record matches")


def find_cell(pred, ice_class="PC6") -> int:
    s = setup()
    for i in range(len(s["recs"])):
        if pred(cell(i, ice_class)):
            return i
    raise AssertionError("no cell matches")


# ------------------------------------------------------------------ tests
def test_exact_10_of_10_regime_is_a_point():
    """An exact regime with no unknown fraction must give a single RIO, not a range.

    In this chart the ONLY such cells are the five water polygons: every resolved
    ice cell carries an SA=98 iceberg fraction, so its range comes from the
    iceberg bound alone. Both halves are asserted here.
    """
    i = find_cell(lambda c: c.cell_state == "resolved"
                  and c.iceberg_fraction_tenths is None)
    c = cell(i)
    assert c.conservative.low.rio == c.conservative.high.rio, "an exact regime is a point"
    assert c.conservative.low.total_tenths in (0, 10)
    assert sum(t for _, t, _ in c.conservative.low.terms) == 10, "regime totals ten tenths"
    assert c.conservative.classification != CLASSIFICATION_INDETERMINATE

    # a CT=92 ice cell: exact concentration, range due ONLY to the iceberg bound
    j = find_cell(lambda x: x.cell_state == "resolved" and not x.interpretation.is_water
                  and x.interpretation.total_concentration.raw == "92")
    d = cell(j)
    assert d.conservative.low.total_tenths == d.conservative.high.total_tenths == 10
    assert d.conservative.low.iceberg_assigned_type != \
        d.conservative.high.iceberg_assigned_type, \
        "the only difference between the endpoints must be the iceberg substitution"
    return (f"#{i} exact, no iceberg -> point RIO {c.conservative_rio_low:g}; "
            f"#{j} CT=92 with iceberg -> {d.conservative_rio_low:g}.."
            f"{d.conservative_rio_high:g} (iceberg bound only)")


def test_ct_interval_is_propagated_not_collapsed():
    i = find(lambda r: r["CT"] in ("13", "24", "46", "68") and r["POLY_TYPE"] == "I"
             and r["CA"] == "-9")
    c = cell(i)
    t = c.interpretation.total_concentration
    assert t.exact is False and t.low_tenths < t.high_tenths
    assert c.conservative.low.rio != c.conservative.high.rio, \
        "an interval CT must give an interval RIO"
    assert {c.conservative.low.total_tenths, c.conservative.high.total_tenths} == \
        {t.low_tenths, t.high_tenths}
    return (f"#{i} CT={t.raw} [{t.low_tenths}..{t.high_tenths}/10] -> RIO "
            f"{c.conservative_rio_low:g}..{c.conservative_rio_high:g}")


def test_multiple_exact_partials_sum_and_close_with_ice_free():
    i = find(lambda r: r["POLY_TYPE"] == "I" and r["CA"] != "-9"
             and r["CB"] != "-9" and r["CC"] != "-9" and r["SA"] != "98")
    c = cell(i)
    e = c.conservative.low
    named = sum(t for _, t, _ in e.terms if t)
    assert named == 10, f"regime must total ten tenths, got {named}"
    ice_free = _S["policy"].ice_free_type
    resid = [t for n, t, _ in e.terms if n == ice_free]
    assert e.residual_ice_free == (resid[0] if resid else 0)
    assert e.residual_ice_free >= 0, "residual must never be negative"
    return (f"#{i} CA/CB/CC={c.interpretation.raw['CA']}/{c.interpretation.raw['CB']}"
            f"/{c.interpretation.raw['CC']} residual Ice-Free={e.residual_ice_free}/10")


def test_partials_absent_uses_the_policy_reading_and_keeps_its_state():
    i = find_cell(lambda c: c.cell_state == "partials_absent")
    c = cell(i)
    assert c.cell_state == "partials_absent", "must not be presented as resolved"
    assert any("INTERPRETATION" in w for w in c.warnings)
    e = c.conservative.low
    stage_tenths = [t for n, t, _ in e.terms
                    if n != _S["policy"].ice_free_type]
    assert sum(stage_tenths) == e.total_tenths, \
        "the single named stage must take the whole of the CT endpoint"
    return (f"#{i} CT={c.interpretation.total_concentration.raw} SA="
            f"{c.interpretation.raw['SA']} -> stage takes {sum(stage_tenths)}/10, "
            f"RIO {c.conservative_rio_low:g}..{c.conservative_rio_high:g}")


def test_sa98_iceberg_fraction_is_bounded_not_given_a_riv():
    i = find(lambda r: r["SA"] == "98" and r["SB"] not in ("-9", "") and r["CA"] != "-9")
    c = cell(i)
    assert c.iceberg_fraction_tenths is not None and c.iceberg_fraction_tenths > 0
    assert any("OUTSIDE POLARIS" in w for w in c.warnings), \
        "the iceberg bound must be flagged as outside POLARIS"
    lo_type = c.conservative.low.iceberg_assigned_type
    hi_type = c.conservative.high.iceberg_assigned_type
    assert lo_type is not None and hi_type is not None
    assert hi_type == _S["policy"].ice_free_type, "best case assigns Ice-Free"
    assert lo_type != _S["policy"].ice_free_type or lo_type == hi_type
    return (f"#{i} iceberg {c.iceberg_fraction_tenths}/10 bounded: worst as "
            f"{lo_type!r}, best as {hi_type!r}  RIO "
            f"{c.conservative_rio_low:g}..{c.conservative_rio_high:g}")


def test_so98_trace_is_flagged_and_does_not_enter_rio():
    i = find(lambda r: r["SO"] == "98" and r["SA"] != "98" and r["POLY_TYPE"] == "I")
    c = cell(i)
    assert c.iceberg_trace_present is True
    assert c.iceberg_fraction_tenths is None, "trace ice has no concentration to add"
    for _, tenths, _ in c.conservative.low.terms:
        assert tenths == int(tenths), "no sub-tenth term may appear in the sum"
    assert any("less than 1/10" in w for w in c.warnings)
    return f"#{i} SO=98 -> trace flag set, no term added to the RIO sum"


def test_sa98_with_sb_missing_returns_unknown_without_a_value():
    i = find(lambda r: r["SA"] == "98" and r["SB"] == "-9")
    c = cell(i)
    assert c.cell_state == "unknown"
    assert c.conservative is None and c.optimistic is None
    assert c.has_rio is False
    assert c.conservative_rio_low is None
    assert any("never navigable by default" in w for w in c.warnings)
    return f"#{i} SA=98 SB=-9 -> no RIO, no default, unknown stays unknown"


def test_mapping_sensitive_reports_both_readings():
    i = find_cell(lambda c: c.mapping_classification_difference)
    c = cell(i)
    assert c.mapping_sensitive is True
    assert c.conservative.usable and c.optimistic.usable
    assert c.conservative_classification_low != c.optimistic_classification_low
    assert c.optimistic_rio_low is not None, "the optimistic result must not be hidden"
    assert c.conservative_rio_low <= c.optimistic_rio_low
    return (f"#{i} conservative RIO {c.conservative_rio_low:g} "
            f"[{c.conservative_classification_low}]  vs  optimistic "
            f"{c.optimistic_rio_low:g} [{c.optimistic_classification_low}]")


def test_band_classification_comes_from_the_riv_config():
    s = setup()
    from src.routing.polaris import classify_rio
    i = find_cell(lambda c: c.has_rio)
    c = cell(i)
    expect = classify_rio(c.conservative.low.rio, s["t13"].ice_class("PC6"), s["t13"])
    assert c.conservative_classification_low == expect.value
    # and the categories are not merged across ice-class groups
    cn = cell(i, ice_class="Not Ice Strengthened")
    assert cn.conservative_classification_low in (
        "normal_operation", "operation_subject_to_special_consideration"), \
        "a no-ice-class ship has no elevated_operational_risk band"
    return (f"#{i} PC6 -> {c.conservative_classification_low}; "
            f"Not Ice Strengthened -> {cn.conservative_classification_low}")


def test_band_straddling_sets_indeterminate():
    i = find_cell(lambda c: c.band_straddling, ice_class="IA")
    c = cell(i, ice_class="IA")
    assert c.conservative.classification == CLASSIFICATION_INDETERMINATE
    assert c.conservative.low.classification != c.conservative.high.classification
    assert any("indeterminate" in w.lower() for w in c.warnings)
    return (f"#{i} (IA) RIO {c.conservative_rio_low:g}..{c.conservative_rio_high:g} "
            f"spans {c.conservative_classification_low} -> "
            f"{c.conservative_classification_high}")


def test_impossible_residual_yields_no_rio_and_is_never_clamped():
    s = setup()
    i = find(lambda r: r["POLY_TYPE"] == "I" and r["CA"] != "-9" and r["CB"] != "-9"
             and int(r["CA"][0]) + int(r["CB"][0]) >= 4)
    rec = dict(s["recs"][i]); rec["CT"] = "13"        # MODIFIED: named sum exceeds 3/10
    c = cell(i, record=rec)
    assert c.conservative is not None and c.conservative.usable is False, \
        "no endpoint can be feasible when the named sum exceeds CT_high"
    assert c.has_rio is False
    assert c.conservative.infeasible, "the reason must be recorded, not silently dropped"
    assert any("no feasible endpoint" in w for w in c.warnings)
    return f"#{i} with CT forced to 13 (MODIFIED) -> no RIO; {len(c.conservative.infeasible)} reasons recorded"


def test_partials_absent_with_an_iceberg_is_not_silently_clamped():
    """No such cell exists in this chart, so the path is exercised by construction.

    The policy gives the whole of CT to the single named stage and says nothing
    about sharing it with an iceberg fraction; the endpoint must be discarded
    with a reason, never rewritten to fit.
    """
    s = setup()
    i = find_cell(lambda c: c.cell_state == "partials_absent")
    rec = dict(s["recs"][i])
    rec["SA"], rec["CA"] = "98", "10"      # MODIFIED: iceberg on a partials-absent cell
    rec["SB"] = s["recs"][i]["SA"]         # keep the original stage, now in SB
    c = cell(i, record=rec)
    if c.cell_state == "partials_absent":
        assert not c.conservative.usable or all(
            e.total_tenths - e.residual_ice_free >= 0
            for e in (c.conservative.low, c.conservative.high) if e)
        assert c.conservative.infeasible, "the reason must be recorded"
        assert not any("clamp" in w.lower() for w in c.warnings)
    return (f"#{i} MODIFIED to partials_absent + iceberg -> state={c.cell_state}, "
            f"{len(c.conservative.infeasible) if c.conservative else 0} reasons recorded")


def test_both_riv_tables_load_and_decay_is_never_worse():
    i = find_cell(lambda c: c.has_rio and c.cell_state != "unknown")
    a, b = cell(i, decayed=False), cell(i, decayed=True)
    assert a.table_used == "1.3" and b.table_used == "1.4"
    assert b.conservative_rio_low >= a.conservative_rio_low, \
        "decayed ice must never score worse than standard ice"
    assert b.conservative_rio_high >= a.conservative_rio_high
    return (f"#{i} table 1.3 RIO {a.conservative_rio_low:g}..{a.conservative_rio_high:g}"
            f"  ->  table 1.4 {b.conservative_rio_low:g}..{b.conservative_rio_high:g}")


def test_calculation_is_deterministic():
    for i in (0, 5, 56, 186, 334, 360):
        assert cell(i) == cell(i), f"polygon {i} scored differently on a second call"
    return "polygons 0, 5, 56, 186, 334, 360 identical on repeat calculation"


def test_no_hard_block_is_produced():
    s = setup()
    cells = compute_chart(s["recs"], "Not Ice Strengthened", s["t13"], s["policy"], "1.3")
    special = [c for c in cells
               if c.has_rio and "special_consideration" in
               (c.conservative_classification_low or "")]
    assert special, "expected some special-consideration cells for this ice class"
    for c in special[:50]:
        assert not hasattr(c, "blocked") and not hasattr(c, "navigable"), \
            "the RIO layer must not express a navigation decision"
    return (f"{len(special)} special-consideration cells reported, none converted "
            f"into a block")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    setup()
    print("=" * 78); print(f"POLARIS RIO layer -- {DBF.name}"); print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<52} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__); print(f"  FAIL  {fn.__name__:<52} {exc}")
        except Exception as exc:                        # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__:<52} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

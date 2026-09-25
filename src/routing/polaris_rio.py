"""
RIO calculation over interpreted SIGRID-3 polygons. Arithmetic only, no routing.

Input : a PolygonInterpretation from src/data/sigrid3_interpret
        a RIVTable loaded by src/routing/polaris.load_riv_table
Output: a RIOCell -- both endpoint RIOs, both mappings, their Table 1.1
        classifications, and every reason the answer is uncertain.

NOTHING IS DUPLICATED HERE
  No RIV value, no SIGRID code, no POLARIS ice-type name and no band limit is
  written into this file. RIVs and bands come from the RIV JSON through
  polaris.RIVTable; stage mappings and concentration decoding arrive already
  resolved on the interpretation. The sum itself is polaris.calculate_rio and
  the classification is polaris.classify_rio; this module decides only WHICH
  regimes to hand them.

WHY EVERY CELL PRODUCES A RANGE
  Three separate things are unknown per cell, and all three are propagated
  rather than averaged away:
    1. CT is usually an interval (only 92 is exact).
    2. Nearly every SIGRID stage in an Antarctic chart is an aggregate category,
       so each cell has a conservative and an optimistic POLARIS reading.
    3. An iceberg fraction has no POLARIS RIV at all.
  The result is a RIO interval per mapping, classified at each endpoint. Where
  the endpoints fall in different Table 1.1 bands the classification is
  INDETERMINATE -- which is information for the master, not a failure.

THE ICEBERG BOUND IS NOT POLARIS
  POLARIS assigns no RIV to glacial ice. To bound its effect this module
  computes each cell twice: once assigning the iceberg fraction the most severe
  RIV present in that cell's own sea-ice regime, once assigning it the Ice-Free
  RIV. Both bounds use published RIVs, so no value is invented -- but the
  substitution itself is this project's policy and is recorded as a warning on
  every affected cell.

ENDPOINT FEASIBILITY, NOT CLAMPING
  residual = CT_endpoint - (named components + iceberg). Where a chart states
  partial concentrations AND an interval CT, the partials are the more precise
  statement: in ANTARC20201224, 146 of 146 interval cells have a named sum above
  the CT low endpoint. Such an endpoint is INFEASIBLE and is discarded with a
  recorded reason -- it is never clamped to zero and the cell is never silently
  narrowed. A cell whose named sum exceeds CT_high has no feasible endpoint at
  all and is returned as INVALID with no RIO. (In this chart: zero such cells.)

WHAT THIS MODULE DOES NOT DO
  It does not decide navigability, does not block anything, does not produce a
  cost, does not touch geometry and does not know what a route is. A
  special-consideration outcome is reported, never enforced: MSC.1/Circ.1519
  section 1.4 calls POLARIS a decision support tool, not a Go/No Go tool.

Usage:
    python -m src.routing.polaris_rio --dbf data/raw/usnic/.../ANTARC20201224.dbf \
        --ice-class PC6 --limit 5
    python -m src.routing.polaris_rio --dbf ... --ice-class IA --summary
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

from src.data.sigrid3_interpret import (Policy, PolygonInterpretation, load_policy,
                                        interpret_polygon, read_dbf_records,
                                        DEFAULT_POLICY)
from src.routing.polaris import (IceClass, IceRegime, IceTypeConcentration, RIVTable,
                                 OperationalCategory, PolarisError, calculate_rio,
                                 classify_rio, load_riv_table)

ROOT = Path(__file__).resolve().parents[2]
TENTHS_TOTAL = 10

STATE_UNKNOWN = "unknown"
STATE_PARTIALS_ABSENT = "partials_absent"
CLASSIFICATION_INDETERMINATE = "indeterminate"
BRANCH_CONSERVATIVE = "conservative"
BRANCH_OPTIMISTIC = "optimistic"


class RIOEngineError(ValueError):
    """A RIO cannot be computed as specified."""


# ------------------------------------------------------------------ result
@dataclass(frozen=True)
class Endpoint:
    """One fully-specified regime and the RIO it produces."""

    rio: float
    classification: str
    total_tenths: int
    residual_ice_free: int
    iceberg_assigned_type: str | None
    terms: tuple[tuple[str, float, float], ...]      # (ice type, tenths, RIV)

    def describe(self) -> str:
        parts = ", ".join(f"{t}x{c:g}@{r:g}" for t, c, r in self.terms)
        return f"RIO {self.rio:g} [{self.classification}]  {parts}"


@dataclass(frozen=True)
class Branch:
    """One mapping reading (conservative or optimistic) as a RIO interval."""

    label: str
    low: Endpoint | None            # worst RIO
    high: Endpoint | None           # best RIO
    classification: str
    infeasible: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.low is not None and self.high is not None


@dataclass(frozen=True)
class RIOCell:
    """A per-polygon RIO result carrying its own justification."""

    index: int
    cell_state: str
    table_used: str
    ice_class: str
    conservative: Branch | None
    optimistic: Branch | None
    mapping_sensitive: bool
    mapping_classification_difference: bool
    iceberg_fraction_tenths: int | None
    iceberg_trace_present: bool
    warnings: tuple[str, ...]
    interpretation: PolygonInterpretation

    # --- flat accessors, for tabular export and the audit trail ----------
    @property
    def conservative_rio_low(self) -> float | None:
        return self.conservative.low.rio if self.conservative and self.conservative.usable else None

    @property
    def conservative_rio_high(self) -> float | None:
        return self.conservative.high.rio if self.conservative and self.conservative.usable else None

    @property
    def conservative_classification_low(self) -> str | None:
        return self.conservative.low.classification if self.conservative and self.conservative.usable else None

    @property
    def conservative_classification_high(self) -> str | None:
        return self.conservative.high.classification if self.conservative and self.conservative.usable else None

    @property
    def optimistic_rio_low(self) -> float | None:
        return self.optimistic.low.rio if self.optimistic and self.optimistic.usable else None

    @property
    def optimistic_rio_high(self) -> float | None:
        return self.optimistic.high.rio if self.optimistic and self.optimistic.usable else None

    @property
    def optimistic_classification_low(self) -> str | None:
        return self.optimistic.low.classification if self.optimistic and self.optimistic.usable else None

    @property
    def optimistic_classification_high(self) -> str | None:
        return self.optimistic.high.classification if self.optimistic and self.optimistic.usable else None

    @property
    def band_straddling(self) -> bool:
        return any(b is not None and b.classification == CLASSIFICATION_INDETERMINATE
                   for b in (self.conservative, self.optimistic))

    @property
    def has_rio(self) -> bool:
        return self.conservative is not None and self.conservative.usable

    def explain(self) -> str:
        i = self.interpretation
        lines = [
            f"polygon #{self.index}  state={self.cell_state}  "
            f"ice class={self.ice_class}  table={self.table_used}",
            f"  raw: " + " ".join(f"{k}={i.raw[k]}" for k in
                                  ("CT", "CA", "CB", "CC", "SA", "SB", "SC", "SO", "POLY_TYPE")),
        ]
        if not self.has_rio:
            lines.append("  RIO: NOT COMPUTED")
        for b in (self.conservative, self.optimistic):
            if b is None:
                continue
            if not b.usable:
                lines.append(f"  {b.label:<13}: no feasible endpoint")
            else:
                lines.append(f"  {b.label:<13}: RIO {b.low.rio:g} .. {b.high.rio:g}"
                             f"   -> {b.classification}")
                lines.append(f"      worst: {b.low.describe()}")
                lines.append(f"      best : {b.high.describe()}")
            for reason in b.infeasible:
                lines.append(f"      infeasible endpoint discarded: {reason}")
        lines.append(f"  mapping_sensitive={self.mapping_sensitive}  "
                     f"classification differs={self.mapping_classification_difference}  "
                     f"band straddling={self.band_straddling}")
        lines.append(f"  iceberg fraction={self.iceberg_fraction_tenths}  "
                     f"trace present={self.iceberg_trace_present}")
        if self.warnings:
            lines.append("  warnings:")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


# ------------------------------------------------------------- calculation
def _named_components(interp: PolygonInterpretation, branch: str,
                      ct_low: int, ct_high: int) -> list[tuple[str, int, int]]:
    """[(polaris type, low tenths, high tenths)] for the sea-ice components."""
    sea = interp.sea_ice_components
    pick = (lambda c: c.polaris_conservative) if branch == BRANCH_CONSERVATIVE \
        else (lambda c: c.polaris_optimistic)

    if interp.state == STATE_PARTIALS_ABSENT:
        # Policy: the single named stage takes the whole of CT; remainder Ice-Free.
        if len(sea) != 1:
            raise RIOEngineError(
                f"polygon {interp.index}: state is {STATE_PARTIALS_ABSENT} but "
                f"{len(sea)} sea-ice stages are named; the policy covers one")
        return [(pick(sea[0]), ct_low, ct_high)]

    out: list[tuple[str, int, int]] = []
    for c in sea:
        if not c.concentration.usable:
            continue
        out.append((pick(c), c.concentration.low_tenths, c.concentration.high_tenths))
    return out


def _build_regime(chosen: list[tuple[str, int]], iceberg_tenths: int,
                  iceberg_type: str | None, residual: int,
                  ice_free_type: str, label: str) -> IceRegime:
    """Merge duplicate ice types and close the regime with Ice-Free."""
    merged: dict[str, int] = {}
    for ice_type, tenths in chosen:
        merged[ice_type] = merged.get(ice_type, 0) + tenths
    if iceberg_tenths and iceberg_type:
        merged[iceberg_type] = merged.get(iceberg_type, 0) + iceberg_tenths
    if residual:
        merged[ice_free_type] = merged.get(ice_free_type, 0) + residual
    if not merged:
        merged[ice_free_type] = TENTHS_TOTAL
    return IceRegime(tuple(IceTypeConcentration(k, v) for k, v in merged.items()), label)


def _branch(interp: PolygonInterpretation, branch: str, ice_class: IceClass,
            table: RIVTable, ice_free_type: str) -> Branch:
    """Enumerate every feasible endpoint for one mapping and bracket the RIO."""
    ct = interp.total_concentration
    ct_ends = sorted({ct.low_tenths, ct.high_tenths})
    comps = _named_components(interp, branch, ct.low_tenths, ct.high_tenths)

    ib = interp.iceberg_concentration
    ib_tenths = ib.low_tenths if (ib is not None and ib.usable) else 0

    # The iceberg fraction has no RIV. Bound it: most severe type present, and Ice-Free.
    if ib_tenths:
        present = [t for t, _, _ in comps] or [ice_free_type]
        severest = min(present, key=lambda t: table.riv_for(ice_class, t))
        iceberg_types = [severest, ice_free_type]
    else:
        iceberg_types = [None]

    endpoints: list[Endpoint] = []
    infeasible: list[str] = []
    per_comp = [sorted({lo, hi}) for _, lo, hi in comps]

    for ct_choice in ct_ends:
        for combo in itertools.product(*per_comp) if per_comp else [()]:
            chosen = [(comps[k][0], combo[k]) for k in range(len(comps))]
            named = sum(v for _, v in chosen)
            if interp.state == STATE_PARTIALS_ABSENT:
                # the single stage already equals the CT endpoint it was given
                if combo and combo[0] != ct_choice:
                    continue
                named_total = named + ib_tenths
                if named_total > ct_choice:
                    # The policy gives the WHOLE of CT to the single named stage. It
                    # says nothing about how an iceberg fraction would share that
                    # total, so shrinking the stage to make room would be an
                    # assumption this module is not entitled to make. Discard the
                    # endpoint and record why. (No such cell exists in
                    # ANTARC20201224: partials_absent cells carry no iceberg.)
                    infeasible.append(
                        f"CT endpoint {ct_choice}/10 is given entirely to the single "
                        f"named stage by missing_partials_policy, leaving no room for "
                        f"the {ib_tenths}/10 iceberg fraction; the policy does not say "
                        f"how the two share CT")
                    continue
            else:
                named_total = named + ib_tenths
            # Feasibility is measured against CT: the named ice cannot exceed the
            # stated total. The regime itself is closed to TEN TENTHS, because the
            # ice not named inside CT and the open water outside CT both carry the
            # Ice-Free RIV, and (CT - named) + (10 - CT) == 10 - named.
            if named_total > ct_choice:
                infeasible.append(
                    f"CT endpoint {ct_choice}/10 < named components + iceberg "
                    f"({named_total}/10): the stated partials are more precise than "
                    f"the CT interval, so this endpoint cannot occur")
                continue
            if named_total > TENTHS_TOTAL:
                infeasible.append(
                    f"named components + iceberg = {named_total}/10 exceeds ten tenths")
                continue
            residual = TENTHS_TOTAL - named_total
            for ib_type in iceberg_types:
                try:
                    regime = _build_regime(chosen, ib_tenths, ib_type, residual,
                                           ice_free_type, f"#{interp.index} {branch}")
                    result = calculate_rio(regime, ice_class, table)
                except PolarisError as exc:
                    infeasible.append(f"CT {ct_choice}/10: {exc}")
                    continue
                endpoints.append(Endpoint(
                    rio=result.rio,
                    classification=classify_rio(result.rio, ice_class, table).value,
                    total_tenths=ct_choice, residual_ice_free=residual,
                    iceberg_assigned_type=ib_type,
                    terms=tuple((t.ice_type, t.tenths, t.riv) for t in result.terms)))

    if not endpoints:
        return Branch(label=branch, low=None, high=None,
                      classification=STATE_UNKNOWN, infeasible=tuple(infeasible))
    low = min(endpoints, key=lambda e: e.rio)
    high = max(endpoints, key=lambda e: e.rio)
    cls = low.classification if low.classification == high.classification \
        else CLASSIFICATION_INDETERMINATE
    return Branch(label=branch, low=low, high=high, classification=cls,
                  infeasible=tuple(dict.fromkeys(infeasible)))


def compute_rio(interp: PolygonInterpretation, ice_class: IceClass,
                table: RIVTable, policy: Policy, table_label: str) -> RIOCell:
    """One polygon -> one auditable RIO result. Never invents a value."""
    warnings = list(interp.warnings)
    ice_free = policy.ice_free_type

    if interp.state == STATE_UNKNOWN:
        warnings.append(str(policy.raw["unknown_policy"]["rule"]))
        return RIOCell(index=interp.index, cell_state=interp.state,
                       table_used=table_label, ice_class=ice_class.name,
                       conservative=None, optimistic=None,
                       mapping_sensitive=interp.mapping_sensitive,
                       mapping_classification_difference=False,
                       iceberg_fraction_tenths=(interp.iceberg_concentration.low_tenths
                                                if interp.iceberg_concentration
                                                and interp.iceberg_concentration.usable else None),
                       iceberg_trace_present=interp.icebergs_trace_present,
                       warnings=tuple(warnings), interpretation=interp)

    cons = _branch(interp, BRANCH_CONSERVATIVE, ice_class, table, ice_free)
    opt = _branch(interp, BRANCH_OPTIMISTIC, ice_class, table, ice_free)

    if interp.iceberg_concentration is not None and interp.iceberg_concentration.usable:
        warnings.append(
            "iceberg fraction bounded by assigning it the most severe RIV present "
            "(worst case) and the Ice-Free RIV (best case): PROJECT BOUNDING POLICY, "
            "OUTSIDE POLARIS -- POLARIS assigns no RIV to glacial ice")
    if cons.classification == CLASSIFICATION_INDETERMINATE:
        warnings.append(str(policy.raw["rio_reporting"]["flag_band_straddle"]))
    diff = bool(cons.usable and opt.usable and cons.classification != opt.classification)
    if diff:
        warnings.append("conservative and optimistic mappings fall in DIFFERENT "
                        "Table 1.1 bands; both are reported")
    if not cons.usable:
        warnings.append("no feasible endpoint for the conservative mapping; RIO not computed")

    return RIOCell(index=interp.index, cell_state=interp.state, table_used=table_label,
                   ice_class=ice_class.name, conservative=cons, optimistic=opt,
                   mapping_sensitive=interp.mapping_sensitive,
                   mapping_classification_difference=diff,
                   iceberg_fraction_tenths=(interp.iceberg_concentration.low_tenths
                                            if interp.iceberg_concentration
                                            and interp.iceberg_concentration.usable else None),
                   iceberg_trace_present=interp.icebergs_trace_present,
                   warnings=tuple(warnings), interpretation=interp)


def compute_chart(records, ice_class_name: str, table: RIVTable, policy: Policy,
                  table_label: str) -> list[RIOCell]:
    """Interpret and score every polygon. Reads only; writes nothing."""
    ice_class = table.ice_class(ice_class_name)
    return [compute_rio(interpret_polygon(r, i, policy), ice_class, table, policy,
                        table_label)
            for i, r in enumerate(records)]


def summarise(cells: list[RIOCell]) -> dict:
    from collections import Counter
    states = Counter(c.cell_state for c in cells)
    return {
        "cells": len(cells),
        "by_state": dict(sorted(states.items())),
        "rio_computed": sum(1 for c in cells if c.has_rio),
        "mapping_sensitive": sum(1 for c in cells if c.mapping_sensitive),
        "mapping_classification_difference": sum(
            1 for c in cells if c.mapping_classification_difference),
        "band_straddling": sum(1 for c in cells if c.band_straddling),
        "iceberg_bearing": sum(1 for c in cells if c.iceberg_fraction_tenths),
        "iceberg_trace": sum(1 for c in cells if c.iceberg_trace_present),
        "conservative_classification": dict(sorted(Counter(
            c.conservative.classification for c in cells if c.has_rio).items())),
        "optimistic_classification": dict(sorted(Counter(
            c.optimistic.classification for c in cells
            if c.optimistic and c.optimistic.usable).items())),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Compute POLARIS RIO over interpreted SIGRID-3 polygons. "
                    "Reports only; decides nothing about navigation.")
    ap.add_argument("--dbf", required=True)
    ap.add_argument("--ice-class", required=True,
                    help="vessel ice class as named in the RIV table (no default)")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY))
    ap.add_argument("--decayed-ice", action="store_true",
                    help="use Table 1.4. Requires a qualified-personnel confirmation "
                         "of ice decay; never inferred from date or season.")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    policy = load_policy(args.policy)
    key = "decayed_ice" if args.decayed_ice else "default"
    table_path = ROOT / policy.raw["riv_tables"][key]
    table = load_riv_table(table_path)
    label = f"{key} ({Path(table_path).name})"
    if args.decayed_ice:
        print("NOTE: Table 1.4 selected. " +
              str(policy.raw["riv_tables"]["selection_policy"]["evidence"]) + "\n")

    records = read_dbf_records(args.dbf)
    cells = compute_chart(records, args.ice_class, table, policy, label)

    if args.summary:
        import json
        print(json.dumps(summarise(cells), indent=2))
        return 0
    picks = [args.index] if args.index is not None else range(min(args.limit, len(cells)))
    for i in picks:
        print(cells[i].explain()); print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (RIOEngineError, PolarisError) as exc:
        print(f"\nCANNOT COMPUTE RIO\n\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)

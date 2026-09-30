"""
Turn a POLARIS RIO result into a configurable navigation PENALTY. Nothing more.

Input : a RIOCell from src/routing/polaris_rio
        configs/polaris_risk_penalties.json
Output: a RiskAssessment -- a penalty per mapping branch, the classification that
        produced each one, and every flag a reviewer needs to challenge it.

THE NUMBER IS OURS, NOT IMO'S
  MSC.1/Circ.1519 defines RIO and three operational categories. It defines no
  penalty, no weight and no cost. The category -> number mapping lives entirely
  in the config file and is a tunable project parameter. This module holds no
  default: with no config there is no penalty.

THIS LAYER CANNOT BLOCK ANYTHING
  There is no `blocked`, `no_go`, `navigable`, `passable` or `forbidden` here,
  and no mask is produced. A penalty is a preference, and the loader refuses a
  non-finite or out-of-range value precisely because an infinite penalty is a
  hard block wearing a number. POLARIS is "not... a 'Go/No Go' tool but... a
  decision support tool" (MSC.1/Circ.1519 section 1.4); the master decides.

UNCERTAINTY IS PRESERVED, NOT AVERAGED
  Every cell carries two mappings (conservative and optimistic) and each mapping
  carries two endpoints. All four penalties are reported. Where a mapping's RIO
  interval crosses a Table 1.1 boundary the mapping's classification stays
  `indeterminate` and BOTH endpoint classifications remain readable. Collapsing
  conservative and optimistic into one number is a decision for the cost layer,
  taken deliberately and with the pair still in hand.

WHAT THIS MODULE DOES NOT DO
  No RIV, no SIGRID code, no band threshold and no RIO arithmetic appears here;
  all of that arrives on the RIOCell. It does not touch A*, cost, the grid, the
  navigation domain or any constraint mask, and it does not rasterise anything.

Usage:
    from src.routing.polaris_risk import load_risk_config, assess
    risk = assess(rio_cell, load_risk_config())
    print(risk.explain())

    python -m src.routing.polaris_risk --dbf ... --ice-class PC6 --summary
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from src.routing.polaris_rio import (CLASSIFICATION_INDETERMINATE, Branch, RIOCell,
                                     compute_chart)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RISK_CONFIG = ROOT / "configs" / "polaris_risk_penalties.json"

STATE_UNKNOWN = "unknown"


class RiskConfigError(ValueError):
    """The penalty configuration cannot be used as supplied."""


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class RiskConfig:
    """Penalties per POLARIS outcome. Every value comes from the file."""

    raw: dict
    path: str

    @property
    def penalties(self) -> dict:
        return self.raw["penalties"]

    def penalty_for(self, classification: str) -> float:
        if classification not in self.penalties:
            raise RiskConfigError(
                f"no penalty configured for classification {classification!r}; "
                f"known: {sorted(self.penalties)}. This module has no default.")
        return float(self.penalties[classification])

    def rationale_for(self, classification: str) -> str:
        return str(self.raw.get("rationale", {}).get(classification, ""))


def load_risk_config(path: Path | str = DEFAULT_RISK_CONFIG) -> RiskConfig:
    """Read and validate the penalty configuration. No built-in fallback."""
    path = Path(path)
    if not path.exists():
        raise RiskConfigError(
            f"No penalty configuration at {path}. This module holds no default "
            f"penalties: the category -> number mapping is a project parameter and "
            f"must be stated in a file a reviewer can read.")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RiskConfigError(f"{path.name} is not valid JSON: {exc}")
    if "penalties" not in raw:
        raise RiskConfigError(f"{path.name} has no 'penalties'")

    limits = raw.get("limits", {})
    lo = float(limits.get("min_penalty", 0.0))
    hi = float(limits.get("max_penalty", math.inf))
    for name, value in raw["penalties"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RiskConfigError(f"penalty {name!r} is {value!r}, not a number")
        v = float(value)
        if not math.isfinite(v):
            raise RiskConfigError(
                f"penalty {name!r} is {v}. An infinite penalty is a hard navigation "
                f"block wearing a number; POLARIS does not authorise one.")
        if v < lo or v > hi:
            raise RiskConfigError(
                f"penalty {name!r} = {v} is outside the configured limits "
                f"[{lo}, {hi}]")
    return RiskConfig(raw=raw, path=str(path))


# ------------------------------------------------------------------ result
@dataclass(frozen=True)
class BranchRisk:
    """One mapping's penalties: the branch verdict and both endpoints."""

    label: str
    classification: str                 # may be CLASSIFICATION_INDETERMINATE
    classification_worst: str | None    # at the lowest RIO endpoint
    classification_best: str | None     # at the highest RIO endpoint
    penalty: float
    penalty_worst: float | None
    penalty_best: float | None
    rio_worst: float | None
    rio_best: float | None

    @property
    def indeterminate(self) -> bool:
        return self.classification == CLASSIFICATION_INDETERMINATE


@dataclass(frozen=True)
class RiskAssessment:
    """A per-polygon navigation penalty with its full justification."""

    index: int
    cell_state: str
    table_used: str
    ice_class: str
    has_rio: bool
    conservative: BranchRisk | None
    optimistic: BranchRisk | None
    unknown_penalty: float | None       # set only when no RIO exists
    mapping_sensitive: bool
    mapping_classification_difference: bool
    indeterminate: bool
    iceberg_fraction_tenths: int | None
    iceberg_trace_present: bool
    config_path: str
    warnings: tuple[str, ...]

    # --- flat accessors for a future cost layer --------------------------
    @property
    def conservative_penalty(self) -> float:
        return self.conservative.penalty if self.conservative else self.unknown_penalty

    @property
    def optimistic_penalty(self) -> float:
        return self.optimistic.penalty if self.optimistic else self.unknown_penalty

    @property
    def classification_used_conservative(self) -> str:
        return self.conservative.classification if self.conservative else STATE_UNKNOWN

    @property
    def classification_used_optimistic(self) -> str:
        return self.optimistic.classification if self.optimistic else STATE_UNKNOWN

    def explain(self) -> str:
        lines = [
            f"polygon #{self.index}  state={self.cell_state}  "
            f"class={self.ice_class}  table={self.table_used}",
            f"  config: {self.config_path}",
        ]
        if not self.has_rio:
            lines.append(f"  no RIO -> penalty {self.unknown_penalty:g} "
                         f"(configured for {STATE_UNKNOWN!r}) -- a penalty, not an exclusion")
        for b in (self.conservative, self.optimistic):
            if b is None:
                continue
            lines.append(
                f"  {b.label:<13} RIO {b.rio_worst:g}..{b.rio_best:g}  "
                f"classification={b.classification}  penalty={b.penalty:g}")
            lines.append(
                f"      endpoints: {b.classification_worst} (penalty "
                f"{b.penalty_worst:g})  ..  {b.classification_best} (penalty "
                f"{b.penalty_best:g})")
        lines.append(f"  mapping_sensitive={self.mapping_sensitive}  "
                     f"classification differs={self.mapping_classification_difference}  "
                     f"indeterminate={self.indeterminate}")
        lines.append(f"  iceberg fraction={self.iceberg_fraction_tenths}  "
                     f"trace={self.iceberg_trace_present}")
        if self.warnings:
            lines.append("  warnings:")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


# ------------------------------------------------------------- assessment
def _branch_risk(branch: Branch | None, config: RiskConfig) -> BranchRisk | None:
    if branch is None or not branch.usable:
        return None
    worst, best = branch.low, branch.high
    return BranchRisk(
        label=branch.label,
        classification=branch.classification,
        classification_worst=worst.classification,
        classification_best=best.classification,
        penalty=config.penalty_for(branch.classification),
        penalty_worst=config.penalty_for(worst.classification),
        penalty_best=config.penalty_for(best.classification),
        rio_worst=worst.rio, rio_best=best.rio)


def assess(cell: RIOCell, config: RiskConfig) -> RiskAssessment:
    """One RIO result -> one auditable penalty. Never an exclusion."""
    cons = _branch_risk(cell.conservative, config)
    opt = _branch_risk(cell.optimistic, config)
    has_rio = cons is not None

    warnings = list(cell.warnings)
    if not has_rio:
        warnings.append(
            f"no RIO: the configured {STATE_UNKNOWN!r} penalty "
            f"{config.penalty_for(STATE_UNKNOWN):g} is applied. It is a preference "
            f"weight, NOT an exclusion; unknown remains unknown.")
    if cons is not None and cons.indeterminate:
        warnings.append(
            f"conservative RIO interval crosses a Table 1.1 boundary: classification "
            f"stays {CLASSIFICATION_INDETERMINATE!r}; both endpoint classifications "
            f"({cons.classification_worst}, {cons.classification_best}) are reported")
    if cons is not None and opt is not None and cons.penalty != opt.penalty:
        warnings.append(
            f"conservative and optimistic mappings give DIFFERENT penalties "
            f"({cons.penalty:g} vs {opt.penalty:g}); both are reported and neither "
            f"may be used without the other being visible")
    warnings.append("penalty values are project parameters from "
                    f"{Path(config.path).name}, not POLARIS quantities")

    return RiskAssessment(
        index=cell.index, cell_state=cell.cell_state, table_used=cell.table_used,
        ice_class=cell.ice_class, has_rio=has_rio, conservative=cons, optimistic=opt,
        unknown_penalty=(None if has_rio else config.penalty_for(STATE_UNKNOWN)),
        mapping_sensitive=cell.mapping_sensitive,
        mapping_classification_difference=cell.mapping_classification_difference,
        indeterminate=bool((cons and cons.indeterminate) or (opt and opt.indeterminate)),
        iceberg_fraction_tenths=cell.iceberg_fraction_tenths,
        iceberg_trace_present=cell.iceberg_trace_present,
        config_path=config.path, warnings=tuple(warnings))


def assess_cells(cells: list[RIOCell], config: RiskConfig) -> list[RiskAssessment]:
    return [assess(c, config) for c in cells]


def summarise(risks: list[RiskAssessment]) -> dict:
    from collections import Counter
    return {
        "cells": len(risks),
        "with_rio": sum(1 for r in risks if r.has_rio),
        "no_rio": sum(1 for r in risks if not r.has_rio),
        "indeterminate": sum(1 for r in risks if r.indeterminate),
        "mapping_sensitive": sum(1 for r in risks if r.mapping_sensitive),
        "penalty_differs_between_branches": sum(
            1 for r in risks if r.conservative and r.optimistic
            and r.conservative.penalty != r.optimistic.penalty),
        "conservative_penalty_histogram": dict(sorted(
            Counter(r.conservative_penalty for r in risks).items())),
        "optimistic_penalty_histogram": dict(sorted(
            Counter(r.optimistic_penalty for r in risks).items())),
        "max_penalty_seen": max(r.conservative_penalty for r in risks) if risks else None,
        "any_infinite_penalty": any(
            not math.isfinite(r.conservative_penalty) for r in risks),
    }


def main(argv: list[str]) -> int:
    from src.data.sigrid3_interpret import load_policy, read_dbf_records
    from src.routing.polaris import load_riv_table

    ap = argparse.ArgumentParser(
        description="Convert POLARIS RIO results into configurable navigation "
                    "penalties. Produces preferences, never exclusions.")
    ap.add_argument("--dbf", required=True)
    ap.add_argument("--ice-class", required=True)
    ap.add_argument("--risk-config", default=str(DEFAULT_RISK_CONFIG))
    ap.add_argument("--decayed-ice", action="store_true")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    policy = load_policy()
    key = "decayed_ice" if args.decayed_ice else "default"
    table = load_riv_table(ROOT / policy.raw["riv_tables"][key])
    config = load_risk_config(args.risk_config)
    cells = compute_chart(read_dbf_records(args.dbf), args.ice_class, table, policy, key)
    risks = assess_cells(cells, config)

    if args.summary:
        print(json.dumps(summarise(risks), indent=2, default=str))
        return 0
    picks = [args.index] if args.index is not None else range(min(args.limit, len(risks)))
    for i in picks:
        print(risks[i].explain()); print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RiskConfigError as exc:
        print(f"\nCANNOT APPLY POLARIS RISK PENALTIES\n\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)

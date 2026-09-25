"""
Compose the project navigation cost: environment + POLARIS + coverage uncertainty.

    composed = environmental_cost
             + polaris_penalty                       (dominant polygon only)
             + coverage_uncertainty_weight * (1 - dominant_fraction)

Every term arrives from a layer that already owns it. This module adds one
addition and a great deal of bookkeeping, and nothing else.

WHAT IT DOES NOT DO
  It does not compute SIC or current cost (src/routing/cost.py owns that
  formula), does not recompute RIO, does not read a RIV table, does not
  introduce a POLARIS category, and does not decide navigability. It produces no
  blocked, no-go or navigable mask: land, bathymetry and ice-shelf exclusions
  belong to NavigationDomain and constraints.py. A cost is a preference.

RIO IS NEVER TRANSFORMED
  A cell takes the DOMINANT polygon's already-configured penalty and nothing
  else. Two polygons' RIOs are never averaged, area-weighted or otherwise
  combined: RIO is a sum over ice-regime concentrations, so blending two
  independently interpreted RIOs by geometry would invent a quantity POLARIS
  does not define. `dominant_fraction` expresses how much of the cell that
  polygon actually covers, as a SEPARATE term.

ABSENCE OF A CHART IS NOT A RISK MEASUREMENT
  An uncharted cell gets NO POLARIS penalty and NO coverage-uncertainty term.
  It is not ice-free, it is not assessed, and charging it would penalise
  everything outside the chart's extent while pretending to know something. Its
  environmental cost still applies, because SIC and currents cover it
  independently. The provenance state says `usnic_uncharted` so a route can
  explain the difference between "assessed and clear" and "never looked at".

FOUR PROVENANCE STATES, KEPT APART
    usnic_uncharted         no USNIC polygon covers this cell
    charted_water           POLY_TYPE=W: charted water, no POLARIS RIO by design
    charted_unknown_regime  charted, but the polygon has no defensible RIO
    charted_ice             charted sea ice with a POLARIS classification
  Only the last two carry a POLARIS penalty, and only the last three carry the
  coverage-uncertainty term.

INFINITY IS INHERITED, NEVER INTRODUCED
  cost.py marks blocked and unpriceable cells +inf. Those pass through
  unchanged, because suppressing them would hand A* a finite cost for a cell it
  must not enter. Every term this module adds is finite and >= 0, and the loader
  refuses a non-finite weight: an infinite weight would be a hard block wearing
  a number.

Usage:
    from src.routing.navigation_cost import load_composition_config, compose_grid
    out = compose_grid(env_cost, usnic_dir, penalties_by_polygon,
                       load_composition_config())
    print(out.explain_cell(row, col))
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from src.data.rasterize_sigrid3 import (AMBIGUITY_TIF, COVERAGE_TIF,
                                        DOMINANT_FRACTION_TIF, FRACTION_NODATA,
                                        POLYGON_ID_NODATA, POLYGON_ID_TIF, SIDECAR,
                                        WATER_TIF)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPOSITION_CONFIG = ROOT / "configs" / "navigation_cost_composition.json"
DEFAULT_USNIC_DIR = ROOT / "data" / "processed" / "usnic"

STATE_UNCHARTED = "usnic_uncharted"
STATE_WATER = "charted_water"
STATE_UNKNOWN_REGIME = "charted_unknown_regime"
STATE_ICE = "charted_ice"
STATES = (STATE_UNCHARTED, STATE_WATER, STATE_UNKNOWN_REGIME, STATE_ICE)


class CompositionError(ValueError):
    """The cost composition cannot be performed as specified."""


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class CompositionConfig:
    raw: dict
    path: str

    @property
    def coverage_uncertainty_weight(self) -> float:
        return float(self.raw["coverage_uncertainty_weight"])


def load_composition_config(path: Path | str = DEFAULT_COMPOSITION_CONFIG
                            ) -> CompositionConfig:
    """Read and validate the composition parameters. No built-in default."""
    path = Path(path)
    if not path.exists():
        raise CompositionError(
            f"No composition configuration at {path}. The coverage-uncertainty "
            f"weight is a project parameter and must be stated in a file, not "
            f"chosen in code.")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CompositionError(f"{path.name} is not valid JSON: {exc}")
    if "coverage_uncertainty_weight" not in raw:
        raise CompositionError(f"{path.name} has no 'coverage_uncertainty_weight'")

    w = raw["coverage_uncertainty_weight"]
    if isinstance(w, bool) or not isinstance(w, (int, float)):
        raise CompositionError(f"coverage_uncertainty_weight is {w!r}, not a number")
    w = float(w)
    if not math.isfinite(w):
        raise CompositionError(
            f"coverage_uncertainty_weight is {w}. An infinite weight is a hard "
            f"navigation block wearing a number; hard constraints belong to "
            f"NavigationDomain, not to a cost.")
    limits = raw.get("limits", {})
    lo = float(limits.get("min_weight", 0.0))
    hi = float(limits.get("max_weight", math.inf))
    if w < lo or w > hi:
        raise CompositionError(
            f"coverage_uncertainty_weight {w} is outside the configured limits "
            f"[{lo}, {hi}]")
    return CompositionConfig(raw=raw, path=str(path))


# ------------------------------------------------------------------ result
@dataclass(frozen=True)
class CellComposition:
    """One cell's composed cost with every term that produced it."""

    row: int
    col: int
    state: str
    environmental_cost: float
    polaris_penalty: float
    coverage_uncertainty: float           # the raw 1 - dominant_fraction signal
    coverage_uncertainty_cost: float      # that signal times the weight
    composed_cost: float
    polygon_id: int | None
    dominant_fraction: float | None
    charted: bool
    mixed: bool
    partial: bool
    is_water: bool
    polaris_classification: str | None
    branch: str
    weight: float

    def explain(self) -> str:
        pid = "none (uncharted)" if self.polygon_id is None else str(self.polygon_id)
        df = "n/a" if self.dominant_fraction is None else f"{self.dominant_fraction:.3f}"
        return "\n".join([
            f"cell (row {self.row}, col {self.col})  state={self.state}  "
            f"branch={self.branch}",
            f"  USNIC coverage      : charted={self.charted}  polygon_id={pid}  "
            f"dominant_fraction={df}  mixed={self.mixed}  partial={self.partial}",
            f"  POLARIS             : classification={self.polaris_classification}  "
            f"penalty={self.polaris_penalty:g}",
            f"  environmental cost  : {self.environmental_cost:g}",
            f"  coverage uncertainty: (1 - {df}) = {self.coverage_uncertainty:g}  "
            f"x weight {self.weight:g} = {self.coverage_uncertainty_cost:g}",
            f"  composed cost       : {self.environmental_cost:g} + "
            f"{self.polaris_penalty:g} + {self.coverage_uncertainty_cost:g} = "
            f"{self.composed_cost:g}",
        ])


@dataclass(frozen=True)
class GridComposition:
    """Composed cost surfaces plus every input layer, for explanation."""

    conservative_cost: np.ndarray
    optimistic_cost: np.ndarray
    environmental_cost: np.ndarray
    polaris_conservative: np.ndarray
    polaris_optimistic: np.ndarray
    coverage_uncertainty: np.ndarray
    coverage_uncertainty_cost: np.ndarray
    state: np.ndarray                      # uint8 index into STATES
    polygon_id: np.ndarray
    dominant_fraction: np.ndarray
    charted: np.ndarray
    mixed: np.ndarray
    water: np.ndarray
    weight: float
    config_path: str
    classification: dict                   # polygon_id -> classification string

    def state_name(self, row: int, col: int) -> str:
        return STATES[int(self.state[row, col])]

    def cell(self, row: int, col: int, branch: str = "conservative") -> CellComposition:
        if branch not in ("conservative", "optimistic"):
            raise CompositionError(f"branch must be conservative or optimistic, "
                                   f"got {branch!r}")
        pid = int(self.polygon_id[row, col])
        pen = (self.polaris_conservative if branch == "conservative"
               else self.polaris_optimistic)[row, col]
        env = float(self.environmental_cost[row, col])
        cu_cost = float(self.coverage_uncertainty_cost[row, col])
        return CellComposition(
            row=row, col=col, state=self.state_name(row, col),
            environmental_cost=env, polaris_penalty=float(pen),
            coverage_uncertainty=float(self.coverage_uncertainty[row, col]),
            coverage_uncertainty_cost=cu_cost,
            composed_cost=float((self.conservative_cost if branch == "conservative"
                                 else self.optimistic_cost)[row, col]),
            polygon_id=None if pid == POLYGON_ID_NODATA else pid,
            dominant_fraction=(None if not self.charted[row, col]
                               else float(self.dominant_fraction[row, col])),
            charted=bool(self.charted[row, col]), mixed=bool(self.mixed[row, col]),
            partial=bool(self.charted[row, col]
                         and self.dominant_fraction[row, col] < 1.0),
            is_water=bool(self.water[row, col]),
            polaris_classification=self.classification.get(pid),
            branch=branch, weight=self.weight)

    def explain_cell(self, row: int, col: int, branch: str = "conservative") -> str:
        return self.cell(row, col, branch).explain()


# ------------------------------------------------------------- composition
def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def compose_grid(environmental_cost: np.ndarray,
                 penalties: dict,
                 config: CompositionConfig,
                 usnic_dir: Path | str = DEFAULT_USNIC_DIR) -> GridComposition:
    """Add the POLARIS and coverage terms onto an existing environmental cost.

    `penalties` maps polygon_id -> {"conservative": float, "optimistic": float,
    "classification": str, "is_water": bool, "has_rio": bool}. Build it with
    penalties_from_risk() so the numbers stay owned by polaris_risk.py.
    """
    usnic_dir = Path(usnic_dir)
    env = np.asarray(environmental_cost, dtype="float64")
    coverage = _read(usnic_dir / COVERAGE_TIF).astype(bool)
    poly_id = _read(usnic_dir / POLYGON_ID_TIF).astype("int32")
    water = _read(usnic_dir / WATER_TIF).astype(bool)
    ambiguity = _read(usnic_dir / AMBIGUITY_TIF)
    fraction = _read(usnic_dir / DOMINANT_FRACTION_TIF).astype("float64")
    for name, arr in (("coverage", coverage), ("polygon_id", poly_id),
                      ("water", water), ("dominant_fraction", fraction)):
        if arr.shape != env.shape:
            raise CompositionError(
                f"{name} shape {arr.shape} != environmental cost shape {env.shape}; "
                f"this module never resamples")

    mixed = ambiguity >= 1
    charted = coverage & (poly_id != POLYGON_ID_NODATA)

    # --- POLARIS: the dominant polygon's configured penalty, nothing else ----
    pol_cons = np.zeros(env.shape, dtype="float64")
    pol_opt = np.zeros(env.shape, dtype="float64")
    state = np.full(env.shape, STATES.index(STATE_UNCHARTED), dtype="uint8")
    for pid, info in penalties.items():
        sel = charted & (poly_id == pid)
        if not sel.any():
            continue
        if info.get("is_water"):
            state[sel] = STATES.index(STATE_WATER)          # no POLARIS RIO by design
            continue
        if not info.get("has_rio", True):
            state[sel] = STATES.index(STATE_UNKNOWN_REGIME)
        else:
            state[sel] = STATES.index(STATE_ICE)
        pol_cons[sel] = float(info["conservative"])
        pol_opt[sel] = float(info["optimistic"])

    # --- coverage uncertainty: charted cells only ---------------------------
    cu = np.zeros(env.shape, dtype="float64")
    cu[charted] = 1.0 - fraction[charted]
    cu = np.clip(cu, 0.0, 1.0)
    cu[~charted] = 0.0                       # absence of a chart is not uncertainty
    w = config.coverage_uncertainty_weight
    cu_cost = w * cu

    for name, arr in (("polaris_conservative", pol_cons),
                      ("polaris_optimistic", pol_opt),
                      ("coverage_uncertainty_cost", cu_cost)):
        if not np.isfinite(arr).all():
            raise CompositionError(f"{name} contains a non-finite value; every term "
                                   f"this layer adds must be finite")
        if (arr < 0).any():
            raise CompositionError(f"{name} contains a negative value")

    cons = env + pol_cons + cu_cost
    opt = env + pol_opt + cu_cost
    finite = np.isfinite(env)
    for name, arr in (("conservative", cons), ("optimistic", opt)):
        if (arr[finite] < 0).any():
            raise CompositionError(f"{name} composed cost went negative")
        if not np.isfinite(arr[finite]).all():
            raise CompositionError(
                f"{name} composed cost is non-finite where the environmental cost "
                f"was finite; this layer must never introduce an infinity")

    return GridComposition(
        conservative_cost=cons, optimistic_cost=opt, environmental_cost=env,
        polaris_conservative=pol_cons, polaris_optimistic=pol_opt,
        coverage_uncertainty=cu, coverage_uncertainty_cost=cu_cost, state=state,
        polygon_id=poly_id, dominant_fraction=fraction, charted=charted,
        mixed=mixed, water=water, weight=w, config_path=config.path,
        classification={pid: info.get("classification")
                        for pid, info in penalties.items()})


def penalties_from_risk(risks, sidecar_path: Path | str | None = None) -> dict:
    """polygon_id -> the penalty record, taken straight from polaris_risk output.

    `risks` is an iterable of RiskAssessment, one per polygon, in polygon order.
    No penalty value is computed here; they are read off the assessments.
    """
    water_ids: set[int] = set()
    if sidecar_path is not None:
        side = json.loads(Path(sidecar_path).read_text())
        water_ids = {int(k) for k, v in side["polygons"].items() if v.get("is_water")}
    out = {}
    for r in risks:
        out[int(r.index)] = {
            "conservative": float(r.conservative_penalty),
            "optimistic": float(r.optimistic_penalty),
            "classification": r.classification_used_conservative,
            "has_rio": bool(r.has_rio),
            "is_water": int(r.index) in water_ids,
        }
    return out


def summarise(out: GridComposition) -> dict:
    from collections import Counter
    finite = np.isfinite(out.environmental_cost)
    st = Counter(STATES[i] for i in out.state.ravel().tolist())
    return {
        "cells": int(out.state.size),
        "by_state": dict(sorted(st.items())),
        "charted": int(out.charted.sum()),
        "mixed": int(out.mixed.sum()),
        "partial": int((out.charted & (out.dominant_fraction < 1.0)).sum()),
        "environmental_finite": int(finite.sum()),
        "environmental_infinite_passed_through": int((~finite).sum()),
        "coverage_uncertainty_weight": out.weight,
        "coverage_uncertainty_cost_max": float(out.coverage_uncertainty_cost.max()),
        "polaris_penalty_histogram": dict(sorted(
            Counter(out.polaris_conservative[finite].ravel().tolist()).items())),
        "composed_conservative_finite_max": (
            float(out.conservative_cost[finite].max()) if finite.any() else None),
        "composed_optimistic_finite_max": (
            float(out.optimistic_cost[finite].max()) if finite.any() else None),
        "any_negative": bool((out.conservative_cost[finite] < 0).any()),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Compose environmental + POLARIS + coverage-uncertainty cost. "
                    "Produces preferences only; no mask, no route.")
    ap.add_argument("--date", required=True, help="SIC/currents date, YYYY-MM-DD")
    ap.add_argument("--ice-class", required=True)
    ap.add_argument("--usnic-dir", default=str(DEFAULT_USNIC_DIR))
    ap.add_argument("--config", default=str(DEFAULT_COMPOSITION_CONFIG))
    ap.add_argument("--cell", nargs=2, type=int, default=None, metavar=("ROW", "COL"))
    args = ap.parse_args(argv)
    print(json.dumps({"note": "use the demo script; this CLI is a thin wrapper"},
                     indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except CompositionError as exc:
        print(f"\nCANNOT COMPOSE THE NAVIGATION COST\n\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)

"""
Hard navigation constraints: which cells a route may never enter.

This is the "absolutely not" layer, kept separate from cost. A cell here is
either passable or it is not; nothing in this module makes a cell merely
expensive. It combines caller-supplied masks into the single blocked_mask that
RoutingGrid carries, and does nothing else -- no A*, no cost surface, no vessel
performance, no iceberg prediction, no UI.

    from src.routing.constraints import NavigationConstraints
    c = NavigationConstraints(land_mask=land, shallow_mask=shallow)
    c.apply_to(grid)

THREE INDEPENDENT MASKS, NONE OF THEM INVENTED HERE
    land_mask            True where the cell is land or ice shelf
    shallow_mask         True where draught is insufficient
    polaris_no_go_mask   True where the vessel's ice class may not operate

This module does not download, derive, infer or approximate any of them. In
particular it never treats a NaN in SIC as land: SIC is NaN over land, over the
pole hole and over a one-day swath gap, and collapsing those three into "land"
would quietly wall off open ocean that happens to have been missed by a satellite
pass that day. Each mask must arrive from a real source -- a coastline product,
a bathymetry grid, an actual RIO calculation -- supplied by the caller.

NOT SUPPLIED IS NOT THE SAME AS NOTHING BLOCKED
    land_mask=None            we have no coastline data
    land_mask=<all False>     we have coastline data and it says no land here
Both produce the same combined mask, and they mean completely different things.
`supplied`, `missing` and `summary()` keep the two distinguishable, and
`apply_to()` refuses by default to write an unconstrained mask onto a grid --
pass allow_unconstrained=True if routing over open ocean is genuinely intended.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                       # keeps this module numpy-only at runtime
    from src.routing.grid import RoutingGrid

MASK_NAMES = ("land_mask", "shallow_mask", "polaris_no_go_mask")


class ConstraintError(ValueError):
    """A supplied mask is not usable as a hard constraint."""


def validate_mask(name: str, mask, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Check one mask. Boolean and 2-D always; shape too when one is given.

    dtype is checked rather than cast: silently calling .astype(bool) on a
    float array turns every NaN into True, which would block cells nobody
    intended to block. If you mean to convert, convert at the call site.
    """
    arr = np.asarray(mask)
    if arr.dtype != np.bool_:
        raise ConstraintError(
            f"{name} must be a boolean array, got dtype {arr.dtype}. "
            f"Convert explicitly at the call site if that is what you mean -- "
            f"note that astype(bool) maps NaN to True.")
    if arr.ndim != 2:
        raise ConstraintError(f"{name} must be 2-D, got {arr.ndim}-D {arr.shape}")
    if shape is not None and arr.shape != shape:
        raise ConstraintError(
            f"{name} shape {arr.shape} != grid shape {shape}")
    return arr


@dataclass
class NavigationConstraints:
    """Optional hard-constraint masks, OR-ed into one blocked_mask."""

    land_mask: np.ndarray | None = None
    shallow_mask: np.ndarray | None = None
    polaris_no_go_mask: np.ndarray | None = None
    shape: tuple[int, int] | None = None     # optional; validates as early as possible

    def __post_init__(self) -> None:
        for name in MASK_NAMES:
            m = getattr(self, name)
            if m is not None:
                setattr(self, name, validate_mask(name, m, self.shape))
        self._check_mutual_shapes()

    def _check_mutual_shapes(self) -> None:
        shapes = {name: getattr(self, name).shape
                  for name in MASK_NAMES if getattr(self, name) is not None}
        if len(set(shapes.values())) > 1:
            raise ConstraintError(
                "supplied masks disagree on shape: "
                + ", ".join(f"{k} {v}" for k, v in shapes.items()))

    # ------------------------------------------------------------- state
    @property
    def supplied(self) -> tuple[str, ...]:
        """Masks the caller actually provided, all-False ones included."""
        return tuple(n for n in MASK_NAMES if getattr(self, n) is not None)

    @property
    def missing(self) -> tuple[str, ...]:
        """Masks with no data behind them at all."""
        return tuple(n for n in MASK_NAMES if getattr(self, n) is None)

    @property
    def is_unconstrained(self) -> bool:
        """True when NOTHING was supplied -- every cell would be passable."""
        return not self.supplied

    def set_mask(self, name: str, mask) -> None:
        if name not in MASK_NAMES:
            raise ConstraintError(f"unknown mask {name!r}; expected one of {MASK_NAMES}")
        setattr(self, name, validate_mask(name, mask, self.shape))
        self._check_mutual_shapes()

    # ---------------------------------------------------------- combining
    def combined(self, shape: tuple[int, int] | None = None) -> np.ndarray:
        """Logical OR of every supplied mask; all-False when none were supplied.

        `shape` is the grid being routed on. When given, every mask is
        re-validated against it, so a mask built for a different grid cannot
        reach a planner.
        """
        target = shape or self.shape
        for name in self.supplied:
            validate_mask(name, getattr(self, name), target)

        if target is None:
            if self.is_unconstrained:
                raise ConstraintError(
                    "cannot build a combined mask: no masks supplied and no "
                    "shape given, so the result has no size.")
            target = getattr(self, self.supplied[0]).shape

        out = np.zeros(target, dtype=bool)
        for name in self.supplied:
            out |= getattr(self, name)
        return out

    # ------------------------------------------------------------- apply
    def apply_to(self, grid: "RoutingGrid",
                 allow_unconstrained: bool = False) -> np.ndarray:
        """Write the combined mask onto grid.blocked_mask and return it."""
        if self.is_unconstrained and not allow_unconstrained:
            raise ConstraintError(
                "no constraint masks supplied, so every cell -- including land "
                "-- would be navigable.\n"
                "  Supply land_mask / shallow_mask / polaris_no_go_mask, or pass "
                "allow_unconstrained=True if that is genuinely intended.")
        mask = self.combined(grid.shape)
        grid.set_blocked_mask(mask)
        return mask

    # ----------------------------------------------------------- summary
    def summary(self, shape: tuple[int, int] | None = None) -> dict:
        """Per-mask and combined cell counts. Distinguishes absent from empty."""
        target = shape or self.shape
        info: dict[str, object] = {"supplied": list(self.supplied),
                                   "missing": list(self.missing)}
        per: dict[str, object] = {}
        for name in MASK_NAMES:
            m = getattr(self, name)
            per[name] = None if m is None else int(m.sum())   # None = not supplied
        info["blocked_cells"] = per

        if self.supplied or target is not None:
            c = self.combined(target)
            info["combined_blocked"] = int(c.sum())
            info["combined_fraction"] = float(c.mean())
            info["shape"] = list(c.shape)
        return info

    def report(self, shape: tuple[int, int] | None = None) -> str:
        """One-screen human summary, for printing before a route is planned."""
        s = self.summary(shape)
        lines = ["Navigation constraints:"]
        for name in MASK_NAMES:
            n = s["blocked_cells"][name]
            lines.append(f"  {name:<20} " + ("NOT SUPPLIED (no data)" if n is None
                                             else f"{n:>12,} cells blocked"))
        if "combined_blocked" in s:
            lines.append(f"  {'COMBINED':<20} {s['combined_blocked']:>12,} cells "
                         f"({100 * s['combined_fraction']:.2f}% of the grid)")
        if self.is_unconstrained:
            lines.append("  -> nothing is blocked; a route may cross land.")
        elif self.missing:
            lines.append(f"  -> no data for: {', '.join(self.missing)}; "
                         f"those hazards are NOT excluded.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"NavigationConstraints(supplied={list(self.supplied)}, "
                f"missing={list(self.missing)})")
"""
Navigation domain: where a ship is permitted to be, from REAL data only.

SIC and current coverage do not define a navigable sea. A cell with valid SIC
may be land, an ice shelf, or 20 m of water under a ship that draws 7 m. This
module is the place where the actual exclusion datasets are loaded, checked
against the routing grid, and handed to NavigationConstraints -- and it is
deliberately incapable of producing a mask from anything but a real file or a
real array the caller already has.

    from src.routing.navigation_domain import NavigationDomain
    dom = NavigationDomain.from_sources(grid, land_mask="data/raw/land.tif")
    print(dom.report())
    dom.to_constraints().apply_to(grid)

BOOLEAN SEMANTICS -- one rule, everywhere
    True  = cell is EXCLUDED from navigation
    False = cell is not excluded BY THIS LAYER (it may still be by another)
False never means "safe". It means this particular dataset has no objection.

MISSING IS NOT PERMISSIVE
  A layer that was never supplied stays None all the way through to
  NavigationConstraints, which already reports `supplied` separately from
  `missing`. Nothing here converts absence into an all-False mask, because an
  all-False mask is a positive claim ("checked, nothing excluded") and absence
  is not. `missing`, `is_complete` and `require_complete()` let a caller decide
  what to do about the gap; this module will not decide for them.

WHAT THIS MODULE WILL NOT DO
  It does not download anything. It does not derive land from SIC NaNs. It does
  not threshold a depth raster into a shallow mask -- choosing the draught
  cut-off is a vessel and bathymetry decision, and a continuous raster handed
  in here is rejected rather than silently binarised at zero. It applies no
  POLARIS rule. It never resamples or reprojects: a mask on the wrong grid is
  an error, not something to fix by interpolation, because quietly warping a
  coastline is how a route ends up crossing one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio

from src.routing.constraints import MASK_NAMES, NavigationConstraints
from src.routing.grid import BOUNDS_TOL, TRANSFORM_TOL, _bounds

if TYPE_CHECKING:
    from src.routing.grid import RoutingGrid

NODATA_POLICIES = ("error", "excluded", "allowed")
MAX_MASK_VALUES = 2          # a real mask has at most two distinct values


class DomainError(ValueError):
    """A navigation mask cannot be accepted as supplied."""


@dataclass(frozen=True)
class MaskLayer:
    """One exclusion dataset, already validated against the grid."""

    name: str
    mask: np.ndarray            # bool (H, W); True = excluded
    source: str                 # a path, or "array" when passed in memory

    @property
    def excluded(self) -> int:
        return int(self.mask.sum())


# --------------------------------------------------------------- validation
def _check_alignment(src, grid: "RoutingGrid", path: Path) -> None:
    """Every way a raster can disagree with the grid, reported together."""
    problems: list[str] = []
    h, w = tuple(grid.shape)

    s_epsg = src.crs.to_epsg() if src.crs else None
    g_epsg = grid.crs.to_epsg() if grid.crs else None
    if s_epsg != g_epsg or src.crs != grid.crs:
        problems.append(f"CRS: {src.crs} vs grid {grid.crs}")
    if (src.height, src.width) != (h, w):
        problems.append(f"shape: {(src.height, src.width)} vs grid {(h, w)}")
    for i, (a, b) in enumerate(zip(list(src.transform)[:6],
                                   list(grid.transform)[:6])):
        if abs(a - b) > TRANSFORM_TOL:
            problems.append(f"transform[{i}]: {a} vs grid {b}")
    rb = _bounds(src.transform, src.height, src.width)
    gb = _bounds(grid.transform, h, w)
    if any(abs(a - b) > BOUNDS_TOL for a, b in zip(rb, gb)):
        problems.append(f"extent: {rb} vs grid {gb}")

    if problems:
        raise DomainError(
            f"{path.name} is not on the routing grid:\n  " + "\n  ".join(problems)
            + "\n  -> this module does not resample or reproject. Regrid the mask "
              "onto the routing grid first, with a method you have chosen "
              "deliberately for a categorical layer (nearest, not bilinear).")


def load_mask_geotiff(path, grid: "RoutingGrid", *, band: int = 1,
                      nodata_policy: str = "error", invert: bool = False,
                      allow_multivalued: bool = False) -> np.ndarray:
    """Read one already-boolean exclusion raster onto the routing grid.

    Nonzero means excluded, zero means not excluded; pass invert=True for a
    raster stored the other way round (1 = navigable).

    nodata_policy decides cells with no value, and defaults to "error" because
    a hole in a coastline mask is not information -- guessing either way is a
    silent safety decision. Use "excluded" (conservative) or "allowed" only
    when you know what the nodata region actually is.

    A raster with more than two distinct finite values is rejected: that is a
    continuous field, most likely depth, and turning it into a mask requires a
    draught threshold this module refuses to invent.
    """
    if nodata_policy not in NODATA_POLICIES:
        raise DomainError(
            f"nodata_policy must be one of {NODATA_POLICIES}, got {nodata_policy!r}")

    path = Path(path)
    if not path.exists():
        raise DomainError(f"Not found: {path}")

    with rasterio.open(path) as src:
        _check_alignment(src, grid, path)
        if band > src.count:
            raise DomainError(
                f"{path.name} has {src.count} band(s); band {band} requested")
        raw = src.read(band)
        nodata = src.nodata

    arr = raw.astype("float64")
    missing = ~np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        missing |= raw == nodata

    present = arr[~missing]
    if present.size == 0:
        raise DomainError(f"{path.name} has no valid pixels at all.")
    distinct = np.unique(present)
    if distinct.size > MAX_MASK_VALUES and not allow_multivalued:
        raise DomainError(
            f"{path.name} holds {distinct.size:,} distinct values "
            f"({distinct[:4]} ...), so it is a continuous raster, not a mask. "
            f"If this is bathymetry, threshold it against a real draught "
            f"yourself and supply the resulting boolean array; choosing that "
            f"cut-off is not this module's decision. Pass "
            f"allow_multivalued=True only if nonzero genuinely means excluded.")

    excluded = arr != 0
    if invert:
        excluded = ~excluded

    n_missing = int(missing.sum())
    if n_missing:
        if nodata_policy == "error":
            first = tuple(int(v) for v in np.argwhere(missing)[0])
            raise DomainError(
                f"{path.name} has {n_missing:,} nodata cell(s), first at "
                f"(row, col) = {first}. Set nodata_policy='excluded' to treat "
                f"them as unnavigable or 'allowed' to ignore them -- but decide "
                f"it explicitly, because it is a safety choice.")
        excluded[missing] = (nodata_policy == "excluded")

    return excluded.astype(bool, copy=False)


# ------------------------------------------------------------------ domain
class NavigationDomain:
    """The real exclusion layers available for one routing grid. Possibly none."""

    def __init__(self, grid: "RoutingGrid"):
        self.grid = grid
        self.layers: dict[str, MaskLayer] = {}

    # ----------------------------------------------------------- building
    @classmethod
    def from_sources(cls, grid: "RoutingGrid",
                     land_mask=None, shallow_mask=None, polaris_no_go_mask=None,
                     **load_kwargs) -> "NavigationDomain":
        """Each source may be a boolean array, a GeoTIFF path, or None."""
        dom = cls(grid)
        for name, src in (("land_mask", land_mask),
                          ("shallow_mask", shallow_mask),
                          ("polaris_no_go_mask", polaris_no_go_mask)):
            if src is None:
                continue
            if isinstance(src, (str, Path)):
                dom.add_geotiff(name, src, **load_kwargs)
            else:
                dom.add_array(name, src)
        return dom

    def add_array(self, name: str, mask, source: str = "array") -> "NavigationDomain":
        self._check_name(name)
        arr = np.asarray(mask)
        if arr.dtype != np.bool_:
            raise DomainError(
                f"{name} must be a boolean array, got dtype {arr.dtype}. "
                f"Convert at the call site so the rule is visible -- note that "
                f"astype(bool) maps NaN to True.")
        if arr.shape != tuple(self.grid.shape):
            raise DomainError(
                f"{name} shape {arr.shape} != grid shape {tuple(self.grid.shape)}; "
                f"this module does not resample.")
        self.layers[name] = MaskLayer(name=name, mask=arr, source=source)
        return self

    def add_geotiff(self, name: str, path, **kwargs) -> "NavigationDomain":
        self._check_name(name)
        mask = load_mask_geotiff(path, self.grid, **kwargs)
        self.layers[name] = MaskLayer(name=name, mask=mask, source=str(path))
        return self

    @staticmethod
    def _check_name(name: str) -> None:
        if name not in MASK_NAMES:
            raise DomainError(f"unknown layer {name!r}; expected one of {MASK_NAMES}")

    # -------------------------------------------------------------- state
    @property
    def supplied(self) -> tuple[str, ...]:
        return tuple(n for n in MASK_NAMES if n in self.layers)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(n for n in MASK_NAMES if n not in self.layers)

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def require_complete(self) -> "NavigationDomain":
        """For callers that must not route with a hazard class unchecked."""
        if self.missing:
            raise DomainError(
                "navigation domain is incomplete; no data for: "
                + ", ".join(self.missing)
                + ". Those hazards are NOT excluded from the route.")
        return self

    def combined(self) -> np.ndarray:
        """OR of the supplied layers; all-False when none were supplied.

        All-False here means "nothing supplied objected", which is not the same
        as "nothing is excluded" -- read `missing` alongside it.
        """
        out = np.zeros(tuple(self.grid.shape), dtype=bool)
        for name in self.supplied:
            out |= self.layers[name].mask
        return out

    # --------------------------------------------------------- handover
    def to_constraints(self) -> NavigationConstraints:
        """Build the NavigationConstraints for this grid. Missing stays None."""
        return NavigationConstraints(
            shape=tuple(self.grid.shape),
            **{n: self.layers[n].mask for n in self.supplied})

    # ---------------------------------------------------------- reporting
    def summary(self) -> dict:
        h, w = tuple(self.grid.shape)
        combined = self.combined()
        return {
            "grid": {
                "shape": [h, w],
                "crs": self.grid.crs.to_string() if self.grid.crs else None,
                "pixel_size": list(self.grid.pixel_size),
                "bounds": list(self.grid.bounds),
                "day": str(self.grid.day) if self.grid.day else None,
            },
            "supplied": list(self.supplied),
            "missing": list(self.missing),
            "layers": {n: (None if n not in self.layers else
                           {"excluded_cells": self.layers[n].excluded,
                            "source": self.layers[n].source})
                       for n in MASK_NAMES},
            "combined_excluded": int(combined.sum()),
            "combined_fraction": float(combined.mean()),
            "navigable_cells": int((~combined).sum()),
            "is_complete": self.is_complete,
        }

    def report(self) -> str:
        s = self.summary()
        g = s["grid"]
        lines = ["Navigation domain:",
                 f"  grid        {g['shape'][0]} x {g['shape'][1]}  {g['crs']}  "
                 f"{g['pixel_size'][0]:g} m",
                 f"  bounds      {tuple(round(v, 1) for v in g['bounds'])}",
                 f"  day         {g['day']}",
                 ""]
        for name in MASK_NAMES:
            info = s["layers"][name]
            if info is None:
                lines.append(f"  {name:<20} NOT SUPPLIED (no dataset)")
            else:
                lines.append(f"  {name:<20} {info['excluded_cells']:>12,} excluded"
                             f"   [{info['source']}]")
        lines.append(f"  {'COMBINED':<20} {s['combined_excluded']:>12,} excluded "
                     f"({100 * s['combined_fraction']:.2f}%), "
                     f"{s['navigable_cells']:,} navigable")
        if not s["supplied"]:
            lines.append("\n  -> NO exclusion data at all: every cell would be "
                         "navigable, including land.")
        elif s["missing"]:
            lines.append(f"\n  -> no data for: {', '.join(s['missing'])}; "
                         f"those hazards are NOT excluded.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"NavigationDomain(supplied={list(self.supplied)}, "
                f"missing={list(self.missing)})")
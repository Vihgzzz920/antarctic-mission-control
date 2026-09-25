"""
Routing foundation: one day's SIC and currents on the common EPSG:3976 grid.

This module does ONE thing -- it hands the routing layer a validated, aligned
pair of rasters plus the two caller-supplied fields a planner needs
(blocked_mask, cost_grid) and the coordinate conversions between projected
metres and row/col. It knows nothing about A*, POLARIS, bathymetry, vessel
performance, iceberg prediction or the UI, and it must stay that way.

    from src.routing.grid import RoutingGrid
    g = RoutingGrid.for_date(date(2025, 1, 15))
    r, c = g.xy_to_rowcol(-1_200_000.0, -800_000.0)

TWO FIELDS ARE DELIBERATELY EMPTY
  blocked_mask defaults to all-False and cost_grid defaults to None. Neither is
  inferred from SIC. A land mask guessed from "SIC is NaN here" would silently
  conflate land, the pole hole and a one-day swath gap, and a cost surface
  invented from SIC alone would look like a vessel model without being one.
  Both are the caller's to supply, from a real source, in a later module.
  cost_grid is None rather than ones-everywhere so that a planner run before a
  cost model exists fails loudly instead of returning a confident straight line.

SIC loading is delegated to preprocess.load_sic, so the project's single
masking rule applies here too: 0.0 is VALID open water, and only >100, <0 and
non-finite are dropped. Nothing is interpolated or filled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.data.preprocess import load_sic

ROOT = Path(__file__).resolve().parents[2]
SIC_DIR = ROOT / "data" / "raw"
CURRENTS_DIR = ROOT / "data" / "processed" / "currents"

TRANSFORM_TOL = 1e-6        # metres; two rasters must align well inside a pixel
BOUNDS_TOL = 1e-3           # metres


class GridAlignmentError(ValueError):
    """SIC and currents are not on the same grid, so they cannot be routed on."""


def _as_date(day: date | str) -> date:
    return day if isinstance(day, date) else datetime.strptime(
        str(day).replace("-", ""), "%Y%m%d").date()


def _bounds(transform: Affine, height: int, width: int) -> tuple[float, ...]:
    a, _, c, _, e, f = (transform.a, transform.b, transform.c,
                        transform.d, transform.e, transform.f)
    x0, x1 = c, c + a * width
    y0, y1 = f, f + e * height
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _check_aligned(sic_src, cur_src, sic_name: str, cur_name: str) -> None:
    """Every way these two rasters could disagree, reported together."""
    problems: list[str] = []

    s_epsg = sic_src.crs.to_epsg() if sic_src.crs else None
    c_epsg = cur_src.crs.to_epsg() if cur_src.crs else None
    if s_epsg != c_epsg or sic_src.crs != cur_src.crs:
        problems.append(f"CRS: {sic_src.crs} vs {cur_src.crs}")

    if (sic_src.height, sic_src.width) != (cur_src.height, cur_src.width):
        problems.append(
            f"shape: {(sic_src.height, sic_src.width)} vs "
            f"{(cur_src.height, cur_src.width)}")

    for i, (s, c) in enumerate(zip(list(sic_src.transform)[:6],
                                   list(cur_src.transform)[:6])):
        if abs(s - c) > TRANSFORM_TOL:
            problems.append(f"transform[{i}]: {s} vs {c}")

    sb = _bounds(sic_src.transform, sic_src.height, sic_src.width)
    cb = _bounds(cur_src.transform, cur_src.height, cur_src.width)
    if any(abs(a - b) > BOUNDS_TOL for a, b in zip(sb, cb)):
        problems.append(f"extent: {sb} vs {cb}")

    if problems:
        raise GridAlignmentError(
            f"{sic_name} and {cur_name} are not on the same grid:\n  "
            + "\n  ".join(problems)
            + "\n  -> routing needs both fields addressable by the same (row, col); "
              "regrid the currents before routing.")


@dataclass
class RoutingGrid:
    """One day of SIC and currents, aligned, plus the planner's two inputs."""

    transform: Affine
    crs: CRS
    shape: tuple[int, int]
    sic: np.ndarray                     # float32 (H, W): NaN = no data, 0.0 = open water
    current_u: np.ndarray               # float32 (H, W): +x / eastward, m/s
    current_v: np.ndarray               # float32 (H, W): +y / northward, m/s
    blocked_mask: np.ndarray            # bool  (H, W): True = impassable
    cost_grid: np.ndarray | None = None  # float32 (H, W); None until a caller sets it
    day: date | None = None
    sources: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------- loading
    @classmethod
    def for_date(cls, day: date | str,
                 sic_dir: Path = SIC_DIR,
                 currents_dir: Path = CURRENTS_DIR,
                 blocked_mask: np.ndarray | None = None,
                 cost_grid: np.ndarray | None = None) -> "RoutingGrid":
        """Load sic_YYYYMMDD.tif and currents_YYYYMMDD.tif and check they align."""
        d = _as_date(day)
        sic_path = Path(sic_dir) / f"sic_{d:%Y%m%d}.tif"
        cur_path = Path(currents_dir) / f"currents_{d:%Y%m%d}.tif"
        for p in (sic_path, cur_path):
            if not p.exists():
                raise FileNotFoundError(f"Not found: {p}")

        with rasterio.open(sic_path) as s, rasterio.open(cur_path) as c:
            _check_aligned(s, c, sic_path.name, cur_path.name)
            if c.count < 2:
                raise GridAlignmentError(
                    f"{cur_path.name} has {c.count} band(s); expected 2 "
                    f"(band 1 = u / eastward, band 2 = v / northward)")
            transform, crs = s.transform, s.crs
            shape = (int(s.height), int(s.width))
            current_u = c.read(1).astype("float32")
            current_v = c.read(2).astype("float32")

        # SIC goes through the project's single masking rule, not a local one.
        sic = load_sic(sic_path).sic

        return cls(
            transform=transform, crs=crs, shape=shape,
            sic=sic, current_u=current_u, current_v=current_v,
            blocked_mask=cls._as_mask(blocked_mask, shape),
            cost_grid=cls._as_cost(cost_grid, shape),
            day=d,
            sources={"sic": str(sic_path), "currents": str(cur_path)},
        )

    @staticmethod
    def _as_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
        if mask is None:
            return np.zeros(shape, dtype=bool)      # nothing blocked until told
        mask = np.asarray(mask)
        if mask.shape != shape:
            raise GridAlignmentError(
                f"blocked_mask shape {mask.shape} != grid shape {shape}")
        return mask.astype(bool, copy=False)

    @staticmethod
    def _as_cost(cost: np.ndarray | None, shape: tuple[int, int]):
        if cost is None:
            return None
        cost = np.asarray(cost, dtype="float32")
        if cost.shape != shape:
            raise GridAlignmentError(
                f"cost_grid shape {cost.shape} != grid shape {shape}")
        return cost

    # ---------------------------------------------------------- properties
    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def pixel_size(self) -> tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> tuple[float, ...]:
        return _bounds(self.transform, self.height, self.width)

    @property
    def current_speed(self) -> np.ndarray:
        """Derived, not stored: |(u, v)| in m/s."""
        return np.hypot(self.current_u, self.current_v)

    # ------------------------------------------------------- caller inputs
    def set_blocked_mask(self, mask: np.ndarray) -> None:
        self.blocked_mask = self._as_mask(mask, self.shape)

    def set_cost_grid(self, cost: np.ndarray) -> None:
        self.cost_grid = self._as_cost(cost, self.shape)

    def require_cost_grid(self) -> np.ndarray:
        """Fail loudly rather than let a planner run on an unset cost surface."""
        if self.cost_grid is None:
            raise ValueError(
                "cost_grid is not set. This module deliberately does not invent "
                "one; supply a real cost surface via set_cost_grid() before "
                "planning a route.")
        return self.cost_grid

    # -------------------------------------------------------- coordinates
    def xy_to_rowcol(self, x, y):
        """Projected metres -> integer (row, col) of the containing cell.

        floor(), not round(): a cell owns [edge, edge + pixel), so a point
        exactly on a boundary belongs to one cell and only one. Scalars in,
        scalars out; arrays in, integer arrays out. Results are NOT clipped --
        use in_bounds() to test them.
        """
        inv = ~self.transform
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        col = inv.a * x + inv.b * y + inv.c
        row = inv.d * x + inv.e * y + inv.f
        row = np.floor(row).astype(np.intp)
        col = np.floor(col).astype(np.intp)
        return (row.item(), col.item()) if row.ndim == 0 else (row, col)

    def rowcol_to_xy(self, row, col):
        """Integer (row, col) -> projected metres at the CELL CENTRE.

        Same +0.5 convention as preprocess_currents.py, so a round trip through
        xy_to_rowcol returns the cell you started from.
        """
        t = self.transform
        row = np.asarray(row, dtype="float64")
        col = np.asarray(col, dtype="float64")
        x = t.c + (col + 0.5) * t.a + (row + 0.5) * t.b
        y = t.f + (col + 0.5) * t.d + (row + 0.5) * t.e
        return (x.item(), y.item()) if x.ndim == 0 else (x, y)

    def in_bounds(self, row, col):
        """True where (row, col) indexes a real cell."""
        row = np.asarray(row)
        col = np.asarray(col)
        ok = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        return bool(ok) if ok.ndim == 0 else ok

    # ------------------------------------------------------------- summary
    def __repr__(self) -> str:
        cost = "unset" if self.cost_grid is None else "set"
        return (f"RoutingGrid(day={self.day}, {self.shape}, "
                f"{self.crs.to_string() if self.crs else 'no CRS'}, "
                f"{self.pixel_size[0]:g} m, blocked={int(self.blocked_mask.sum()):,}, "
                f"cost_grid={cost})")
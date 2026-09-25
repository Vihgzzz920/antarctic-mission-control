"""
Environmental traversal cost: how difficult each cell is, not how far it is.

    from src.routing.cost import CostParams, build_cost_grid
    result = build_cost_grid(grid, constraints)
    result.apply_to(grid)          # sets cost_grid AND blocks unpriceable cells
    # grid is now ready for astar()

WHAT A COST HERE MEANS
  A dimensionless difficulty multiplier, >= base_cost. astar.py multiplies it
  by the distance travelled, so distance belongs there and only environmental
  difficulty belongs here. A cost of 2.0 means "a metre through this cell costs
  what two metres of clear water cost".

      cost = base_cost + ice_weight * (sic/100)^ice_exponent
                       + current_weight * min(speed / current_reference, 1)

  Blocked and unpriceable cells get +inf.

CURRENTS ARE A MAGNITUDE PENALTY, NEVER A DIRECTIONAL BENEFIT
  A per-cell scalar cost is isotropic by construction: astar.py prices a move
  as distance x the mean of the two cells' costs, which is identical in both
  directions. So this grid CANNOT express "with the current" versus "against
  it", and pretending otherwise would be the most flattering-looking error in
  the whole pipeline -- a route that claims to ride a favourable current while
  the arithmetic knows nothing about heading.

  What the current term can honestly represent is that strong flow is harder
  regardless of heading: more set and drift, more station-keeping, less
  predictable ice motion. That is why current_weight DEFAULTS TO 0.0 -- it is
  opt-in, so nobody ships a number implying directional modelling by accident.
  Real directional current effects need an anisotropic, edge-based cost, which
  is a different cost model inside astar.py, not a different array here.

NO DATA IS NOT FREE WATER
  SIC is NaN over land, over the pole hole and in a one-day swath gap; currents
  are NaN everywhere outside the GLORYS latitude band. A cell whose inputs are
  missing cannot be priced, and this module refuses to invent a value for it.
  Such cells are returned in `no_data_mask` and marked +inf, and apply_to()
  ORs them into blocked_mask -- which is also what keeps this compatible with
  astar.StaticCostModel: that class rejects a non-finite cost on any cell which
  is not blocked.

  Currents are exempt while current_weight == 0.0: if currents do not affect
  the cost, their NaNs must not shrink the navigable area.

This module does not read bathymetry, apply POLARIS rules, guess a land mask,
model a vessel, or know anything about time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                       # keeps this module numpy-only at runtime
    from src.routing.constraints import NavigationConstraints
    from src.routing.grid import RoutingGrid


class CostError(ValueError):
    """The cost grid cannot be built as specified."""


@dataclass(frozen=True)
class CostParams:
    """Every number the model uses. All of them are choices, none are measured."""

    base_cost: float = 1.0          # cost of clear open water; must be > 0
    ice_weight: float = 1.0         # extra cost at 100% SIC
    ice_exponent: float = 1.0       # >1 makes ice bite harder near full cover
    current_weight: float = 0.0     # extra cost at/above current_reference; opt-in
    current_reference: float = 0.5  # m/s at which the current term saturates

    def validate(self) -> None:
        for name in ("base_cost", "ice_exponent", "current_reference"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise CostError(f"{name} must be finite and > 0, got {v!r}")
        for name in ("ice_weight", "current_weight"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise CostError(f"{name} must be finite and >= 0, got {v!r}")


@dataclass
class CostResult:
    """The cost surface plus the cells it could not price."""

    cost: np.ndarray                # float64 (H, W); +inf where unusable
    no_data_mask: np.ndarray        # bool: navigable-looking but unpriceable
    blocked_mask: np.ndarray        # bool: the hard constraints used
    params: CostParams

    @property
    def unusable_mask(self) -> np.ndarray:
        """Everything A* must never enter: hard constraints plus missing data."""
        return self.blocked_mask | self.no_data_mask

    def apply_to(self, grid: "RoutingGrid") -> "CostResult":
        """Write both fields onto the grid, in the order astar.py requires.

        blocked_mask MUST absorb no_data_mask: astar.StaticCostModel rejects a
        non-finite cost on any cell that is not blocked, so an unpriced cell
        left unblocked would fail the search rather than be avoided by it.
        """
        grid.set_blocked_mask(self.unusable_mask)
        grid.set_cost_grid(self.cost)
        return self

    def summary(self) -> dict:
        finite = np.isfinite(self.cost)
        v = self.cost[finite]
        return {
            "shape": list(self.cost.shape),
            "navigable_cells": int(finite.sum()),
            "blocked_cells": int(self.blocked_mask.sum()),
            "no_data_cells": int(self.no_data_mask.sum()),
            "cost_min": float(v.min()) if v.size else None,
            "cost_mean": float(v.mean()) if v.size else None,
            "cost_max": float(v.max()) if v.size else None,
            "params": vars(self.params),
        }

    def report(self) -> str:
        s = self.summary()
        n = self.cost.size
        lines = ["Traversal cost grid:",
                 f"  navigable   {s['navigable_cells']:>12,}  "
                 f"({100 * s['navigable_cells'] / n:5.2f}%)",
                 f"  blocked     {s['blocked_cells']:>12,}  "
                 f"(hard constraints)",
                 f"  no data     {s['no_data_cells']:>12,}  "
                 f"(inputs missing; NOT navigable)"]
        if s["cost_min"] is not None:
            lines.append(f"  cost        min {s['cost_min']:.3f}  "
                         f"mean {s['cost_mean']:.3f}  max {s['cost_max']:.3f}")
        if self.params.current_weight == 0.0:
            lines.append("  -> current_weight is 0: currents do not affect cost, "
                         "and their NaNs do not block cells.")
        return "\n".join(lines)


def build_cost_grid(grid: "RoutingGrid",
                    constraints: "NavigationConstraints | None" = None,
                    params: CostParams | None = None) -> CostResult:
    """Build the environmental cost surface for one RoutingGrid.

    `constraints`, when given, supplies the hard blocked mask; otherwise the
    grid's current blocked_mask is used as-is. Nothing is fabricated either way.
    """
    p = params or CostParams()
    p.validate()

    shape = tuple(grid.shape)
    sic = np.asarray(grid.sic, dtype="float64")
    u = np.asarray(grid.current_u, dtype="float64")
    v = np.asarray(grid.current_v, dtype="float64")
    for name, arr in (("sic", sic), ("current_u", u), ("current_v", v)):
        if arr.shape != shape:
            raise CostError(f"grid.{name} shape {arr.shape} != grid shape {shape}")

    if constraints is not None:
        blocked = np.asarray(constraints.combined(shape), dtype=bool)
    else:
        blocked = np.asarray(grid.blocked_mask, dtype=bool)
    if blocked.shape != shape:
        raise CostError(f"blocked mask shape {blocked.shape} != grid shape {shape}")

    sic_ok = np.isfinite(sic)
    speed = np.hypot(u, v)
    speed_ok = np.isfinite(speed)

    # Only inputs the model actually uses can make a cell unpriceable.
    missing = ~sic_ok
    if p.current_weight > 0:
        missing |= ~speed_ok
    no_data = missing & ~blocked          # blocked cells were already excluded

    # Price the cells we can. Work on filled copies so no NaN reaches the maths;
    # the fills only ever land on cells that end up +inf anyway.
    sic_f = np.where(sic_ok, np.clip(sic, 0.0, 100.0), 0.0)
    cost = np.full(shape, float(p.base_cost), dtype="float64")
    cost += p.ice_weight * np.power(sic_f / 100.0, p.ice_exponent)

    if p.current_weight > 0:
        sp = np.where(speed_ok, speed, 0.0)
        cost += p.current_weight * np.clip(sp / p.current_reference, 0.0, 1.0)

    cost[blocked | no_data] = np.inf

    # The invariant astar.StaticCostModel will check, asserted here where the
    # error can still name the cause.
    usable = np.isfinite(cost)
    if usable.any() and (cost[usable] <= 0).any():
        raise CostError("produced a non-positive traversal cost; check that "
                        "base_cost > 0 and the weights are >= 0.")

    return CostResult(cost=cost, no_data_mask=no_data, blocked_mask=blocked, params=p)


# --------------------------------------------------------------- smoke test
def _smoke_test() -> None:
    """Deterministic checks on synthetic arrays. Writes nothing, saves nothing."""
    from datetime import date

    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from src.routing.astar import StaticCostModel, astar
    from src.routing.constraints import NavigationConstraints
    from src.routing.grid import RoutingGrid

    H = W = 12
    sic = np.linspace(0.0, 100.0, H * W).reshape(H, W)
    sic[0, 0] = np.nan                                   # a swath gap, not land
    cur = np.full((H, W), 0.25)
    g = RoutingGrid(transform=Affine(6250.0, 0, 0, 0, -6250.0, 0),
                    crs=CRS.from_epsg(3976), shape=(H, W),
                    sic=sic.astype("float32"),
                    current_u=cur.astype("float32"),
                    current_v=np.zeros((H, W), "float32"),
                    blocked_mask=np.zeros((H, W), bool), day=date(2025, 1, 15))

    land = np.zeros((H, W), bool)
    land[5, :8] = True
    con = NavigationConstraints(land_mask=land)

    r = build_cost_grid(g, con)
    assert np.isinf(r.cost[5, 0]), "blocked cell must not have a finite cost"
    assert np.isinf(r.cost[0, 0]) and r.no_data_mask[0, 0], "NaN SIC must be unpriceable"
    assert r.no_data_mask.sum() == 1, "currents are NaN-free here; nothing else missing"
    # compare against the value the grid actually stores (float32), not the
    # float64 original, or the round trip fails the tolerance rather than the maths
    assert abs(r.cost[0, 1] - (1.0 + float(g.sic[0, 1]) / 100.0)) < 1e-9, "ice term"
    assert r.cost[-1, -1] > r.cost[0, 1], "higher SIC must cost more"

    r.apply_to(g)
    assert g.blocked_mask[0, 0] and g.blocked_mask[5, 0], "unusable cells must be blocked"
    StaticCostModel.from_grid(g)                          # the compatibility contract
    res = astar(g, (0, 1), (H - 1, W - 1))
    assert res.success and not any(g.blocked_mask[rr, cc] for rr, cc in res.path)

    # current_weight == 0 must not let NaN currents shrink the navigable area
    g2 = RoutingGrid(transform=g.transform, crs=g.crs, shape=(H, W),
                     sic=np.zeros((H, W), "float32"),
                     current_u=np.full((H, W), np.nan, "float32"),
                     current_v=np.full((H, W), np.nan, "float32"),
                     blocked_mask=np.zeros((H, W), bool))
    assert build_cost_grid(g2).no_data_mask.sum() == 0
    assert build_cost_grid(g2, params=CostParams(current_weight=0.5)
                           ).no_data_mask.sum() == H * W

    for bad in (CostParams(base_cost=0.0), CostParams(ice_weight=-1.0),
                CostParams(ice_exponent=0.0), CostParams(current_reference=np.nan)):
        try:
            build_cost_grid(g, con, bad)
            raise AssertionError(f"should have rejected {bad}")
        except CostError:
            pass

    print(r.report())
    print(f"\nastar over the synthetic cost grid: {res!r}")
    print("smoke test OK")


if __name__ == "__main__":
    _smoke_test()
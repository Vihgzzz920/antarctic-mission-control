"""
Baseline spatial A* over the routing grid. 8-connected, static cost.

    from src.routing.astar import astar
    result = astar(grid, start=(600, 500), goal=(900, 800))
    if result.success:
        print(result.total_cost, len(result.path))

WHAT THIS IS NOT
  Static only. Cost depends on where you are, never on when you get there. A
  time-dependent search needs a different closed-set rule (a cell can be worth
  revisiting at a later time), so it is a separate implementation, not a flag
  on this one. The seam for it is the cost model below.

THE COST MODEL IS THE SEAM
  The search itself only ever calls two methods:
      model.step(r0, c0, r1, c1)   cost of one move between adjacent cells
      model.heuristic(r, c, gr, gc) optimistic cost of the rest of the journey
  StaticCostModel implements them from grid.cost_grid. Replacing it with a
  time-aware model later means writing a new class with those two methods, not
  rewriting the search. Anything that is not geometry -- POLARIS, bathymetry,
  fuel, ice drift -- belongs in whatever builds cost_grid, never in here.

ADMISSIBILITY
  A step costs distance x the mean of the two cells' costs, so the cheapest any
  metre of travel can ever be is the smallest valid cell cost on the grid.
  The heuristic is therefore straight-line distance x that minimum, which can
  never overestimate, so the path A* returns is optimal for this cost model.
  A replacement model that overestimates gives up that guarantee.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from math import hypot
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.routing.grid import RoutingGrid

# (d_row, d_col) for 8-connected movement.
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1), (1, 0), (1, 1))


class AStarError(ValueError):
    """The search cannot be run as specified."""


# --------------------------------------------------------------- cost model
@dataclass
class StaticCostModel:
    """Per-cell traversal cost that does not change with time.

    A move costs the geometric distance between the two cell centres times the
    mean of their costs. Symmetric by construction, so the graph is undirected
    and the same route costs the same in either direction.
    """

    cost: np.ndarray            # float64 (H, W), validated where not blocked
    dx: float                   # pixel width, metres
    dy: float                   # pixel height, metres
    min_cost: float             # cheapest valid cell; scales the heuristic

    @classmethod
    def from_grid(cls, grid: "RoutingGrid") -> "StaticCostModel":
        cost = np.asarray(grid.require_cost_grid(), dtype="float64")
        if cost.shape != tuple(grid.shape):
            raise AStarError(
                f"cost_grid shape {cost.shape} != grid shape {tuple(grid.shape)}")

        # Only cells a route may actually enter have to be valid. Blocked cells
        # are never stepped on, so NaN there is fine and common.
        open_cells = ~np.asarray(grid.blocked_mask, dtype=bool)
        vals = cost[open_cells]
        if vals.size == 0:
            raise AStarError("every cell is blocked; there is nothing to search.")

        bad_nonfinite = int((~np.isfinite(vals)).sum())
        bad_nonpositive = int((np.isfinite(vals) & (vals <= 0)).sum())
        if bad_nonfinite or bad_nonpositive:
            where = np.argwhere(open_cells & (~np.isfinite(cost) | (cost <= 0)))
            raise AStarError(
                f"cost_grid has invalid traversal costs on navigable cells: "
                f"{bad_nonfinite:,} non-finite, {bad_nonpositive:,} <= 0 "
                f"(of {vals.size:,} navigable). First at (row, col) = "
                f"{tuple(int(v) for v in where[0])}. Traversal cost must be "
                f"finite and strictly positive; block the cell instead of "
                f"pricing it at zero or infinity.")

        dx, dy = grid.pixel_size
        return cls(cost=cost, dx=float(dx), dy=float(dy), min_cost=float(vals.min()))

    def distance(self, r0: int, c0: int, r1: int, c1: int) -> float:
        """Metres between two cell centres. Diagonals are sqrt(dx^2+dy^2)."""
        return hypot((c1 - c0) * self.dx, (r1 - r0) * self.dy)

    def step(self, r0: int, c0: int, r1: int, c1: int) -> float:
        return self.distance(r0, c0, r1, c1) * 0.5 * (
            self.cost[r0, c0] + self.cost[r1, c1])

    def heuristic(self, r: int, c: int, gr: int, gc: int) -> float:
        return self.min_cost * hypot((gc - c) * self.dx, (gr - r) * self.dy)


# -------------------------------------------------------------------- result
@dataclass
class AStarResult:
    """Everything a later comparison or plot needs, and nothing more."""

    success: bool
    path: list[tuple[int, int]] = field(default_factory=list)
    total_cost: float = float("inf")
    expanded_nodes: int = 0
    reason: str = ""
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None

    def __repr__(self) -> str:
        if self.success:
            return (f"AStarResult(success, {len(self.path)} cells, "
                    f"cost={self.total_cost:.3f}, expanded={self.expanded_nodes:,})")
        return (f"AStarResult(FAILED: {self.reason}, "
                f"expanded={self.expanded_nodes:,})")


# -------------------------------------------------------------- validation
def _check_endpoint(grid: "RoutingGrid", rc, label: str) -> tuple[int, int]:
    try:
        r, c = int(rc[0]), int(rc[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise AStarError(f"{label} must be a (row, col) pair, got {rc!r}") from exc
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        raise AStarError(
            f"{label} (row, col) = ({r}, {c}) is outside the grid {(h, w)}")
    if bool(grid.blocked_mask[r, c]):
        raise AStarError(
            f"{label} (row, col) = ({r}, {c}) is on a blocked cell; a route "
            f"cannot begin or end inside a hard constraint.")
    return r, c


# ------------------------------------------------------------------ search
def astar(grid: "RoutingGrid",
          start: tuple[int, int],
          goal: tuple[int, int],
          cost_model: StaticCostModel | None = None,
          allow_corner_cutting: bool = False) -> AStarResult:
    """Least-cost 8-connected path from start to goal over grid.cost_grid.

    cost_model defaults to StaticCostModel.from_grid(grid), which raises if
    grid.cost_grid is unset -- no cost is ever invented here.

    allow_corner_cutting=False (the default) forbids a diagonal move when BOTH
    of the orthogonal cells it passes between are blocked, i.e. squeezing
    through the corner where two obstacles touch. That is a legal graph edge
    and a physically impossible transit.
    """
    s = _check_endpoint(grid, start, "start")
    g = _check_endpoint(grid, goal, "goal")
    model = cost_model or StaticCostModel.from_grid(grid)

    h, w = grid.shape
    blocked = np.asarray(grid.blocked_mask, dtype=bool)

    if s == g:
        return AStarResult(success=True, path=[s], total_cost=0.0,
                           expanded_nodes=0, reason="start == goal",
                           start=s, goal=g)

    # Flat (row * w + col) indexing keeps the arrays compact on a 1.7M-cell grid.
    g_score = np.full(h * w, np.inf, dtype="float64")
    came_from = np.full(h * w, -1, dtype="int64")
    closed = np.zeros(h * w, dtype=bool)

    s_flat, g_flat = s[0] * w + s[1], g[0] * w + g[1]
    g_score[s_flat] = 0.0

    counter = 0                      # deterministic tie-break, never compares nodes
    heap = [(model.heuristic(s[0], s[1], g[0], g[1]), counter, s_flat)]
    expanded = 0

    while heap:
        _, _, current = heapq.heappop(heap)
        if closed[current]:
            continue                 # a stale duplicate left behind by a better path
        closed[current] = True
        expanded += 1

        if current == g_flat:
            path, node = [], current
            while node != -1:
                path.append((node // w, node % w))
                node = came_from[node]
            path.reverse()
            return AStarResult(success=True, path=path,
                               total_cost=float(g_score[g_flat]),
                               expanded_nodes=expanded, reason="reached goal",
                               start=s, goal=g)

        r, c = current // w, current % w
        g_here = g_score[current]

        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc]:
                continue
            if dr and dc and not allow_corner_cutting:
                if blocked[r, nc] and blocked[nr, c]:
                    continue         # both shoulders blocked: no transit through
            nxt = nr * w + nc
            if closed[nxt]:
                continue

            tentative = g_here + model.step(r, c, nr, nc)
            if tentative < g_score[nxt]:
                g_score[nxt] = tentative
                came_from[nxt] = current
                counter += 1
                heapq.heappush(
                    heap,
                    (tentative + model.heuristic(nr, nc, g[0], g[1]), counter, nxt))

    return AStarResult(success=False, expanded_nodes=expanded,
                       reason="goal is unreachable: every route from start is "
                              "closed off by blocked cells",
                       start=s, goal=g)
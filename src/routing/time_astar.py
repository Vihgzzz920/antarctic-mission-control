"""
Time-dependent A*: the cost of a cell is evaluated at the time the ship gets there.

    from src.routing.time_astar import time_astar
    res = time_astar(grid, (10, 3), (10, 40),
                     cost_fn=my_forecast_cost,      # (row, col, t_seconds) -> float
                     min_cost=1.0,                  # true lower bound of cost_fn
                     vessel_speed_mps=5.0,
                     time_step_s=3600.0,
                     max_horizon_s=10 * 86400.0)

This is a SEPARATE implementation from astar.py, not a flag on it. A static
search may close a cell forever the first time it is settled, because nothing
about that cell can ever change. Here a cell reached two hours later may sit
under different ice, so closing on (row, col) alone would silently throw away
the better route.

SEARCH STATE: LABELS, AND NO CROSS-TIME DOMINANCE
  Each state is a label (row, col, arrival_time, cost_so_far) and every cell
  keeps a SET of labels rather than a single best value.

  There is deliberately NO rule discarding a label because another reached the
  same cell no later and for no more. That rule (t' <= t and g' <= g) is valid
  when minimising TIME -- constant speed means leaving earlier never means
  arriving later -- but it is NOT valid for an arbitrary time-dependent cost
  field, and an earlier version of this file applied it anyway. Counterexample:

      label A reaches cell X at t = 1 h having spent 10
      label B reaches cell X at t = 5 h having spent 20
      the only onward cell costs 1000 before t = 4 h and 1 after

  A is both earlier and cheaper, so that rule discards B -- yet A must move on
  immediately and pays 1000, while B would have paid 1. This search has no
  WAITING action, so an early label cannot reproduce a late label's schedule
  and therefore has no claim on its future. Discarding B loses the optimum.

  The only cross-time comparison that would be sound is between labels with
  EQUAL arrival times, and those necessarily share a bucket, so it is already
  covered below. Nothing else is pruned on the basis of time.

WHERE THE APPROXIMATION LIVES -- read this before choosing time_step_s
  With no cross-time dominance, labels would grow without bound, so exactly one
  bound is imposed: AT MOST ONE LABEL PER (cell, time bucket), where

      bucket = floor((arrival_time - start_time) / time_step_s)

  When two labels land on the same cell in the same bucket, only the cheaper is
  kept. This is the ONLY pruning the search performs, and it IS AN
  APPROXIMATION, not a dominance claim: the discarded label arrived at a
  different instant within the bucket and would have met a different forecast.
  The one case where it is exact is two labels with the SAME arrival time --
  identical futures, so the cheaper genuinely dominates.

  Choose time_step_s no coarser than the timescale on which the forecast
  actually moves. Against a daily forecast a 24 h bucket erases the point of
  the exercise. Finer buckets are more faithful and cost memory and time in
  direct proportion; time_step_s is the single dial trading one for the other.

  Arrival times themselves are never quantised. They are carried and reported
  as exact floats; the bucket is only a bookkeeping device for the label cap.

  Floor, not round: consistent with the cell binning elsewhere in the project.

STALE HEAP ENTRIES DIE BY IDENTITY
  Every label gets a unique id and the heap carries ids, not coordinates. A
  label retired by the bucket cap is removed from the live set, so when its
  heap entry surfaces it is dropped. Correctness therefore does not rest on an
  argument about the order in which equal-keyed entries happen to pop.

BOUNDS ARE MANDATORY, NOT OPTIONAL
  time_step_s and max_horizon_s are required with no defaults, because the right
  values depend on the forecast and the voyage and this module has no basis to
  guess either. Labels arriving after start_time + max_horizon_s are discarded;
  max_expansions caps the work regardless. Labels per cell are bounded by
  max_horizon_s / time_step_s.

THE FORECAST IS ENTIRELY THE CALLER'S
  cost_fn is injected and is the ONLY source of environmental cost. This module
  never reads grid.cost_grid, never interpolates a forecast, never extrapolates
  past the end of one, and holds no opinion about ice, currents, POLARIS,
  bathymetry, fuel or vessel behaviour. A non-finite return means the cell is
  impassable at that time.

  min_cost is the caller's promise that cost_fn never returns less than it. The
  heuristic is straight-line distance x min_cost, admissible only if that
  promise holds, so every evaluation is checked against it by default.

SPEED IS CONSTANT HERE ON PURPOSE
  dt = distance / vessel_speed_mps. A speed that degrades in ice is a vessel
  performance model; it belongs in whatever owns the vessel, not in the search.
  Note that constant speed does NOT license cross-time dominance; see above.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from math import hypot
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from src.routing.grid import RoutingGrid

# Same 8-connected neighbourhood and ordering as astar.py.
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1), (1, 0), (1, 1))

CostFn = Callable[[int, int, float], float]


class TimeAStarError(ValueError):
    """The time-dependent search cannot be run as specified."""


@dataclass
class TimeAStarResult:
    """Everything a later comparison, plot or schedule needs."""

    success: bool
    path: list[tuple[int, int]] = field(default_factory=list)
    arrival_times: list[float] = field(default_factory=list)   # seconds, same order
    total_cost: float = float("inf")
    expanded_nodes: int = 0
    reason: str = ""
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None
    start_time: float = 0.0
    goal_arrival_time: float | None = None
    labels_created: int = 0
    labels_retired: int = 0

    @property
    def duration_s(self) -> float | None:
        if self.goal_arrival_time is None:
            return None
        return self.goal_arrival_time - self.start_time

    def __repr__(self) -> str:
        if self.success:
            return (f"TimeAStarResult(success, {len(self.path)} cells, "
                    f"cost={self.total_cost:.3f}, "
                    f"duration={self.duration_s / 3600.0:.2f} h, "
                    f"expanded={self.expanded_nodes:,}, "
                    f"labels={self.labels_created:,})")
        return (f"TimeAStarResult(FAILED: {self.reason}, "
                f"expanded={self.expanded_nodes:,})")


def _check_endpoint(grid: "RoutingGrid", rc, label: str) -> tuple[int, int]:
    try:
        r, c = int(rc[0]), int(rc[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise TimeAStarError(f"{label} must be a (row, col) pair, got {rc!r}") from exc
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        raise TimeAStarError(
            f"{label} (row, col) = ({r}, {c}) is outside the grid {(h, w)}")
    if bool(grid.blocked_mask[r, c]):
        raise TimeAStarError(
            f"{label} (row, col) = ({r}, {c}) is on a blocked cell; a route "
            f"cannot begin or end inside a hard constraint.")
    return r, c


def time_astar(grid: "RoutingGrid",
               start: tuple[int, int],
               goal: tuple[int, int],
               *,
               cost_fn: CostFn,
               min_cost: float,
               vessel_speed_mps: float,
               time_step_s: float,
               max_horizon_s: float,
               start_time: float = 0.0,
               allow_corner_cutting: bool = False,
               max_expansions: int = 5_000_000,
               validate_lower_bound: bool = True) -> TimeAStarResult:
    """Least-cost 8-connected path where cost is sampled at the arrival time.

    An edge from A to B costs

        distance(A, B) * 0.5 * (cost_fn(A, t_depart) + cost_fn(B, t_arrive))

    which is the time-dependent reading of the static rule in astar.py: each
    cell is priced when the ship is actually in it.
    """
    for name, v in (("min_cost", min_cost), ("vessel_speed_mps", vessel_speed_mps),
                    ("time_step_s", time_step_s), ("max_horizon_s", max_horizon_s)):
        if not np.isfinite(v) or v <= 0:
            raise TimeAStarError(f"{name} must be finite and > 0, got {v!r}")
    if max_expansions <= 0:
        raise TimeAStarError(f"max_expansions must be > 0, got {max_expansions!r}")
    if not callable(cost_fn):
        raise TimeAStarError("cost_fn must be callable as cost_fn(row, col, time)")

    s = _check_endpoint(grid, start, "start")
    g = _check_endpoint(grid, goal, "goal")

    h, w = grid.shape
    blocked = np.asarray(grid.blocked_mask, dtype=bool)
    dx, dy = grid.pixel_size
    deadline = start_time + max_horizon_s

    def evaluate(r: int, c: int, t: float) -> float:
        """One forecast sample, with the caller's lower-bound promise enforced."""
        v = float(cost_fn(r, c, t))
        if not np.isfinite(v):
            return float("inf")          # impassable at this time; not an error
        if v <= 0:
            raise TimeAStarError(
                f"cost_fn returned {v!r} at (row, col, t) = ({r}, {c}, {t}); "
                f"traversal cost must be strictly positive or non-finite "
                f"(non-finite meaning impassable at that time).")
        if validate_lower_bound and v < min_cost:
            raise TimeAStarError(
                f"cost_fn returned {v!r} at (row, col, t) = ({r}, {c}, {t}), "
                f"below the declared min_cost {min_cost!r}. The heuristic would "
                f"overestimate and the route would no longer be optimal. Lower "
                f"min_cost, or pass validate_lower_bound=False if you knowingly "
                f"accept a suboptimal path.")
        return v

    def heuristic(r: int, c: int) -> float:
        return min_cost * hypot((g[1] - c) * dx, (g[0] - r) * dy)

    def bucket(t: float) -> int:
        return int((t - start_time) // time_step_s)

    if s == g:
        return TimeAStarResult(success=True, path=[s], arrival_times=[start_time],
                               total_cost=0.0, expanded_nodes=0,
                               reason="start == goal", start=s, goal=g,
                               start_time=start_time, goal_arrival_time=start_time,
                               labels_created=1)

    # ---- label store -------------------------------------------------------
    # One label = one (cell, arrival_time, cost). Cells keep a SET of labels.
    lab_rc: list[tuple[int, int]] = []
    lab_t: list[float] = []
    lab_g: list[float] = []
    lab_parent: list[int] = []
    alive: set[int] = set()                    # not retired by the bucket cap
    settled: set[int] = set()                  # already expanded
    # per cell: bucket -> label id. One label per bucket is the ONLY pruning.
    by_cell: dict[tuple[int, int], dict[int, int]] = {}
    retired = 0

    def add_label(r: int, c: int, t: float, gg: float, parent: int) -> int | None:
        """Insert a label, subject only to the one-label-per-bucket cap.

        NO cross-time dominance: a label is never discarded merely because
        another arrived earlier and cost less. Without a waiting action the
        later label may be the only one that meets a future cheap window, so
        that comparison is unsound (see the module docstring). The sole pruning
        is within a bucket, where the cheaper label wins -- exact when the two
        arrival times are equal, an approximation otherwise.

        Returns the new label id, or None when the candidate lost its bucket.
        """
        nonlocal retired
        slots = by_cell.setdefault((r, c), {})
        b = bucket(t)
        held = slots.get(b)
        if held is not None:
            if lab_g[held] <= gg:
                return None
            alive.discard(held)
            retired += 1

        new_id = len(lab_rc)
        lab_rc.append((r, c)); lab_t.append(t); lab_g.append(gg)
        lab_parent.append(parent)
        alive.add(new_id)
        slots[b] = new_id
        return new_id

    root = add_label(s[0], s[1], start_time, 0.0, -1)
    counter = 0
    heap = [(heuristic(*s), counter, root)]
    expanded = 0
    horizon_pruned = 0

    while heap:
        _, _, lid = heapq.heappop(heap)
        # Stale entries die by identity, not by heap-order argument.
        if lid not in alive or lid in settled:
            continue
        settled.add(lid)
        expanded += 1

        r, c = lab_rc[lid]
        t, g_here = lab_t[lid], lab_g[lid]

        if (r, c) == g:
            chain = []
            node = lid
            while node != -1:
                chain.append(node)
                node = lab_parent[node]
            chain.reverse()
            return TimeAStarResult(
                success=True,
                path=[lab_rc[i] for i in chain],
                arrival_times=[lab_t[i] for i in chain],
                total_cost=float(g_here),
                expanded_nodes=expanded,
                reason="reached goal",
                start=s, goal=g, start_time=start_time,
                goal_arrival_time=t,
                labels_created=len(lab_rc), labels_retired=retired)

        if expanded >= max_expansions:
            return TimeAStarResult(
                success=False, expanded_nodes=expanded,
                reason=f"gave up after max_expansions={max_expansions:,} without "
                       f"reaching the goal",
                start=s, goal=g, start_time=start_time,
                labels_created=len(lab_rc), labels_retired=retired)

        cost_here = evaluate(r, c, t)
        if not np.isfinite(cost_here):
            continue                      # this cell is impassable at this time

        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc]:
                continue
            if dr and dc and not allow_corner_cutting:
                if blocked[r, nc] and blocked[nr, c]:
                    continue              # both shoulders blocked: no transit

            dist = hypot(dc * dx, dr * dy)
            t_arrive = t + dist / vessel_speed_mps
            if t_arrive > deadline:
                horizon_pruned += 1
                continue

            cost_there = evaluate(nr, nc, t_arrive)
            if not np.isfinite(cost_there):
                continue                  # impassable at the time we would arrive

            gg = g_here + dist * 0.5 * (cost_here + cost_there)
            new_id = add_label(nr, nc, t_arrive, gg, lid)
            if new_id is None:
                continue
            counter += 1
            heapq.heappush(heap, (gg + heuristic(nr, nc), counter, new_id))

    reason = "goal is unreachable: every route from start is closed off"
    if horizon_pruned:
        reason += (f" or falls beyond max_horizon_s={max_horizon_s:g} s "
                   f"({horizon_pruned:,} expansions pruned by the horizon)")
    return TimeAStarResult(success=False, expanded_nodes=expanded, reason=reason,
                           start=s, goal=g, start_time=start_time,
                           labels_created=len(lab_rc), labels_retired=retired)


# --------------------------------------------------------------- smoke test
def _smoke_test() -> None:
    """Deterministic synthetic checks. Writes nothing, touches no project data."""
    from datetime import date

    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from src.routing.grid import RoutingGrid

    px, HOUR = 10_000.0, 3600.0                     # 10 km cells, tidy arithmetic
    SPEED = px / HOUR                               # 1 cell per hour orthogonally

    def make(h: int, w: int, blocked=None) -> "RoutingGrid":
        return RoutingGrid(
            transform=Affine(px, 0, 0, 0, -px, 0), crs=CRS.from_epsg(3976),
            shape=(h, w), sic=np.zeros((h, w), "float32"),
            current_u=np.zeros((h, w), "float32"),
            current_v=np.zeros((h, w), "float32"),
            blocked_mask=np.zeros((h, w), bool) if blocked is None else blocked,
            day=date(2025, 1, 15))

    # ---- 1. NO CROSS-TIME DOMINANCE, verified against brute force ----------
    # A gate column that is shut early and opens at 6 h. The direct run reaches
    # it at 5 h and pays 1000; the optimum burns time and slips through after it
    # opens. At cell (1,4) the direct route produces an EARLIER and CHEAPER
    # label than the time-burning route -- the exact pair the old dominance rule
    # collapsed, which made it return a route 78.7x more expensive. The answer
    # is checked against an exhaustive search, so a label wrongly discarded
    # anywhere shows up as a mismatch.
    H, W = 3, 6
    gate_grid = make(H, W)
    GATE_OPENS = 6.0 * HOUR

    def gate_cost(r: int, c: int, t: float) -> float:
        if c == W - 1:
            return 1.0 if t >= GATE_OPENS else 1000.0
        return 1.0

    gate = time_astar(gate_grid, (1, 0), (1, W - 1), cost_fn=gate_cost,
                      min_cost=1.0, vessel_speed_mps=SPEED,
                      time_step_s=HOUR, max_horizon_s=48 * HOUR)
    assert gate.success

    best = [float("inf"), None]

    def brute(r, c, t, gg, depth):
        if gg >= best[0]:
            return                                  # branch and bound
        if (r, c) == (1, W - 1):
            best[0], best[1] = gg, t
            return
        if depth == 0:
            return
        ch = gate_cost(r, c, t)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                d = hypot(dc * px, dr * px)
                ta = t + d / SPEED
                if ta > 48 * HOUR:
                    continue
                brute(nr, nc, ta, gg + d * 0.5 * (ch + gate_cost(nr, nc, ta)),
                      depth - 1)

    brute(1, 0, 0.0, 0.0, 7)
    assert abs(gate.total_cost - best[0]) < 1e-6, (
        f"not optimal: search {gate.total_cost} vs exhaustive {best[0]} -- a "
        f"label was discarded that should have been kept")
    assert gate.goal_arrival_time >= GATE_OPENS, (
        "the optimum does not wait out the gate; scenario no longer exercises "
        "the case that cross-time dominance breaks")
    assert gate.goal_arrival_time > (W - 1) * HOUR, "took the direct early route"

    # ---- 2. a route around a wall, with a forecast that changes ------------
    H2, W2 = 9, 24
    wall = np.zeros((H2, W2), bool)
    wall[0:6, 12] = True                            # gap only at rows 6+
    grid = make(H2, W2, wall)

    def cost_fn(r: int, c: int, t: float) -> float:
        if c <= 12:
            return 1.0
        return 1.0 if t >= 10 * HOUR else 9.0

    common = dict(cost_fn=cost_fn, min_cost=1.0, vessel_speed_mps=SPEED,
                  time_step_s=HOUR, max_horizon_s=48 * HOUR)
    res = time_astar(grid, (0, 0), (0, 23), **common)
    assert res.success, res.reason
    assert res.path[0] == (0, 0) and res.path[-1] == (0, 23)
    assert len(res.arrival_times) == len(res.path), "one arrival time per cell"
    assert res.arrival_times[0] == 0.0
    assert all(b > a for a, b in zip(res.arrival_times, res.arrival_times[1:])), \
        "arrival times must increase along the path"
    assert not any(wall[r, c] for r, c in res.path), "path entered a blocked cell"
    assert res.duration_s <= 48 * HOUR
    assert all(r >= 6 for r, c in res.path if c == 12), "did not use the gap"

    # arrival times are EXACT distance/speed, never snapped to the bucket grid
    for (r0, c0), (r1, c1), t0, t1 in zip(res.path, res.path[1:],
                                          res.arrival_times, res.arrival_times[1:]):
        expect = hypot((c1 - c0) * px, (r1 - r0) * px) / SPEED
        assert abs((t1 - t0) - expect) < 1e-9, "arrival time is not distance / speed"
    assert any(abs(t / HOUR - round(t / HOUR)) > 1e-6 for t in res.arrival_times), \
        "no fractional arrival hour: exact diagonal timing not exercised"

    # freezing the forecast at departure must give a strictly worse total
    frozen = time_astar(grid, (0, 0), (0, 23),
                        **{**common, "cost_fn": lambda r, c, t: cost_fn(r, c, 0.0)})
    assert frozen.success and res.total_cost < frozen.total_cost, (
        f"time dependence had no effect: {res.total_cost} vs {frozen.total_cost}")

    # ---- 3. two arrival times, one cell, one time bucket -------------------
    # min_cost is tiny here ON PURPOSE: still an admissible lower bound for a
    # cost_fn returning 1.0, but weak enough that the search explores broadly
    # instead of running straight down the diagonal and never generating the
    # second arrival at all.
    seen: dict[tuple[int, int], set[float]] = {}

    def recording(r: int, c: int, t: float) -> float:
        seen.setdefault((r, c), set()).add(round(t, 6))
        return 1.0

    coarse = time_astar(make(6, 6), (0, 0), (5, 5), cost_fn=recording,
                        min_cost=1e-9, vessel_speed_mps=SPEED,
                        time_step_s=4 * HOUR, max_horizon_s=48 * HOUR)
    assert coarse.success
    diag, ortho = round(hypot(px, px) / SPEED, 6), round(2 * px / SPEED, 6)
    assert {diag, ortho} <= seen.get((1, 1), set()), (
        f"both arrival times at (1,1) should have been generated; saw "
        f"{sorted(seen.get((1, 1), set()))}")
    assert int(diag // (4 * HOUR)) == int(ortho // (4 * HOUR)) == 0, \
        "the two arrivals are no longer in the same bucket; test is meaningless"
    assert all(b > a for a, b in zip(coarse.arrival_times, coarse.arrival_times[1:]))
    assert abs(coarse.arrival_times[-1] - coarse.goal_arrival_time) < 1e-12

    # ---- 4. the bucket cap is the only pruning, and it is real -------------
    def windowed(r: int, c: int, t: float) -> float:
        return 1.0 if (int(t // HOUR) % 3 == 0) else 4.0   # a window that shuts

    wgrid = make(7, 7)
    runs = {}
    for step in (4 * HOUR, 0.25 * HOUR):
        runs[step] = time_astar(wgrid, (0, 0), (6, 6), cost_fn=windowed,
                                min_cost=1.0, vessel_speed_mps=SPEED,
                                time_step_s=step, max_horizon_s=72 * HOUR)
        assert runs[step].success
    assert runs[0.25 * HOUR].labels_created > runs[4 * HOUR].labels_created, (
        "finer buckets did not retain more labels; the one-label-per-bucket cap "
        "is not doing what the docstring claims")
    assert runs[4 * HOUR].labels_retired > 0, "no label was ever retired"

    # ---- 5. guards ---------------------------------------------------------
    short = time_astar(grid, (0, 0), (0, 23), **{**common, "max_horizon_s": 3 * HOUR})
    assert not short.success and "horizon" in short.reason
    assert short.expanded_nodes < 5000, "horizon must bound the work"

    for a, b, why in (((0, 12), (0, 23), "blocked start"),
                      ((0, 0), (0, 12), "blocked goal"),
                      ((-1, 0), (0, 23), "start out of bounds"),
                      ((0, 0), (9, 99), "goal out of bounds")):
        try:
            time_astar(grid, a, b, **common)
            raise AssertionError(f"should have rejected {why}")
        except TimeAStarError:
            pass
    for bad in ({"time_step_s": 0.0}, {"max_horizon_s": -1.0},
                {"vessel_speed_mps": 0.0}, {"min_cost": np.nan}):
        try:
            time_astar(grid, (0, 0), (0, 23), **{**common, **bad})
            raise AssertionError(f"should have rejected {bad}")
        except TimeAStarError:
            pass
    try:
        time_astar(grid, (0, 0), (0, 23),
                   **{**common, "cost_fn": lambda r, c, t: 0.1})
        raise AssertionError("should have caught the lower-bound violation")
    except TimeAStarError:
        pass

    def closed_early(r: int, c: int, t: float) -> float:
        return float("inf") if (c == 13 and t < 4 * HOUR) else 1.0
    late = time_astar(grid, (0, 0), (0, 23), **{**common, "cost_fn": closed_early})
    assert late.success, "a temporarily impassable cell should not fail the search"

    print(f"gate scenario  : {gate!r}")
    print(f"  cost {gate.total_cost:,.1f} == exhaustive optimum {best[0]:,.1f}; "
          f"arrives {gate.goal_arrival_time / HOUR:.4f} h, gate opens "
          f"{GATE_OPENS / HOUR:.0f} h")
    print(f"  (with the removed cross-time rule this returned 5,045,000.0 -- 78.7x worse)")
    print(f"wall detour    : {res!r}")
    print(f"frozen at t=0  : {frozen!r}  -> {frozen.total_cost:,.0f} vs "
          f"{res.total_cost:,.0f}")
    print(f"same bucket    : (1,1) generated at {diag / HOUR:.4f} h AND "
          f"{ortho / HOUR:.4f} h, both in bucket 0")
    print(f"bucket cap     : 4.00 h -> {runs[4 * HOUR].labels_created:,} labels, "
          f"0.25 h -> {runs[0.25 * HOUR].labels_created:,} labels")
    print(f"short horizon  : {short!r}")
    print("smoke test OK")


if __name__ == "__main__":
    _smoke_test()
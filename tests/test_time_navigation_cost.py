"""
Focused tests for the time-indexed navigation-cost adapter, plus two
integration tests that drive the UNMODIFIED src/routing/time_astar.py.

Every composed field in here is produced by the real
navigation_cost.compose_grid over small synthetic USNIC rasters, so the
adapter is exercised against genuine GridComposition objects and no cost
formula is restated in the test either.

    python tests/test_time_navigation_cost.py
"""
from __future__ import annotations

import ast
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.rasterize_sigrid3 import (AMBIGUITY_TIF, COVERAGE_TIF,          # noqa: E402
                                        DOMINANT_FRACTION_TIF, FRACTION_NODATA,
                                        POLYGON_ID_NODATA, POLYGON_ID_TIF,
                                        WATER_TIF)
from src.routing.navigation_cost import (compose_grid,                        # noqa: E402
                                         load_composition_config)
from src.routing.time_astar import time_astar                                 # noqa: E402
from src.routing.time_navigation_cost import (BRANCHES, ComposedField,        # noqa: E402
                                              ForecastUnavailable,
                                              TimeCostError,
                                              TimeIndexedNavigationCost)

CONFIG = ROOT / "configs" / "navigation_cost_composition.json"
MODULE = ROOT / "src" / "routing" / "time_navigation_cost.py"

EPSG = 3976
PX = 10_000.0                    # 10 km cells -> tidy hours
HOUR = 3600.0
SPEED = PX / HOUR                # exactly one cell per hour orthogonally
TF = (PX, 0.0, 0.0, 0.0, -PX, 0.0)

DAY = date(2025, 1, 15)          # the environmental layers' date
CHART_2025 = date(2025, 1, 15)   # an aligned chart
CHART_2020 = date(2020, 12, 24)  # the real ANTARC20201224 chart date

_DIRS: dict = {}


# --------------------------------------------------------------- fixtures
def _usnic_dir(key: str, height: int, width: int, layout: dict) -> Path:
    """Write a tiny synthetic USNIC raster set; cached per key."""
    if key in _DIRS:
        return _DIRS[key]
    d = Path(tempfile.mkdtemp(prefix=f"usnic_{key}_"))
    prof = dict(driver="GTiff", height=height, width=width, count=1,
                crs=CRS.from_epsg(EPSG), transform=Affine(*TF))
    for name, k, dt, nd in ((COVERAGE_TIF, "coverage", "uint8", None),
                            (POLYGON_ID_TIF, "polygon_id", "int32", POLYGON_ID_NODATA),
                            (WATER_TIF, "water", "uint8", None),
                            (AMBIGUITY_TIF, "ambiguity", "uint8", None),
                            (DOMINANT_FRACTION_TIF, "fraction", "float32",
                             FRACTION_NODATA)):
        with rasterio.open(d / name, "w", dtype=dt, nodata=nd, **prof) as dst:
            dst.write(np.asarray(layout[k], dtype=dt), 1)
    _DIRS[key] = d
    return d


# ---- the 1 x 3 unit-test chart: uncharted | ice poly 10 | ice poly 11 ----
SMALL = dict(coverage=[[0, 1, 1]],
             polygon_id=[[POLYGON_ID_NODATA, 10, 11]],
             water=[[0, 0, 0]],
             ambiguity=[[0, 0, 0]],
             fraction=[[FRACTION_NODATA, 1.0, 1.0]])
SMALL_PEN = {
    10: dict(conservative=25.0, optimistic=0.0,
             classification="elevated_operational_risk", has_rio=True, is_water=False),
    11: dict(conservative=100.0, optimistic=0.0,
             classification="indeterminate", has_rio=True, is_water=False),
}
ENV_EARLY = np.array([[1.0, 2.0, 3.0]], dtype="float64")
ENV_LATE = np.array([[1.0, 20.0, 30.0]], dtype="float64")


def _field(bucket: int, env: np.ndarray, chart: date = CHART_2025,
           env_date: date = DAY, label: str = "", key: str = "small",
           shape: tuple[int, int] = (1, 3), layout: dict | None = None,
           transform: tuple = TF, epsg: int = EPSG,
           penalties: dict | None = None) -> ComposedField:
    d = _usnic_dir(key, shape[0], shape[1], layout or SMALL)
    comp = compose_grid(env, SMALL_PEN if penalties is None else penalties,
                        load_composition_config(CONFIG), d)
    return ComposedField(bucket=bucket, composition=comp, environment_date=env_date,
                         chart_date=chart, epsg=epsg, shape=shape,
                         transform=transform, label=label or f"bucket {bucket}")


def two_bucket_provider(**kw) -> TimeIndexedNavigationCost:
    """bucket 0 = the early field, bucket 1 = a different, later field."""
    return TimeIndexedNavigationCost.from_fields(
        [_field(0, ENV_EARLY, label="early"), _field(1, ENV_LATE, label="late")],
        bucket_seconds=HOUR, **kw)


# ============================================================ unit tests
def test_bucket_lookup_is_floor_of_elapsed_time():
    p = two_bucket_provider()
    assert p.bucket_for(0.0) == 0
    assert p.bucket_for(HOUR - 1e-9) == 0
    assert p.bucket_for(HOUR) == 1, "the boundary belongs to the LATER bucket"
    assert p.bucket_for(2 * HOUR - 1) == 1
    off = TimeIndexedNavigationCost.from_fields(
        [_field(0, ENV_EARLY), _field(1, ENV_LATE)],
        bucket_seconds=HOUR, start_time=1000.0)
    assert off.bucket_for(1000.0) == 0 and off.bucket_for(1000.0 + HOUR) == 1
    return "floor((t - start_time)/bucket_seconds); boundary goes to the later bucket"


def test_lookup_uses_arrival_time_not_departure_or_start_time():
    p = two_bucket_provider()
    fn = p.cost_fn("conservative")
    t_depart, t_arrive = 0.5 * HOUR, 1.5 * HOUR      # depart bucket 0, arrive bucket 1
    early = float(p.fields[0].cost("conservative")[0, 1])     # 2 + 25
    late = float(p.fields[1].cost("conservative")[0, 1])      # 20 + 25
    assert early != late
    assert fn(0, 1, t_arrive) == late, "must price the cell at the ARRIVAL time"
    assert fn(0, 1, t_arrive) != early, "must not fall back to the departure bucket"
    assert fn(0, 1, p.start_time) == early, "start time is only ever its own bucket"
    return (f"t_arrive={t_arrive:g}s -> {late:g} (bucket 1), not {early:g} "
            f"(bucket 0 = departure/start)")


def test_changing_forecast_changes_the_selected_cost():
    p = two_bucket_provider()
    fn = p.cost_fn("conservative")
    a, b = fn(0, 2, 0.0), fn(0, 2, HOUR)
    assert a == 3.0 + 100.0 and b == 30.0 + 100.0
    assert a != b, "the same cell must change cost when the forecast changes"
    return f"cell (0,2): {a:g} at bucket 0 -> {b:g} at bucket 1"


def test_unavailable_bucket_raises_and_never_substitutes():
    # a deliberate hole: buckets 0 and 5 exist, 1-4 do not.
    p = TimeIndexedNavigationCost.from_fields(
        [_field(0, ENV_EARLY, label="early"), _field(5, ENV_LATE, label="late")],
        bucket_seconds=HOUR)
    fn = p.cost_fn("conservative")
    newest = float(p.fields[5].cost("conservative")[0, 1])
    previous = float(p.fields[0].cost("conservative")[0, 1])
    for t, why in ((2.5 * HOUR, "a hole between two fields"),
                   (9 * HOUR, "beyond the last field")):
        try:
            fn(0, 1, t)
            raise AssertionError(f"no exception for {why}")
        except ForecastUnavailable as exc:
            msg = str(exc)
            assert "does not" in msg and "extrapolate" in msg
    # and it is the refusal, not a quiet reuse of either neighbour
    assert newest != previous
    return ("a missing bucket raises ForecastUnavailable; no newest-field, "
            "previous-day, interpolated or extrapolated substitute")


def test_impassable_mode_returns_inf_and_is_not_a_hard_block():
    p = two_bucket_provider(on_unavailable="impassable")
    fn = p.cost_fn("conservative")
    assert fn(0, 1, 9 * HOUR) == float("inf"), "unpriceable at that time"
    assert np.isfinite(fn(0, 1, 0.0)), "the SAME cell is priceable inside the horizon"
    # nothing about the cell itself was marked: no mask, no persistent state
    assert np.isfinite(fn(0, 1, HOUR))
    return ("+inf only for a time with no forecast; the same cell stays finite "
            "inside the horizon, so no cell is ever marked unnavigable")


def test_grid_alignment_is_validated():
    ok = _field(0, ENV_EARLY)
    wide = _field(1, np.array([[1.0, 2.0, 3.0, 4.0]]), key="wide", shape=(1, 4),
                  layout=dict(coverage=[[0, 1, 1, 0]],
                              polygon_id=[[POLYGON_ID_NODATA, 10, 11,
                                           POLYGON_ID_NODATA]],
                              water=[[0, 0, 0, 0]], ambiguity=[[0, 0, 0, 0]],
                              fraction=[[FRACTION_NODATA, 1.0, 1.0,
                                         FRACTION_NODATA]]))
    bad_crs = _field(1, ENV_LATE, epsg=3031)
    bad_tf = _field(1, ENV_LATE, transform=(PX, 0.0, 5000.0, 0.0, -PX, 0.0))
    caught = []
    for label, other in (("shape", wide), ("CRS", bad_crs), ("transform", bad_tf)):
        try:
            TimeIndexedNavigationCost.from_fields([ok, other], bucket_seconds=HOUR)
            raise AssertionError(f"mismatched {label} was accepted")
        except TimeCostError as exc:
            caught.append(f"{label}: {str(exc).splitlines()[0][:44]}")
    # a duplicated bucket is also refused
    try:
        TimeIndexedNavigationCost.from_fields([ok, _field(0, ENV_LATE)],
                                              bucket_seconds=HOUR)
        raise AssertionError("two fields claiming one bucket were accepted")
    except TimeCostError:
        caught.append("duplicate bucket")
    return "; ".join(caught)


def test_both_branches_are_preserved_and_independently_selectable():
    p = two_bucket_provider()
    cons, opt = p.cost_fn("conservative"), p.cost_fn("optimistic")
    # polygon 11 is conservative 100 / optimistic 0 -- the branches must not merge
    assert cons(0, 2, 0.0) == 3.0 + 100.0
    assert opt(0, 2, 0.0) == 3.0 + 0.0
    assert cons(0, 2, HOUR) == 30.0 + 100.0 and opt(0, 2, HOUR) == 30.0
    assert p.min_cost("conservative") == p.min_cost("optimistic") == 1.0
    assert set(BRANCHES) == {"conservative", "optimistic"}
    try:
        p.cost_fn("middle")
        raise AssertionError("an invented branch was accepted")
    except TimeCostError:
        pass
    return "conservative 103 vs optimistic 3 on the same cell/time; neither collapsed"


def test_temporal_mismatch_is_rejected_by_default():
    mismatched = [_field(0, ENV_EARLY, chart=CHART_2020),
                  _field(1, ENV_LATE, chart=CHART_2020)]
    try:
        TimeIndexedNavigationCost.from_fields(mismatched, bucket_seconds=HOUR)
        raise AssertionError("2020 chart under a 2025 environment was accepted")
    except TimeCostError as exc:
        assert "HISTORICAL DEMONSTRATION" in str(exc)
    p = TimeIndexedNavigationCost.from_fields(
        mismatched, bucket_seconds=HOUR, allow_historical_demo_mismatch=True)
    assert p.historical_demo_override is True
    assert p.provenance()["historical_demo_override"] is True
    assert "HISTORICAL DEMO OVERRIDE" in p.explain_lookup(0, 1, 0.0)
    aligned = two_bucket_provider()
    assert aligned.historical_demo_override is False
    return ("2020-12-24 chart + 2025-01-15 environment refused by default; the "
            "override is explicit and stamped on the provider and every explanation")


def test_repeated_lookup_is_deterministic():
    p = two_bucket_provider()
    fn = p.cost_fn("conservative")
    times = [0.0, 0.5 * HOUR, HOUR, 1.9 * HOUR]
    first = [[fn(0, c, t) for c in range(3)] for t in times]
    for _ in range(4):
        assert [[fn(0, c, t) for c in range(3)] for t in times] == first
    q = two_bucket_provider()
    fn2 = q.cost_fn("conservative")
    assert [[fn2(0, c, t) for c in range(3)] for t in times] == first
    return f"{5 * len(times) * 3} lookups, identical every time and across rebuilds"


def test_adapter_returns_the_composed_value_untouched():
    """No formula is re-applied: the adapter hands back exactly what it holds."""
    p = two_bucket_provider()
    for branch in BRANCHES:
        fn = p.cost_fn(branch)
        for b, t in ((0, 0.0), (1, HOUR)):
            arr = p.fields[b].cost(branch)
            for c in range(3):
                assert fn(0, c, t) == float(arr[0, c])
    return "every lookup equals the GridComposition value bit for bit"


def test_no_cost_formula_or_policy_value_is_duplicated():
    """Scan the AST, not the prose: no identifier from the owning layers."""
    tree = ast.parse(MODULE.read_text())
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            ids.add(n.id)
        elif isinstance(n, ast.Attribute):
            ids.add(n.attr)
        elif isinstance(n, ast.arg):
            ids.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            ids.add(n.name)
        elif isinstance(n, ast.keyword) and n.arg:
            ids.add(n.arg)
    forbidden = {"sic", "ice_weight", "ice_exponent", "current_weight",
                 "current_reference", "base_cost", "riv", "rio", "compute_rio",
                 "compute_chart", "assess", "assess_cells", "interpret_polygon",
                 "coverage_uncertainty_weight", "penalties", "normal_operation",
                 "elevated_operational_risk", "special_consideration"}
    hit = sorted(ids & forbidden)
    assert not hit, f"the adapter references owned-elsewhere names: {hit}"
    imports = {f"{n.module}.{a.name}" for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("src.")
               for a in n.names}
    assert imports == {"src.routing.navigation_cost.GridComposition"}, imports
    # and no penalty magic number in the code
    nums = {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)}
    assert not (nums & {25.0, 100.0, 25, 100}), f"penalty values inlined: {nums}"
    return f"only {sorted(imports)[0].split('.')[-1]} imported from src; no penalty literals"


def test_no_hard_block_vocabulary_or_mask_is_produced():
    tree = ast.parse(MODULE.read_text())
    ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    ids |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    ids |= {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    banned = {"blocked", "blocked_mask", "no_go", "nogo", "navigable", "passable",
              "forbidden", "mask", "exclusion", "keep_out"}
    hit = sorted(ids & banned)
    assert not hit, f"hard-navigation vocabulary present: {hit}"
    p = two_bucket_provider()
    assert not any(k in p.provenance() for k in banned)
    fn = p.cost_fn("conservative")
    vals = [fn(0, c, t) for c in range(3) for t in (0.0, HOUR)]
    assert all(np.isfinite(v) for v in vals), "no cell was made impassable"
    return "no mask, no block vocabulary, every in-horizon lookup finite"


def test_min_cost_is_a_true_lower_bound_over_every_bucket():
    p = two_bucket_provider()
    for branch in BRANCHES:
        m = p.min_cost(branch)
        for f in p.fields.values():
            arr = f.cost(branch)
            assert (arr[np.isfinite(arr)] >= m - 1e-12).all()
        assert m == min(float(f.cost(branch)[np.isfinite(f.cost(branch))].min())
                        for f in p.fields.values())
    return f"conservative min {p.min_cost('conservative'):g} holds across all buckets"


def test_invalid_construction_is_refused():
    caught = []
    for kw, label in (({"bucket_seconds": 0.0}, "bucket_seconds=0"),
                      ({"bucket_seconds": -HOUR}, "negative bucket_seconds"),
                      ({"bucket_seconds": float("inf")}, "infinite bucket_seconds")):
        try:
            TimeIndexedNavigationCost.from_fields([_field(0, ENV_EARLY)], **kw)
            raise AssertionError(f"{label} accepted")
        except TimeCostError:
            caught.append(label)
    try:
        TimeIndexedNavigationCost.from_fields([], bucket_seconds=HOUR)
        raise AssertionError("empty field list accepted")
    except TimeCostError:
        caught.append("no fields")
    try:
        two_bucket_provider(on_unavailable="use_latest")
        raise AssertionError("on_unavailable='use_latest' accepted")
    except TimeCostError:
        caught.append("on_unavailable='use_latest'")
    return "; ".join(caught) + " all refused"


def test_nan_and_negative_costs_are_refused():
    f = _field(0, ENV_EARLY)
    for bad, label in ((np.nan, "NaN"), (-1.0, "negative")):
        arr = f.composition.conservative_cost.copy()
        arr[0, 1] = bad
        broken = ComposedField(bucket=0, composition=type(f.composition)(
            **{**f.composition.__dict__, "conservative_cost": arr}),
            environment_date=DAY, chart_date=CHART_2025, epsg=EPSG,
            shape=(1, 3), transform=TF)
        try:
            TimeIndexedNavigationCost.from_fields([broken], bucket_seconds=HOUR)
            raise AssertionError(f"{label} cost accepted")
        except TimeCostError as exc:
            if label == "NaN":
                assert "+inf, not NaN" in str(exc)
    return "NaN and negative composed costs both refused at construction"


def test_explain_lookup_and_provenance_are_traceable():
    p = two_bucket_provider()
    txt = p.explain_lookup(0, 2, 1.5 * HOUR)
    for token in ("bucket 1", "2025-01-15", "dates aligned", "conservative",
                  "composed cost"):
        assert token in txt, f"explain_lookup omits {token!r}\n{txt}"
    prov = p.provenance()
    assert prov["buckets"] == [0, 1] and prov["bucket_seconds"] == HOUR
    assert prov["fields"][1]["label"] == "late"
    miss = p.explain_lookup(0, 2, 9 * HOUR)
    assert "UNAVAILABLE" in miss and "bucket 9" in miss
    return ("explain_lookup names the bucket, both dates and every term, and says "
            "UNAVAILABLE rather than a number when there is no field")


def test_horizon_s_is_the_last_time_that_can_actually_be_priced():
    """time_astar's deadline is inclusive, so horizon_s must be too."""
    p = two_bucket_provider()
    assert p.horizon_s < 2 * HOUR, "the bucket boundary itself has no field"
    assert p.bucket_for(p.horizon_s) == max(p.buckets)
    p.cost_fn("conservative")(0, 1, p.horizon_s)          # must not raise
    try:
        p.cost_fn("conservative")(0, 1, 2 * HOUR)
        raise AssertionError("the boundary instant was priced")
    except ForecastUnavailable:
        pass
    return (f"horizon_s={p.horizon_s!r} lands in bucket {max(p.buckets)}; "
            f"{2 * HOUR:g}s does not")


# ====================================================== integration: A*
#   row 0:  .  A  A  A  .        corridor A
#   row 1:  .  #  #  #  .
#   row 2:  S  #  #  #  G        the wall is the GRID's blocked_mask, which
#   row 3:  .  #  #  #  .        belongs to NavigationDomain/constraints --
#   row 4:  .  B  B  B  .        this adapter contributes no mask at all.
#
# Corridor A is cheap in bucket 0 and expensive from bucket 1 on; corridor B is
# the reverse. Every cell after the start is entered at t >= 3600 s, i.e. bucket
# 1 or later, so a search that prices cells at their ARRIVAL time must take B,
# and one frozen at bucket 0 must take A.
BIG = (5, 5)
A_CELLS = [(0, 1), (0, 2), (0, 3)]
B_CELLS = [(4, 1), (4, 2), (4, 3)]
WALL = [(r, c) for r in (1, 2, 3) for c in (1, 2, 3)]
START, GOAL = (2, 0), (2, 4)

CORRIDOR_PEN = {
    10: dict(conservative=5.0, optimistic=5.0, classification="normal_operation",
             has_rio=True, is_water=False),
    11: dict(conservative=5.0, optimistic=5.0, classification="normal_operation",
             has_rio=True, is_water=False),
}


def _corridor_layout() -> dict:
    cov = np.zeros(BIG, "uint8"); pid = np.full(BIG, POLYGON_ID_NODATA, "int32")
    frac = np.full(BIG, FRACTION_NODATA, "float32")
    for (r, c) in A_CELLS:
        cov[r, c], pid[r, c], frac[r, c] = 1, 10, 1.0
    for (r, c) in B_CELLS:
        cov[r, c], pid[r, c], frac[r, c] = 1, 11, 1.0
    return dict(coverage=cov, polygon_id=pid, water=np.zeros(BIG, "uint8"),
                ambiguity=np.zeros(BIG, "uint8"), fraction=frac)


def _corridor_env(a_cost: float, b_cost: float) -> np.ndarray:
    env = np.ones(BIG, dtype="float64")
    for (r, c) in A_CELLS:
        env[r, c] = a_cost
    for (r, c) in B_CELLS:
        env[r, c] = b_cost
    return env


def _corridor_field(bucket: int, a_cost: float, b_cost: float) -> ComposedField:
    return _field(bucket, _corridor_env(a_cost, b_cost), key="corridor", shape=BIG,
                  layout=_corridor_layout(), penalties=CORRIDOR_PEN,
                  label=f"A={a_cost:g} B={b_cost:g}")


def _corridor_provider() -> TimeIndexedNavigationCost:
    fields = [_corridor_field(0, 1.0, 50.0)]                       # A cheap at t0
    fields += [_corridor_field(b, 50.0, 1.0) for b in range(1, 10)]  # B cheap later
    return TimeIndexedNavigationCost.from_fields(fields, bucket_seconds=HOUR)


def _corridor_grid():
    from src.routing.grid import RoutingGrid
    blocked = np.zeros(BIG, bool)
    for (r, c) in WALL:
        blocked[r, c] = True
    z = np.zeros(BIG, "float32")
    return RoutingGrid(transform=Affine(*TF), crs=CRS.from_epsg(EPSG), shape=BIG,
                       sic=z.copy(), current_u=z.copy(), current_v=z.copy(),
                       blocked_mask=blocked, day=DAY)


def _run(cost_fn, min_cost, horizon):
    return time_astar(_corridor_grid(), START, GOAL, cost_fn=cost_fn,
                      min_cost=min_cost, vessel_speed_mps=SPEED,
                      time_step_s=HOUR, max_horizon_s=horizon)


def test_integration_time_dependent_route_takes_the_later_cheap_corridor():
    p = _corridor_provider()
    res = _run(p.cost_fn("conservative"), p.min_cost("conservative"), p.horizon_s)
    assert res.success, res.reason
    cells = set(res.path)
    assert cells & set(B_CELLS), f"did not use corridor B: {res.path}"
    assert not (cells & set(A_CELLS)), f"used corridor A: {res.path}"
    return (f"cost {res.total_cost:,.0f} via corridor B, "
            f"{res.duration_s / 3600:.2f} h, {len(res.path)} cells")


def test_integration_each_cell_is_priced_at_its_own_arrival_time():
    p = _corridor_provider()
    res = _run(p.cost_fn("conservative"), p.min_cost("conservative"), p.horizon_s)
    assert res.success, res.reason
    seen = []
    for (r, c), t in zip(res.path, res.arrival_times):
        b = p.bucket_for(t)
        paid = float(p.field_for(t).cost("conservative")[r, c])
        assert paid == float(p.fields[b].cost("conservative")[r, c])
        if (r, c) in B_CELLS:
            assert b >= 1, f"corridor cell {(r, c)} entered in bucket {b}"
            assert paid == 6.0, f"{(r, c)} paid {paid}, not the later field's 6.0"
            assert float(p.fields[0].cost("conservative")[r, c]) == 55.0
            seen.append(f"{(r, c)}@{t / 3600:.2f}h=bucket{b}:{paid:g}")
    assert seen, "no corridor-B cell on the path"
    return ("later arrival -> later field: " + ", ".join(seen)
            + "  (bucket 0 would have charged 55.0)")


def test_integration_frozen_cost_routing_takes_the_other_corridor():
    p = _corridor_provider()
    frozen_field = p.fields[0].cost("conservative")

    def frozen(r: int, c: int, t: float) -> float:       # ignores time entirely
        return float(frozen_field[r, c])

    hot = _run(p.cost_fn("conservative"), p.min_cost("conservative"), p.horizon_s)
    cold = _run(frozen, float(frozen_field.min()), p.horizon_s)
    assert hot.success and cold.success, (hot.reason, cold.reason)
    assert set(cold.path) & set(A_CELLS), f"frozen route left corridor A: {cold.path}"
    assert not (set(cold.path) & set(B_CELLS))
    assert hot.path != cold.path, "the two searches produced the same route"

    # What the frozen plan would actually have cost, priced as the ship arrives.
    fn = p.cost_fn("conservative")
    repriced = 0.0
    for (a, ta), (b, tb) in zip(zip(cold.path, cold.arrival_times),
                                zip(cold.path[1:], cold.arrival_times[1:])):
        d = float(np.hypot((b[1] - a[1]) * PX, (b[0] - a[0]) * PX))
        repriced += d * 0.5 * (fn(a[0], a[1], ta) + fn(b[0], b[1], tb))
    assert repriced > hot.total_cost, (repriced, hot.total_cost)
    return (f"frozen -> corridor A ({cold.total_cost:,.0f} as planned, but "
            f"{repriced:,.0f} once each cell is priced on arrival); "
            f"time-dependent -> corridor B ({hot.total_cost:,.0f}); "
            f"same grid, same solver, different routes")


def test_integration_horizon_is_refused_not_extrapolated():
    """Only one bucket of forecast: the search must not invent the rest."""
    short = TimeIndexedNavigationCost.from_fields(
        [_corridor_field(0, 1.0, 50.0)], bucket_seconds=HOUR)
    try:
        _run(short.cost_fn("conservative"), short.min_cost("conservative"),
             10 * HOUR)                     # a horizon far past the last field
        raise AssertionError("the search ran past the forecast without complaint")
    except ForecastUnavailable as exc:
        assert "extrapolate" in str(exc)
    lenient = TimeIndexedNavigationCost.from_fields(
        [_corridor_field(0, 1.0, 50.0)], bucket_seconds=HOUR,
        on_unavailable="impassable")
    res = _run(lenient.cost_fn("conservative"), lenient.min_cost("conservative"),
               10 * HOUR)
    assert not res.success, "a route was found with no forecast to price it"
    return ("default: ForecastUnavailable propagates out of time_astar; "
            "impassable mode: search simply finds nothing beyond the forecast")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78); print("time-indexed navigation cost adapter"); print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<58} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__); print(f"  FAIL  {fn.__name__:<58} {exc}")
        except Exception as exc:                        # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__:<58} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

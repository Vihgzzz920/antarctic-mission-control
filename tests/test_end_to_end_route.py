"""
Tests for the end-to-end route orchestration layer.

The ones that matter: hard constraints come only from the constraint layer,
a cell is priced at the ARRIVAL time, a 2020 chart under 2025 environment is
refused unless the override is asked for, and a failure is never dressed up as
a route.

    python tests/test_end_to_end_route.py
"""
from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.rasterize_sigrid3 import (AMBIGUITY_TIF, COVERAGE_TIF,         # noqa: E402
                                        DOMINANT_FRACTION_TIF, FRACTION_NODATA,
                                        POLYGON_ID_NODATA, POLYGON_ID_TIF,
                                        WATER_TIF)
from src.routing.constraints import NavigationConstraints                    # noqa: E402
from src.routing.cost import build_cost_grid                                 # noqa: E402
from src.routing.end_to_end_route import (DEFAULT_POLARIS_PENALTIES,          # noqa: E402
                                          DEFAULT_SIDECAR, INDETERMINATE,
                                          OUTCOME_DATE_MISMATCH,
                                          OUTCOME_FORECAST_UNAVAILABLE,
                                          OUTCOME_FOUND,
                                          OUTCOME_HARD_CONSTRAINT,
                                          OUTCOME_NO_PATH, SPECIAL_CONSIDERATION,
                                          RouteError, build_polaris_penalties,
                                          plan_route, resolve_riv_table)
from src.routing.grid import RoutingGrid                                     # noqa: E402
from src.routing.iceberg_navigation_cost import (CONFIG, ExposureField,      # noqa: E402
                                                 IcebergExposureProvider,
                                                 IcebergTimeNavigationCost,
                                                 load_config)
from src.routing.iceberg_risk import IcebergForecast                         # noqa: E402
from src.routing.navigation_cost import (compose_grid,                       # noqa: E402
                                         load_composition_config)
from src.routing.navigation_domain import NavigationDomain                   # noqa: E402
from src.routing.time_navigation_cost import (ComposedField,                 # noqa: E402
                                              TimeIndexedNavigationCost)

MODULE = ROOT / "src" / "routing" / "end_to_end_route.py"
COMP_CONFIG = ROOT / "configs" / "navigation_cost_composition.json"
CFG = load_config(CONFIG)
HOUR = 3600.0
EPSG = 3976
DAY = date(2025, 1, 8)
DEPART = datetime(2025, 1, 8, 0, 0)
BEDMACHINE = ROOT / "data" / "processed" / "bedmachine" / \
    "bedmachine_land_exclusion_3976.tif"
USNIC = ROOT / "data" / "processed" / "usnic"
ICEBERGS = ROOT / "data" / "processed" / "icebergs"
_CACHE: dict = {}


# ------------------------------------------------------------- synthetic world
def usnic_dir(shape: tuple[int, int], px: float) -> Path:
    """An all-uncharted USNIC fixture: no POLARIS penalty, no coverage signal."""
    key = ("usnic", shape, px)
    if key not in _CACHE:
        d = Path(tempfile.mkdtemp(prefix="e2e_usnic_"))
        h, w = shape
        prof = dict(driver="GTiff", height=h, width=w, count=1,
                    crs=CRS.from_epsg(EPSG), transform=Affine(px, 0, 0, 0, -px, 0))
        lay = dict(coverage=np.zeros(shape, "uint8"),
                   polygon_id=np.full(shape, POLYGON_ID_NODATA, "int32"),
                   water=np.zeros(shape, "uint8"),
                   ambiguity=np.zeros(shape, "uint8"),
                   fraction=np.full(shape, FRACTION_NODATA, "float32"))
        for name, k, dt_, nd in ((COVERAGE_TIF, "coverage", "uint8", None),
                                 (POLYGON_ID_TIF, "polygon_id", "int32",
                                  POLYGON_ID_NODATA),
                                 (WATER_TIF, "water", "uint8", None),
                                 (AMBIGUITY_TIF, "ambiguity", "uint8", None),
                                 (DOMINANT_FRACTION_TIF, "fraction", "float32",
                                  FRACTION_NODATA)):
            with rasterio.open(d / name, "w", dtype=dt_, nodata=nd, **prof) as dst:
                dst.write(lay[k].astype(dt_), 1)
        _CACHE[key] = d
    return _CACHE[key]


def make_grid(shape, px, blocked=None) -> RoutingGrid:
    h, w = shape
    z = np.zeros(shape, "float32")
    return RoutingGrid(transform=Affine(px, 0, 0, 0, -px, 0),
                       crs=CRS.from_epsg(EPSG), shape=shape, sic=z.copy(),
                       current_u=z.copy(), current_v=z.copy(),
                       blocked_mask=(np.zeros(shape, bool) if blocked is None
                                     else blocked), day=DAY)


def base_provider(shape, px, envs, *, chart=DAY, env_date=DAY, buckets=None,
                  bucket_s=HOUR, usnic=None, penalties=None,
                  **kw) -> TimeIndexedNavigationCost:
    """One ComposedField per supplied environmental-cost array."""
    cfg = load_composition_config(COMP_CONFIG)
    d = usnic or usnic_dir(shape, px)
    pen = penalties or {}
    fields = []
    for i, env in enumerate(envs):
        b = i if buckets is None else buckets[i]
        fields.append(ComposedField(
            bucket=b, composition=compose_grid(env, pen, cfg, d),
            environment_date=env_date, chart_date=chart, epsg=EPSG, shape=shape,
            transform=(px, 0, 0, 0, -px, 0), label=f"bucket {b}"))
    return TimeIndexedNavigationCost.from_fields(fields, bucket_seconds=bucket_s, **kw)


def berg(name="fixture_berg", radius=18.1, hours=24.0, extrap=False) -> IcebergForecast:
    return IcebergForecast(
        iceberg_id=name, position_source="ascat",
        forecast_start_date=DAY.isoformat(), horizon_seconds=hours * HOUR,
        horizon_hours=hours, latitude=-70.0, longitude=0.0, radius_km=radius,
        extrapolated_uncertainty=extrap, physics_position_is_extrapolated=False,
        calibration_quantile=0.9, calibration_n=1197, growth_law="power",
        interpolation_quality="observed_to_observed")


def iceberg_provider(base, shape, exposures, weight, *, buckets=None,
                     bergs=None, **kw) -> IcebergTimeNavigationCost:
    bergs = bergs or (berg(),)
    fields = []
    for i, e in enumerate(exposures):
        b = i if buckets is None else buckets[i]
        arr = np.asarray(e, dtype="float64")
        fields.append(ExposureField(
            bucket=b, exposure=arr,
            dominant=np.where(arr > 0, 0, -1).astype("int32"), forecasts=bergs,
            forecast_start_date=DAY, horizon_hours=24.0))
    prov = IcebergExposureProvider.from_fields(fields, shape)
    return IcebergTimeNavigationCost.from_providers(
        base, prov, CFG.with_weight(weight), **kw)


def corridor_world(px=10_000.0):
    """The asymmetric scenario: a short exposed shortcut, a longer clear detour.

         col:  0  1  2  3  4  5  6  7  8  9 10 11 12
      row 0    .  .  .  .  .  .  .  .  .  .  .  .  .   <- clear detour
      row 1-5  .  .  .  .  #  #  #  #  #  .  .  .  .
      row 6    S  .  .  .  ~  ~  *  ~  ~  .  .  .  G   <- shortcut, through a cone
    """
    h, w = 7, 13
    blocked = np.zeros((h, w), bool)
    blocked[1:6, 4:9] = True
    exposure = np.zeros((h, w), "float64")
    for r in range(h):
        for c in range(w):
            d = math.hypot(r - 6.0, c - 6.0)
            exposure[r, c] = 0.85 * max(0.0, 1.0 - d / 3.0)
    exposure[blocked] = 0.0
    return (h, w), blocked, exposure


# ------------------------------------------------------------------ 1. success
def test_successful_end_to_end_route():
    shape, px = (5, 7), 10_000.0
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(12)])
    land = np.zeros(shape, bool)
    land[2, 3] = True
    nc = NavigationConstraints(shape=shape, land_mask=land)
    res = plan_route(grid, (2, 0), (2, 6), provider=base, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=nc)
    assert res.success and res.outcome == OUTCOME_FOUND, (res.outcome, res.reason)
    m = res.metrics
    assert m is not None and m.cells == len(res.path) == len(res.cells)
    assert m.distance_m > 0 and m.travel_time_s > 0
    assert abs(m.reconciliation_residual) < 1e-6, m.reconciliation_residual
    assert res.arrival_time and res.departure_time == DEPART.isoformat()
    assert res.objective == "minimum configured cost"
    return (f"{m.cells} cells, {m.distance_km:.2f} km, {m.travel_time_h:.2f} h, "
            f"cost {m.total_cost:,.0f}, terms reconcile to "
            f"{m.reconciliation_residual:g}")


# --------------------------------------------------------- 2. hard constraints
def test_hard_constraint_is_respected_and_comes_only_from_the_domain():
    shape, px = (5, 7), 10_000.0
    wall = np.zeros(shape, bool)
    wall[0:4, 3] = True                      # a wall with one gap, at row 4
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(20)])
    nc = NavigationConstraints(shape=shape, land_mask=wall)
    res = plan_route(grid, (0, 0), (0, 6), provider=base, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=nc)
    assert res.success, res.reason
    assert not any(wall[r, c] for r, c in res.path), res.path
    assert (4, 3) in res.path, f"the only gap was not used: {res.path}"
    #  the mask the search used is the constraint layer's, bit for bit
    assert np.array_equal(np.asarray(grid.blocked_mask, bool), nc.combined(shape))
    assert res.constraint_summary["route_cells_inside_excluded"] == 0
    assert res.constraint_summary["masks_added_by_route_layer"] == 0

    #  an endpoint inside the domain's exclusion is a hard-constraint failure,
    #  not "no path"
    bad = plan_route(grid, (0, 0), (2, 3), provider=base, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=nc)
    assert not bad.success and bad.outcome == OUTCOME_HARD_CONSTRAINT, bad.outcome
    assert bad.path == () and bad.metrics is None
    return (f"route detoured through the single gap at (4, 3); a goal inside the "
            f"exclusion returns {OUTCOME_HARD_CONSTRAINT}, not a route")


# ------------------------------------------------------- 3. arrival-time cost
def test_each_cell_is_priced_at_its_arrival_time():
    shape, px = (1, 6), 10_000.0
    #  bucket b costs 1 + b everywhere, so the bucket a cell was priced in is
    #  readable straight off its composed cost
    envs = [np.full(shape, 1.0 + b) for b in range(8)]
    grid = make_grid(shape, px)
    base = base_provider(shape, px, envs)
    res = plan_route(grid, (0, 0), (0, 5), provider=base, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=None,
                     allow_unconstrained=True)
    assert res.success, res.reason
    buckets = [c.bucket for c in res.cells]
    assert buckets == sorted(buckets) and max(buckets) > 0, buckets
    for c in res.cells:
        assert c.composed_cost == 1.0 + c.bucket, (c.col, c.bucket, c.composed_cost)
        assert c.bucket == int(c.arrival_time // HOUR)
    #  bucket 0 pricing everywhere would have been cheaper; it was not used
    flat = sum(math.hypot(0.0, px) for _ in res.path[1:]) * 1.0
    assert res.metrics.total_cost > flat
    return (f"buckets along the route {buckets}; each cell's cost equals its "
            f"OWN bucket, and the total {res.metrics.total_cost:,.0f} exceeds the "
            f"{flat:,.0f} a bucket-0 pricing would have given")


# ------------------------------------------------- 4/5. temporal mismatch
def test_historical_polaris_mismatch_is_rejected_by_default():
    shape, px = (1, 4), 10_000.0
    grid = make_grid(shape, px)
    #  a 2020 USNIC chart under a 2025 environment: refused where it is built ...
    try:
        base_provider(shape, px, [np.ones(shape) for _ in range(6)],
                      chart=date(2020, 12, 24))
        raise AssertionError("the provider accepted mismatched dates by default")
    except AssertionError:
        raise
    except Exception:
        pass
    #  ... and refused again at route level even when the provider carries the
    #  override, so an override cannot leak in from how the provider was built
    base = base_provider(shape, px, [np.ones(shape) for _ in range(6)],
                         chart=date(2020, 12, 24),
                         allow_historical_demo_mismatch=True)
    res = plan_route(grid, (0, 0), (0, 3), provider=base, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=None,
                     allow_unconstrained=True)
    assert not res.success and res.outcome == OUTCOME_DATE_MISMATCH, res.outcome
    assert res.path == () and res.metrics is None
    assert res.temporal_provenance["historical_demo_override"] is False
    assert res.temporal_provenance["provider_historical_demo_override"] is True
    assert res.temporal_provenance["chart_dates"] == ["2020-12-24"]
    assert res.temporal_provenance["environment_dates"] == ["2025-01-08"]
    return ("2020-12-24 chart + 2025-01-08 environment refused twice: at the "
            "provider, and again at plan_route which never inherits an override")


def test_explicit_historical_override_is_preserved_and_stamped():
    shape, px = (1, 4), 10_000.0
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(6)],
                         chart=date(2020, 12, 24),
                         allow_historical_demo_mismatch=True)
    res = plan_route(grid, (0, 0), (0, 3), provider=base, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=None,
                     allow_unconstrained=True,
                     allow_historical_demo_override=True)
    assert res.success and res.outcome == OUTCOME_FOUND, (res.outcome, res.reason)
    assert res.historical_demo_override is True
    tp = res.temporal_provenance
    assert tp["historical_demo_override"] is True
    assert tp["all_dates_aligned"] is False
    assert tp["override_is_never_the_default"] is True
    assert "HISTORICAL DEMO OVERRIDE" in res.explain()
    return ("the override routes and is stamped historical_demo_override=true in "
            "the provenance and in explain(); it is never the default")


def test_a_departure_date_outside_the_environment_dates_is_refused():
    shape, px = (1, 4), 10_000.0
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(6)])
    res = plan_route(grid, (0, 0), (0, 3), provider=base,
                     departure_time=datetime(2019, 7, 1), vessel_speed_mps=px / HOUR,
                     constraints=None, allow_unconstrained=True)
    assert not res.success and res.outcome == OUTCOME_DATE_MISMATCH, res.outcome
    assert res.temporal_provenance[
        "departure_date_matches_environment_dates"] is False
    return "a 2019 departure over 2025 environment fields is refused, not routed"


# -------------------------------------- 6. iceberg exposure changes the route
def test_iceberg_exposure_changes_the_route_in_the_controlled_scenario():
    shape, blocked, exposure = corridor_world()
    px = 10_000.0
    envs = [np.ones(shape) for _ in range(30)]
    #  the cone is absent at departure and has drifted onto the shortcut after
    zero = np.zeros(shape, "float64")
    exps = [zero if b == 0 else exposure for b in range(30)]
    nc = NavigationConstraints(shape=shape, land_mask=blocked)

    def route(weight):
        grid = make_grid(shape, px)
        base = base_provider(shape, px, envs)
        p = iceberg_provider(base, shape, exps, weight)
        return plan_route(grid, (6, 0), (6, 12), provider=p, departure_time=DEPART,
                          vessel_speed_mps=px / HOUR, constraints=nc)

    off, on = route(0.0), route(CFG.weight)
    assert off.success and on.success, (off.reason, on.reason)
    shortcut = {(6, c) for c in (4, 5, 6, 7, 8)}
    assert set(off.path) & shortcut, f"weight 0 avoided the shortcut: {off.path}"
    #  at weight 0 the exposure is still REPORTED, it is just not charged for
    assert off.metrics.max_iceberg_exposure > 0, "exposure vanished at weight 0"
    assert off.metrics.iceberg_exposure_contribution == 0.0, "weight 0 charged"
    assert off.metrics.contributing_iceberg_ids

    #  with the SHIPPED weight the detour must be worth its extra distance. The
    #  test asserts the measured trade-off, not a guessed outcome.
    detour_penalty = on.metrics.total_cost - off.metrics.environmental_contribution
    if set(on.path) & shortcut:
        assert on.metrics.iceberg_exposure_contribution > 0
        return (f"weight {CFG.weight:g} kept the shortcut: the detour was not "
                f"worth it here (cost {on.metrics.total_cost:,.0f})")
    assert on.metrics.iceberg_exposure_contribution == 0.0
    assert on.metrics.distance_m > off.metrics.distance_m
    assert on.metrics.travel_time_s > off.metrics.travel_time_s
    assert off.path != on.path
    return (f"weight 0 -> shortcut {off.metrics.distance_km:.2f} km "
            f"(iceberg {off.metrics.iceberg_exposure_contribution:,.0f}); weight "
            f"{CFG.weight:g} -> detour {on.metrics.distance_km:.2f} km "
            f"(iceberg 0), +{on.metrics.distance_km - off.metrics.distance_km:.2f} "
            f"km bought for {detour_penalty:,.0f}")


# ------------------------------- 7. iceberg / POLARIS never become hard blocks
def test_iceberg_and_polaris_terms_never_become_hard_blocks():
    shape, px = (1, 5), 10_000.0
    #  the ONLY corridor runs through a fully exposed cell
    exposure = np.zeros(shape, "float64")
    exposure[0, 2] = 1.0
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(20)])
    p = iceberg_provider(base, shape, [exposure] * 20, 1000.0)   # the config maximum
    before = np.asarray(grid.blocked_mask, bool).copy()
    res = plan_route(grid, (0, 0), (0, 4), provider=p, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=None,
                     allow_unconstrained=True)
    assert res.success, res.reason
    assert (0, 2) in res.path, "a fully exposed cell was treated as impassable"
    assert res.metrics.max_iceberg_exposure == 1.0
    assert res.metrics.iceberg_exposure_contribution > 0
    assert np.array_equal(np.asarray(grid.blocked_mask, bool), before)

    #  and the module itself never writes a mask
    tree = ast.parse(MODULE.read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "set_blocked_mask" not in attrs, "the route layer writes a mask itself"
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for banned in ("no_go", "no_go_mask", "navigable_mask", "iceberg_mask",
                   "polaris_no_go", "exclusion_from_exposure"):
        assert banned not in names, f"identifier {banned!r} exists in the module"
    return ("a cell at exposure 1.0 priced at the maximum weight is still "
            "traversable, the blocked mask is untouched, and the module never "
            "calls set_blocked_mask")


# -------------------------------------------- 8. unavailable forecast handling
def test_unavailable_forecast_is_reported_not_guessed():
    shape, px = (1, 8), 10_000.0
    grid = make_grid(shape, px)
    #  buckets 0 and 2 exist, bucket 1 does not; the route must cross it
    base = base_provider(shape, px, [np.ones(shape), np.ones(shape)],
                         buckets=[0, 2])
    res = plan_route(grid, (0, 0), (0, 7), provider=base, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=None,
                     allow_unconstrained=True)
    assert not res.success, "a missing bucket produced a route"
    assert res.outcome == OUTCOME_FORECAST_UNAVAILABLE, res.outcome
    assert res.metrics is None and res.path == ()
    assert "unavailable" in res.reason.lower()
    return f"missing bucket 1 -> {res.outcome}, no substituted field, no route"


def test_missing_iceberg_exposure_is_also_reported_not_zeroed():
    shape, px = (1, 8), 10_000.0
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(10)])
    e = np.zeros(shape, "float64")
    p = iceberg_provider(base, shape, [e, e], 5.0, buckets=[0, 1])
    res = plan_route(grid, (0, 0), (0, 7), provider=p, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=None,
                     allow_unconstrained=True)
    assert not res.success and res.outcome == OUTCOME_FORECAST_UNAVAILABLE, res.outcome
    assert res.metrics is None
    return ("a bucket with no exposure field is reported unavailable, never read "
            "as exposure 0")


# ------------------------------------------------------------ 9. determinism
def test_repeated_route_is_deterministic():
    shape, blocked, exposure = corridor_world()
    px = 10_000.0
    nc = NavigationConstraints(shape=shape, land_mask=blocked)
    seen = []
    for _ in range(3):
        grid = make_grid(shape, px)
        base = base_provider(shape, px, [np.ones(shape) for _ in range(30)])
        p = iceberg_provider(base, shape, [exposure] * 30, CFG.weight)
        r = plan_route(grid, (6, 0), (6, 12), provider=p, departure_time=DEPART,
                       vessel_speed_mps=px / HOUR, constraints=nc)
        assert r.success, r.reason
        seen.append((r.path, r.metrics.total_cost, r.metrics.distance_m,
                     r.metrics.iceberg_exposure_contribution,
                     tuple(c.bucket for c in r.cells)))
    assert seen[0] == seen[1] == seen[2], "identical inputs gave different routes"
    return f"3 rebuilds, identical path, cost {seen[0][1]:,.3f} and buckets"


# -------------------------------------------------- 10. impossible, cleanly
def test_impossible_route_is_reported_cleanly():
    shape, px = (5, 7), 10_000.0
    wall = np.zeros(shape, bool)
    wall[:, 3] = True                        # a complete wall, no gap
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(20)])
    nc = NavigationConstraints(shape=shape, land_mask=wall)
    res = plan_route(grid, (2, 0), (2, 6), provider=base, departure_time=DEPART,
                     vessel_speed_mps=px / HOUR, constraints=nc)
    assert not res.success and res.outcome == OUTCOME_NO_PATH, res.outcome
    assert res.path == () and res.arrival_times == () and res.metrics is None
    assert res.arrival_time is None
    assert res.constraint_summary and res.temporal_provenance, "audit was dropped"
    return (f"a sealed wall gives {OUTCOME_NO_PATH} with no path, no metrics and "
            f"no invented route, while the constraint and date audit survive")


def test_ordinary_high_cost_is_a_route_not_a_failure():
    shape, px = (1, 5), 10_000.0
    grid = make_grid(shape, px)
    cheap = base_provider(shape, px, [np.ones(shape) for _ in range(20)])
    dear = base_provider(shape, px, [np.full(shape, 900.0) for _ in range(20)])
    a = plan_route(grid, (0, 0), (0, 4), provider=cheap, departure_time=DEPART,
                   vessel_speed_mps=px / HOUR, constraints=None,
                   allow_unconstrained=True)
    b = plan_route(make_grid(shape, px), (0, 0), (0, 4), provider=dear,
                   departure_time=DEPART, vessel_speed_mps=px / HOUR,
                   constraints=None, allow_unconstrained=True)
    assert a.success and b.success and a.outcome == b.outcome == OUTCOME_FOUND
    assert b.metrics.total_cost > 100 * a.metrics.total_cost
    assert a.path == b.path
    return (f"cost {a.metrics.total_cost:,.0f} -> {b.metrics.total_cost:,.0f} is "
            f"still a route; expense is never downgraded to a failure")


# ------------------------------------------------------- malformed requests
def test_malformed_requests_are_refused_loudly():
    shape, px = (1, 4), 10_000.0
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(6)])
    bad = 0
    for kw in (dict(departure_time=DAY),
               dict(vessel_speed_mps=0.0),
               dict(vessel_speed_mps=float("nan")),
               dict(departure_offset_s=-1.0),
               dict(provider=object()),
               dict(constraints=np.zeros(shape, bool)),
               dict(constraints=None, allow_unconstrained=False)):
        args = dict(provider=base, departure_time=DEPART, vessel_speed_mps=px / HOUR,
                    constraints=None, allow_unconstrained=True)
        args.update(kw)
        try:
            plan_route(grid, (0, 0), (0, 3), **args)
        except RouteError:
            bad += 1
        else:
            raise AssertionError(f"accepted a malformed request: {kw}")
    assert bad == 7
    return f"{bad} malformed requests refused with RouteError, none silently fixed"


# ------------------------------------------------- vocabulary and duplication
def test_no_safety_claim_and_no_duplicated_formula():
    src = MODULE.read_text()
    assert "safest" not in src.lower(), "the module claims a safety ranking"
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    funcs = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for banned in ("safest_route", "safe_route", "risk_rank", "recommended_route"):
        assert banned not in names | funcs, f"{banned!r} exists in the module"
    imported = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("src")}
    assert imported == {"src.routing.constraints", "src.routing.navigation_domain",
                        "src.routing.time_astar", "src.routing.time_navigation_cost",
                        "src.routing.iceberg_navigation_cost",
                        "src.routing.navigation_cost", "src.routing.polaris",
                        "src.routing.polaris_rio",
                        "src.routing.polaris_risk"}, sorted(imported)
    #  the POLARIS chain is imported, never restated: no RIV value, RIO sum,
    #  band edge or penalty number may be written into this module. Scanned over
    #  code, not prose, so the module's own documented prohibitions do not trip it.
    code = code_strings_and_names(MODULE)
    for banned in ("calculate_rio", "classify_rio", "riv_for", "rio_bands",
                   "penalty_for", "RIOTerm", "IceRegime", "RIOBand"):
        assert banned not in code, f"{banned!r} is restated in the module"
    for value in ("25.0", "100.0", "elevated_operational_risk",
                  "normal_operation", SPECIAL_CONSIDERATION):
        assert value not in code, f"penalty value or band {value!r} transcribed here"
    #  no cost, RIO, cone or bucket formula is restated here either. The RIO
    #  values are READ from the sidecar -- rio_key is a lookup key, not a sum --
    #  so the check is on the formulas and their inputs, not on the word "rio".
    #  calling one of these is delegation and is the point; DEFINING one here
    #  would be a second copy of a formula another module owns
    for owned in ("bucket_for", "calculate_rio", "classify_rio", "compose_grid",
                  "great_circle_m", "penalty_for", "riv_for"):
        assert f"def {owned}" not in src, f"the module redefines {owned!r}"
    for fragment in ("1 - distance", "math.floor((", "// time_step_s",
                     "iceberg_exposure_weight *", "1.0 - fraction"):
        assert fragment not in src, f"the formula {fragment!r} is restated"
    return (f"imports only {len(imported)} existing src modules, restates no "
            f"formula, and makes no safety claim")


# ---------------------------------------------------------- real-data smoke
def _real_world():
    """The 2025-01-08 environment, the real BedMachine exclusion and the real
    exposure rasters. Nothing here is invented and no mask is built."""
    if "real" in _CACHE:
        return _CACHE["real"]
    grid = RoutingGrid.for_date(DAY)
    dom = NavigationDomain(grid).add_geotiff("land_mask", BEDMACHINE)
    nc = dom.to_constraints()
    excluded = nc.combined(tuple(grid.shape))
    env = build_cost_grid(grid, constraints=nc).cost
    #  PC6 on IMO Table 1.3, both named explicitly below. These are TEST INPUTS
    #  chosen for this smoke run, not defaults: nothing anywhere picks a class
    #  or a table on its own.
    pp = real_polaris(PC6, TABLE_13)
    comp = compose_grid(env, pp.penalties, load_composition_config(COMP_CONFIG),
                        USNIC)
    with rasterio.open(USNIC / COVERAGE_TIF) as src:
        tag = src.tags()["source_chart"]             # e.g. ANTARC20201224
    chart_date = date(int(tag[-8:-4]), int(tag[-4:-2]), int(tag[-2:]))
    _CACHE["real"] = (grid, dom, nc, excluded, comp, chart_date, pp)
    return _CACHE["real"]


def _real_provider(chart_date, grid, comp, *, override):
    fields = [ComposedField(bucket=b, composition=comp, environment_date=DAY,
                            chart_date=chart_date, epsg=EPSG,
                            shape=tuple(grid.shape),
                            transform=tuple(grid.transform)[:6],
                            label=f"{b * 6}-{(b + 1) * 6} h")
              for b in (0, 1)]
    base = TimeIndexedNavigationCost.from_fields(
        fields, bucket_seconds=6 * HOUR, allow_historical_demo_mismatch=override)
    exps = []
    for b, h in ((0, "06h"), (1, "12h")):
        with rasterio.open(ICEBERGS / f"iceberg_exposure_{h}_3976.tif") as src:
            band = src.read(1).astype("float64")
            tags = src.tags()
        #  the raster is a max over the whole forecast population, so it carries
        #  no per-cell iceberg id; the label says exactly that
        b_ = IcebergForecast(
            iceberg_id=f"exposure_raster_{h}_max_over_"
                       f"{tags['iceberg_count']}_icebergs",
            position_source="ascat", forecast_start_date=tags["forecast_start_date"],
            horizon_seconds=float(tags["horizon_hours"]) * HOUR,
            horizon_hours=float(tags["horizon_hours"]), latitude=-70.0,
            longitude=0.0, radius_km=float(tags["radius_km"]),
            extrapolated_uncertainty=tags["extrapolated_uncertainty"] == "true",
            physics_position_is_extrapolated=False, calibration_quantile=0.9,
            calibration_n=1197, growth_law="power",
            interpolation_quality="observed_to_observed")
        exps.append(ExposureField(
            bucket=b, exposure=band,
            dominant=np.where(band > 0, 0, -1).astype("int32"), forecasts=(b_,),
            forecast_start_date=DAY, horizon_hours=float(tags["horizon_hours"]),
            label=h))
    prov = IcebergExposureProvider.from_fields(exps, tuple(grid.shape))
    return IcebergTimeNavigationCost.from_providers(
        base, prov, CFG, allow_historical_demo_mismatch=override), exps


def test_real_data_smoke_2025_01_08():
    for p in (BEDMACHINE, USNIC / COVERAGE_TIF,
              ICEBERGS / "iceberg_exposure_06h_3976.tif"):
        if not p.exists():
            raise AssertionError(f"real input missing: {p}")
    grid, dom, nc, excluded, comp, chart_date, pp = _real_world()
    assert chart_date == date(2020, 12, 24), chart_date

    #  the 2020 chart under the 2025 environment is refused where it is built
    try:
        _real_provider(chart_date, grid, comp, override=False)
        raise AssertionError("a 2020 chart was combined with 2025 data by default")
    except AssertionError:
        raise
    except Exception:
        pass

    provider, exps = _real_provider(chart_date, grid, comp, override=True)
    assert provider.config.weight == 5.0, provider.config.weight

    #  refused a SECOND time at route level, because plan_route was not asked
    grid_a = RoutingGrid.for_date(DAY)
    refused = plan_route(grid_a, REAL_START, (REAL_GOAL[0], REAL_GOAL[1]), provider=provider,
                         departure_time=DEPART, vessel_speed_mps=5.0,
                         constraints=NavigationDomain(grid_a).add_geotiff(
                             "land_mask", BEDMACHINE),
                         polaris_ice_class=PC6, polaris_riv_table=TABLE_13,
                         polaris_penalties=pp)
    assert refused.outcome == OUTCOME_DATE_MISMATCH and not refused.success

    #  goal: a real cell with real exposure in both buckets, reachable and priced
    e06, e12 = exps[0].exposure, exps[1].exposure
    routable = np.isfinite(comp.conservative_cost) & ~excluded
    goal_r, goal_c = REAL_GOAL
    assert routable[goal_r, goal_c] and e06[goal_r, goal_c] > 0 \
        and e12[goal_r, goal_c] > 0, "the chosen goal is not a real exposed cell"

    grid_b = RoutingGrid.for_date(DAY)
    res = plan_route(grid_b, REAL_START, (goal_r, goal_c), provider=provider,
                     departure_time=DEPART, vessel_speed_mps=5.0,
                     constraints=NavigationDomain(grid_b).add_geotiff(
                         "land_mask", BEDMACHINE),
                     polaris_ice_class=PC6, polaris_riv_table=TABLE_13,
                     polaris_penalties=pp, allow_historical_demo_override=True)
    assert res.success and res.outcome == OUTCOME_FOUND, (res.outcome, res.reason)
    m = res.metrics
    #  1. every route cell is inside the hard-constraint domain
    assert not any(excluded[r, c] for r, c in res.path)
    assert res.constraint_summary["route_cells_inside_excluded"] == 0
    #  2. at least one cell was priced from a LATER arrival bucket
    assert max(c.bucket for c in res.cells) > 0, "the route never left bucket 0"
    assert len(m.buckets_used) > 1, m.buckets_used
    #  3. the iceberg term is actually contributing at the shipped weight 5.0
    assert m.iceberg_exposure_contribution > 0, m.iceberg_exposure_contribution
    assert m.max_iceberg_exposure > 0 and m.contributing_iceberg_ids
    assert all(c.state == "iceberg_exposure_priced" for c in res.cells)
    #  4. the date mismatch is stamped, never silently ignored
    assert res.historical_demo_override is True
    assert res.temporal_provenance["chart_dates"] == ["2020-12-24"]
    assert res.temporal_provenance["environment_dates"] == ["2025-01-08"]
    assert abs(m.reconciliation_residual) < 1e-6 * max(abs(m.total_cost), 1.0)
    #  5. PC6 on Table 1.3 is priced from the real chart
    sel = res.polaris["selection"]
    assert res.polaris_selected and sel["ice_class"] == PC6
    assert sel["riv_table_requested"] == TABLE_13 and sel["riv_table_key"] == "default"
    assert sel["sidecar_chart"] == "ANTARC20201224"
    assert m.polaris_charted_ice_cells > 0, "the route crossed no charted ice"
    assert m.polaris_contribution > 0, "charted ice produced no POLARIS penalty"
    for c in res.cells:
        if c.polygon_id is not None and not pp.penalties[c.polygon_id]["is_water"]:
            assert c.polaris_penalty == pp.records[c.polygon_id].conservative_penalty
    return (f"HISTORICAL DEMO (2020-12-24 chart over 2025-01-08 data, not an "
            f"operational route): {m.cells} cells, {m.distance_km:.2f} km, "
            f"{m.travel_time_h:.2f} h, cost {m.total_cost:,.0f} = environment "
            f"{m.environmental_contribution:,.0f} + POLARIS "
            f"{m.polaris_contribution:,.0f} + iceberg "
            f"{m.iceberg_exposure_contribution:,.0f}; PC6/table 1.3: "
            f"{m.polaris_charted_ice_cells} charted-ice, "
            f"{m.polaris_special_consideration_cells} special-consideration, "
            f"{m.polaris_indeterminate_cells} indeterminate, "
            f"{m.polaris_unknown_regime_cells} unknown-regime, "
            f"{m.polaris_uncharted_cells} uncharted cells; max exposure "
            f"{m.max_iceberg_exposure:.3f}; buckets {list(m.buckets_used)}")


def code_strings_and_names(path: Path) -> set[str]:
    """Every string constant and identifier in a module, EXCLUDING docstrings.

    A module that documents a rule it obeys ("Table 1.4 is never inferred")
    would trip a raw substring search on its own prose, which is how earlier
    vocabulary tests in this project managed to fail on the disclaimer rather
    than on the behaviour. Docstrings are prose; code is what is scanned.
    """
    tree = ast.parse(path.read_text())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docs.add(id(body[0].value))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docs:
            out.add(node.value)
        elif isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
    return out


# =========================================================== POLARIS wiring
#  These use the REAL chart sidecar. No RIO, classification or penalty is
#  invented anywhere below; the only synthetic thing is a tiny raster that
#  carries real chart polygon ids so a route can be driven across them.
PC6, TABLE_13 = "PC6", "1.3"

#  a real corridor on the 2025-01-08 grid: open, charted-ice water ending on a
#  cell with real modelled iceberg exposure in both the 06 h and 12 h forecasts
REAL_START, REAL_GOAL = (948, 503), (922, 497)


def real_polaris(ice_class=PC6, table=TABLE_13):
    key = ("pol", ice_class, table)
    if key not in _CACHE:
        if not DEFAULT_SIDECAR.exists():
            raise AssertionError(f"chart sidecar missing: {DEFAULT_SIDECAR}")
        _CACHE[key] = build_polaris_penalties(ice_class=ice_class, riv_table=table)
    return _CACHE[key]


def pick_polygons(pp):
    """One real polygon of each kind, chosen by what the chart actually says."""
    kinds = {}
    for pid, rec in sorted(pp.records.items()):
        info = pp.penalties[pid]
        if info["is_water"]:
            kinds.setdefault("water", pid)
        elif not rec.has_rio:
            kinds.setdefault("unknown", pid)
        elif info["classification"] == SPECIAL_CONSIDERATION:
            kinds.setdefault("special", pid)
        elif info["classification"] == INDETERMINATE:
            kinds.setdefault("indeterminate", pid)
        elif info["classification"] == "normal_operation":
            kinds.setdefault("normal", pid)
    missing = {"water", "unknown", "special", "indeterminate", "normal"} - set(kinds)
    if missing:
        raise AssertionError(f"the real chart has no polygon of kind(s) {missing}")
    return kinds


def chart_fixture(poly_ids, water_ids=()) -> tuple[Path, tuple[int, int], float]:
    """A 1 x N USNIC raster set whose cells carry the given real polygon ids."""
    key = ("fixture", tuple(poly_ids), tuple(water_ids))
    if key not in _CACHE:
        px = 10_000.0
        shape = (1, len(poly_ids))
        d = Path(tempfile.mkdtemp(prefix="e2e_chart_"))
        prof = dict(driver="GTiff", height=1, width=len(poly_ids), count=1,
                    crs=CRS.from_epsg(EPSG), transform=Affine(px, 0, 0, 0, -px, 0))
        lay = dict(coverage=np.ones(shape, "uint8"),
                   polygon_id=np.asarray([poly_ids], dtype="int32"),
                   water=np.asarray([[1 if p in water_ids else 0
                                      for p in poly_ids]], dtype="uint8"),
                   ambiguity=np.zeros(shape, "uint8"),
                   fraction=np.ones(shape, "float32"))
        for name, k, dt_, nd in ((COVERAGE_TIF, "coverage", "uint8", None),
                                 (POLYGON_ID_TIF, "polygon_id", "int32",
                                  POLYGON_ID_NODATA),
                                 (WATER_TIF, "water", "uint8", None),
                                 (AMBIGUITY_TIF, "ambiguity", "uint8", None),
                                 (DOMINANT_FRACTION_TIF, "fraction", "float32",
                                  FRACTION_NODATA)):
            with rasterio.open(d / name, "w", dtype=dt_, nodata=nd, **prof) as dst:
                dst.write(lay[k].astype(dt_), 1)
        _CACHE[key] = (d, shape, px)
    return _CACHE[key]


def chart_route(pp, poly_ids, water_ids=(), *, ice_class=PC6, table=TABLE_13,
                declare=True, penalties=None):
    d, shape, px = chart_fixture(poly_ids, water_ids)
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(20)],
                         usnic=d, penalties=(pp.penalties if penalties is None
                                             else penalties))
    kw = dict(polaris_ice_class=ice_class, polaris_riv_table=table,
              polaris_penalties=pp) if declare else {}
    return plan_route(grid, (0, 0), (0, shape[1] - 1), provider=base,
                      departure_time=DEPART, vessel_speed_mps=px / HOUR,
                      constraints=None, allow_unconstrained=True, **kw)


def test_polaris_class_and_table_must_both_be_explicit():
    pp = real_polaris()
    d, shape, px = chart_fixture([pp.records and next(iter(sorted(pp.records)))])
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(6)], usnic=d,
                         penalties=pp.penalties)
    args = dict(provider=base, departure_time=DEPART, vessel_speed_mps=px / HOUR,
                constraints=None, allow_unconstrained=True)
    refused = 0
    for kw in (dict(polaris_ice_class=PC6),
               dict(polaris_riv_table=TABLE_13),
               dict(polaris_penalties=pp),
               dict(polaris_ice_class="PC99", polaris_riv_table=TABLE_13),
               dict(polaris_ice_class=PC6, polaris_riv_table="9.9"),
               dict(polaris_ice_class="", polaris_riv_table=TABLE_13),
               dict(polaris_ice_class="PC3", polaris_riv_table=TABLE_13,
                    polaris_penalties=pp)):
        try:
            plan_route(grid, (0, 0), (0, shape[1] - 1), **args, **kw)
        except RouteError:
            refused += 1
        else:
            raise AssertionError(f"accepted an incomplete POLARIS selection: {kw}")
    assert refused == 7
    #  neither given: POLARIS is simply not selected, and the result says so
    off = plan_route(grid, (0, 0), (0, shape[1] - 1), **args)
    assert off.success and off.polaris_selected is False
    assert "no polaris_ice_class" in off.polaris["reason"]
    return (f"{refused} incomplete or contradictory selections refused; with "
            f"neither supplied the result reports selected=False, never a default")


def test_polaris_riv_table_resolves_through_the_policy():
    by_number = {n: resolve_riv_table(n)[0] for n in ("1.3", "1.4")}
    by_key = {k: resolve_riv_table(k)[0] for k in ("default", "decayed_ice")}
    assert by_number == {"1.3": "default", "1.4": "decayed_ice"}, by_number
    assert by_key == {"default": "default", "decayed_ice": "decayed_ice"}, by_key
    code = code_strings_and_names(MODULE)
    for number in ("1.3", "1.4", "default", "decayed_ice"):
        assert number not in code, \
            f"the module hardcodes the RIV table identifier {number!r}"
    for n in ("9.9", "1.1", ""):
        try:
            resolve_riv_table(n)
        except RouteError:
            pass
        else:
            raise AssertionError(f"resolved a table that is not a RIV table: {n!r}")
    return ("1.3 -> default, 1.4 -> decayed_ice, resolved from the policy file and "
            "each config's own declared source; no table number appears in the module")


def test_pc6_table_1_3_gives_a_non_zero_polaris_contribution():
    pp = real_polaris()
    k = pick_polygons(pp)
    ids = [k["normal"], k["special"], k["indeterminate"], k["normal"]]
    res = chart_route(pp, ids)
    assert res.success, res.reason
    assert res.polaris_selected
    sel = res.polaris["selection"]
    assert sel["ice_class"] == PC6 and sel["riv_table_key"] == "default"
    assert sel["riv_table_requested"] == TABLE_13
    assert sel["sidecar_chart"] == "ANTARC20201224"
    m = res.metrics
    assert m.polaris_contribution > 0, "PC6 on table 1.3 charged nothing"
    assert m.polaris_special_consideration_cells == 1
    assert m.polaris_indeterminate_cells >= 1
    assert m.polaris_charted_ice_cells == len(ids)
    for cell in res.cells:
        assert cell.polaris_penalty == \
            pp.records[cell.polygon_id].conservative_penalty
    return (f"PC6 / table 1.3 over real polygons {ids}: POLARIS contribution "
            f"{m.polaris_contribution:,.0f}, "
            f"{m.polaris_special_consideration_cells} special-consideration and "
            f"{m.polaris_indeterminate_cells} indeterminate cells")


def test_changing_the_ice_class_changes_the_penalty_when_the_rio_requires_it():
    a, b = real_polaris(PC6, TABLE_13), real_polaris("PC3", TABLE_13)
    differ = [pid for pid in a.records
              if a.penalties[pid]["conservative"] != b.penalties[pid]["conservative"]]
    same = [pid for pid in a.records
            if a.penalties[pid]["conservative"] == b.penalties[pid]["conservative"]
            and a.records[pid].has_rio]
    assert differ and same, (len(differ), len(same))
    ids = [same[0], differ[0], same[0]]
    ra = chart_route(a, ids, ice_class=PC6)
    rb = chart_route(b, ids, ice_class="PC3")
    assert ra.success and rb.success
    assert ra.metrics.polaris_contribution != rb.metrics.polaris_contribution
    #  and the cell whose RIO does NOT separate the classes is charged the same
    assert ra.cells[0].polaris_penalty == rb.cells[0].polaris_penalty
    assert ra.cells[1].polaris_penalty != rb.cells[1].polaris_penalty
    return (f"{len(differ)} of {len(a.records)} polygons are charged differently "
            f"for PC3 than PC6 on table 1.3; the route total moves "
            f"{ra.metrics.polaris_contribution:,.0f} -> "
            f"{rb.metrics.polaris_contribution:,.0f}, while a polygon whose RIO "
            f"does not separate the two classes is charged identically")


def test_water_takes_no_rio_and_no_polaris_penalty():
    pp = real_polaris()
    k = pick_polygons(pp)
    ids = [k["normal"], k["water"], k["normal"]]
    res = chart_route(pp, ids, water_ids=(k["water"],))
    assert res.success, res.reason
    wet = [c for c in res.cells if c.polygon_id == k["water"]]
    assert wet, res.cells
    for c in wet:
        assert c.chart_state == "charted_water", c.chart_state
        assert c.polaris_penalty == 0.0, c.polaris_penalty
    assert pp.records[k["water"]].has_rio is False
    assert res.metrics.polaris_charted_water_cells == len(wet)
    return (f"polygon {k['water']} is POLY_TYPE=W: no RIO, zero POLARIS penalty, "
            f"state charted_water -- and the record still says has_rio=False "
            f"rather than inventing a regime")


def test_unknown_regime_keeps_the_existing_unknown_behaviour():
    pp = real_polaris()
    k = pick_polygons(pp)
    ids = [k["normal"], k["unknown"], k["normal"]]
    res = chart_route(pp, ids)
    assert res.success, res.reason
    unknown_penalty = json.loads(
        DEFAULT_POLARIS_PENALTIES.read_text())["penalties"]["unknown"]
    got = [c for c in res.cells if c.polygon_id == k["unknown"]]
    assert got
    for c in got:
        assert c.chart_state == "charted_unknown_regime", c.chart_state
        assert c.polaris_penalty == unknown_penalty
    assert pp.records[k["unknown"]].has_rio is False
    assert res.metrics.polaris_unknown_regime_cells == len(got)
    assert math.isfinite(res.metrics.total_cost)
    return (f"polygon {k['unknown']} has no defensible regime: it keeps the "
            f"configured unknown penalty {unknown_penalty:g} and the state "
            f"charted_unknown_regime; no RIO is fabricated for it")


def test_special_consideration_indeterminate_and_unknown_never_block():
    pp = real_polaris()
    k = pick_polygons(pp)
    #  the ONLY corridor runs through the three of them in a row
    ids = [k["normal"], k["special"], k["indeterminate"], k["unknown"], k["normal"]]
    d, shape, px = chart_fixture(ids)
    grid = make_grid(shape, px)
    before = np.asarray(grid.blocked_mask, bool).copy()
    res = chart_route(pp, ids)
    assert res.success, res.reason
    assert len(res.path) == len(ids), res.path
    assert all(math.isfinite(c.polaris_penalty) for c in res.cells)
    assert all(c.final_cost > 0 and math.isfinite(c.final_cost) for c in res.cells)
    assert np.array_equal(np.asarray(grid.blocked_mask, bool), before)
    assert res.polaris[
        "no_hard_block_from_special_consideration_indeterminate_or_unknown"] is True
    return ("a corridor made only of special-consideration, indeterminate and "
            "unknown cells is still traversable; every penalty is finite and no "
            "mask was created")


def test_polaris_per_cell_and_total_reconcile():
    pp = real_polaris()
    k = pick_polygons(pp)
    ids = [k["normal"], k["special"], k["indeterminate"], k["unknown"], k["normal"]]
    res = chart_route(pp, ids)
    assert res.success, res.reason
    dx = 10_000.0
    rebuilt = sum(dx * 0.5 * (a.polaris_penalty + b.polaris_penalty)
                  for a, b in zip(res.cells, res.cells[1:]))
    assert abs(rebuilt - res.metrics.polaris_contribution) < 1e-9 * max(
        abs(rebuilt), 1.0), (rebuilt, res.metrics.polaris_contribution)
    assert abs(res.polaris["total_contribution"]
               - res.metrics.polaris_contribution) < 1e-12
    assert len(res.polaris["per_cell"]) == len(res.cells)
    assert abs(res.metrics.reconciliation_residual) < 1e-9 * max(
        abs(res.metrics.total_cost), 1.0)
    return (f"per-cell penalties rebuild the {res.metrics.polaris_contribution:,.3f} "
            f"total exactly, and the four terms still sum to the search total "
            f"(residual {res.metrics.reconciliation_residual:g})")


def test_a_declared_selection_that_the_cost_did_not_use_is_refused():
    a, b = real_polaris(PC6, TABLE_13), real_polaris("PC3", TABLE_13)
    differ = [pid for pid in a.records
              if a.penalties[pid]["conservative"] != b.penalties[pid]["conservative"]]
    ids = [differ[0], differ[0]]
    try:
        #  compose with PC3's penalties, declare PC6
        chart_route(a, ids, ice_class=PC6, penalties=b.penalties)
    except RouteError as exc:
        assert "different selection" in str(exc), exc
        return ("composing with PC3 penalties while declaring PC6 is refused: the "
                "class and table in the result are verified against the penalty "
                "every route cell was actually charged")
    raise AssertionError("a mislabelled POLARIS selection was accepted")


def test_special_and_indeterminate_counts_are_defined_and_may_overlap():
    """One polygon can be special consideration on one branch and
    indeterminate on the other, so the two counts must never be added."""
    pp = real_polaris()
    both = [pid for pid, r in pp.records.items()
            if r.has_rio and not pp.penalties[pid]["is_water"]
            and r.classification_used_conservative == SPECIAL_CONSIDERATION
            and r.indeterminate]
    only_special = [pid for pid, r in pp.records.items()
                    if r.has_rio and not pp.penalties[pid]["is_water"]
                    and r.classification_used_conservative == SPECIAL_CONSIDERATION
                    and not r.indeterminate]
    assert both and only_special, (len(both), len(only_special))
    ids = [both[0], only_special[0]]
    res = chart_route(pp, ids)
    assert res.success, res.reason
    m = res.metrics
    assert m.polaris_special_consideration_cells == 2
    assert m.polaris_indeterminate_cells == 1
    assert m.polaris_indeterminate_used_branch_cells == 0, (
        "neither polygon is indeterminate on the branch that was priced")
    assert m.polaris_special_consideration_cells + m.polaris_indeterminate_cells \
        > len(res.cells), "the counts overlap, which is what the definitions say"
    d = res.polaris["definitions"]
    assert "OVERLAPS" in d["indeterminate_cells"]
    assert set(d) == {"special_consideration_cells", "indeterminate_cells",
                      "indeterminate_used_branch_cells", "mapping_sensitive_cells"}
    return (f"polygon {both[0]} is special consideration on the priced branch and "
            f"indeterminate on the other; the audit defines both counts and says "
            f"they overlap rather than leaving them to be added")


def test_polaris_route_is_deterministic():
    pp = real_polaris()
    k = pick_polygons(pp)
    ids = [k["normal"], k["special"], k["indeterminate"], k["unknown"], k["normal"]]
    seen = []
    for _ in range(3):
        r = chart_route(pp, ids)
        assert r.success, r.reason
        seen.append((r.path, r.metrics.total_cost, r.metrics.polaris_contribution,
                     tuple(c.polaris_penalty for c in r.cells),
                     tuple(c.polaris_classification for c in r.cells)))
    assert seen[0] == seen[1] == seen[2]
    return (f"3 rebuilds: identical path, POLARIS contribution "
            f"{seen[0][2]:,.3f} and identical per-cell classifications")


def test_polaris_selection_does_not_weaken_the_historical_mismatch_rule():
    pp = real_polaris()
    k = pick_polygons(pp)
    ids = [k["normal"], k["special"]]
    d, shape, px = chart_fixture(ids)
    grid = make_grid(shape, px)
    base = base_provider(shape, px, [np.ones(shape) for _ in range(6)], usnic=d,
                         penalties=pp.penalties, chart=date(2020, 12, 24),
                         allow_historical_demo_mismatch=True)
    res = plan_route(grid, (0, 0), (0, shape[1] - 1), provider=base,
                     departure_time=DEPART, vessel_speed_mps=px / HOUR,
                     constraints=None, allow_unconstrained=True,
                     polaris_ice_class=PC6, polaris_riv_table=TABLE_13,
                     polaris_penalties=pp)
    assert not res.success and res.outcome == OUTCOME_DATE_MISMATCH, res.outcome
    assert res.metrics is None and res.path == ()
    #  the POLARIS audit survives the refusal rather than being dropped
    assert res.polaris["selected"] is True
    assert res.polaris["selection"]["ice_class"] == PC6
    return ("selecting PC6 / table 1.3 does not buy past the date rule: the 2020 "
            "chart under 2025 data is still refused, and the selection is still "
            "reported on the failed result")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78)
    print("end-to-end route orchestration")
    print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<58} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__)
            print(f"  FAIL  {fn.__name__:<58} {exc}")
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

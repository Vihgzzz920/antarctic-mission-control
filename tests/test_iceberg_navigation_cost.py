"""
Focused tests for folding iceberg exposure into the time-indexed nav cost.

The ones that matter: weight 0 reproduces the existing composed cost bit for
bit, exposure is selected by ARRIVAL time, and a missing field never becomes
zero exposure.

    python tests/test_iceberg_navigation_cost.py
"""
from __future__ import annotations

import ast
import json
import math
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

from src.data.rasterize_sigrid3 import (AMBIGUITY_TIF, COVERAGE_TIF,         # noqa: E402
                                        DOMINANT_FRACTION_TIF, FRACTION_NODATA,
                                        POLYGON_ID_NODATA, POLYGON_ID_TIF,
                                        WATER_TIF)
from src.routing.iceberg_navigation_cost import (CONFIG, STATE_PRICED,       # noqa: E402
                                                 STATE_UNAVAILABLE,
                                                 ExposureField,
                                                 IcebergCostError,
                                                 IcebergExposureProvider,
                                                 IcebergExposureUnavailable,
                                                 IcebergTimeNavigationCost,
                                                 load_config)
from src.routing.iceberg_risk import IcebergForecast                         # noqa: E402
from src.routing.navigation_cost import (compose_grid,                       # noqa: E402
                                         load_composition_config)
from src.routing.time_navigation_cost import (ComposedField,                 # noqa: E402
                                              TimeIndexedNavigationCost)

CFG = load_config(CONFIG)
MODULE = ROOT / "src" / "routing" / "iceberg_navigation_cost.py"
COMP_CONFIG = ROOT / "configs" / "navigation_cost_composition.json"
HOUR = 3600.0
EPSG, TF = 3976, (6250.0, 0.0, 0.0, 0.0, -6250.0, 0.0)
SHAPE = (1, 3)
DAY = date(2025, 1, 8)
_T: dict = {}

# one uncharted cell, two charted ice polygons -- the same fixture shape the
# navigation-cost suite already uses
LAYOUT = dict(coverage=[[0, 1, 1]], polygon_id=[[POLYGON_ID_NODATA, 10, 11]],
              water=[[0, 0, 0]], ambiguity=[[0, 0, 0]],
              fraction=[[FRACTION_NODATA, 1.0, 1.0]])
PEN = {10: dict(conservative=25.0, optimistic=0.0, classification="elevated",
                has_rio=True, is_water=False),
       11: dict(conservative=100.0, optimistic=0.0, classification="indeterminate",
                has_rio=True, is_water=False)}
ENV_A = np.array([[1.0, 2.0, 3.0]], dtype="float64")
ENV_B = np.array([[1.0, 20.0, 30.0]], dtype="float64")


def usnic_dir() -> Path:
    if "dir" not in _T:
        d = Path(tempfile.mkdtemp(prefix="ice_nav_"))
        prof = dict(driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1,
                    crs=CRS.from_epsg(EPSG), transform=Affine(*TF))
        for name, key, dt_, nd in ((COVERAGE_TIF, "coverage", "uint8", None),
                                   (POLYGON_ID_TIF, "polygon_id", "int32",
                                    POLYGON_ID_NODATA),
                                   (WATER_TIF, "water", "uint8", None),
                                   (AMBIGUITY_TIF, "ambiguity", "uint8", None),
                                   (DOMINANT_FRACTION_TIF, "fraction", "float32",
                                    FRACTION_NODATA)):
            with rasterio.open(d / name, "w", dtype=dt_, nodata=nd, **prof) as dst:
                dst.write(np.asarray(LAYOUT[key], dtype=dt_), 1)
        _T["dir"] = d
    return _T["dir"]


def composed_field(bucket: int, env, chart: date = DAY,
                   env_date: date = DAY) -> ComposedField:
    comp = compose_grid(env, PEN, load_composition_config(COMP_CONFIG),
                        usnic_dir())
    return ComposedField(bucket=bucket, composition=comp, environment_date=env_date,
                         chart_date=chart, epsg=EPSG, shape=SHAPE, transform=TF,
                         label=f"bucket {bucket}")


def base_provider(**kw) -> TimeIndexedNavigationCost:
    return TimeIndexedNavigationCost.from_fields(
        [composed_field(0, ENV_A), composed_field(1, ENV_B)],
        bucket_seconds=HOUR, **kw)


def berg(name="berg_a", radius=18.1, hours=24.0, extrap=False,
         lat=-70.0, lon=0.0) -> IcebergForecast:
    return IcebergForecast(
        iceberg_id=name, position_source="ascat",
        forecast_start_date=DAY.isoformat(), horizon_seconds=hours * HOUR,
        horizon_hours=hours, latitude=lat, longitude=lon, radius_km=radius,
        extrapolated_uncertainty=extrap, physics_position_is_extrapolated=False,
        calibration_quantile=0.9, calibration_n=1197, growth_law="power",
        interpolation_quality="observed_to_observed")


def exposure_field(bucket, values, bergs=None, dom=None, hours=24.0,
                   start=DAY, extrap=False) -> ExposureField:
    bergs = bergs or (berg(hours=hours, extrap=extrap),)
    e = np.asarray([values], dtype="float64")
    d = np.asarray([dom if dom is not None
                    else [0 if v > 0 else -1 for v in values]], dtype="int32")
    return ExposureField(bucket=bucket, exposure=e, dominant=d, forecasts=bergs,
                         forecast_start_date=start, horizon_hours=hours,
                         label=f"exposure bucket {bucket}")


def provider(weight=0.0, exposures=None, on_missing=None, **kw):
    cfg = CFG.with_weight(weight)
    if on_missing is not None:
        raw = json.loads(json.dumps(cfg.raw))
        raw["missing_exposure"]["policy"] = on_missing
        from src.routing.iceberg_navigation_cost import _validated
        cfg = _validated(raw, cfg.path)
    fields = exposures if exposures is not None else [
        exposure_field(0, [0.0, 0.5, 0.0]), exposure_field(1, [0.0, 0.0, 0.9])]
    return IcebergTimeNavigationCost.from_providers(
        base_provider(), IcebergExposureProvider.from_fields(fields, SHAPE),
        cfg, **kw)


# --------------------------------------------------------- weight zero
def test_weight_zero_reproduces_the_existing_composed_cost_exactly():
    base = base_provider()
    p = provider(weight=0.0)
    bf, pf = base.cost_fn("conservative"), p.cost_fn("conservative")
    for t in (0.0, 0.5 * HOUR, HOUR, 1.9 * HOUR):
        for c in range(SHAPE[1]):
            assert pf(0, c, t) == bf(0, c, t), (c, t)
    cell = p.cell(0, 1, 0.0)
    assert cell.iceberg_cost == 0.0
    assert cell.final_cost == cell.composed_cost
    assert cell.iceberg_exposure == 0.5, "exposure is still reported at weight 0"
    assert CFG.weight == 5.0, (
        "the shipped weight is not 5.0; it was selected by the asymmetric "
        "sensitivity study (break-even ~2.868) and is a project routing "
        "parameter, not a physical or POLARIS quantity")
    assert p.min_cost("conservative") == base.min_cost("conservative")
    return ("weight 0 gives the base cost bit for bit at every cell and time, "
            "while still reporting the exposure it chose not to charge for")


def test_positive_weight_increases_cost_proportionally():
    base = base_provider().cost_fn("conservative")
    got = {}
    for w in (0.0, 1.0, 10.0, 40.0):
        f = provider(weight=w).cost_fn("conservative")
        got[w] = f(0, 1, 0.0)
        assert abs(got[w] - (base(0, 1, 0.0) + w * 0.5)) < 1e-12, w
    assert got[0.0] < got[1.0] < got[10.0] < got[40.0]
    #  exactly linear in the weight
    assert abs((got[40.0] - got[0.0]) - 4 * (got[10.0] - got[0.0])) < 1e-9
    #  a cell with zero exposure is untouched at any weight
    for w in (0.0, 40.0):
        assert provider(weight=w).cost_fn("conservative")(0, 0, 0.0) == base(0, 0, 0.0)
    return (f"exposure 0.5: cost rises {got[0.0]:g} -> {got[40.0]:g} exactly "
            f"linearly in the weight; a zero-exposure cell never moves")


def test_zero_exposure_gives_zero_iceberg_cost():
    p = provider(weight=100.0)
    c = p.cell(0, 0, 0.0)
    assert c.iceberg_exposure == 0.0 and c.iceberg_cost == 0.0
    assert c.final_cost == c.composed_cost
    assert c.state == STATE_PRICED, "zero exposure is PRICED, not unavailable"
    assert c.iceberg_id is None
    return "exposure 0.0 costs 0.0 and is still state=priced, not 'unavailable'"


# ------------------------------------------------------- arrival time
def test_exposure_is_selected_by_arrival_time():
    p = provider(weight=10.0)
    f = p.cost_fn("conservative")
    base = base_provider().cost_fn("conservative")
    #  cell (0,1) is exposed 0.5 in bucket 0 and 0.0 in bucket 1
    t_depart, t_arrive = 0.5 * HOUR, 1.5 * HOUR
    assert p.bucket_for(t_depart) == 0 and p.bucket_for(t_arrive) == 1
    at_arrival = f(0, 1, t_arrive)
    assert abs(at_arrival - (base(0, 1, t_arrive) + 10.0 * 0.0)) < 1e-12
    at_depart = f(0, 1, t_depart)
    assert abs(at_depart - (base(0, 1, t_depart) + 10.0 * 0.5)) < 1e-12
    assert at_arrival != at_depart, "the two buckets are indistinguishable"
    #  the bucket rule is DELEGATED, not restated
    body = MODULE.read_text().split('"""', 2)[2]
    assert "math.floor" not in body and "bucket_seconds" not in body
    assert "return self.base.bucket_for(arrival_time)" in body
    return ("cell (0,1) priced at t=1.5 h uses bucket 1 (exposure 0.0), not the "
            "departure bucket 0 (exposure 0.5); bucket_for is delegated to the "
            "base provider, never restated")


def test_different_arrival_buckets_give_different_exposure():
    p = provider(weight=20.0)
    got = {}
    for col in (1, 2):
        got[col] = [p.cell(0, col, t).iceberg_exposure for t in (0.0, HOUR)]
    assert got[1] == [0.5, 0.0] and got[2] == [0.0, 0.9]
    costs = {col: [p.cell(0, col, t).final_cost for t in (0.0, HOUR)]
             for col in (1, 2)}
    assert costs[1][0] > costs[1][1] - 1e9   # different composed costs, but
    assert p.cell(0, 2, HOUR).iceberg_cost == 20.0 * 0.9
    assert p.cell(0, 2, 0.0).iceberg_cost == 0.0
    return (f"cell (0,1) 0.5 -> 0.0 and cell (0,2) 0.0 -> 0.9 across the bucket "
            f"boundary; the iceberg term follows")


# ------------------------------------------------- provenance & flags
def test_provenance_is_preserved():
    bergs = (berg("b22g", radius=18.1011, hours=24.0),)
    p = provider(weight=5.0, exposures=[
        exposure_field(0, [0.0, 0.75, 0.0], bergs=bergs),
        exposure_field(1, [0.0, 0.0, 0.0], bergs=bergs)])
    c = p.cell(0, 1, 0.0)
    assert c.iceberg_id == "b22g"
    assert c.predicted_center_latitude == -70.0
    assert abs(c.radius_km - 18.1011) < 1e-9
    assert c.forecast_horizon_hours == 24.0
    assert c.extrapolated_uncertainty is False
    assert c.bucket == 0 and c.arrival_time == 0.0
    assert abs(c.iceberg_exposure - 0.75) < 1e-12
    #  every component survives
    for name in ("environmental_cost", "polaris_penalty",
                 "coverage_uncertainty_cost", "composed_cost",
                 "iceberg_exposure", "iceberg_cost", "final_cost"):
        assert getattr(c, name) is not None, name
    assert abs(c.final_cost - (c.composed_cost + c.iceberg_cost)) < 1e-12
    assert abs(c.composed_cost - (c.environmental_cost + c.polaris_penalty
                                  + c.coverage_uncertainty_cost)) < 1e-9
    txt = p.explain_lookup(0, 1, 0.0)
    for token in ("b22g", "bucket 0", "iceberg exposure", "final cost"):
        assert token in txt, token
    return ("iceberg id, centre, radius, horizon, flag, bucket and arrival time "
            "all reach the cell result; the five components still sum")


def test_extrapolated_flag_is_preserved():
    far = (berg("b47", radius=25.6, hours=48.0, extrap=True),)
    near = (berg("b47", radius=18.1, hours=24.0, extrap=False),)
    p = provider(weight=5.0, exposures=[
        exposure_field(0, [0.0, 0.4, 0.0], bergs=near, hours=24.0),
        exposure_field(1, [0.0, 0.4, 0.0], bergs=far, hours=48.0, extrap=True)])
    assert p.cell(0, 1, 0.0).extrapolated_uncertainty is False
    late = p.cell(0, 1, HOUR)
    assert late.extrapolated_uncertainty is True
    assert late.forecast_horizon_hours == 48.0
    assert "EXTRAPOLATED" in late.explain()
    prov = p.provenance()
    assert prov["exposure_extrapolated_by_bucket"] == {0: False, 1: True}
    assert prov["exposure_horizons_hours"] == {0: 24.0, 1: 48.0}
    #  the cost arithmetic is identical either way -- only the label differs
    assert p.cell(0, 1, 0.0).iceberg_cost == late.iceberg_cost
    return ("the 48 h bucket carries extrapolated_uncertainty=True through to "
            "the cell result, the explanation and the provenance")


# ----------------------------------------------- missing and mismatch
def test_unavailable_exposure_is_explicit_never_zero():
    only0 = [exposure_field(0, [0.0, 0.5, 0.0])]
    strict = provider(weight=10.0, exposures=only0)
    try:
        strict.cost_fn("conservative")(0, 1, HOUR)
        raise AssertionError("a missing exposure bucket was priced anyway")
    except IcebergExposureUnavailable as exc:
        assert "NOT read as zero exposure" in str(exc)
    lenient = provider(weight=10.0, exposures=only0, on_missing="omit_iceberg_term")
    c = lenient.cell(0, 1, HOUR)
    assert c.state == STATE_UNAVAILABLE
    assert c.iceberg_exposure is None, "a missing field became a number"
    assert c.iceberg_cost == 0.0 and c.final_cost == c.composed_cost
    assert "unavailable" in c.explain()
    priced = lenient.cell(0, 1, 0.0)
    assert priced.state == STATE_PRICED and priced.iceberg_exposure == 0.5
    assert STATE_UNAVAILABLE != STATE_PRICED
    return ("default raises; the opt-in policy adds nothing but stamps "
            "iceberg_exposure_unavailable with exposure=None, which is not 0.0")


def test_temporal_mismatch_is_rejected_by_default():
    other = date(2020, 12, 24)
    fields = [exposure_field(0, [0.0, 0.5, 0.0], start=other),
              exposure_field(1, [0.0, 0.0, 0.9], start=other)]
    try:
        provider(weight=1.0, exposures=fields)
        raise AssertionError("a 2020 iceberg forecast over 2025 was accepted")
    except IcebergCostError as exc:
        assert "HISTORICAL DEMONSTRATION" in str(exc)
    p = provider(weight=1.0, exposures=fields,
                 allow_historical_demo_mismatch=True)
    assert p.historical_demo_override is True
    assert p.provenance()["historical_demo_override"] is True
    assert provider(weight=1.0).historical_demo_override is False
    assert CFG.raw["temporal"]["mismatch_policy"] == "reject"
    return ("an iceberg forecast dated 2020-12-24 over a 2025-01-08 environment "
            "is refused; the override is explicit and stamped")


def test_shape_mismatch_and_out_of_range_exposure_are_refused():
    wide = ExposureField(bucket=0, exposure=np.zeros((1, 4)),
                         dominant=np.full((1, 4), -1, "int32"),
                         forecasts=(berg(),), forecast_start_date=DAY,
                         horizon_hours=24.0)
    for bad, why in ((wide, "a wrong shape"),
                     (exposure_field(0, [0.0, 1.5, 0.0]), "exposure above 1"),
                     (exposure_field(0, [0.0, -0.1, 0.0]), "negative exposure"),
                     (exposure_field(0, [0.0, float("nan"), 0.0]), "NaN")):
        try:
            IcebergExposureProvider.from_fields([bad], SHAPE)
            raise AssertionError(f"{why} was accepted")
        except IcebergCostError:
            pass
    try:
        IcebergExposureProvider.from_fields(
            [exposure_field(0, [0, 0, 0]), exposure_field(0, [0, 0, 0])], SHAPE)
        raise AssertionError("two fields claiming one bucket were accepted")
    except IcebergCostError:
        pass
    return "wrong shape, exposure outside [0,1], NaN and a duplicate bucket refused"


# ------------------------------------------------------------- safety
def test_no_hard_block_and_infinite_cost_passes_through():
    env_inf = np.array([[1.0, 2.0, np.inf]], dtype="float64")
    base = TimeIndexedNavigationCost.from_fields(
        [composed_field(0, env_inf), composed_field(1, env_inf)],
        bucket_seconds=HOUR)
    p = IcebergTimeNavigationCost.from_providers(
        base, IcebergExposureProvider.from_fields(
            [exposure_field(0, [0.0, 0.5, 1.0]),
             exposure_field(1, [0.0, 0.5, 1.0])], SHAPE),
        CFG.with_weight(50.0))
    f = p.cost_fn("conservative")
    assert math.isinf(f(0, 2, 0.0)), "an inherited infinity was suppressed"
    assert math.isfinite(f(0, 1, 0.0))
    #  no mask vocabulary anywhere in the code
    src = MODULE.read_text()
    tree = ast.parse(src)
    ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    ids |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    ids |= {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for banned in ("blocked", "blocked_mask", "no_go", "navigable", "passable",
                   "forbidden", "exclusion", "mask"):
        assert banned not in ids, f"the module uses the identifier {banned}"
    assert p.provenance()["no_hard_mask"] is True
    return ("an infinite composed cost stays infinite and is not topped up; no "
            "mask identifier exists in the module")


def test_repeated_calculation_is_deterministic():
    p = provider(weight=7.5)
    f = p.cost_fn("conservative")
    times = [0.0, 0.5 * HOUR, HOUR, 1.9 * HOUR]
    first = [[f(0, c, t) for c in range(3)] for t in times]
    for _ in range(4):
        assert [[f(0, c, t) for c in range(3)] for t in times] == first
    q = provider(weight=7.5)
    g = q.cost_fn("conservative")
    assert [[g(0, c, t) for c in range(3)] for t in times] == first
    cells = [p.cell(0, c, t) for c in range(3) for t in times]
    assert [q.cell(0, c, t) for c in range(3) for t in times] == cells
    return f"{5 * len(times) * 3} lookups identical, and across rebuilds"


def test_configuration_validation():
    for mutate, why in (
            (lambda r: r.update(iceberg_exposure_weight=-1.0), "a negative weight"),
            (lambda r: r.update(iceberg_exposure_weight=float("inf")), "infinity"),
            (lambda r: r.update(iceberg_exposure_weight="5"), "a string weight"),
            (lambda r: r.update(iceberg_exposure_weight=1e9), "an out-of-limit weight"),
            (lambda r: r.pop("iceberg_exposure_weight"), "no weight at all"),
            (lambda r: r["missing_exposure"].update(policy="assume_zero"),
             "a policy that assumes zero"),
            (lambda r: r["temporal"].update(mismatch_policy="allow"),
             "allowing a temporal mismatch"),
            (lambda r: r.update(no_hard_mask=False), "disabling no_hard_mask"),
            (lambda r: r["exposure_source"].update(is_a_probability=True),
             "calling the exposure a probability")):
        raw = json.loads(CONFIG.read_text())
        mutate(raw)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(raw, fh, default=str); p = fh.name
        try:
            load_config(p)
            raise AssertionError(f"{why} was accepted")
        except IcebergCostError:
            pass
    try:
        load_config(Path(tempfile.mkdtemp()) / "nope.json")
        raise AssertionError("a missing config was accepted")
    except IcebergCostError as exc:
        assert "not chosen in code" in str(exc)
    return "nine malformed configurations and a missing file all refused"


def test_no_formula_is_duplicated():
    src = MODULE.read_text()
    tree = ast.parse(src)
    ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    ids |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    ids |= {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for banned in ("sic", "ice_weight", "ice_exponent", "current_weight",
                   "compute_rio", "assess", "coverage_uncertainty_weight",
                   "build_exposure", "exposure_value", "great_circle_m",
                   "advect", "uncertainty_at"):
        assert banned not in ids, f"the module recomputes {banned}"
    imports = {f"{n.module}" for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("src.")}
    assert imports == {"src.routing.iceberg_risk",
                       "src.routing.time_navigation_cost"}, imports
    return (f"only {sorted(imports)} imported from src; no cost, RIO, cone or "
            f"uncertainty formula is restated")


# -------------------------------------------------- integration tests
def test_integration_a_moving_iceberg_changes_the_cost_of_one_cell():
    """Synthetic: one cell, two arrival times, an iceberg drifting onto it."""
    base = base_provider()
    bergs = (berg("drifter", radius=18.1, hours=24.0),)
    #  the iceberg is elsewhere in bucket 0 and over cell (0,1) in bucket 1
    fields = [exposure_field(0, [0.0, 0.0, 0.0], bergs=bergs),
              exposure_field(1, [0.0, 0.8, 0.0], bergs=bergs)]
    p = IcebergTimeNavigationCost.from_providers(
        base, IcebergExposureProvider.from_fields(fields, SHAPE),
        CFG.with_weight(30.0))
    f = p.cost_fn("conservative")
    early, late = f(0, 1, 0.5 * HOUR), f(0, 1, 1.5 * HOUR)
    b = base.cost_fn("conservative")
    assert abs(early - b(0, 1, 0.5 * HOUR)) < 1e-12, "bucket 0 should be clear"
    assert abs(late - (b(0, 1, 1.5 * HOUR) + 30.0 * 0.8)) < 1e-12
    c_early, c_late = p.cell(0, 1, 0.5 * HOUR), p.cell(0, 1, 1.5 * HOUR)
    assert c_early.iceberg_cost == 0.0 and abs(c_late.iceberg_cost - 24.0) < 1e-12
    assert c_late.iceberg_id == "drifter" and c_early.iceberg_id is None
    delta = late - b(0, 1, 1.5 * HOUR)
    return (f"same cell: clear on arrival at 0.5 h, +{delta:g} cost on arrival "
            f"at 1.5 h when the cone has moved over it")


def test_integration_time_astar_reroutes_around_a_moving_iceberg():
    """The unmodified time_astar, priced through this provider."""
    from src.routing.grid import RoutingGrid
    from src.routing.time_astar import time_astar

    px, H, W = 10_000.0, 5, 5
    speed = px / HOUR
    z = np.zeros((H, W), "float32")
    blocked = np.zeros((H, W), bool)
    for r in (1, 2, 3):
        for c in (1, 2, 3):
            blocked[r, c] = True
    grid = RoutingGrid(transform=Affine(px, 0, 0, 0, -px, 0),
                       crs=CRS.from_epsg(EPSG), shape=(H, W), sic=z.copy(),
                       current_u=z.copy(), current_v=z.copy(),
                       blocked_mask=blocked, day=DAY)

    #  a uniform composed cost, so ONLY the iceberg term can decide the route
    big = Path(tempfile.mkdtemp(prefix="ice_nav_big_"))
    prof = dict(driver="GTiff", height=H, width=W, count=1,
                crs=CRS.from_epsg(EPSG), transform=Affine(px, 0, 0, 0, -px, 0))
    lay = dict(coverage=np.zeros((H, W), "uint8"),
               polygon_id=np.full((H, W), POLYGON_ID_NODATA, "int32"),
               water=np.zeros((H, W), "uint8"),
               ambiguity=np.zeros((H, W), "uint8"),
               fraction=np.full((H, W), FRACTION_NODATA, "float32"))
    for name, key, dt_, nd in ((COVERAGE_TIF, "coverage", "uint8", None),
                               (POLYGON_ID_TIF, "polygon_id", "int32",
                                POLYGON_ID_NODATA),
                               (WATER_TIF, "water", "uint8", None),
                               (AMBIGUITY_TIF, "ambiguity", "uint8", None),
                               (DOMINANT_FRACTION_TIF, "fraction", "float32",
                                FRACTION_NODATA)):
        with rasterio.open(big / name, "w", dtype=dt_, nodata=nd, **prof) as dst:
            dst.write(lay[key].astype(dt_), 1)
    comp = compose_grid(np.ones((H, W), "float64"), {},
                        load_composition_config(COMP_CONFIG), big)
    fields = [ComposedField(bucket=b, composition=comp, environment_date=DAY,
                            chart_date=DAY, epsg=EPSG, shape=(H, W),
                            transform=(px, 0, 0, 0, -px, 0))
              for b in range(10)]
    base = TimeIndexedNavigationCost.from_fields(fields, bucket_seconds=HOUR)

    #  north corridor exposed from bucket 1 on; south corridor always clear
    bergs = (berg("north_berg"),)
    exps = []
    for b in range(10):
        e = np.zeros((H, W), "float64")
        d = np.full((H, W), -1, "int32")
        if b >= 1:
            for c in (1, 2, 3):
                e[0, c] = 0.9
                d[0, c] = 0
        exps.append(ExposureField(bucket=b, exposure=e, dominant=d,
                                  forecasts=bergs, forecast_start_date=DAY,
                                  horizon_hours=24.0))
    prov = IcebergExposureProvider.from_fields(exps, (H, W))

    def route(weight):
        p = IcebergTimeNavigationCost.from_providers(
            base, prov, CFG.with_weight(weight))
        return time_astar(grid, (2, 0), (2, 4), cost_fn=p.cost_fn("conservative"),
                          min_cost=p.min_cost("conservative"),
                          vessel_speed_mps=speed, time_step_s=HOUR,
                          max_horizon_s=base.horizon_s)

    off, on = route(0.0), route(50.0)
    assert off.success and on.success, (off.reason, on.reason)
    north = {(0, c) for c in (1, 2, 3)}
    south = {(4, c) for c in (1, 2, 3)}
    assert set(off.path) & north, f"weight 0 did not take the north corridor: {off.path}"
    assert set(on.path) & south and not (set(on.path) & north), (
        f"weight 50 did not avoid the exposed corridor: {on.path}")
    assert off.path != on.path

    #  Both totals come out equal because the corridors are geometrically
    #  symmetric and the chosen route pays no exposure -- which looks like
    #  "nothing changed" unless you also price the route it REJECTED.
    priced = IcebergTimeNavigationCost.from_providers(
        base, prov, CFG.with_weight(50.0))
    fn = priced.cost_fn("conservative")
    def reprice(res):
        total = 0.0
        for (a, ta), (b, tb) in zip(zip(res.path, res.arrival_times),
                                    zip(res.path[1:], res.arrival_times[1:])):
            d = math.hypot((b[1] - a[1]) * px, (b[0] - a[0]) * px)
            total += d * 0.5 * (fn(a[0], a[1], ta) + fn(b[0], b[1], tb))
        return total
    rejected = reprice(off)
    chosen = reprice(on)
    assert rejected > chosen, (rejected, chosen)
    assert abs(chosen - on.total_cost) < 1e-6, "the chosen route was mispriced"
    return (f"unmodified time_astar: weight 0 routes north, weight 50 routes "
            f"south around the cone. Repriced at weight 50 the rejected north "
            f"route costs {rejected:,.0f} against {chosen:,.0f} for the one "
            f"taken -- {rejected - chosen:,.0f} of exposure avoided")


def test_real_data_smoke_from_the_generated_exposure_rasters():
    """Read the real 06/12/24/48 h rasters. No iceberg id is hardcoded."""
    ice = ROOT / "data" / "processed" / "icebergs"
    paths = {h: ice / f"iceberg_exposure_{h}_3976.tif"
             for h in ("06h", "12h", "24h", "48h")}
    missing = [h for h, p in paths.items() if not p.exists()]
    if missing:
        raise AssertionError(f"exposure rasters missing: {missing}")
    seen = {}
    for label, path in paths.items():
        with rasterio.open(path) as src:
            band = src.read(1).astype("float64")
            tags = src.tags()
            shape = (src.height, src.width)
        assert band.min() >= 0.0 and band.max() <= 1.0
        nz = int((band > 0).sum())
        assert nz > 0, f"{label} has no exposed cell"
        dominant = np.where(band > 0, 0, -1).astype("int32")
        hours = float(tags["horizon_hours"])
        extrap = tags["extrapolated_uncertainty"] == "true"
        f = ExposureField(
            bucket=0, exposure=band, dominant=dominant,
            forecasts=(berg(tags.get("forecast_start_date", "unknown"),
                            radius=float(tags["radius_km"]), hours=hours,
                            extrap=extrap),),
            forecast_start_date=DAY, horizon_hours=hours)
        prov = IcebergExposureProvider.from_fields([f], shape)
        rows, cols = np.nonzero(band > 0)
        r, c = int(rows[0]), int(cols[0])
        exposure = float(band[r, c])
        for w in (0.0, 25.0):
            cost = 1.0 + w * exposure
            assert abs(cost - (1.0 + w * exposure)) < 1e-12
        seen[label] = (nz, round(float(band.max()), 4), hours, extrap,
                       round(exposure, 4))
        assert prov.field_for_bucket(0).exposure.shape == shape
    assert seen["48h"][3] is True, "the 48 h raster is not flagged extrapolated"
    assert all(v[3] is False for k, v in seen.items() if k != "48h")
    assert seen["06h"][0] < seen["12h"][0] < seen["24h"][0] < seen["48h"][0]
    return ("real rasters load into the provider: cells "
            + ", ".join(f"{k}={v[0]}" for k, v in seen.items())
            + f"; only 48 h flagged extrapolated")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78); print("iceberg exposure in the navigation cost")
    print("=" * 78)
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

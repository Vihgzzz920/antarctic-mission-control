"""
Focused tests for the dynamic iceberg exposure layer.

The ones that matter: overlapping cones take the MAX and never the sum, the
distance rule is great-circle metres and never projected metres or degrees,
and nothing here produces a navigability mask.

    python tests/test_iceberg_risk.py
"""
from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess_icebergs import great_circle_m                    # noqa: E402
from src.models.iceberg_ml_residual import (CONFIG as SPLIT_CONFIG,        # noqa: E402
                                            build_population,
                                            chronological_split,
                                            load_config as load_split_config)
from src.models.iceberg_uncertainty import (CONFIG as UNC_CONFIG,          # noqa: E402
                                            SECONDS_PER_HOUR, calibrate,
                                            load_config as load_unc_config)
from src.routing.iceberg_risk import (CONFIG, EXPOSURE_CONSTANT,           # noqa: E402
                                      EXPOSURE_LINEAR, ExposureError,
                                      IcebergForecast, RoutingGridGeometry,
                                      _great_circle_grid, build_exposure,
                                      build_provenance, exposure_value,
                                      forecast_icebergs, load_config,
                                      write_raster)

CFG = load_config(CONFIG)
UCFG = load_unc_config(UNC_CONFIG)
MODULE = ROOT / "src" / "routing" / "iceberg_risk.py"
_C: dict = {}


def state():
    if "grid" not in _C:
        pop, _ = build_population()
        chronological_split(pop, load_split_config(SPLIT_CONFIG))
        _C["pop"] = [r for r in pop if r["moved"]]
        _C["cal"] = calibrate(pop, UCFG)
        _C["grid"] = RoutingGridGeometry()
    return _C


def fc(lat, lon, radius_km, berg="test", hours=24.0, extrap=False):
    return IcebergForecast(
        iceberg_id=berg, position_source="ascat",
        forecast_start_date="2025-01-08", horizon_seconds=hours * 3600.0,
        horizon_hours=hours, latitude=lat, longitude=lon, radius_km=radius_km,
        extrapolated_uncertainty=extrap, physics_position_is_extrapolated=False,
        calibration_quantile=0.9, calibration_n=1197, growth_law="power",
        interpolation_quality="observed_to_observed")


# ----------------------------------------------------- the exposure rule
def test_center_edge_and_outside():
    r = 18_000.0
    assert exposure_value(0.0, r) == 1.0, "the centre is not maximally exposed"
    assert exposure_value(r, r) == 0.0, "the edge is not zero"
    assert exposure_value(r + 1.0, r) == 0.0
    assert exposure_value(r * 1e6, r) == 0.0
    assert abs(exposure_value(r / 2, r) - 0.5) < 1e-12
    assert abs(exposure_value(r * 0.9, r) - 0.1) < 1e-12
    assert exposure_value(0.0, r, EXPOSURE_CONSTANT) == 1.0
    assert exposure_value(r * 0.99, r, EXPOSURE_CONSTANT) == 1.0
    assert exposure_value(r, r, EXPOSURE_CONSTANT) == 0.0
    return ("centre 1.0, half-radius 0.5, edge exactly 0.0, outside 0.0; the "
            "constant variant is 1 inside and 0 at the edge")


def test_zero_radius_exposes_nothing():
    for d in (0.0, 1.0, 1e6):
        assert exposure_value(d, 0.0) == 0.0, d
    arr = np.array([[0.0, 100.0]])
    from src.routing.iceberg_risk import _exposure_array
    assert (_exposure_array(arr, 0.0, EXPOSURE_LINEAR) == 0.0).all()
    g = state()["grid"]
    e, dom, _ = build_exposure([fc(-70.0, 0.0, 0.0)], g, CFG)
    assert e.max() == 0.0 and (dom == -1).all()
    return "a zero radius exposes no cell and divides by nothing"


def test_exposure_is_finite_and_within_zero_one():
    g = state()["grid"]
    forecasts = [fc(-70.0, 0.0, 18.1, "a"), fc(-70.05, 0.1, 25.6, "b"),
                 fc(-64.0, 179.98, 12.8, "c")]
    e, _, _ = build_exposure(forecasts, g, CFG)
    assert np.isfinite(e).all(), "a non-finite exposure appeared"
    assert e.min() >= 0.0 and e.max() <= 1.0, (e.min(), e.max())
    for bad in ((-1.0, 10.0), (float("nan"), 10.0), (1.0, float("inf"))):
        try:
            exposure_value(*bad)
            raise AssertionError(f"{bad} was accepted")
        except ExposureError:
            pass
    return f"grid exposure in [{e.min():.1f}, {e.max():.4f}]; bad inputs refused"


# ------------------------------------------------- overlap uses the max
def test_two_overlapping_cones_take_the_max_not_the_sum():
    """Synthetic: two icebergs 10 km apart, both with a 20 km radius."""
    g = state()["grid"]
    lat, lon, r = -70.0, 0.0, 20.0
    dlat = math.degrees(10_000.0 / 6_371_008.8)          # 10 km north
    a, b = fc(lat, lon, r, "berg_a"), fc(lat + dlat, lon, r, "berg_b")
    ea, _, _ = build_exposure([a], g, CFG)
    eb, _, _ = build_exposure([b], g, CFG)
    both, dom, _ = build_exposure([a, b], g, CFG)
    overlap = (ea > 0) & (eb > 0)
    assert overlap.sum() > 0, "the synthetic cones do not overlap"
    assert np.allclose(both, np.maximum(ea, eb)), "combination is not max"
    summed = ea + eb
    assert (summed[overlap] > both[overlap] + 1e-12).all(), (
        "the test point is not diagnostic: sum equals max there")
    assert both.max() <= 1.0 < summed.max(), (
        f"the sum reaches {summed.max():.3f}, outside [0,1]; max reaches "
        f"{both.max():.3f}")
    #  each overlapping cell is attributed to the iceberg that actually won
    ids = ["berg_a", "berg_b"]
    rows, cols = np.nonzero(overlap)
    for rr, cc in list(zip(rows.tolist(), cols.tolist()))[:50]:
        want = 0 if ea[rr, cc] >= eb[rr, cc] else 1
        assert ids[int(dom[rr, cc])] == ids[want], (rr, cc)
    return (f"{int(overlap.sum())} overlapping cells: combined == "
            f"max(a, b) everywhere, sum would reach {summed.max():.3f} > 1, "
            f"and each cell names the iceberg that won")


def test_contributing_iceberg_provenance():
    g = state()["grid"]
    a, b = fc(-70.0, 0.0, 20.0, "berg_a"), fc(-70.0, 0.6, 20.0, "berg_b")
    e, dom, _ = build_exposure([a, b], g, CFG)
    recs = build_provenance(e, dom, [a, b], g)
    assert recs and len(recs) == int((e > 0).sum())
    assert {r["dominant_iceberg_id"] for r in recs} == {"berg_a", "berg_b"}
    for rec in recs[:200]:
        want = 1.0 - rec["distance_km"] / rec["radius_km"]
        assert abs(rec["exposure"] - want) < 1e-5, rec
        assert rec["distance_km"] < rec["radius_km"]
        for key in ("row", "col", "predicted_center_latitude", "horizon_hours",
                    "extrapolated_uncertainty", "position_source"):
            assert key in rec
    assert not any(r["exposure"] == 0.0 for r in recs), "a zero cell was listed"
    return (f"{len(recs)} non-zero cells, each naming its iceberg with a "
            f"distance and radius that reproduce its exposure")


# ------------------------------------------- distance and the projection
def test_the_vectorised_distance_matches_the_project_scalar_function():
    g = state()["grid"]
    lat0, lon0 = -70.0, 12.0
    r0, c0 = g.rowcol_of(lat0, lon0)
    win = (slice(r0 - 4, r0 + 5), slice(c0 - 4, c0 + 5))
    got = _great_circle_grid(g.lat[win], g.lon[win], lat0, lon0)
    worst = 0.0
    for i in range(got.shape[0]):
        for j in range(got.shape[1]):
            want = great_circle_m(float(g.lat[win][i, j]),
                                  float(g.lon[win][i, j]), lat0, lon0)
            worst = max(worst, abs(got[i, j] - want))
    assert worst < 1e-6, f"vectorised distance differs by {worst} m"
    return f"81 cells: array haversine matches great_circle_m to {worst:.2e} m"


def test_no_degree_metre_mix_up_and_no_projected_distance_test():
    """Every exposure is a function of TRUE distance, not projected distance.

    Rather than assert that two snapped cell centres give the same number --
    they do not, because a 6.25 km cell quantises the offset differently at
    each latitude -- this measures what the projected-metre method WOULD have
    produced for the very same cells, and shows the two disagree exactly where
    the projection says they should.
    """
    from rasterio.warp import transform as warp_transform
    g = state()["grid"]
    scale_north, scale_south = g.local_scale(-62.0, 20.0), g.local_scale(-76.0, 20.0)
    assert scale_north > 1.03 and scale_south < 1.0, (scale_north, scale_south)

    r_km, report = 18.0, {}
    for lat in (-62.0, -70.0, -76.0):
        f = fc(lat, 20.0, r_km)
        e, dom, _ = build_exposure([f], g, CFG)
        rows, cols = np.nonzero(e > 0.0)
        assert rows.size > 0, lat
        cx, cy = warp_transform("EPSG:4326", "EPSG:3976", [20.0], [lat])
        worst_true = worst_proj = 0.0
        for rr, cc in zip(rows.tolist(), cols.tolist()):
            clat, clon = float(g.lat[rr, cc]), float(g.lon[rr, cc])
            d_true = great_circle_m(clat, clon, lat, 20.0)
            #  what the module actually produced must be the TRUE-distance answer
            worst_true = max(worst_true,
                             abs(float(e[rr, cc]) - (1.0 - d_true / (r_km * 1000))))
            px, py = warp_transform("EPSG:4326", "EPSG:3976", [clon], [clat])
            d_proj = math.hypot(px[0] - cx[0], py[0] - cy[0])
            worst_proj = max(worst_proj,
                             abs(float(e[rr, cc]) - (1.0 - d_proj / (r_km * 1000))))
        assert worst_true < 1e-6, (lat, worst_true)
        report[lat] = (rows.size, worst_proj)
    #  the wrong method is materially wrong in the north, less so in the south
    assert report[-62.0][1] > 0.02, report
    assert report[-62.0][1] > report[-76.0][1], report

    #  and the code converts km to metres and scales only the WINDOW
    body = MODULE.read_text().split('"""', 2)[2]
    assert "radius_km * 1000.0" in body, "the radius is not converted to metres"
    tree = ast.parse(MODULE.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "build_exposure")
    names = {m.attr for m in ast.walk(node) if isinstance(m, ast.Attribute)}
    assert "local_scale" in names, "the window is not scaled by the projection"
    return (f"exposure reproduces the TRUE-distance formula to <1e-6 at every "
            f"latitude; the projected-metre method would have been off by up to "
            f"{report[-62.0][1]:.3f} at 62S against {report[-76.0][1]:.3f} at 76S")


def test_search_window_never_clips_a_cone():
    """A generous window is the point: no cell inside the radius may be missed."""
    g = state()["grid"]
    for lat in (-62.0, -70.0, -76.0):
        f = fc(lat, 30.0, 25.6)
        e, _, _ = build_exposure([f], g, CFG)
        r0, c0 = g.rowcol_of(lat, 30.0)
        wide = 14
        win = (slice(max(0, r0 - wide), r0 + wide + 1),
               slice(max(0, c0 - wide), c0 + wide + 1))
        d = _great_circle_grid(g.lat[win], g.lon[win], lat, 30.0)
        should = d < f.radius_km * 1000.0
        got = e[win] > 0.0
        assert (should == got).all(), (
            f"at {lat}: {int((should & ~got).sum())} cells inside the radius "
            f"were missed by the window")
    return ("scanning a window far wider than the one the code uses finds no "
            "cell inside the radius that the code missed, at three latitudes")


def test_antimeridian():
    g = state()["grid"]
    f = fc(-68.0, 179.97, 20.0, "dateline")
    e, dom, _ = build_exposure([f], g, CFG)
    assert (e > 0).sum() > 0, "a cone on the dateline exposed nothing"
    recs = build_provenance(e, dom, [f], g)
    lons = [float(g.lon[r["row"], r["col"]]) for r in recs]
    assert any(l > 179.0 for l in lons) and any(l < -179.0 for l in lons), (
        f"the cone did not straddle 180: lon range {min(lons)} .. {max(lons)}")
    for rec in recs:
        assert rec["distance_km"] < rec["radius_km"]
        assert rec["distance_km"] < 25.0, "a wrapped longitude inflated a distance"
    assert abs(_great_circle_grid(np.array([[-68.0]]), np.array([[-179.98]]),
                                 -68.0, 179.98)[0, 0] - 1668.0) < 200.0
    return (f"{len(recs)} cells span longitudes {min(lons):.2f} to "
            f"{max(lons):.2f}; every distance stays under the radius")


# -------------------------------------------------------- the horizons
def test_zero_horizon_and_the_24h_and_48h_labels():
    s = state()
    rows = s["pop"][:3]
    z = forecast_icebergs(rows, 0.0, s["cal"], UCFG, CFG)
    assert all(f.radius_km == 0.0 for f in z)
    assert all(f.extrapolated_uncertainty is False for f in z)
    assert all(f.latitude == r["start_latitude"] for f, r in zip(z, rows))
    got = {}
    for hours in (6.0, 12.0, 24.0, 48.0):
        f = forecast_icebergs(rows, hours * SECONDS_PER_HOUR, s["cal"],
                              UCFG, CFG)[0]
        got[hours] = (round(f.radius_km, 4), f.extrapolated_uncertainty)
    assert got[24.0][1] is False and got[48.0][1] is True
    assert got[6.0][1] is False and got[12.0][1] is False
    assert abs(got[24.0][0] - s["cal"].radius_km) < 1e-3
    assert abs(got[48.0][0] / got[24.0][0] - math.sqrt(2)) < 1e-3
    assert got[6.0][0] < got[12.0][0] < got[24.0][0] < got[48.0][0]
    try:
        forecast_icebergs(rows, -1.0, s["cal"], UCFG, CFG)
        raise AssertionError("a negative horizon was accepted")
    except ExposureError:
        pass
    return (f"t=0 -> radius 0 at the start position; 6/12/24 h calibrated "
            f"{got[6.0][0]}/{got[12.0][0]}/{got[24.0][0]} km; 48 h "
            f"{got[48.0][0]} km flagged extrapolated")


def test_determinism():
    s = state()
    rows = s["pop"][:5]
    f1 = forecast_icebergs(rows, 24 * SECONDS_PER_HOUR, s["cal"], UCFG, CFG)
    f2 = forecast_icebergs(rows, 24 * SECONDS_PER_HOUR, s["cal"], UCFG, CFG)
    assert [x.as_dict() for x in f1] == [x.as_dict() for x in f2]
    e1, d1, st1 = build_exposure(f1, s["grid"], CFG)
    e2, d2, st2 = build_exposure(f2, s["grid"], CFG)
    assert np.array_equal(e1, e2) and np.array_equal(d1, d2) and st1 == st2
    assert build_provenance(e1, d1, f1, s["grid"]) == build_provenance(
        e2, d2, f2, s["grid"])
    return "identical forecasts, grids, dominance maps and provenance on a rerun"


# --------------------------------------------------------- the raster
def test_raster_alignment_and_that_zero_is_not_nodata():
    import rasterio
    from src.data.preprocess_currents import target_grid
    s = state()
    f = forecast_icebergs(s["pop"][:4], 24 * SECONDS_PER_HOUR, s["cal"],
                          UCFG, CFG)
    e, _, _ = build_exposure(f, s["grid"], CFG)
    d = Path(tempfile.mkdtemp())
    path = write_raster(e, s["grid"], 24.0, f, d)
    epsg, h, w, transform = target_grid()
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == epsg == 3976
        assert (src.height, src.width) == (h, w) == (1328, 1264)
        assert src.dtypes[0] == "float32"
        assert src.nodata is None, "a nodata value was written; 0 is real data"
        assert abs(abs(src.transform.a) - 6250.0) < 1e-9
        for i, (a, b) in enumerate(zip(list(src.transform)[:6],
                                       list(transform)[:6])):
            assert abs(a - b) < 1e-6, i
        tags = src.tags()
        assert tags["is_a_probability"] == "false"
        assert "NOT a statement that the water is clear" in tags["zero_meaning"]
        assert "no blocked" in tags["no_hard_mask"]
        band = src.read(1)
    assert np.isfinite(band).all() and band.min() >= 0.0 and band.max() <= 1.0
    return ("EPSG:3976, 1328x1264, 6250 m, float32, nodata=None; the tags say "
            "0 means no modeled exposure, not clear water")


def test_no_hard_mask_no_probability_no_routing_code():
    src = MODULE.read_text()
    body = src.split('"""', 2)[2]
    #  Scan IDENTIFIERS, not prose: the module's own disclaimer says it makes
    #  "no blocked/no-go/navigable mask", and that sentence is the opposite of
    #  a violation.
    tree = ast.parse(src)
    ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    ids |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    ids |= {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    ids |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    for banned in ("blocked_mask", "blocked", "no_go", "navigable", "passable",
                   "collision_probability", "astar", "time_astar",
                   "navigation_cost", "polaris", "RoutingGrid", "sklearn",
                   "size_major_km", "size_minor_km", "predict_corrected"):
        assert banned not in ids, f"the module uses the identifier {banned}"
    for banned in ("import sklearn", "from sklearn", "def astar",
                   "probability_of("):
        assert banned not in body, f"the module references {banned}"
    #  Where these words DO appear they must be DENIALS, not uses. Scan the
    #  AST's string CONSTANTS, not source lines: Python joins implicitly
    #  concatenated literals, so "not a " / "collision probability" split
    #  across two lines is one string to the parser and two to a line reader.
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    for text in literals:
        low = " ".join(text.lower().split())
        if "collision probability" in low:
            assert ("not a collision probability" in low
                    or "no collision probability" in low
                    or "not a probability" in low), text
        if "navigable" in low or "no-go" in low:
            assert ("no blocked" in low or "produces no" in low
                    or "never" in low), text
    #  and the one identifier carrying the phrase is an explicit False
    for line in body.splitlines():
        if "collision_prob" in line:
            assert line.strip().startswith(
                '"is_a_collision_probability": False'), line
    assert CFG.raw["exposure"]["is_a_probability"] is False
    assert CFG.raw["population"]["uses_ml_residual_model"] is False
    assert CFG.raw["population"]["uses_iceberg_size"] is False
    #  it imports the frozen models rather than restating them
    src = MODULE.read_text()
    assert "from src.models.iceberg_physics import" in src
    assert "from src.models.iceberg_uncertainty import" in src
    assert "def advect" not in body and "def uncertainty_at" not in body
    return ("no mask, no probability, no router, no ML, no size; physics and "
            "uncertainty are imported, not restated")


def test_configuration_validation():
    for mutate, why in (
            (lambda r: r["exposure"].update(function="magic"), "unknown function"),
            (lambda r: r["exposure"].update(combination_rule="sum"), "sum"),
            (lambda r: r["exposure"].update(combination_rule="mean"), "mean"),
            (lambda r: r["exposure"].update(is_a_probability=True),
             "claiming it is a probability"),
            (lambda r: r["output"].update(nodata=0.0), "nodata = 0"),
            (lambda r: r.update(horizons_hours=[]), "no horizons"),
            (lambda r: r.update(horizons_hours=[-6.0]), "a negative horizon"),
            (lambda r: r["population"].update(uses_ml_residual_model=True),
             "enabling the ML residual"),
            (lambda r: r["population"].update(uses_iceberg_size=True),
             "enabling iceberg size")):
        raw = json.loads(CONFIG.read_text())
        mutate(raw)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(raw, fh); p = fh.name
        try:
            load_config(p)
            raise AssertionError(f"{why} was accepted")
        except ExposureError:
            pass
    try:
        load_config(Path(tempfile.mkdtemp()) / "nope.json")
        raise AssertionError("a missing config was accepted")
    except ExposureError:
        pass
    return ("nine malformed configurations refused, including sum, mean, "
            "nodata=0, and any attempt to enable ML or iceberg size")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78); print("dynamic iceberg exposure layer"); print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<54} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__); print(f"  FAIL  {fn.__name__:<54} {exc}")
        except Exception as exc:                        # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__:<54} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

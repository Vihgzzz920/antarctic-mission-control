"""
Focused tests for the SIGRID-3 -> EPSG:3976 rasterisation layer, on the real chart.

Cells are assigned by TRUE AREA INTERSECTION. Cases the real chart does not
contain (a genuine source overlap) are built from grid-aligned boxes and are
labelled CONSTRUCTED in the output.

    python tests/test_rasterize_sigrid3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess_currents import target_grid                      # noqa: E402
from src.data.rasterize_sigrid3 import (AMBIGUITY_MIXED,                  # noqa: E402
                                        AMBIGUITY_SINGLE,
                                        AMBIGUITY_SOURCE_OVERLAP,
                                        AMBIGUITY_TIF, COVERAGE_TIF,
                                        DOMINANT_FRACTION_TIF, FRACTION_NODATA,
                                        MIXED_SIDECAR, POLYGON_ID_NODATA,
                                        POLYGON_ID_TIF, SIDECAR, WATER_TIF,
                                        DOWNSTREAM_POLICY,
                                        assign_cells, assign_cells_by_area,
                                        build_rasters, read_shapefile_polygons,
                                        read_source_crs)

CHART = ROOT / "data" / "raw" / "usnic" / "ANTARC20201224"
OUT = ROOT / "data" / "processed" / "usnic"
_S: dict = {}


def setup():
    if not _S:
        if not CHART.exists():
            raise SystemExit(f"missing chart: {CHART}")
        res, recs = build_rasters(CHART)
        _S["res"], _S["recs"], _S["grid"] = res, recs, target_grid()
    return _S


def n(v):
    return "" if v is None else str(v).strip()


def box(x0, y0, x1, y1):
    return {"type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def cell_x(k):           # left edge of column k
    _, _, _, tf = setup()["grid"]
    return tf.c + k * tf.a


def cell_y(k):           # top edge of row k
    _, _, _, tf = setup()["grid"]
    return tf.f + k * tf.e


# --------------------------------------------------------- grid / plumbing
def test_outputs_are_on_the_exact_routing_grid():
    epsg, h, w, tf = setup()["grid"]
    names = (COVERAGE_TIF, POLYGON_ID_TIF, WATER_TIF, AMBIGUITY_TIF,
             DOMINANT_FRACTION_TIF)
    for name in names:
        p = OUT / name
        assert p.exists(), f"{name} not written"
        with rasterio.open(p) as src:
            assert src.crs.to_epsg() == epsg
            assert (src.height, src.width) == (h, w)
            assert all(abs(a - b) <= 1e-6 for a, b in
                       zip(list(src.transform)[:6], list(tf)[:6]))
            assert abs(src.transform.a) == 6250.0 and abs(src.transform.e) == 6250.0
    return f"{len(names)} rasters on EPSG:{epsg} {h}x{w} @6250 m"


def test_nodata_is_explicit_and_correct():
    with rasterio.open(OUT / POLYGON_ID_TIF) as src:
        assert src.dtypes[0] == "int32" and src.nodata == POLYGON_ID_NODATA
    with rasterio.open(OUT / DOMINANT_FRACTION_TIF) as src:
        assert src.dtypes[0] == "float32" and src.nodata == FRACTION_NODATA
        a = src.read(1)
        assert ((a == FRACTION_NODATA) | ((a > 0) & (a <= 1.0))).all()
    for name, allowed in ((COVERAGE_TIF, {0, 1}), (WATER_TIF, {0, 1}),
                          (AMBIGUITY_TIF, {0, 1, 2})):
        with rasterio.open(OUT / name) as src:
            assert src.dtypes[0] == "uint8" and src.nodata is None
            assert set(np.unique(src.read(1)).tolist()) <= allowed
    return (f"polygon_id nodata={POLYGON_ID_NODATA}, fraction nodata="
            f"{FRACTION_NODATA}, categorical bands carry none")


def test_source_crs_is_read_and_is_not_the_target():
    d = read_source_crs(CHART / f"{CHART.name}.prj").to_dict()
    epsg, *_ = setup()["grid"]
    assert abs(float(d["lat_ts"]) + 60.0) < 1e-9 and abs(float(d["lon_0"]) - 180.0) < 1e-9
    return (f"source lat_ts={d['lat_ts']} lon_0={d['lon_0']} -> EPSG:{epsg} "
            f"(lat_ts -70, lon_0 0): real reprojection")


def test_geometry_reader_handles_holes():
    from src.data.rasterize_sigrid3 import _signed_area
    rings = read_shapefile_polygons(CHART / f"{CHART.name}.shp")
    assert len(rings) == 680
    multi = [r for r in rings if len(r) > 1]
    for r in rings[:200]:
        assert _signed_area(r[0]) < 0
        for hole in r[1:]:
            assert _signed_area(hole) >= 0
    return f"680 records; {len(multi)} with holes; max rings {max(len(r) for r in rings)}"


# ------------------------------------------------- the area assignment rule
def test_polygon_covering_a_cell_without_its_centre_is_still_coverage():
    _, h, w, tf = setup()["grid"]
    # a sliver over the LEFT 30% of cell (10,10): never contains the centre
    g = box(cell_x(10), cell_y(11), cell_x(10) + 0.30 * tf.a, cell_y(10))
    pid, dom, tot, ncon, mixed, per = assign_cells_by_area([g], h, w, tf)
    assert pid[10, 10] == 0, "area intersection must find it"
    frac = dom[10, 10] / per
    assert 0.25 < frac < 0.35, f"expected ~0.30 of the cell, got {frac}"
    ccount, cid, _, _ = assign_cells([g], h, w, tf)
    assert cid[10, 10] == POLYGON_ID_NODATA, "the centre rule misses it"
    res = setup()["res"]
    real = int(((res.coverage == 1) & (res.centre_polygon_id == POLYGON_ID_NODATA)).sum())
    return (f"CONSTRUCTED sliver: area frac {frac:.2f} charted, centre rule blind; "
            f"real chart: {real:,} such cells")


def test_dominant_area_wins_and_ties_go_to_the_lowest_id():
    _, h, w, tf = setup()["grid"]
    small = box(cell_x(20), cell_y(21), cell_x(20) + 0.25 * tf.a, cell_y(20))
    big = box(cell_x(20) + 0.25 * tf.a, cell_y(21), cell_x(21), cell_y(20))
    pid, dom, tot, ncon, mixed, per = assign_cells_by_area([small, big], h, w, tf)
    assert pid[20, 20] == 1, "the 75% polygon must win over the 25% one"
    assert abs(dom[20, 20] / per - 0.75) < 0.02
    # exact halves -> lowest id wins, deterministically
    lo = box(cell_x(30), cell_y(31), cell_x(30) + 0.5 * tf.a, cell_y(30))
    hi = box(cell_x(30) + 0.5 * tf.a, cell_y(31), cell_x(31), cell_y(30))
    pid2, _, _, _, _, _ = assign_cells_by_area([lo, hi], h, w, tf)
    pid3, _, _, _, _, _ = assign_cells_by_area([hi, lo], h, w, tf)
    assert pid2[30, 30] == 0 and pid3[30, 30] == 0, "an exact tie goes to index 0"
    return "75/25 -> larger wins; exact 50/50 tie -> lowest polygon_id, both orders"


def test_adjacent_polygons_in_one_cell_are_mixed_not_overlap():
    _, h, w, tf = setup()["grid"]
    left = box(cell_x(40), cell_y(41), cell_x(40) + 0.5 * tf.a, cell_y(40))
    right = box(cell_x(40) + 0.5 * tf.a, cell_y(41), cell_x(41), cell_y(40))
    pid, dom, tot, ncon, mixed, per = assign_cells_by_area([left, right], h, w, tf)
    assert ncon[40, 40] == 2
    share = tot[40, 40] / per
    assert abs(share - 1.0) < 1e-9, f"adjacent polygons must partition the cell, got {share}"
    assert (40, 40) in mixed and len(mixed[(40, 40)]) == 2
    return f"CONSTRUCTED adjacency: 2 contributors, total area share {share:.3f} (not >1)"


def test_genuine_source_overlap_is_distinguished():
    _, h, w, tf = setup()["grid"]
    a = box(cell_x(50), cell_y(51), cell_x(51), cell_y(50))          # whole cell
    b = box(cell_x(50), cell_y(51), cell_x(50) + 0.5 * tf.a, cell_y(50))  # left half, on top
    pid, dom, tot, ncon, mixed, per = assign_cells_by_area([a, b], h, w, tf)
    share = tot[50, 50] / per
    assert share > 1.0, f"overlapping polygons must sum above 1, got {share}"
    assert ncon[50, 50] == 2
    assert pid[50, 50] == 0, "the larger area still wins, deterministically"
    res = setup()["res"]
    assert int((res.ambiguity == AMBIGUITY_SOURCE_OVERLAP).sum()) == 0
    assert res.total_fraction.max() <= 1.0 + 1e-9
    return (f"CONSTRUCTED overlap: area share {share:.2f} > 1; real chart max share "
            f"{res.total_fraction.max():.4f} -> 0 genuine overlaps")


def test_boundary_only_contact_is_not_an_intersection():
    _, h, w, tf = setup()["grid"]
    a = box(cell_x(60), cell_y(61), cell_x(61), cell_y(60))
    b = box(cell_x(61), cell_y(61), cell_x(62), cell_y(60))   # shares the edge exactly
    pid, dom, tot, ncon, mixed, per = assign_cells_by_area([a, b], h, w, tf)
    assert ncon[60, 60] == 1 and ncon[60, 61] == 1, "a shared edge adds no area"
    assert not mixed, "boundary contact must not produce a mixed cell"
    return "CONSTRUCTED shared edge: 1 contributor per cell, 0 mixed cells"


def test_real_chart_mixed_cells_are_all_adjacency():
    res = setup()["res"]
    mixed = json.loads((OUT / MIXED_SIDECAR).read_text())
    assert mixed["by_kind"].get("source_overlap", 0) == 0
    assert mixed["cells_listed"] == int((res.counts >= 2).sum())
    assert mixed["cells_listed"] == len(res.mixed)
    for c in mixed["cells"][:200]:
        assert c["total_area_share"] <= 1.0 + 1e-6
        assert c["kind"] == "adjacent"
        assert len(c["intersecting"]) >= 2
        assert c["assigned_polygon_id"] == c["intersecting"][0]["polygon_id"], \
            "the listed polygons must be ordered with the winner first"
    return (f"{mixed['cells_listed']:,} mixed cells, all adjacency; max contributors "
            f"{int(res.counts.max())}")


# ---------------------------------------------------------- semantics kept
def test_coverage_and_unknown_are_complementary():
    res = setup()["res"]
    cov = res.coverage == 1
    assert np.array_equal(cov, res.polygon_id != POLYGON_ID_NODATA)
    return f"charted {int(cov.sum()):,}  uncharted {int((~cov).sum()):,}"


def test_missing_coverage_never_becomes_ice_free():
    res = setup()["res"]
    unc = res.coverage == 0
    assert (res.polygon_id[unc] == POLYGON_ID_NODATA).all()
    assert (res.water[unc] == 0).all()
    assert (res.dominant_fraction[unc] == FRACTION_NODATA).all(), \
        "an uncovered cell must carry fraction nodata, never 0.0"
    side = json.loads((OUT / SIDECAR).read_text())
    for i in np.unique(res.polygon_id[res.coverage == 1]).tolist():
        assert str(i) in side["polygons"]
    with rasterio.open(OUT / COVERAGE_TIF) as src:
        assert "NOT ice-free" in src.tags().get("critical", "")
    return (f"{int(unc.sum()):,} uncovered: id=-1, water=0, fraction={FRACTION_NODATA}, "
            f"metadata states 0 is NOT ice-free")


def test_water_polygons_are_preserved_as_water():
    res, recs = setup()["res"], setup()["recs"]
    wids = {i for i, r in enumerate(recs) if n(r.get("POLY_TYPE")) == "W"}
    assert len(wids) == 5
    wc = res.water == 1
    assert set(np.unique(res.polygon_id[wc]).tolist()) <= wids
    assert (res.coverage[wc] == 1).all()
    side = json.loads((OUT / SIDECAR).read_text())
    for i in wids:
        e = side["polygons"][str(i)]
        assert e["is_water"] and e["rio"] is None
        assert "not a sea-ice regime" in e["rio_omitted_reason"]
        assert e["interpretation_state"] != "unknown"
    return f"5 water polygons -> {int(wc.sum()):,} cells, charted, no RIO, not unknown"


def test_polygon_provenance_is_intact():
    res, recs = setup()["res"], setup()["recs"]
    used = set(np.unique(res.polygon_id[res.polygon_id != POLYGON_ID_NODATA]).tolist())
    assert used <= set(range(len(recs)))
    side = json.loads((OUT / SIDECAR).read_text())
    assert len(side["polygons"]) == len(recs) == 680
    for i in list(used)[:50]:
        assert side["polygons"][str(i)]["routing_cells"] == res.cells_per_polygon[i]
    return f"{len(used)} ids in the raster, all in the chart; sidecar has all 680"


def test_unknown_interpretation_stays_unknown():
    side = json.loads((OUT / SIDECAR).read_text())
    unk = [e for e in side["polygons"].values() if e["interpretation_state"] == "unknown"]
    assert unk
    for e in unk:
        assert e["rio"] is None and not e["is_water"]
        assert "no POLARIS ice type and no RIO are invented" in e["rio_omitted_reason"]
    return f"{len(unk)} unknown polygons: no RIO, no invented ice type"


def test_uncertainty_is_not_collapsed():
    side = json.loads((OUT / SIDECAR).read_text())
    withrio = [e for e in side["polygons"].values() if e["rio"]]
    e = next(x for x in withrio if x["mapping_sensitive"])
    r = e["rio"][next(iter(e["rio"]))]
    for f in ("conservative_rio", "conservative_classification", "optimistic_rio",
              "optimistic_classification", "mapping_classification_difference",
              "band_straddling"):
        assert f in r
    assert any(x["iceberg_bearing"] for x in side["polygons"].values())
    return (f"{len(withrio)} polygons with RIO for {len(side['ice_classes'])} classes "
            f"x {len(side['riv_tables'])} tables, both branches kept")


def test_no_hard_block_or_navigability_output():
    written = {p.name for p in OUT.glob("*")}
    for f in written:
        for k in ("navig", "no_go", "nogo", "block", "forbid", "passab", "cost"):
            assert k not in f.lower(), f"output looks like a navigation decision: {f}"
    for name in (COVERAGE_TIF, WATER_TIF, AMBIGUITY_TIF, POLYGON_ID_TIF,
                 DOMINANT_FRACTION_TIF):
        with rasterio.open(OUT / name) as src:
            blob = " ".join(src.tags().values()).lower()
            for k in ("no_go", "blocked", "navigable", "passable"):
                assert k not in blob
    return f"{len(written)} outputs, none a navigability or cost layer"


def test_rasterisation_is_deterministic():
    a, _ = build_rasters(CHART)
    b = setup()["res"]
    for f in ("coverage", "polygon_id", "water", "ambiguity", "dominant_fraction",
              "counts", "total_fraction"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f"{f} differs on a rerun"
    assert a.cells_per_polygon == b.cells_per_polygon
    assert a.mixed == b.mixed
    return "all bands, counts, fractions and the mixed-cell record identical on a rerun"


def test_mixed_cells_are_not_presented_as_full_coverage():
    """Regression: a consumer must not be able to read assignment as full coverage.

    Gap found in audit: 8,599 cells are partially covered by a SINGLE polygon, so
    the ambiguity flag is legitimately 0 for them and they are absent from the
    mixed sidecar. Only dominant_fraction records their partial coverage, and
    nothing said so where a consumer would look.
    """
    res = setup()["res"]
    charted = res.coverage == 1
    partial = charted & (res.dominant_fraction < 1.0)
    mixed = res.counts >= 2
    partial_single = partial & ~mixed
    assert partial_single.any(), "expected partially covered single-polygon cells"
    assert (mixed & charted).all() == (mixed & charted).all()
    assert not (mixed & (res.dominant_fraction >= 1.0)).any(), \
        "a mixed cell cannot be fully covered by its winner"

    # the contract must be stated where a consumer reads it
    for name in (POLYGON_ID_TIF, COVERAGE_TIF, DOMINANT_FRACTION_TIF, AMBIGUITY_TIF):
        with rasterio.open(OUT / name) as src:
            pol = src.tags().get("downstream_policy", "")
        assert "DOES NOT IMPLY FULL COVERAGE" in pol, f"{name} lacks the contract"
        assert "NEVER combine" in pol and "arithmetic" in pol.lower(), \
            f"{name} does not forbid arithmetic combination of RIOs"
        assert "never turn a mixed cell into a hard navigation block" in pol

    side = json.loads((OUT / SIDECAR).read_text())
    assert DOWNSTREAM_POLICY == side["downstream_policy"]
    mixed_side = json.loads((OUT / MIXED_SIDECAR).read_text())
    assert "not_listed_here" in mixed_side and "do not use the ambiguity flag alone" \
        in mixed_side["not_listed_here"]

    # per-polygon coverage quality must reconcile with the rasters
    for i, e in list(side["polygons"].items())[:200]:
        i = int(i)
        assert e["routing_cells_full"] + e["routing_cells_partial"] == e["routing_cells"]
        assert e["routing_cells_mixed"] <= e["routing_cells_partial"], \
            "every mixed cell is by definition partial"
        assert e["routing_cells_full"] == int(((res.polygon_id == i) &
                                               (res.dominant_fraction >= 1.0)).sum())
    return (f"{int(partial.sum()):,} partial cells ({int(partial_single.sum()):,} of "
            f"them single-polygon and NOT flagged mixed); contract present on 4 "
            f"rasters and both sidecars; per-polygon counts reconcile")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    setup()
    print("=" * 78); print(f"SIGRID-3 rasterisation (area rule) -- {CHART.name}")
    print("=" * 78)
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

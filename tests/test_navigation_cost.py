"""
Focused tests for the navigation cost composer.

Most cases use a small synthetic USNIC raster set so every situation (dominant
fraction 0.8, 0.3, water, unknown, uncharted) can be hit exactly. Two tests run
against the real ANTARC20201224 rasters.

    python tests/test_navigation_cost.py
"""
from __future__ import annotations

import json
import sys
import tempfile
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
from src.routing.navigation_cost import (STATE_ICE, STATE_UNCHARTED,          # noqa: E402
                                         STATE_UNKNOWN_REGIME, STATE_WATER,
                                         CompositionError, compose_grid,
                                         load_composition_config, summarise)

REAL_USNIC = ROOT / "data" / "processed" / "usnic"
CONFIG = ROOT / "configs" / "navigation_cost_composition.json"

# synthetic layout: one row, six cells
#  col 0: uncharted
#  col 1: charted ice, fully covered            (fraction 1.0)
#  col 2: charted ice, MIXED, fraction 0.8
#  col 3: charted ice, MIXED, fraction 0.3
#  col 4: charted ice, PARTIAL single polygon   (fraction 0.6, not mixed)
#  col 5: charted water
#  col 6: charted, polygon has no RIO (unknown regime)
LAYOUT = dict(
    coverage=[0, 1, 1, 1, 1, 1, 1],
    polygon_id=[POLYGON_ID_NODATA, 10, 11, 12, 13, 14, 15],
    water=[0, 0, 0, 0, 0, 1, 0],
    ambiguity=[0, 0, 1, 1, 0, 0, 0],
    fraction=[FRACTION_NODATA, 1.0, 0.8, 0.3, 0.6, 1.0, 1.0],
)
PENALTIES = {
    10: dict(conservative=25.0, optimistic=0.0, classification="elevated_operational_risk",
             has_rio=True, is_water=False),
    11: dict(conservative=100.0, optimistic=0.0, classification="indeterminate",
             has_rio=True, is_water=False),
    12: dict(conservative=0.0, optimistic=0.0, classification="normal_operation",
             has_rio=True, is_water=False),
    13: dict(conservative=25.0, optimistic=25.0, classification="elevated_operational_risk",
             has_rio=True, is_water=False),
    14: dict(conservative=0.0, optimistic=0.0, classification=None,
             has_rio=True, is_water=True),
    15: dict(conservative=100.0, optimistic=100.0, classification="unknown",
             has_rio=False, is_water=False),
}
ENV = np.array([[2.0, 1.0, 1.5, 3.0, 1.2, 1.0, np.inf]], dtype="float64")
_TMP: dict = {}


def synthetic_dir() -> Path:
    if "dir" not in _TMP:
        d = Path(tempfile.mkdtemp(prefix="usnic_synth_"))
        tf = Affine(6250.0, 0, 0, 0, -6250.0, 0)
        prof = dict(driver="GTiff", height=1, width=7, count=1,
                    crs=CRS.from_epsg(3976), transform=tf)
        for name, key, dt, nd in ((COVERAGE_TIF, "coverage", "uint8", None),
                                  (POLYGON_ID_TIF, "polygon_id", "int32",
                                   POLYGON_ID_NODATA),
                                  (WATER_TIF, "water", "uint8", None),
                                  (AMBIGUITY_TIF, "ambiguity", "uint8", None),
                                  (DOMINANT_FRACTION_TIF, "fraction", "float32",
                                   FRACTION_NODATA)):
            with rasterio.open(d / name, "w", dtype=dt, nodata=nd, **prof) as dst:
                dst.write(np.array([LAYOUT[key]], dtype=dt), 1)
        _TMP["dir"] = d
    return _TMP["dir"]


def compose(weight=None):
    cfg = load_composition_config(CONFIG)
    if weight is not None:
        raw = dict(cfg.raw); raw["coverage_uncertainty_weight"] = weight
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(raw, f); p = f.name
        cfg = load_composition_config(p)
    return compose_grid(ENV, PENALTIES, cfg, synthetic_dir()), cfg


# ------------------------------------------------------------------ tests
def test_ordinary_fully_covered_cell():
    out, _ = compose()
    c = out.cell(0, 1)
    assert c.state == STATE_ICE and c.charted and not c.mixed and not c.partial
    assert c.dominant_fraction == 1.0 and c.coverage_uncertainty == 0.0
    assert c.composed_cost == c.environmental_cost + c.polaris_penalty
    return f"env 1.0 + POLARIS 25 + cover 0 = {c.composed_cost:g}"


def test_mixed_cell_fraction_08():
    out, _ = compose()
    c = out.cell(0, 2)
    assert c.mixed and c.partial and abs(c.dominant_fraction - 0.8) < 1e-6
    assert abs(c.coverage_uncertainty - 0.2) < 1e-6
    assert c.polaris_penalty == 100.0, "dominant polygon's penalty, unmodified"
    return (f"frac 0.8 -> uncertainty 0.2; penalty {c.polaris_penalty:g} unchanged; "
            f"composed {c.composed_cost:g}")


def test_mixed_cell_fraction_03():
    out, _ = compose()
    c = out.cell(0, 3)
    assert c.mixed and abs(c.dominant_fraction - 0.3) < 1e-6
    assert abs(c.coverage_uncertainty - 0.7) < 1e-6
    assert c.polaris_penalty == 0.0, "a 30% winner still supplies the whole penalty"
    return f"frac 0.3 -> uncertainty 0.7; penalty {c.polaris_penalty:g}; not averaged"


def test_partial_single_polygon_cell_is_not_mixed():
    out, _ = compose()
    c = out.cell(0, 4)
    assert c.partial and not c.mixed, "partial coverage != mixed polygons"
    assert abs(c.coverage_uncertainty - 0.4) < 1e-6
    assert c.state == STATE_ICE
    return "fraction 0.6, single polygon: partial=True mixed=False, uncertainty 0.4"


def test_uncharted_cell_gets_no_polaris_and_no_uncertainty():
    out, _ = compose(weight=10.0)
    c = out.cell(0, 0)
    assert c.state == STATE_UNCHARTED and not c.charted
    assert c.polygon_id is None and c.dominant_fraction is None
    assert c.polaris_penalty == 0.0, "absence of a chart is not a POLARIS finding"
    assert c.coverage_uncertainty == 0.0 and c.coverage_uncertainty_cost == 0.0
    assert c.composed_cost == c.environmental_cost, "environment still applies"
    return (f"uncharted: POLARIS 0, uncertainty 0 even at weight 10, composed "
            f"{c.composed_cost:g} = environmental only")


def test_water_cell_keeps_water_state_and_no_rio():
    out, _ = compose(weight=10.0)
    c = out.cell(0, 5)
    assert c.state == STATE_WATER and c.is_water and c.charted
    assert c.polaris_penalty == 0.0, "no fabricated sea-ice cost for charted water"
    assert c.coverage_uncertainty_cost == 0.0, "fully covered water: 1-1.0 = 0"
    return f"charted water: state={c.state}, POLARIS 0, composed {c.composed_cost:g}"


def test_unknown_regime_uses_the_risk_adapters_unknown_penalty():
    out, _ = compose()
    c = out.cell(0, 6)
    assert c.state == STATE_UNKNOWN_REGIME
    assert c.state != STATE_UNCHARTED, "unknown regime is NOT uncharted"
    assert c.polaris_penalty == 100.0, "the adapter's configured unknown penalty"
    assert not np.isfinite(c.environmental_cost)
    assert not np.isfinite(c.composed_cost), "an inherited +inf passes through"
    return "charted-but-unknown kept distinct from uncharted; unknown penalty 100"


def test_polaris_penalty_changes_the_final_cost():
    out, _ = compose()
    a = out.cell(0, 1)
    alt = dict(PENALTIES); alt[10] = dict(PENALTIES[10]); alt[10]["conservative"] = 75.0
    cfg = load_composition_config(CONFIG)
    out2 = compose_grid(ENV, alt, cfg, synthetic_dir())
    b = out2.cell(0, 1)
    assert b.composed_cost - a.composed_cost == 50.0
    assert b.environmental_cost == a.environmental_cost
    return f"penalty 25 -> 75 moves composed {a.composed_cost:g} -> {b.composed_cost:g}"


def test_weight_zero_contributes_nothing():
    out, cfg = compose()
    assert cfg.coverage_uncertainty_weight == 0.0, "the shipped default must be 0.0"
    assert float(out.coverage_uncertainty_cost.max()) == 0.0
    for col in (2, 3, 4):
        c = out.cell(0, col)
        assert c.coverage_uncertainty > 0, "the signal is still measured"
        assert c.coverage_uncertainty_cost == 0.0, "but contributes nothing at w=0"
        assert c.composed_cost == c.environmental_cost + c.polaris_penalty
    return "weight 0.0: signal preserved, cost contribution exactly zero"


def test_changing_the_weight_changes_only_that_component():
    a, _ = compose(weight=0.0)
    b, _ = compose(weight=5.0)
    assert np.array_equal(a.polaris_conservative, b.polaris_conservative)
    assert np.array_equal(a.environmental_cost, b.environmental_cost)
    assert np.array_equal(a.coverage_uncertainty, b.coverage_uncertainty)
    c = b.cell(0, 3)
    assert abs(c.coverage_uncertainty_cost - 5.0 * 0.7) < 1e-5
    delta = b.conservative_cost[0, 3] - a.conservative_cost[0, 3]
    assert abs(delta - 3.5) < 1e-5   # dominant_fraction is stored float32
    return "w 0 -> 5: POLARIS and environmental untouched; only the third term moves"


def test_no_area_weighted_rio():
    """A cell's penalty equals the dominant polygon's, never a blend."""
    out, _ = compose()
    for col, pid in ((1, 10), (2, 11), (3, 12), (4, 13)):
        c = out.cell(0, col)
        assert c.polaris_penalty == PENALTIES[pid]["conservative"], \
            "the penalty must be taken verbatim from the dominant polygon"
    # a cell whose winner holds 80% of it still carries 100% of that penalty
    c2 = out.cell(0, 2)
    assert c2.dominant_fraction < 1.0 and PENALTIES[11]["conservative"] > 0
    assert c2.polaris_penalty == PENALTIES[11]["conservative"] == 100.0
    assert c2.polaris_penalty != c2.dominant_fraction * PENALTIES[11]["conservative"], \
        "the penalty must not be scaled by the area fraction"
    return ("every cell's penalty is its dominant polygon's, unscaled and unblended "
            "(0.8-covered cell still carries the full 100)")


def test_no_hard_block_output():
    out, _ = compose(weight=5.0)
    names = {n.lower() for n in dir(out) if not n.startswith("_")}
    for k in ("blocked", "no_go", "nogo", "navigable", "passable", "forbidden", "mask"):
        assert k not in names, f"the composer exposes {k}"
    finite = np.isfinite(out.environmental_cost)
    assert np.isfinite(out.conservative_cost[finite]).all(), \
        "no infinity may be introduced where the environment was priceable"
    assert not np.isfinite(out.conservative_cost[~finite]).any(), \
        "an inherited infinity must survive"
    try:
        load_composition_config.__wrapped__      # noqa: B018
    except AttributeError:
        pass
    return "no mask attribute; +inf only where cost.py already had it"


def test_rejects_invalid_configuration():
    for bad, why in ((float("inf"), "infinite"), (-1.0, "negative"), ("x", "not a number")):
        raw = json.loads(CONFIG.read_text())
        raw["coverage_uncertainty_weight"] = bad
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(raw, f, default=str); p = f.name
        try:
            load_composition_config(p)
            raise AssertionError(f"a {why} weight must be rejected")
        except CompositionError:
            pass
    return "infinite, negative and non-numeric weights all rejected by the loader"


def test_composition_is_deterministic_and_non_negative():
    a, _ = compose(weight=3.0)
    b, _ = compose(weight=3.0)
    for f in ("conservative_cost", "optimistic_cost", "polaris_conservative",
              "coverage_uncertainty", "coverage_uncertainty_cost", "state"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f"{f} differs on a rerun"
    finite = np.isfinite(a.environmental_cost)
    assert (a.conservative_cost[finite] >= 0).all()
    assert (a.optimistic_cost[finite] >= 0).all()
    return "identical on a rerun; all finite composed costs >= 0"


def test_real_chart_composition_runs_and_balances():
    if not (REAL_USNIC / COVERAGE_TIF).exists():
        raise AssertionError("real USNIC rasters missing")
    env = np.full((1328, 1264), 1.5, dtype="float64")
    pen = {i: dict(conservative=25.0, optimistic=0.0, classification="x",
                   has_rio=True, is_water=False) for i in range(680)}
    out = compose_grid(env, pen, load_composition_config(CONFIG), REAL_USNIC)
    s = summarise(out)
    assert s["cells"] == 1678592
    assert s["by_state"][STATE_UNCHARTED] == 575962
    assert s["charted"] == 1102630
    assert s["coverage_uncertainty_cost_max"] == 0.0, "shipped weight is 0"
    assert not s["any_negative"]
    return (f"{s['charted']:,} charted / {s['by_state'][STATE_UNCHARTED]:,} uncharted; "
            f"max composed {s['composed_conservative_finite_max']:g}")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("=" * 78); print("navigation cost composer"); print("=" * 78)
    failures = []
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<52} {fn() or ''}")
        except AssertionError as exc:
            failures.append(fn.__name__); print(f"  FAIL  {fn.__name__:<52} {exc}")
        except Exception as exc:                        # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__:<52} {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

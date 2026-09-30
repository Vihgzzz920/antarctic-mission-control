"""
Dynamic iceberg EXPOSURE field on the routing grid.

    data/processed/icebergs/iceberg_exposure_{06,12,24,48}h_3976.tif
    data/processed/icebergs/iceberg_risk_evaluation.json
    data/processed/icebergs/iceberg_risk_provenance.json

THIS IS AN EXPOSURE INDEX, NOT A PROBABILITY
  The value in a cell is how deep inside one iceberg's calibrated uncertainty
  cone that cell lies:

      exposure = 1 - distance / radius     for distance < radius, else 0

  0.8 does not mean an 80% chance of ice. It means the cell sits 80% of the way
  from the edge of a cone to its predicted centre. It is not a collision
  probability, not a hit rate, and not an official maritime safety metric.

  It is also not a mask. This layer produces no blocked, no-go or navigable
  field; hard constraints belong to NavigationDomain and constraints.py.

MAX, NOT SUM, NOT MEAN
  Where cones overlap, the cell takes the LARGEST single-iceberg exposure and
  records which iceberg produced it. Summing would make two distant icebergs
  look worse than one adjacent one; averaging would let a far iceberg dilute a
  near one. Neither is defensible, and both would quietly leave [0, 1].

DISTANCE IS GREAT-CIRCLE, BECAUSE THE RADIUS IS
  The uncertainty radius is the q90 of RADIAL GREAT-CIRCLE endpoint errors.
  Testing a cell against it in projected metres would compare two different
  quantities. EPSG:3976 is polar stereographic with its standard parallel at
  70S: measured over the real iceberg band, a kilometre of ground is 0.987 to
  1.042 projected kilometres, so a Euclidean test would be wrong by up to ~4%,
  worst at the northern edge where the bergs are.

  So the projection is used ONLY to find an over-inclusive window of candidate
  cells around each iceberg, and every distance that decides a value is a true
  great-circle distance between the cell centre and the predicted centre. Cell
  centre latitude/longitude are read from the grid geometry preprocess_currents
  already cached; nothing recomputes them, and no degree is ever compared with
  a metre.

NOTHING HERE IS NEW PHYSICS
  Centres come from the frozen physics baseline via iceberg_physics.advect,
  radii from iceberg_uncertainty.uncertainty_at. No drift formula, no radius
  formula and no calibration is restated in this file. The ML residual model is
  deliberately not used: it measured worse than the physics baseline on the
  held-out moved 24 h rows.

WHAT 0 MEANS, AND WHAT IT DOES NOT
  0 is a real value: no modeled exposure from the icebergs in this forecast.
  The rasters therefore carry NO nodata. It does NOT mean the water is clear --
  only 37 distinct icebergs have clean moved observations in this dataset, all
  ASCAT, and anything outside that population is invisible here. That
  distinction lives in the raster tags and the evaluation JSON, not in a
  sentinel that would quietly turn absence into safety.

NOT IN SCOPE
  No collision probability, no navigation-cost integration, no A*, no route
  selection, no replanning, no ML, no new uncertainty calibration.

Usage:
    python -m src.routing.iceberg_risk --audit          # counts, writes nothing
    python -m src.routing.iceberg_risk --date 2025-01-08
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.preprocess_currents import grid_geometry, target_grid
from src.data.preprocess_icebergs import great_circle_m, normalise_longitude
from src.models.iceberg_ml_residual import (CONFIG as SPLIT_CONFIG,
                                            build_population,
                                            chronological_split,
                                            load_config as load_split_config)
from src.models.iceberg_physics import ICE_DIR, advect
from src.models.iceberg_uncertainty import (CONFIG as UNC_CONFIG,
                                            SECONDS_PER_HOUR, Calibration,
                                            calibrate,
                                            load_config as load_unc_config,
                                            uncertainty_at)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "iceberg_risk.json"
EVAL_JSON = ICE_DIR / "iceberg_risk_evaluation.json"
PROV_JSON = ICE_DIR / "iceberg_risk_provenance.json"
CURRENTS_DIR = ROOT / "data" / "processed" / "currents"

EXPOSURE_LINEAR, EXPOSURE_CONSTANT = "linear", "constant"


class ExposureError(ValueError):
    """The iceberg exposure field cannot be built as specified."""


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class RiskConfig:
    raw: dict
    path: str

    @property
    def function(self) -> str:
        return str(self.raw["exposure"]["function"])

    @property
    def horizons_hours(self) -> list[float]:
        return [float(h) for h in self.raw["horizons_hours"]]

    @property
    def window_scale(self) -> float:
        return float(self.raw["distance"]["window_safety_scale"])

    @property
    def window_extra(self) -> int:
        return int(self.raw["distance"]["window_extra_cells"])

    @property
    def max_icebergs(self) -> int:
        return int(self.raw["limits"]["max_icebergs"])

    @property
    def max_radius_km(self) -> float:
        return float(self.raw["limits"]["max_radius_km"])

    def with_function(self, name: str) -> "RiskConfig":
        raw = json.loads(json.dumps(self.raw))
        raw["exposure"]["function"] = name
        return RiskConfig(raw=raw, path=self.path)


def load_config(path: Path | str = CONFIG) -> RiskConfig:
    """Read and validate. A config that claims a probability is refused."""
    path = Path(path)
    if not path.exists():
        raise ExposureError(
            f"No exposure configuration at {path}. The exposure function and "
            f"the combination rule are project parameters and must be stated "
            f"in a file, not chosen in code.")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ExposureError(f"{path.name} is not valid JSON: {exc}") from None
    exp = raw.get("exposure", {})
    if exp.get("function") not in exp.get("available_functions", []):
        raise ExposureError(
            f"exposure.function is {exp.get('function')!r}; available_functions "
            f"are {exp.get('available_functions')}")
    if exp.get("combination_rule") != "max":
        raise ExposureError(
            f"exposure.combination_rule is {exp.get('combination_rule')!r}. "
            f"Only 'max' is supported: summing or averaging cone exposures "
            f"leaves [0, 1] and misrepresents overlap.")
    if exp.get("is_a_probability") is not False:
        raise ExposureError(
            "exposure.is_a_probability must be false. This index is not a "
            "collision probability and the config must say so.")
    if raw.get("output", {}).get("nodata") is not None:
        raise ExposureError(
            "output.nodata must be null: 0 means 'no modeled exposure' and is "
            "a real value, never a sentinel for missing data.")
    if not raw.get("horizons_hours"):
        raise ExposureError("horizons_hours is empty")
    for h in raw["horizons_hours"]:
        if not isinstance(h, (int, float)) or h < 0 or not math.isfinite(h):
            raise ExposureError(f"horizon {h!r} must be finite and >= 0")
    if raw.get("population", {}).get("uses_ml_residual_model") is not False:
        raise ExposureError("population.uses_ml_residual_model must be false")
    if raw.get("population", {}).get("uses_iceberg_size") is not False:
        raise ExposureError("population.uses_iceberg_size must be false")
    return RiskConfig(raw=raw, path=str(path))


# ------------------------------------------------------------- exposure
def exposure_value(distance_m: float, radius_m: float,
                   function: str = EXPOSURE_LINEAR) -> float:
    """The project exposure index for one cell against one cone.

    At or beyond the radius: 0. A zero radius exposes nothing, because no
    point is strictly inside it -- that is also what a zero forecast horizon
    produces, and it must not become a divide-by-zero.
    """
    if distance_m < 0 or not math.isfinite(distance_m):
        raise ExposureError(f"distance {distance_m} must be finite and >= 0")
    if radius_m < 0 or not math.isfinite(radius_m):
        raise ExposureError(f"radius {radius_m} must be finite and >= 0")
    if radius_m == 0.0 or distance_m >= radius_m:
        return 0.0
    if function == EXPOSURE_CONSTANT:
        return 1.0
    if function == EXPOSURE_LINEAR:
        return 1.0 - distance_m / radius_m
    raise ExposureError(f"unknown exposure function {function!r}")


def _exposure_array(distance_m: np.ndarray, radius_m: float,
                    function: str) -> np.ndarray:
    if radius_m <= 0.0:
        return np.zeros(distance_m.shape, dtype="float64")
    inside = distance_m < radius_m
    out = np.zeros(distance_m.shape, dtype="float64")
    if function == EXPOSURE_CONSTANT:
        out[inside] = 1.0
    else:
        out[inside] = 1.0 - distance_m[inside] / radius_m
    return out


# ------------------------------------------------------------- forecast
@dataclass(frozen=True)
class IcebergForecast:
    """One iceberg's predicted centre and cone at one horizon."""

    iceberg_id: str
    position_source: str
    forecast_start_date: str
    horizon_seconds: float
    horizon_hours: float
    latitude: float
    longitude: float
    radius_km: float
    extrapolated_uncertainty: bool
    physics_position_is_extrapolated: bool
    calibration_quantile: float
    calibration_n: int
    growth_law: str
    interpolation_quality: str

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def forecast_icebergs(rows: list[dict], horizon_seconds: float,
                      cal: Calibration, unc_cfg, cfg: RiskConfig
                      ) -> list[IcebergForecast]:
    """Centres from the frozen physics, radii from the frozen uncertainty."""
    if horizon_seconds < 0 or not math.isfinite(horizon_seconds):
        raise ExposureError(
            f"horizon_seconds is {horizon_seconds}; a negative or non-finite "
            f"forecast horizon is refused.")
    if len(rows) > cfg.max_icebergs:
        raise ExposureError(f"{len(rows)} icebergs exceeds the configured "
                            f"maximum {cfg.max_icebergs}")
    u = uncertainty_at(horizon_seconds, cal, unc_cfg)
    if u.radius_km > cfg.max_radius_km:
        raise ExposureError(
            f"radius {u.radius_km:.1f} km at {horizon_seconds / 3600:.1f} h "
            f"exceeds the configured maximum {cfg.max_radius_km} km")
    out = []
    for r in rows:
        ve = r["physics_disp_east_m"] / r["elapsed_seconds"]
        vn = r["physics_disp_north_m"] / r["elapsed_seconds"]
        if horizon_seconds == 0.0:
            lat, lon = r["start_latitude"], normalise_longitude(
                r["start_longitude"])
        else:
            lat, lon = advect(r["start_latitude"], r["start_longitude"], ve, vn,
                              horizon_seconds)
        out.append(IcebergForecast(
            iceberg_id=r["iceberg_id"], position_source=r["position_source"],
            forecast_start_date=r["start_date"],
            horizon_seconds=float(horizon_seconds),
            horizon_hours=float(horizon_seconds) / SECONDS_PER_HOUR,
            latitude=lat, longitude=lon, radius_km=u.radius_km,
            extrapolated_uncertainty=u.extrapolated_uncertainty,
            physics_position_is_extrapolated=(
                horizon_seconds > r["elapsed_seconds"]),
            calibration_quantile=u.calibration_quantile,
            calibration_n=u.calibration_n, growth_law=u.growth_law,
            interpolation_quality="observed_to_observed"))
    return out


# ----------------------------------------------------------------- grid
class RoutingGridGeometry:
    """The routing grid and its cached per-cell lat/lon. Read only."""

    def __init__(self, cache_dir: Path = CURRENTS_DIR) -> None:
        self.epsg, self.height, self.width, self.transform = target_grid()
        geom = grid_geometry(self.epsg, self.height, self.width, self.transform,
                             cache_dir=cache_dir)
        self.lat, self.lon = geom["lat"], geom["lon"]
        self.cell_size_m = abs(float(self.transform.a))

    def rowcol_of(self, lat: float, lon: float) -> tuple[int, int]:
        """Cell containing a geographic position. Projection used for lookup."""
        from rasterio.warp import transform as warp_transform
        xs, ys = warp_transform("EPSG:4326", f"EPSG:{self.epsg}", [lon], [lat])
        inv = ~self.transform
        col, row = inv * (xs[0], ys[0])
        return int(math.floor(row)), int(math.floor(col))

    def local_scale(self, lat: float, lon: float) -> float:
        """Projected metres per true metre here. Measured, not assumed."""
        from rasterio.warp import transform as warp_transform
        step = 20_000.0
        dlat = math.degrees(step / 6_371_008.8)
        xs, ys = warp_transform("EPSG:4326", f"EPSG:{self.epsg}",
                                [lon, lon], [lat, lat + dlat])
        proj = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
        true = great_circle_m(lat, lon, lat + dlat, lon)
        return proj / true if true > 0 else 1.0


def build_exposure(forecasts: list[IcebergForecast], grid: RoutingGridGeometry,
                   cfg: RiskConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """(exposure, dominant iceberg index, stats). max over icebergs, never sum."""
    exposure = np.zeros((grid.height, grid.width), dtype="float64")
    dominant = np.full((grid.height, grid.width), -1, dtype="int32")
    stats = Counter()
    for idx, f in enumerate(forecasts):
        radius_m = f.radius_km * 1000.0
        if radius_m <= 0.0:
            stats["forecasts_with_zero_radius"] += 1
            continue
        row0, col0 = grid.rowcol_of(f.latitude, f.longitude)
        scale = grid.local_scale(f.latitude, f.longitude)
        half = int(math.ceil(radius_m * scale * cfg.window_scale
                             / grid.cell_size_m)) + cfg.window_extra
        r_lo, r_hi = max(0, row0 - half), min(grid.height, row0 + half + 1)
        c_lo, c_hi = max(0, col0 - half), min(grid.width, col0 + half + 1)
        if r_lo >= r_hi or c_lo >= c_hi:
            stats["forecasts_outside_the_routing_grid"] += 1
            continue
        lat_w = grid.lat[r_lo:r_hi, c_lo:c_hi]
        lon_w = grid.lon[r_lo:r_hi, c_lo:c_hi]
        dist = _great_circle_grid(lat_w, lon_w, f.latitude, f.longitude)
        vals = _exposure_array(dist, radius_m, cfg.function)
        hit = vals > 0.0
        if not hit.any():
            stats["forecasts_touching_no_cell"] += 1
            continue
        stats["forecasts_contributing"] += 1
        sub_e = exposure[r_lo:r_hi, c_lo:c_hi]
        sub_d = dominant[r_lo:r_hi, c_lo:c_hi]
        better = hit & (vals > sub_e)
        sub_e[better] = vals[better]
        sub_d[better] = idx
    return exposure, dominant, dict(sorted(stats.items()))


def _great_circle_grid(lat: np.ndarray, lon: np.ndarray, lat0: float,
                       lon0: float) -> np.ndarray:
    """Haversine, vectorised, on the same sphere great_circle_m uses.

    The scalar function is the project's definition; this is that arithmetic
    over an array. test_iceberg_risk.py checks the two agree cell by cell.
    """
    from src.data.preprocess_icebergs import EARTH_RADIUS_M

    p1 = np.radians(lat)
    p2 = math.radians(lat0)
    dp = p2 - p1
    dl = np.radians(((lon0 - lon) + 180.0) % 360.0 - 180.0)   # antimeridian-safe
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(a)))


# --------------------------------------------------------------- output
def write_raster(exposure: np.ndarray, grid: RoutingGridGeometry, hours: float,
                 forecasts: list[IcebergForecast], out_dir: Path) -> Path:
    import rasterio
    from rasterio.crs import CRS

    path = out_dir / f"iceberg_exposure_{int(round(hours)):02d}h_3976.tif"
    extrap = bool(forecasts and forecasts[0].extrapolated_uncertainty)
    profile = dict(driver="GTiff", height=grid.height, width=grid.width, count=1,
                   dtype="float32", crs=CRS.from_epsg(grid.epsg),
                   transform=grid.transform, nodata=None,
                   compress="deflate", tiled=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(exposure.astype("float32"), 1)
        dst.set_band_description(
            1, "iceberg exposure index [0,1] - NOT a collision probability")
        dst.update_tags(
            index_meaning="1 - distance/radius inside one iceberg's calibrated "
                          "uncertainty cone; max over icebergs",
            is_a_probability="false",
            zero_meaning="no MODELED exposure from the icebergs in this "
                         "forecast; NOT a statement that the water is clear",
            nodata_meaning="none: 0 is a real value and must never be read as "
                           "missing data",
            no_hard_mask="this layer produces no blocked/no-go/navigable mask",
            horizon_hours=f"{hours:g}",
            forecast_start_date=(forecasts[0].forecast_start_date
                                 if forecasts else "none"),
            iceberg_count=str(len(forecasts)),
            radius_km=(f"{forecasts[0].radius_km:.6f}" if forecasts else "0"),
            extrapolated_uncertainty=str(extrap).lower(),
            empirically_calibrated_to="24h",
            population="clean 24h observed->observed MOVED rows, ASCAT only",
            coverage_caveat="only icebergs in that population are represented; "
                            "any other iceberg is invisible to this raster",
            distance_method="great-circle metres against the cell centre",
        )
    return path


def build_provenance(exposure: np.ndarray, dominant: np.ndarray,
                     forecasts: list[IcebergForecast],
                     grid: RoutingGridGeometry) -> list[dict]:
    """One record per NON-ZERO cell. Zero cells are not enumerated."""
    rows, cols = np.nonzero(exposure > 0.0)
    out = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        f = forecasts[int(dominant[r, c])]
        d = great_circle_m(float(grid.lat[r, c]), float(grid.lon[r, c]),
                           f.latitude, f.longitude)
        out.append({
            "row": r, "col": c,
            "dominant_iceberg_id": f.iceberg_id,
            "position_source": f.position_source,
            "exposure": round(float(exposure[r, c]), 6),
            "distance_km": round(d / 1000.0, 6),
            "radius_km": round(f.radius_km, 6),
            "predicted_center_latitude": f.latitude,
            "predicted_center_longitude": f.longitude,
            "horizon_hours": f.horizon_hours,
            "extrapolated_uncertainty": f.extrapolated_uncertainty,
            "physics_position_is_extrapolated": f.physics_position_is_extrapolated,
            "interpolation_quality": f.interpolation_quality,
        })
    return out


def _population(date: str | None, split_config: Path) -> tuple[list[dict], dict]:
    pop, _ = build_population()
    chronological_split(pop, load_split_config(split_config))
    moved = [r for r in pop if r["moved"]]
    by_date = Counter(r["start_date"] for r in moved)
    chosen = date or max(by_date)
    rows = [r for r in moved if r["start_date"] == chosen]
    if not rows:
        raise ExposureError(
            f"no moved iceberg observations start on {chosen}. Dates with the "
            f"most icebergs: {by_date.most_common(5)}")
    return rows, {"forecast_start_date": chosen,
                  "moved_rows_total": len(moved),
                  "distinct_icebergs_total": len({r["iceberg_id"] for r in moved}),
                  "distinct_start_dates": len(by_date),
                  "rows_on_this_date": len(rows),
                  "distinct_icebergs_on_this_date": len(
                      {r["iceberg_id"] for r in rows}),
                  "busiest_dates": by_date.most_common(5),
                  "latest_date": max(by_date)}


def run(date: str | None = None, config: Path = CONFIG,
        unc_config: Path = UNC_CONFIG, split_config: Path = SPLIT_CONFIG,
        out_dir: Path = ICE_DIR) -> dict:
    cfg = load_config(config)
    ucfg = load_unc_config(unc_config)
    pop, _ = build_population()
    chronological_split(pop, load_split_config(split_config))
    cal = calibrate(pop, ucfg)
    rows, pop_info = _population(date, split_config)
    grid = RoutingGridGeometry()

    horizons, prov = {}, {}
    for hours in cfg.horizons_hours:
        secs = hours * SECONDS_PER_HOUR
        fc = forecast_icebergs(rows, secs, cal, ucfg, cfg)
        exposure, dominant, stats = build_exposure(fc, grid, cfg)
        records = build_provenance(exposure, dominant, fc, grid)
        path = write_raster(exposure, grid, hours, fc, out_dir)
        contributors = Counter(r["dominant_iceberg_id"] for r in records)
        nz = exposure[exposure > 0.0]
        horizons[f"{hours:g}h"] = {
            "horizon_hours": hours, "horizon_seconds": secs,
            "icebergs_forecast": len(fc),
            "distinct_icebergs": len({f.iceberg_id for f in fc}),
            "active_icebergs_touching_the_grid":
                stats.get("forecasts_contributing", 0),
            "uncertainty_radius_km": round(fc[0].radius_km, 6) if fc else 0.0,
            "extrapolated_uncertainty": bool(
                fc and fc[0].extrapolated_uncertainty),
            "physics_positions_extrapolated": sum(
                1 for f in fc if f.physics_position_is_extrapolated),
            "affected_cells": int(len(records)),
            "affected_fraction_of_grid": round(
                len(records) / (grid.height * grid.width), 9),
            "max_exposure": round(float(nz.max()), 6) if nz.size else 0.0,
            "mean_exposure_over_affected_cells": round(
                float(nz.mean()), 6) if nz.size else 0.0,
            "contributing_iceberg_ids": sorted(contributors),
            "top_contributing_icebergs_by_cell_count":
                contributors.most_common(10),
            "forecast_stats": stats,
            "raster": str(path.relative_to(ROOT)),
        }
        prov[f"{hours:g}h"] = records

    report = {
        "layer": "dynamic iceberg exposure index on the routing grid",
        "is_a_collision_probability": False,
        "is_an_official_safety_metric": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "exposure_formula": cfg.raw["exposure"]["linear_formula"],
        "exposure_function": cfg.function,
        "combination_rule": cfg.raw["exposure"]["combination_rule"],
        "combination_note": cfg.raw["exposure"]["combination_note"],
        "distance_method": cfg.raw["distance"],
        "grid": {"epsg": grid.epsg, "shape": [grid.height, grid.width],
                 "cell_size_m": grid.cell_size_m,
                 "transform": [float(v) for v in list(grid.transform)[:6]]},
        "population": pop_info,
        "uncertainty": {
            "calibration_radius_km_24h": round(cal.radius_km, 6),
            "calibration_quantile": cal.quantile,
            "calibration_n": cal.n,
            "growth_law": ucfg.law, "growth_exponent": ucfg.exponent,
            "empirically_calibrated_to": f"{cal.horizon_hours:g}h",
            "horizons_above_24h_are_extrapolated": True,
        },
        "physics": {"frozen": True, "windage_coefficient": 0.0,
                    "wind_turn_deg": 0.0, "ml_residual_used": False,
                    "ml_residual_note": "not used: it measured worse than the "
                                        "physics baseline on the held-out "
                                        "moved 24 h rows"},
        "iceberg_size_used": False,
        "horizons": horizons,
        "zero_means": cfg.raw["output"]["zero_means"],
        "nodata_note": cfg.raw["output"]["nodata_note"],
        "no_hard_mask": True,
        "known_limitations": cfg.raw["known_limitations"],
        "not_included": cfg.raw["not_included"],
    }
    (out_dir / EVAL_JSON.name).write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / PROV_JSON.name).write_text(json.dumps({
        "note": "one record per NON-ZERO cell; zero-exposure cells are not "
                "enumerated and are not missing data",
        "forecast_start_date": pop_info["forecast_start_date"],
        "cells_by_horizon": {k: len(v) for k, v in prov.items()},
        "cells": prov}, indent=2) + "\n")
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Dynamic iceberg exposure index on the routing grid. "
                    "Not a collision probability. No A*, no route cost.")
    ap.add_argument("--date", default=None,
                    help="forecast start date; default is the latest available")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--uncertainty-config", default=str(UNC_CONFIG))
    ap.add_argument("--split-config", default=str(SPLIT_CONFIG))
    ap.add_argument("--out-dir", default=str(ICE_DIR))
    ap.add_argument("--audit", action="store_true",
                    help="report the population and exit, writing nothing")
    args = ap.parse_args(argv)

    if args.audit:
        _, info = _population(args.date, Path(args.split_config))
        print(json.dumps(info, indent=2))
        return 0
    report = run(args.date, Path(args.config), Path(args.uncertainty_config),
                 Path(args.split_config), Path(args.out_dir))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("known_limitations", "distance_method")},
                     indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ExposureError as exc:
        print(f"\nCANNOT BUILD THE EXPOSURE FIELD\n\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)

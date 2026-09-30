"""
First-order current + windage iceberg drift baseline, and its 24 h / 48 h score.

    data/processed/icebergs/iceberg_physics_predictions.csv
    data/processed/icebergs/iceberg_physics_evaluation.json

WHAT THIS IS
      iceberg_velocity = ocean_current + windage_coefficient * rotated_10m_wind
      displacement     = iceberg_velocity * elapsed_seconds
  Constant velocity over the whole step. That is all. It is a first-order
  current + windage drift baseline, NOT a physical simulation: there is no
  iceberg mass, draft, sail area, added mass, wave radiation force, sea-ice
  drag, Coriolis term or grounding anywhere in this file. Both parameters come
  from configs/iceberg_physics.json and neither is a scientific constant.

ONE VECTOR FRAME, REUSED -- NEVER A SECOND ONE
  The GLORYS rasters already hold the current in the EPSG:3976 local basis
  (band 1 "u_x eastward-in-EPSG:3976", band 2 "u_y northward-in-EPSG:3976"),
  rotated there by preprocess_currents.rotate(). ERA5 wind arrives in the
  geographic east/north basis. Adding them raw would be adding two different
  arrows.

  So this module IMPORTS preprocess_currents.rotate and its finite-difference
  probe, rebuilds the same local east/north unit vectors at the iceberg's own
  position, and pushes the wind through the identical transform the currents
  went through. `unrotate` is that matrix transposed, which is its exact
  inverse because the basis is orthonormal -- measured at 1e-12 on the cached
  grid geometry, not assumed. tests/test_iceberg_physics.py checks this
  module's basis against the cached _grid_geometry npz so the two can never
  drift apart.

  The two vectors are summed in the EPSG:3976 basis, as asked. The SUM is then
  expressed back in the local geographic tangent plane for integration, because
  EPSG:3976 is polar stereographic with its standard parallel at 70S: projected
  metres are stretched about 4% at 60S, and integrating raw m/s as projected
  metres would bake that into every trajectory. The tangent plane is also
  exactly how preprocess_icebergs derived the observed velocities this model is
  scored against, so prediction and truth share one geometry. Rotation is
  orthogonal, so summing in 3976 and then unrotating is identical arithmetic to
  unrotating first -- the order is chosen to make the shared basis explicit.

THE END POSITION IS SCORING ONLY
  predict_physics() takes a StepInputs -- start latitude, start longitude,
  current, wind, elapsed seconds -- and that frozen dataclass has no field for
  the observed end position. There is nowhere for the target to enter. The
  constant-velocity baseline is PERSISTENCE: it uses the velocity of the
  PREVIOUS step in the same (iceberg, sensor) track, which is observed entirely
  at or before this step's start. A step with no predecessor gets no
  constant-velocity prediction rather than a borrowed one.

NOTHING IS FABRICATED
  A row without both a current and a wind value is excluded from the physics
  evaluation and counted, never filled with a zero or a climatology. Steps
  longer than the window being scored are kept in the dataset and excluded from
  that window's metrics, with their counts reported.

NOT IN SCOPE
  No ML, no residual correction, no uncertainty cone, no iceberg risk raster,
  no routing, no POLARIS.

Usage:
    python -m src.models.iceberg_physics --audit       # counts only, writes nothing
    python -m src.models.iceberg_physics               # predict, score, write
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.preprocess_currents import EARTH_R, FD_STEP_M, rotate
from src.data.preprocess_icebergs import (EARTH_RADIUS_M, great_circle_m,
                                          normalise_longitude, _wrap_delta_deg)

ROOT = Path(__file__).resolve().parents[2]
ICE_DIR = ROOT / "data" / "processed" / "icebergs"
MATCHES_CSV = ICE_DIR / "iceberg_environment_matches.csv"
STEPS_CSV = ICE_DIR / "iceberg_steps.csv"
PRED_CSV = ICE_DIR / "iceberg_physics_predictions.csv"
EVAL_JSON = ICE_DIR / "iceberg_physics_evaluation.json"
CONFIG = ROOT / "configs" / "iceberg_physics.json"

SECONDS_PER_DAY = 86_400.0
WINDOWS = {"24h": 1, "48h": 2}
QUALITY_CLEAN = "observed_to_observed"
QUALITY_INTERP = "touches_interpolated_endpoint"

PRED_FIELDS = [
    "iceberg_id", "position_source", "start_date", "end_date", "elapsed_days",
    "elapsed_seconds", "start_latitude", "start_longitude",
    "observed_end_latitude", "observed_end_longitude",
    "physics_pred_latitude", "physics_pred_longitude",
    "constant_velocity_pred_latitude", "constant_velocity_pred_longitude",
    "physics_error_km", "constant_velocity_error_km", "interpolation_quality",
    "current_u_3976", "current_v_3976", "wind_u10_east", "wind_v10_north",
    "drift_east_mps", "drift_north_mps", "persistence_source",
]


class PhysicsError(ValueError):
    """The drift baseline cannot be built or scored as specified."""


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class PhysicsConfig:
    windage_coefficient: float
    wind_turn_deg: float
    raw: dict
    path: str

    def replace(self, **kw) -> "PhysicsConfig":
        """A calibrated copy. The file on disk is never rewritten by a fit."""
        return PhysicsConfig(
            windage_coefficient=kw.get("windage_coefficient",
                                       self.windage_coefficient),
            wind_turn_deg=kw.get("wind_turn_deg", self.wind_turn_deg),
            raw=self.raw, path=self.path)


def load_config(path: Path | str = CONFIG) -> PhysicsConfig:
    """Read and validate the parameters. No built-in default, no silent clamp."""
    path = Path(path)
    if not path.exists():
        raise PhysicsError(
            f"No physics configuration at {path}. The windage coefficient and "
            f"turning angle are project parameters and must be stated in a "
            f"file, not chosen in code.")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PhysicsError(f"{path.name} is not valid JSON: {exc}") from None

    limits = raw.get("limits", {})
    values = {}
    for key in ("windage_coefficient", "wind_turn_deg"):
        if key not in raw:
            raise PhysicsError(f"{path.name} has no {key!r}")
        v = raw[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise PhysicsError(f"{key} is {v!r}, not a number")
        v = float(v)
        if not math.isfinite(v):
            raise PhysicsError(f"{key} is {v}; it must be finite")
        lim = limits.get(key, {})
        lo, hi = float(lim.get("min", -math.inf)), float(lim.get("max", math.inf))
        if not lo <= v <= hi:
            raise PhysicsError(
                f"{key} {v} is outside its configured limits [{lo}, {hi}]")
        values[key] = v
    return PhysicsConfig(values["windage_coefficient"], values["wind_turn_deg"],
                         raw, str(path))


# ------------------------------------------------------------ vector frame
def local_basis(lat: float, lon: float) -> tuple[float, float, float, float]:
    """(ex, ey, nx, ny): local east and north unit vectors in EPSG:3976 metres.

    The same finite-difference probe preprocess_currents.grid_geometry uses,
    evaluated at one arbitrary position instead of at every grid cell. Step
    size and Earth radius are imported from that module so the two cannot
    diverge.
    """
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform

    src, dst = CRS.from_epsg(4326), CRS.from_epsg(3976)
    dlat = math.degrees(FD_STEP_M / EARTH_R)
    coslat = min(max(math.cos(math.radians(lat)), 1e-3), 1.0)
    dlon = math.degrees(FD_STEP_M / (EARTH_R * coslat))
    lons = [lon + dlon, lon - dlon, lon, lon]
    lats = [lat, lat,
            min(max(lat + dlat, -89.999), 89.999),
            min(max(lat - dlat, -89.999), 89.999)]
    xs, ys = warp_transform(src, dst, lons, lats)

    def unit(dx, dy):
        n = math.hypot(dx, dy)
        if n == 0:
            raise PhysicsError(f"degenerate local basis at ({lat}, {lon})")
        return dx / n, dy / n

    ex, ey = unit(xs[0] - xs[1], ys[0] - ys[1])
    nx, ny = unit(xs[2] - xs[3], ys[2] - ys[3])
    return ex, ey, nx, ny


def to_projected(u_east: float, u_north: float,
                 basis: tuple[float, float, float, float]) -> tuple[float, float]:
    """Geographic east/north -> EPSG:3976 x/y, via preprocess_currents.rotate."""
    ex, ey, nx, ny = basis
    ux, uy = rotate(np.asarray(u_east, dtype="float64"),
                    np.asarray(u_north, dtype="float64"),
                    np.asarray(ex), np.asarray(ey),
                    np.asarray(nx), np.asarray(ny))
    return float(ux), float(uy)


def to_geographic(u_x: float, u_y: float,
                  basis: tuple[float, float, float, float]) -> tuple[float, float]:
    """EPSG:3976 x/y -> geographic east/north. The transpose of `rotate`.

    Valid because the basis is orthonormal; the test suite measures that rather
    than trusting it.
    """
    ex, ey, nx, ny = basis
    return u_x * ex + u_y * ey, u_x * nx + u_y * ny


def turn_wind(w_east: float, w_north: float, turn_deg: float
              ) -> tuple[float, float]:
    """Rotate the wind in the local (east, north) plane. Positive = CCW.

    Configured in configs/iceberg_physics.json; the sign convention is stated
    there. Speed is preserved exactly.
    """
    t = math.radians(float(turn_deg))
    c, s = math.cos(t), math.sin(t)
    return w_east * c - w_north * s, w_east * s + w_north * c


# ----------------------------------------------------------- the predictor
@dataclass(frozen=True)
class StepInputs:
    """Everything the physics model is allowed to see.

    There is deliberately NO field for the observed end position. The target
    cannot leak into a prediction through a struct that has nowhere to put it.
    """

    start_latitude: float
    start_longitude: float
    current_u_3976: float       # as stored in the GLORYS raster
    current_v_3976: float
    wind_u10_east: float        # as served by ERA5, geographic basis
    wind_v10_north: float
    elapsed_seconds: float


@dataclass(frozen=True)
class Prediction:
    latitude: float
    longitude: float
    drift_east_mps: float
    drift_north_mps: float
    drift_x_3976: float
    drift_y_3976: float


def drift_velocity(step: StepInputs, cfg: PhysicsConfig
                   ) -> tuple[float, float, float, float]:
    """current + windage * rotated_wind, summed in the EPSG:3976 basis.

    Returns (east, north, x_3976, y_3976) for the combined velocity, so an
    audit can see both descriptions of the same arrow.
    """
    basis = local_basis(step.start_latitude, step.start_longitude)
    we, wn = turn_wind(step.wind_u10_east, step.wind_v10_north,
                       cfg.wind_turn_deg)
    wx, wy = to_projected(we, wn, basis)            # wind into the current's basis
    vx = step.current_u_3976 + cfg.windage_coefficient * wx
    vy = step.current_v_3976 + cfg.windage_coefficient * wy
    ve, vn = to_geographic(vx, vy, basis)           # back out for integration
    return ve, vn, vx, vy


def advect(lat: float, lon: float, v_east: float, v_north: float,
           elapsed_s: float) -> tuple[float, float]:
    """Constant velocity over the step, in the local tangent plane.

        phi1    = phi0 + v_north * dt / R
        lambda1 = lambda0 + v_east * dt / (R * cos(phi_mid))

    This is the exact inverse of the east/north velocity that
    preprocess_icebergs derived from two observed positions, so a prediction and
    the truth it is scored against are the same kind of quantity. Longitude is
    normalised with that module's own function, which is what carries a
    trajectory across the dateline.
    """
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise PhysicsError(f"elapsed_seconds must be finite and > 0, got {elapsed_s}")
    phi0 = math.radians(lat)
    phi1 = phi0 + (v_north * elapsed_s) / EARTH_RADIUS_M
    phi1 = min(max(phi1, math.radians(-89.999)), math.radians(89.999))
    phi_mid = 0.5 * (phi0 + phi1)
    cos_mid = math.cos(phi_mid)
    if abs(cos_mid) < 1e-12:
        raise PhysicsError("advection is undefined at the pole")
    dlon = math.degrees((v_east * elapsed_s) / (EARTH_RADIUS_M * cos_mid))
    return math.degrees(phi1), normalise_longitude(lon + dlon)


def predict_physics(step: StepInputs, cfg: PhysicsConfig) -> Prediction:
    """Start position + environment + elapsed time -> predicted end position."""
    ve, vn, vx, vy = drift_velocity(step, cfg)
    lat1, lon1 = advect(step.start_latitude, step.start_longitude, ve, vn,
                        step.elapsed_seconds)
    return Prediction(lat1, lon1, ve, vn, vx, vy)


def predict_persistence(lat: float, lon: float, v_east_prev: float,
                        v_north_prev: float, elapsed_s: float
                        ) -> tuple[float, float]:
    """Carry the PREVIOUS step's observed velocity forward. No target used."""
    return advect(lat, lon, v_east_prev, v_north_prev, elapsed_s)


# ------------------------------------------------------------- the dataset
def _previous_step_velocity(steps_csv: Path = STEPS_CSV) -> dict:
    """(iceberg, sensor, date) -> the velocity of the step ENDING on that date.

    Keyed by the predecessor's END date, which is the successor's START date.
    Every value is observed at or before the successor's start, so using it
    cannot leak the successor's target.
    """
    out: dict[tuple[str, str, str], tuple[float, float, str]] = {}
    with Path(steps_csv).open(newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["iceberg_id"], r["position_source"], r["date"])] = (
                float(r["velocity_east_mps"]), float(r["velocity_north_mps"]),
                f"{r['prev_date']}->{r['date']}")
    return out


def load_rows(matches_csv: Path = MATCHES_CSV, steps_csv: Path = STEPS_CSV
              ) -> tuple[list[dict], dict]:
    """Join the matched environment onto the observed steps. Reads only."""
    for p in (matches_csv, steps_csv):
        if not Path(p).exists():
            raise PhysicsError(
                f"not found: {p}\nRun first: python -m src.data.preprocess_era5_wind")
    truth: dict[tuple, tuple[float, float]] = {}
    with Path(steps_csv).open(newline="") as fh:
        for r in csv.DictReader(fh):
            truth[(r["iceberg_id"], r["position_source"], r["prev_date"],
                   r["date"])] = (float(r["latitude_deg"]),
                                  float(r["longitude_deg"]))

    rows, stats = [], Counter()
    with Path(matches_csv).open(newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["iceberg_id"], r["position_source"], r["start_date"],
                   r["end_date"])
            end = truth.get(key)
            if end is None:
                stats["matched_rows_without_an_observed_step"] += 1
                continue
            days = int(r["elapsed_days"])
            stats[f"elapsed_days_{days if days <= 3 else 'gt3'}"] += 1
            quality = (QUALITY_CLEAN
                       if r["either_endpoint_interpolated"] == "False"
                       else QUALITY_INTERP)
            stats[quality] += 1
            have_c = r["current_available"] == "True"
            have_w = r["wind_available"] == "True"
            stats["current_available"] += have_c
            stats["wind_available"] += have_w
            stats["both_environmental_inputs"] += (have_c and have_w)
            rows.append({
                "iceberg_id": r["iceberg_id"],
                "position_source": r["position_source"],
                "start_date": r["start_date"], "end_date": r["end_date"],
                "elapsed_days": days,
                "elapsed_seconds": float(r["elapsed_seconds"]),
                "start_latitude": float(r["start_latitude"]),
                "start_longitude": float(r["start_longitude"]),
                "observed_end_latitude": end[0],
                "observed_end_longitude": end[1],
                "current_u_3976": float(r["current_u"]) if have_c else None,
                "current_v_3976": float(r["current_v"]) if have_c else None,
                "wind_u10_east": float(r["wind_u10"]) if have_w else None,
                "wind_v10_north": float(r["wind_v10"]) if have_w else None,
                "has_environment": have_c and have_w,
                "interpolation_quality": quality,
            })
    return rows, dict(sorted(stats.items()))


def inputs_for(row: dict) -> StepInputs:
    if not row["has_environment"]:
        raise PhysicsError(
            f"{row['iceberg_id']} {row['start_date']}: a current and a wind "
            f"value are both required; neither is ever fabricated")
    return StepInputs(
        start_latitude=row["start_latitude"],
        start_longitude=row["start_longitude"],
        current_u_3976=row["current_u_3976"],
        current_v_3976=row["current_v_3976"],
        wind_u10_east=row["wind_u10_east"],
        wind_v10_north=row["wind_v10_north"],
        elapsed_seconds=row["elapsed_seconds"])


# ------------------------------------------------------------- evaluation
def _component_errors(row: dict, lat_p: float, lon_p: float) -> tuple[float, float]:
    """(east, north) error in metres, in the same tangent plane as everything."""
    lat_o, lon_o = row["observed_end_latitude"], row["observed_end_longitude"]
    phi_mid = math.radians(0.5 * (lat_p + lat_o))
    east = (EARTH_RADIUS_M * math.radians(_wrap_delta_deg(lon_p - lon_o))
            * math.cos(phi_mid))
    north = EARTH_RADIUS_M * math.radians(lat_p - lat_o)
    return east, north


def _summarise(errors_km: list[float], east_m: list[float], north_m: list[float],
               mag_m: list[float]) -> dict:
    if not errors_km:
        return {"n": 0}
    s = sorted(errors_km)
    return {
        "n": len(s),
        "mean_error_km": round(statistics.fmean(s), 4),
        "median_error_km": round(statistics.median(s), 4),
        "rmse_error_km": round(math.sqrt(statistics.fmean([e * e for e in s])), 4),
        "p90_error_km": round(s[min(len(s) - 1, int(math.ceil(0.9 * len(s)) - 1))], 4),
        "max_error_km": round(s[-1], 4),
        "mean_signed_east_error_km": round(statistics.fmean(east_m) / 1000.0, 4),
        "mean_signed_north_error_km": round(statistics.fmean(north_m) / 1000.0, 4),
        "mean_abs_east_error_km": round(
            statistics.fmean([abs(v) for v in east_m]) / 1000.0, 4),
        "mean_abs_north_error_km": round(
            statistics.fmean([abs(v) for v in north_m]) / 1000.0, 4),
        "mean_signed_displacement_magnitude_error_km": round(
            statistics.fmean(mag_m) / 1000.0, 4),
    }


def score(rows: list[dict], cfg: PhysicsConfig, prev_vel: dict,
          window_days: int | None = None, quality: str | None = None) -> dict:
    """Both baselines, on the SAME rows. The end position only ever scores.

    A row enters the comparison only when BOTH baselines can be computed for
    it: it needs a current and a wind value for the physics, and a preceding
    step for persistence. Scoring each baseline over whatever rows it happens
    to support would compare them on different data -- the first version of
    this function did exactly that, putting the physics on 4,645 rows and
    persistence on 433,591, which is not a comparison at all. Rows that only
    one baseline could have handled are counted, not quietly used.
    """
    phys = {"km": [], "e": [], "n": [], "mag": []}
    const = {"km": [], "e": [], "n": [], "mag": []}
    # The same two, restricted to steps the source actually shows moving.
    phys_m = {"km": [], "e": [], "n": [], "mag": []}
    const_m = {"km": [], "e": [], "n": [], "mag": []}
    excluded = Counter()
    zero_move = 0
    for row in rows:
        if window_days is not None and row["elapsed_days"] != window_days:
            excluded["wrong_window"] += 1
            continue
        if quality is not None and row["interpolation_quality"] != quality:
            excluded["wrong_interpolation_quality"] += 1
            continue
        key = (row["iceberg_id"], row["position_source"], row["start_date"])
        pv = prev_vel.get(key)
        has_env, has_prev = row["has_environment"], pv is not None
        if not (has_env and has_prev):
            if not has_env and not has_prev:
                excluded["missing_current_or_wind_and_no_preceding_step"] += 1
            elif not has_env:
                excluded["missing_current_or_wind"] += 1
            else:
                excluded["no_preceding_step_for_persistence"] += 1
            continue

        obs = (row["observed_end_latitude"], row["observed_end_longitude"])
        obs_m = great_circle_m(row["start_latitude"], row["start_longitude"],
                               *obs)
        if obs_m == 0.0:
            # The source reported the identical position on both days. Kept --
            # it is a real observation -- but counted, because persistence
            # scores it exactly and that flatters it.
            zero_move += 1

        p = predict_physics(inputs_for(row), cfg)
        d = great_circle_m(p.latitude, p.longitude, *obs)
        e, n = _component_errors(row, p.latitude, p.longitude)
        pm = great_circle_m(row["start_latitude"], row["start_longitude"],
                            p.latitude, p.longitude)
        phys["km"].append(d / 1000.0); phys["e"].append(e)
        phys["n"].append(n); phys["mag"].append(pm - obs_m)
        if obs_m > 0.0:
            phys_m["km"].append(d / 1000.0); phys_m["e"].append(e)
            phys_m["n"].append(n); phys_m["mag"].append(pm - obs_m)

        lat_c, lon_c = predict_persistence(
            row["start_latitude"], row["start_longitude"], pv[0], pv[1],
            row["elapsed_seconds"])
        d = great_circle_m(lat_c, lon_c, *obs)
        e, n = _component_errors(row, lat_c, lon_c)
        cm = great_circle_m(row["start_latitude"], row["start_longitude"],
                            lat_c, lon_c)
        const["km"].append(d / 1000.0); const["e"].append(e)
        const["n"].append(n); const["mag"].append(cm - obs_m)
        if obs_m > 0.0:
            const_m["km"].append(d / 1000.0); const_m["e"].append(e)
            const_m["n"].append(n); const_m["mag"].append(cm - obs_m)

    n_common = len(phys["km"])
    return {
        "evaluated_on_common_rows": n_common,
        "both_baselines_scored_on_identical_rows": True,
        "steps_with_zero_observed_displacement": zero_move,
        "zero_displacement_note": (
            "the source reported the same position on both dates; persistence "
            "is exactly right on these by construction, so a headline mean "
            "that includes them measures how often the chart repeated a "
            "position as much as it measures drift skill. The moved_only "
            "block below is the same two baselines over the steps that "
            "actually moved."),
        "moved_only": {
            "n": len(phys_m["km"]),
            "physics_current_plus_windage": _summarise(
                phys_m["km"], phys_m["e"], phys_m["n"], phys_m["mag"]),
            "constant_velocity_persistence": _summarise(
                const_m["km"], const_m["e"], const_m["n"], const_m["mag"]),
        },
        "physics_current_plus_windage": _summarise(
            phys["km"], phys["e"], phys["n"], phys["mag"]),
        "constant_velocity_persistence": _summarise(
            const["km"], const["e"], const["n"], const["mag"]),
        "exclusions": dict(sorted(excluded.items())),
    }


# ------------------------------------------------------------ calibration
def calibrate(rows: list[dict], cfg: PhysicsConfig, window_days: int = 1,
              train_fraction: float = 0.7, min_rows: int = 200) -> dict:
    """Grid-search the two parameters on a CHRONOLOGICAL training split.

    Not machine learning: a coarse-then-fine sweep of one or two scalars,
    scored by mean endpoint error. Rows are ordered by start date and the
    EARLIER `train_fraction` is fitted on; the later rows are never touched
    here. If the window has fewer than `min_rows` clean rows the fit is
    declined and the configured values stand.
    """
    usable = [r for r in rows
              if r["elapsed_days"] == window_days
              and r["interpolation_quality"] == QUALITY_CLEAN
              and r["has_environment"]]
    if len(usable) < min_rows:
        # Decline before sorting: a rejected sample should not have to carry
        # every field the chronological ordering needs.
        return {"performed": False, "n_usable": len(usable),
                "min_rows_required": min_rows,
                "reason": (f"only {len(usable)} clean {window_days}-day rows with "
                           f"both environmental inputs; below the {min_rows} "
                           f"threshold, so the configured values stand")}
    usable.sort(key=lambda r: (r["start_date"], r["iceberg_id"],
                               r["position_source"]))
    cut = int(len(usable) * train_fraction)
    train, valid = usable[:cut], usable[cut:]
    if not train or not valid:
        return {"performed": False, "n_usable": len(usable),
                "reason": "the chronological split left one side empty"}

    lim = cfg.raw.get("limits", {})
    c_lo = float(lim.get("windage_coefficient", {}).get("min", 0.0))
    c_hi = float(lim.get("windage_coefficient", {}).get("max", 0.1))
    t_lo = float(lim.get("wind_turn_deg", {}).get("min", -90.0))
    t_hi = float(lim.get("wind_turn_deg", {}).get("max", 90.0))

    def mean_err(subset, coeff, turn) -> float:
        trial = cfg.replace(windage_coefficient=coeff, wind_turn_deg=turn)
        tot = 0.0
        for r in subset:
            p = predict_physics(inputs_for(r), trial)
            tot += great_circle_m(p.latitude, p.longitude,
                                  r["observed_end_latitude"],
                                  r["observed_end_longitude"])
        return tot / len(subset) / 1000.0

    coeffs = [round(c_lo + i * (c_hi - c_lo) / 40.0, 6) for i in range(41)]
    turns = [t_lo + i * (t_hi - t_lo) / 12.0 for i in range(13)]
    best = min(((mean_err(train, c, t), c, t) for c in coeffs for t in turns),
               key=lambda x: x[0])
    # one refinement pass around the winner, still on the training rows only
    _, c0, t0 = best
    dc, dt_ = (c_hi - c_lo) / 40.0, (t_hi - t_lo) / 12.0
    fine_c = [min(max(c0 + k * dc / 8.0, c_lo), c_hi) for k in range(-8, 9)]
    fine_t = [min(max(t0 + k * dt_ / 6.0, t_lo), t_hi) for k in range(-6, 7)]
    best = min(((mean_err(train, c, t), c, t) for c in fine_c for t in fine_t),
               key=lambda x: x[0])
    err, coeff, turn = best
    return {
        "performed": True,
        "method": ("coarse-then-fine grid search over the two configured "
                   "parameters, scored by mean endpoint error. No ML, no "
                   "gradient fitting, no per-iceberg parameters."),
        "window_days": window_days,
        "n_usable_clean_rows": len(usable),
        "train_rows": len(train), "validation_rows": len(valid),
        "train_fraction": train_fraction,
        "split": "chronological by start date; the fit never sees the later rows",
        "train_date_range": [train[0]["start_date"], train[-1]["start_date"]],
        "validation_date_range": [valid[0]["start_date"], valid[-1]["start_date"]],
        "configured_windage_coefficient": cfg.windage_coefficient,
        "configured_wind_turn_deg": cfg.wind_turn_deg,
        "fitted_windage_coefficient": round(coeff, 6),
        "fitted_wind_turn_deg": round(turn, 4),
        "train_mean_error_km_at_fitted": round(err, 4),
        "train_mean_error_km_at_configured": round(
            mean_err(train, cfg.windage_coefficient, cfg.wind_turn_deg), 4),
        "config_file_rewritten": False,
        "note": ("the fitted values are REPORTED, not written back to "
                 "configs/iceberg_physics.json; the configured values remain "
                 "the shipped defaults"),
    }


# ----------------------------------------------------------------- output
def write_predictions(rows: list[dict], cfg: PhysicsConfig, prev_vel: dict,
                      out: Path = PRED_CSV) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PRED_FIELDS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            rec = {k: row.get(k, "") for k in PRED_FIELDS}
            rec["interpolation_quality"] = row["interpolation_quality"]
            obs = (row["observed_end_latitude"], row["observed_end_longitude"])
            if row["has_environment"]:
                p = predict_physics(inputs_for(row), cfg)
                rec["physics_pred_latitude"] = p.latitude
                rec["physics_pred_longitude"] = p.longitude
                rec["physics_error_km"] = great_circle_m(
                    p.latitude, p.longitude, *obs) / 1000.0
                rec["drift_east_mps"] = p.drift_east_mps
                rec["drift_north_mps"] = p.drift_north_mps
            else:
                for k in ("physics_pred_latitude", "physics_pred_longitude",
                          "physics_error_km", "drift_east_mps",
                          "drift_north_mps"):
                    rec[k] = ""
            pv = prev_vel.get((row["iceberg_id"], row["position_source"],
                               row["start_date"]))
            if pv is not None:
                lat_c, lon_c = predict_persistence(
                    row["start_latitude"], row["start_longitude"], pv[0], pv[1],
                    row["elapsed_seconds"])
                rec["constant_velocity_pred_latitude"] = lat_c
                rec["constant_velocity_pred_longitude"] = lon_c
                rec["constant_velocity_error_km"] = great_circle_m(
                    lat_c, lon_c, *obs) / 1000.0
                rec["persistence_source"] = pv[2]
            else:
                for k in ("constant_velocity_pred_latitude",
                          "constant_velocity_pred_longitude",
                          "constant_velocity_error_km", "persistence_source"):
                    rec[k] = ""
            for k in ("current_u_3976", "current_v_3976", "wind_u10_east",
                      "wind_v10_north"):
                rec[k] = "" if row.get(k) is None else row[k]
            w.writerow(rec)
            n += 1
    return n


def run(matches_csv: Path = MATCHES_CSV, steps_csv: Path = STEPS_CSV,
        config: Path = CONFIG, out_dir: Path = ICE_DIR,
        do_calibration: bool = True) -> dict:
    cfg = load_config(config)
    rows, stats = load_rows(matches_csv, steps_csv)
    prev_vel = _previous_step_velocity(steps_csv)

    windows = {}
    for label, days in WINDOWS.items():
        windows[label] = {
            "elapsed_days": days,
            QUALITY_CLEAN: score(rows, cfg, prev_vel, days, QUALITY_CLEAN),
            QUALITY_INTERP: score(rows, cfg, prev_vel, days, QUALITY_INTERP),
        }

    cal = (calibrate(rows, cfg, 1) if do_calibration
           else {"performed": False, "reason": "calibration not requested"})
    cal_windows = {}
    if cal.get("performed"):
        fitted = cfg.replace(
            windage_coefficient=cal["fitted_windage_coefficient"],
            wind_turn_deg=cal["fitted_wind_turn_deg"])
        train_cut = set()
        usable = sorted(
            (r for r in rows if r["elapsed_days"] == 1
             and r["interpolation_quality"] == QUALITY_CLEAN
             and r["has_environment"]),
            key=lambda r: (r["start_date"], r["iceberg_id"],
                           r["position_source"]))
        for r in usable[:cal["train_rows"]]:
            train_cut.add(id(r))
        held_out = [r for r in usable if id(r) not in train_cut]
        cal_windows["24h_validation_rows_only"] = {
            "n": len(held_out),
            "at_configured_parameters": score(held_out, cfg, prev_vel, 1,
                                              QUALITY_CLEAN),
            "at_fitted_parameters": score(held_out, fitted, prev_vel, 1,
                                          QUALITY_CLEAN),
        }
        cal_windows["48h_all_clean_rows_never_fitted"] = {
            "at_configured_parameters": score(rows, cfg, prev_vel, 2,
                                              QUALITY_CLEAN),
            "at_fitted_parameters": score(rows, fitted, prev_vel, 2,
                                          QUALITY_CLEAN),
        }

    n_pred = write_predictions(rows, cfg, prev_vel, out_dir / PRED_CSV.name)

    long_steps = {k: v for k, v in stats.items() if k.startswith("elapsed_days_")}
    report = {
        "model": "first-order current + windage drift baseline",
        "formula": ("iceberg_velocity = ocean_current + windage_coefficient * "
                    "rotate(wind_10m, wind_turn_deg); displacement = "
                    "iceberg_velocity * elapsed_seconds"),
        "is_a_physical_simulation": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            "matches_csv": str(Path(matches_csv).relative_to(ROOT)),
            "steps_csv": str(Path(steps_csv).relative_to(ROOT)),
            "config": str(Path(config).relative_to(ROOT)),
        },
        "provenance": _provenance(),
        "parameters": {
            "windage_coefficient": cfg.windage_coefficient,
            "wind_turn_deg": cfg.wind_turn_deg,
            "source": cfg.path,
            "are_physical_constants": False,
            "turn_sign_convention": cfg.raw.get("turn_sign_convention"),
        },
        "vector_frame_convention": {
            "current_as_stored": "EPSG:3976 local basis (u_x, u_y), rotated "
                                 "there by src/data/preprocess_currents.rotate",
            "wind_as_served": "ERA5 geographic east/north (u10, v10), m/s",
            "conversion": ("the wind is pushed through the SAME transform via "
                           "preprocess_currents.rotate and a local basis "
                           "rebuilt with that module's own finite-difference "
                           "probe; the two are summed in the EPSG:3976 basis"),
            "integration_basis": ("the sum is expressed back in the local "
                                  "geographic east/north tangent plane before "
                                  "integrating, because EPSG:3976 projected "
                                  "metres carry the polar-stereographic scale "
                                  "factor (~+4% at 60S) and because the "
                                  "observed velocities were derived in that "
                                  "same tangent plane"),
            "inverse_is_the_transpose": "valid because the basis is orthonormal",
        },
        "dataset_counts": stats,
        "elapsed_day_breakdown": long_steps,
        "long_and_gap_steps": {
            "kept_in_the_prediction_table": True,
            "excluded_from_24h_and_48h_metrics": True,
            "counts": long_steps,
        },
        "windows_at_configured_parameters": windows,
        "calibration": cal,
        "calibration_effect": cal_windows,
        "predictions_written": n_pred,
        "target_endpoint_usage": ("observed end positions were used ONLY to "
                                  "compute error. StepInputs has no field for "
                                  "them, and the constant-velocity baseline "
                                  "uses the PREVIOUS step's observed velocity, "
                                  "which is complete at this step's start."),
        "no_environmental_value_fabricated": True,
        "not_included": ["ML residual correction", "uncertainty cones",
                         "iceberg risk rasters", "routing", "POLARIS"],
    }
    (out_dir / EVAL_JSON.name).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _provenance() -> dict:
    out = {}
    for name, path in (("iceberg_data_manifest",
                        ICE_DIR / "iceberg_data_manifest.json"),
                       ("environment_match_manifest",
                        ICE_DIR / "environment_match_manifest.json")):
        if path.exists():
            j = json.loads(path.read_text())
            out[name] = {k: j.get(k) for k in
                         ("source_name", "source_version", "source_url",
                          "original_date_range", "processed_observation_count",
                          "step_count", "era5_dataset", "era5_timesteps",
                          "fully_matched") if k in j}
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="First-order current + windage iceberg drift baseline. "
                    "No ML, no cones, no risk raster, no routing.")
    ap.add_argument("--matches", default=str(MATCHES_CSV))
    ap.add_argument("--steps", default=str(STEPS_CSV))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out-dir", default=str(ICE_DIR))
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="report the dataset counts and exit, writing nothing")
    args = ap.parse_args(argv)

    if args.audit:
        _, stats = load_rows(Path(args.matches), Path(args.steps))
        print(json.dumps(stats, indent=2))
        return 0

    report = run(Path(args.matches), Path(args.steps), Path(args.config),
                 Path(args.out_dir), not args.no_calibration)
    slim = {k: v for k, v in report.items()
            if k not in ("provenance", "dataset_counts")}
    print(json.dumps(slim, indent=2))
    print(f"\nwrote {Path(args.out_dir) / PRED_CSV.name}")
    print(f"wrote {Path(args.out_dir) / EVAL_JSON.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except PhysicsError as exc:
        print(f"\nCANNOT RUN THE DRIFT BASELINE\n\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)

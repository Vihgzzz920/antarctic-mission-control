"""
Forecast uncertainty cone for the frozen iceberg physics baseline.

    data/processed/icebergs/iceberg_uncertainty_evaluation.json
    data/processed/icebergs/iceberg_uncertainty_predictions.csv

WHAT IS CALIBRATED, AND WHAT IS ASSUMED
  CALIBRATED: one number. The 90th percentile of the radial endpoint error the
  frozen physics baseline actually made over the clean 24 h moved rows in the
  TRAIN and VALIDATION splits. That is the 24 h radius, and it is the only
  horizon this project has evidence for.

  ASSUMED: everything else. The radius at any other horizon comes from

      radius(t) = radius_24h * (t / 24h) ** exponent      (exponent 0.5)

  which is the growth a random-walk displacement error would follow. It is a
  PROJECT ASSUMPTION written in configs/iceberg_uncertainty.json, not a fitted
  law. There are 6 clean moved 48 h rows in the whole dataset; they cannot tell
  square-root growth from linear growth from anything else, and this module
  will not pretend otherwise. A linear law is available for sensitivity testing
  and must not be chosen using the 48 h rows.

  Every horizon above 24 h is stamped `extrapolated_uncertainty = true` on the
  result itself, not merely mentioned in a footnote.

THE OBSERVED ENDPOINT ONLY EVER SCORES
  `uncertainty_at` takes a horizon and a calibration, and returns a radius. It
  has no parameter for an observed position and no way to receive one: the
  radius at 48 h is the same number for every iceberg. Observed endpoints build
  the calibration errors and measure coverage, and appear nowhere else. The
  TEST split is untouched until the final coverage count.

CIRCULAR ON PURPOSE
  The primary output is a radius, because the collision-risk visualisation this
  feeds consumes a circle. East/north spread and their correlation are computed
  and reported as SECONDARY information, so a later step can decide on evidence
  whether an ellipse is worth the complexity. Nothing here consumes them.

NOT IN SCOPE
  No collision probability, no risk raster, no route cost, no replanning, no
  ML, and no change to the physics model, which is imported and used frozen.

Usage:
    python -m src.models.iceberg_uncertainty --calibrate   # numbers only
    python -m src.models.iceberg_uncertainty               # evaluate + write
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from src.data.preprocess_icebergs import great_circle_m
from src.models.iceberg_ml_residual import (build_population,
                                            chronological_split,
                                            load_config as load_split_config,
                                            CONFIG as SPLIT_CONFIG)
from src.models.iceberg_physics import ICE_DIR, advect

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "iceberg_uncertainty.json"
EVAL_JSON = ICE_DIR / "iceberg_uncertainty_evaluation.json"
PRED_CSV = ICE_DIR / "iceberg_uncertainty_predictions.csv"

SECONDS_PER_HOUR = 3600.0
LAW_POWER, LAW_LINEAR = "power", "linear"

PRED_FIELDS = [
    "iceberg_id", "position_source", "start_date", "horizon_hours",
    "physics_pred_latitude", "physics_pred_longitude", "uncertainty_radius_km",
    "extrapolated_uncertainty", "calibration_quantile", "calibration_n",
    "calibration_horizon_hours", "growth_law", "growth_exponent",
    "interpolation_quality", "split", "physics_position_is_extrapolated",
]


class UncertaintyError(ValueError):
    """The uncertainty cone cannot be calibrated or evaluated as specified."""


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class UncertaintyConfig:
    raw: dict
    path: str

    @property
    def quantile(self) -> float:
        return float(self.raw["calibration"]["quantile"])

    @property
    def reported_quantiles(self) -> list[float]:
        return [float(q) for q in self.raw["calibration"]["reported_quantiles"]]

    @property
    def calibration_hours(self) -> float:
        return float(self.raw["calibration"]["horizon_hours"])

    @property
    def law(self) -> str:
        return str(self.raw["growth"]["law"])

    @property
    def exponent(self) -> float:
        return float(self.raw["growth"]["exponent"])

    @property
    def splits_used(self) -> list[str]:
        return list(self.raw["calibration"]["splits_used"])

    @property
    def max_radius_km(self) -> float:
        return float(self.raw["limits"]["max_radius_km"])

    def with_growth(self, law: str | None = None,
                    exponent: float | None = None) -> "UncertaintyConfig":
        """A copy with a different growth law. The file is never rewritten."""
        raw = json.loads(json.dumps(self.raw))
        if law is not None:
            raw["growth"]["law"] = law
        if exponent is not None:
            raw["growth"]["exponent"] = exponent
        return UncertaintyConfig(raw=raw, path=self.path)


def load_config(path: Path | str = CONFIG) -> UncertaintyConfig:
    """Read and validate. A bad quantile or exponent is refused, not clamped."""
    path = Path(path)
    if not path.exists():
        raise UncertaintyError(
            f"No uncertainty configuration at {path}. The calibration quantile "
            f"and the growth law are project parameters and must be stated in "
            f"a file, not chosen in code.")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise UncertaintyError(f"{path.name} is not valid JSON: {exc}") from None

    cal, grow = raw.get("calibration", {}), raw.get("growth", {})
    q = cal.get("quantile")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0.0 < q < 1.0:
        raise UncertaintyError(f"calibration.quantile is {q!r}; it must be in (0, 1)")
    for r in cal.get("reported_quantiles", []):
        if not 0.0 < float(r) < 1.0:
            raise UncertaintyError(f"reported quantile {r} is outside (0, 1)")
    h = cal.get("horizon_hours")
    if not isinstance(h, (int, float)) or not (math.isfinite(h) and h > 0):
        raise UncertaintyError(f"calibration.horizon_hours is {h!r}; it must be > 0")
    law = grow.get("law")
    if law not in grow.get("available_laws", []):
        raise UncertaintyError(
            f"growth.law is {law!r}; available_laws are "
            f"{grow.get('available_laws')}")
    e = grow.get("exponent")
    if isinstance(e, bool) or not isinstance(e, (int, float)) or not math.isfinite(e):
        raise UncertaintyError(f"growth.exponent is {e!r}; it must be finite")
    lim = grow.get("exponent_limits", {})
    lo, hi = float(lim.get("min", -math.inf)), float(lim.get("max", math.inf))
    if not lo <= float(e) <= hi:
        raise UncertaintyError(
            f"growth.exponent {e} is outside its configured limits [{lo}, {hi}]")
    if grow.get("is_this_measured") is not False:
        raise UncertaintyError(
            "growth.is_this_measured must be false: the growth law is a project "
            "assumption and the config must say so.")
    if not cal.get("splits_used"):
        raise UncertaintyError("calibration.splits_used is empty")
    if "test" in cal["splits_used"]:
        raise UncertaintyError(
            "calibration.splits_used names the TEST split; the held-out rows "
            "may only be used to measure coverage, never to set the radius.")
    return UncertaintyConfig(raw=raw, path=str(path))


# ------------------------------------------------------------- calibration
@dataclass(frozen=True)
class Calibration:
    """One number with everything needed to defend it."""

    radius_km: float
    quantile: float
    horizon_hours: float
    n: int
    splits_used: tuple[str, ...]
    date_min: str
    date_max: str
    quantile_radii_km: dict
    error_summary_km: dict
    component_statistics_km: dict
    physics_config_sha256: str


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile. statistics.quantiles cannot take one q."""
    if not sorted_values:
        raise UncertaintyError("no values to take a quantile of")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _physics_error_km(row: dict) -> float:
    return great_circle_m(row["physics_pred_latitude"],
                          row["physics_pred_longitude"],
                          row["observed_end_latitude"],
                          row["observed_end_longitude"]) / 1000.0


def calibrate(rows: list[dict], cfg: UncertaintyConfig) -> Calibration:
    """The empirical radial-error quantile over the calibration splits."""
    import hashlib

    use = [r for r in rows
           if r["split"] in cfg.splits_used and r["moved"]]
    n_min = int(cfg.raw["calibration"].get("min_calibration_rows", 0))
    if len(use) < n_min:
        raise UncertaintyError(
            f"only {len(use)} calibration rows; below the configured minimum "
            f"{n_min}. A radius from fewer rows would not be a distribution.")
    errs = sorted(_physics_error_km(r) for r in use)
    east, north = [], []
    for r in use:
        de, dn = _endpoint_offset(r)
        east.append(de / 1000.0)
        north.append(dn / 1000.0)
    dates = sorted(r["start_date"] for r in use)
    phys_cfg = ROOT / "configs" / "iceberg_physics.json"
    sha = (hashlib.sha256(phys_cfg.read_bytes()).hexdigest()
           if phys_cfg.exists() else None)
    return Calibration(
        radius_km=_quantile(errs, cfg.quantile),
        quantile=cfg.quantile,
        horizon_hours=cfg.calibration_hours,
        n=len(use),
        splits_used=tuple(cfg.splits_used),
        date_min=dates[0], date_max=dates[-1],
        quantile_radii_km={f"q{int(q * 100)}": round(_quantile(errs, q), 6)
                           for q in cfg.reported_quantiles},
        error_summary_km={
            "n": len(errs),
            "mean": round(statistics.fmean(errs), 4),
            "median": round(statistics.median(errs), 4),
            "p90": round(_quantile(errs, 0.9), 4),
            "max": round(errs[-1], 4),
            "rmse": round(math.sqrt(statistics.fmean([e * e for e in errs])), 4),
        },
        component_statistics_km={
            "status": "SECONDARY - reported for information only; the primary "
                      "output of this layer is the circular radius above",
            "mean_signed_east": round(statistics.fmean(east), 4),
            "mean_signed_north": round(statistics.fmean(north), 4),
            "stdev_east": round(statistics.stdev(east), 4),
            "stdev_north": round(statistics.stdev(north), 4),
            "correlation_east_north": round(
                statistics.correlation(east, north), 4),
            "anisotropy_stdev_ratio": round(
                statistics.stdev(east) / statistics.stdev(north), 4),
        },
        physics_config_sha256=sha)


def _endpoint_offset(row: dict) -> tuple[float, float]:
    """(east, north) metres from the observed endpoint to the predicted one."""
    from src.models.iceberg_ml_residual import displacement_components
    return displacement_components(row["observed_end_latitude"],
                                   row["observed_end_longitude"],
                                   row["physics_pred_latitude"],
                                   row["physics_pred_longitude"])


# ------------------------------------------------------------- the radius
@dataclass(frozen=True)
class UncertaintyResult:
    """A radius and every fact needed to read it correctly."""

    radius_km: float
    horizon_seconds: float
    horizon_hours: float
    extrapolated_uncertainty: bool
    empirically_calibrated_to_hours: float
    calibration_quantile: float
    calibration_n: int
    growth_law: str
    growth_exponent: float
    iceberg_id: str | None = None
    forecast_start_date: str | None = None
    physics_pred_latitude: float | None = None
    physics_pred_longitude: float | None = None
    source_quality_flag: str | None = None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def growth_factor(horizon_seconds: float, cfg: UncertaintyConfig) -> float:
    """(t / 24h) ** exponent, or t / 24h for the linear law. t=0 gives 0."""
    if not isinstance(horizon_seconds, (int, float)) or isinstance(
            horizon_seconds, bool):
        raise UncertaintyError(f"horizon_seconds must be a number, got "
                               f"{horizon_seconds!r}")
    t = float(horizon_seconds)
    if not math.isfinite(t):
        raise UncertaintyError(f"horizon_seconds must be finite, got {t}")
    if t < 0.0:
        raise UncertaintyError(
            f"horizon_seconds is {t}; a negative forecast horizon has no "
            f"uncertainty to report and is refused rather than reflected.")
    base = cfg.calibration_hours * SECONDS_PER_HOUR
    ratio = t / base
    if ratio == 0.0:
        return 0.0
    if cfg.law == LAW_LINEAR:
        return ratio
    if cfg.law == LAW_POWER:
        return ratio ** cfg.exponent
    raise UncertaintyError(f"unknown growth law {cfg.law!r}")


def uncertainty_at(horizon_seconds: float, cal: Calibration,
                   cfg: UncertaintyConfig, iceberg_id: str | None = None,
                   forecast_start_date: str | None = None,
                   physics_pred_latitude: float | None = None,
                   physics_pred_longitude: float | None = None,
                   source_quality_flag: str | None = None) -> UncertaintyResult:
    """The radius at a horizon. No observed position is accepted or consulted.

    The radius depends on the horizon and the calibration alone. The iceberg
    id, date and predicted position are carried through for provenance and
    change nothing.
    """
    radius = cal.radius_km * growth_factor(horizon_seconds, cfg)
    if not math.isfinite(radius) or radius < 0.0:
        raise UncertaintyError(f"computed radius {radius} is not finite and >= 0")
    if radius > cfg.max_radius_km:
        raise UncertaintyError(
            f"radius {radius:.1f} km exceeds the configured maximum "
            f"{cfg.max_radius_km} km at horizon {horizon_seconds / 3600:.1f} h")
    hours = float(horizon_seconds) / SECONDS_PER_HOUR
    return UncertaintyResult(
        radius_km=radius, horizon_seconds=float(horizon_seconds),
        horizon_hours=hours,
        extrapolated_uncertainty=hours > cal.horizon_hours,
        empirically_calibrated_to_hours=cal.horizon_hours,
        calibration_quantile=cal.quantile, calibration_n=cal.n,
        growth_law=cfg.law, growth_exponent=cfg.exponent,
        iceberg_id=iceberg_id, forecast_start_date=forecast_start_date,
        physics_pred_latitude=physics_pred_latitude,
        physics_pred_longitude=physics_pred_longitude,
        source_quality_flag=source_quality_flag)


# -------------------------------------------------------------- coverage
def coverage(rows: list[dict], cal: Calibration, cfg: UncertaintyConfig) -> dict:
    """What fraction of the held-out errors actually fall inside each radius."""
    errs = sorted(_physics_error_km(r) for r in rows)
    if not errs:
        return {"n": 0}
    sel = uncertainty_at(cal.horizon_hours * SECONDS_PER_HOUR, cal, cfg)
    out = {
        "n": len(errs),
        "median_error_km": round(statistics.median(errs), 4),
        "p90_error_km": round(_quantile(errs, 0.9), 4),
        "mean_error_km": round(statistics.fmean(errs), 4),
        "mean_radius_km": round(sel.radius_km, 4),
        "selected_radius_km": round(sel.radius_km, 4),
        "fraction_inside_selected_radius": round(
            sum(1 for e in errs if e <= sel.radius_km) / len(errs), 4),
    }
    for q in cfg.reported_quantiles:
        r = cal.quantile_radii_km[f"q{int(q * 100)}"]
        out[f"radius_q{int(q * 100)}_km"] = round(r, 4)
        out[f"fraction_inside_q{int(q * 100)}_radius"] = round(
            sum(1 for e in errs if e <= r) / len(errs), 4)
    return out


# ---------------------------------------------------------------- output
def _physics_position_at(row: dict, horizon_s: float) -> tuple[float, float, bool]:
    """The frozen baseline's position at a horizon. Constant velocity, as built.

    At 24 h this reproduces the stored prediction. Beyond it, the same constant
    velocity is simply carried further, which is the frozen model's own
    assumption extended -- flagged, because the environment was sampled once at
    the start and is not re-read.
    """
    dt_row = row["elapsed_seconds"]
    ve = row["physics_disp_east_m"] / dt_row
    vn = row["physics_disp_north_m"] / dt_row
    lat, lon = advect(row["start_latitude"], row["start_longitude"], ve, vn,
                      horizon_s)
    return lat, lon, horizon_s > dt_row


def write_predictions(rows: list[dict], cal: Calibration,
                      cfg: UncertaintyConfig, out: Path = PRED_CSV) -> int:
    horizons = [float(h) for h in cfg.raw["horizons_hours_for_output"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PRED_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            for hours in horizons:
                secs = hours * SECONDS_PER_HOUR
                lat, lon, extrap_pos = _physics_position_at(r, secs)
                u = uncertainty_at(secs, cal, cfg, iceberg_id=r["iceberg_id"],
                                   forecast_start_date=r["start_date"],
                                   physics_pred_latitude=lat,
                                   physics_pred_longitude=lon,
                                   source_quality_flag="observed_to_observed")
                w.writerow({
                    "iceberg_id": r["iceberg_id"],
                    "position_source": r["position_source"],
                    "start_date": r["start_date"], "horizon_hours": hours,
                    "physics_pred_latitude": lat,
                    "physics_pred_longitude": lon,
                    "uncertainty_radius_km": u.radius_km,
                    "extrapolated_uncertainty": u.extrapolated_uncertainty,
                    "calibration_quantile": cal.quantile,
                    "calibration_n": cal.n,
                    "calibration_horizon_hours": cal.horizon_hours,
                    "growth_law": cfg.law, "growth_exponent": cfg.exponent,
                    "interpolation_quality": "observed_to_observed",
                    "split": r["split"],
                    "physics_position_is_extrapolated": extrap_pos,
                })
                n += 1
    return n


def run(config: Path = CONFIG, split_config: Path = SPLIT_CONFIG,
        out_dir: Path = ICE_DIR) -> dict:
    cfg = load_config(config)
    pop, _ = build_population()
    chronological_split(pop, load_split_config(split_config))
    cal = calibrate(pop, cfg)

    moved = {s: [r for r in pop if r["split"] == s and r["moved"]]
             for s in ("train", "validation", "test")}
    horizons = [float(h) for h in cfg.raw["horizons_hours_for_output"]]
    curve = {}
    for law, exp_ in ((LAW_POWER, cfg.exponent), (LAW_LINEAR, 1.0)):
        trial = cfg.with_growth(law=law, exponent=exp_)
        curve[f"{law}_exponent_{exp_}"] = {
            f"{h:g}h": round(
                uncertainty_at(h * SECONDS_PER_HOUR, cal, trial).radius_km, 4)
            for h in sorted(set(horizons) | {0.0, 12.0, 24.0, 48.0, 72.0})}

    report = {
        "layer": "iceberg forecast uncertainty cone on the frozen physics baseline",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(Path(config).relative_to(ROOT)),
        "physics_baseline": {
            "frozen": True,
            "windage_coefficient": 0.0, "wind_turn_deg": 0.0,
            "config_sha256": cal.physics_config_sha256,
            "note": "the radius is calibrated against THIS physics "
                    "configuration; changing it voids the calibration",
        },
        "calibration": {
            "population": cfg.raw["calibration"]["population"],
            "splits_used": list(cal.splits_used),
            "n": cal.n,
            "date_range": [cal.date_min, cal.date_max],
            "quantile": cal.quantile,
            "radius_km": round(cal.radius_km, 6),
            "horizon_hours": cal.horizon_hours,
            "quantile_radii_km": cal.quantile_radii_km,
            "error_summary_km": cal.error_summary_km,
            "test_split_used_for_calibration": False,
        },
        "split_counts_moved": {s: len(v) for s, v in moved.items()},
        "validation_coverage": coverage(moved["validation"], cal, cfg),
        "held_out_test_coverage": coverage(moved["test"], cal, cfg),
        "growth_law": {
            "law": cfg.law, "exponent": cfg.exponent,
            "formula": cfg.raw["growth"]["formula"],
            "is_this_measured": False,
            "assumption_statement": cfg.raw["growth"]["assumption_statement"],
            "radius_km_by_horizon": curve,
            "linear_is_sensitivity_only": True,
        },
        "empirically_calibrated_to": f"{cal.horizon_hours:g}h",
        "48h_status": "extrapolated",
        "horizons_above_24h_are_extrapolated": True,
        "secondary_component_statistics_km": cal.component_statistics_km,
        "observed_endpoint_usage": ("observed endpoints produced the "
                                    "calibration errors and the coverage "
                                    "counts, and nothing else. uncertainty_at() "
                                    "has no parameter that could accept one."),
        "known_limitations": cfg.raw["known_limitations"],
        "not_included": cfg.raw["not_included"],
    }
    report["predictions_written"] = write_predictions(
        [r for r in pop if r["moved"]], cal, cfg, out_dir / PRED_CSV.name)
    (out_dir / EVAL_JSON.name).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Forecast uncertainty cone for the frozen iceberg physics "
                    "baseline. No risk raster, no route cost, no ML.")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--split-config", default=str(SPLIT_CONFIG))
    ap.add_argument("--out-dir", default=str(ICE_DIR))
    ap.add_argument("--calibrate", action="store_true",
                    help="report the calibration and exit, writing nothing")
    args = ap.parse_args(argv)

    if args.calibrate:
        cfg = load_config(Path(args.config))
        pop, _ = build_population()
        chronological_split(pop, load_split_config(Path(args.split_config)))
        cal = calibrate(pop, cfg)
        print(json.dumps({"radius_km": cal.radius_km, "quantile": cal.quantile,
                          "n": cal.n, "splits": list(cal.splits_used),
                          "dates": [cal.date_min, cal.date_max],
                          "quantile_radii_km": cal.quantile_radii_km,
                          "error_summary_km": cal.error_summary_km}, indent=2))
        return 0

    report = run(Path(args.config), Path(args.split_config), Path(args.out_dir))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {Path(args.out_dir) / PRED_CSV.name}")
    print(f"wrote {Path(args.out_dir) / EVAL_JSON.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except UncertaintyError as exc:
        print(f"\nCANNOT BUILD THE UNCERTAINTY CONE\n\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)

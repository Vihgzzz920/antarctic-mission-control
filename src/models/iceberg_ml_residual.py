"""
ML residual correction on top of the first-order physics drift baseline.

    data/processed/icebergs/iceberg_ml_predictions.csv
    data/processed/icebergs/iceberg_ml_evaluation.json

WHAT IS LEARNED
  Not the drift. The RESIDUAL of the drift:

      target_east  = observed_displacement_east  - physics_displacement_east
      target_north = observed_displacement_north - physics_displacement_north

  both in metres over a 24 h step, in the same local east/north tangent plane
  src/models/iceberg_physics.py integrates in. The corrected prediction is

      corrected_displacement = physics_displacement + predicted_residual

  so the physics baseline stays the reference and the model can only nudge it.
  One HistGradientBoostingRegressor per component, three fixed configurations,
  chosen on the validation split. Nothing deeper, and nothing tuned harder --
  with ~900 training rows a bigger search would be selecting on noise.

THE TARGET CANNOT REACH THE FEATURES
  The observed endpoint builds the target and scores the result. It is never a
  feature. `build_features` returns a plain matrix from a fixed, configured
  column list, `predict_corrected` takes that matrix and the physics
  displacement and nothing else, and the config carries a list of forbidden
  substrings that the feature names are checked against at load time. The test
  suite re-checks all of it, including that no training row's date is on or
  after a validation or test date.

TRAINED ON MOVED ROWS ONLY, REPORTED ON BOTH
  68% of the clean 24 h rows have ZERO observed displacement: the chart
  repeated the position. Training the residual on those would teach the model
  that the right answer is usually "undo the physics entirely". So the fit uses
  only rows whose observed displacement is greater than zero, by the project's
  own great_circle_m. The EVALUATION then reports both populations -- moved
  rows and all clean rows -- because a moved-only score would hide how the
  correction behaves on the repeated-position records that dominate the data.

CHRONOLOGICAL, NEVER SHUFFLED
  Train is the earliest rows, then validation, then test. Cut points come from
  the 60th and 80th percentile of the moved rows and are snapped to whole
  dates, so every row sharing a start date lands in one split and no same-day
  pair straddles a boundary. The test rows are touched exactly once, at the
  end, to score.

NOT IN SCOPE
  No uncertainty cones, no iceberg risk raster, no routing, no POLARIS, and no
  change to the physics model, which is imported and used as published.

Usage:
    python -m src.models.iceberg_ml_residual --audit   # counts only, writes nothing
    python -m src.models.iceberg_ml_residual           # fit, score, write
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.preprocess_icebergs import (EARTH_RADIUS_M, great_circle_m,
                                          _wrap_delta_deg)
from src.models.iceberg_physics import (CONFIG as PHYSICS_CONFIG, ICE_DIR,
                                        MATCHES_CSV, QUALITY_CLEAN, STEPS_CSV,
                                        _previous_step_velocity, advect,
                                        inputs_for, load_config as load_physics,
                                        load_rows, local_basis,
                                        predict_persistence, predict_physics,
                                        to_geographic)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "iceberg_ml_residual.json"
PRED_CSV = ICE_DIR / "iceberg_ml_predictions.csv"
EVAL_JSON = ICE_DIR / "iceberg_ml_evaluation.json"
HORIZON_S = 86_400.0
SPLITS = ("train", "validation", "test")

PRED_FIELDS = [
    "iceberg_id", "position_source", "start_date", "end_date", "split", "moved",
    "start_latitude", "start_longitude",
    "observed_end_latitude", "observed_end_longitude",
    "physics_pred_latitude", "physics_pred_longitude",
    "ml_pred_latitude", "ml_pred_longitude",
    "constant_velocity_pred_latitude", "constant_velocity_pred_longitude",
    "physics_error_km", "ml_error_km", "constant_velocity_error_km",
    "residual_east_m", "residual_north_m",
    "predicted_residual_east_m", "predicted_residual_north_m",
]


class ResidualError(ValueError):
    """The residual model cannot be built or scored as specified."""


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class ResidualConfig:
    raw: dict
    path: str

    @property
    def features(self) -> list[str]:
        return list(self.raw["features"])

    @property
    def physics_features(self) -> list[str]:
        return list(self.raw.get("physics_displacement_features", []))

    @property
    def grid(self) -> list[dict]:
        return list(self.raw["estimator"]["grid"])

    @property
    def random_state(self) -> int:
        return int(self.raw["estimator"]["random_state"])

    @property
    def min_training_rows(self) -> int:
        return int(self.raw["limits"]["min_training_rows"])


def load_config(path: Path | str = CONFIG) -> ResidualConfig:
    """Read the configuration and refuse anything that smells like the target."""
    path = Path(path)
    if not path.exists():
        raise ResidualError(f"No residual configuration at {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ResidualError(f"{path.name} is not valid JSON: {exc}") from None
    feats = raw.get("features")
    if not feats or not isinstance(feats, list):
        raise ResidualError(f"{path.name} has no feature list")
    if len(set(feats)) != len(feats):
        raise ResidualError(f"{path.name} repeats a feature name")
    banned = raw.get("forbidden_feature_substrings", [])
    leaks = sorted({f for f in feats for b in banned if b in f})
    if leaks:
        raise ResidualError(
            f"{path.name} lists feature(s) {leaks} that match a forbidden "
            f"substring from {banned}; the observed endpoint and the target "
            f"must never be features.")
    missing = [f for f in raw.get("physics_displacement_features", [])
               if f not in feats]
    if missing:
        raise ResidualError(
            f"physics_displacement_features {missing} are not in the feature list")
    if not raw.get("estimator", {}).get("grid"):
        raise ResidualError(f"{path.name} has no estimator grid")
    return ResidualConfig(raw=raw, path=str(path))


# ------------------------------------------------------------------- data
def displacement_components(lat0: float, lon0: float, lat1: float,
                            lon1: float) -> tuple[float, float]:
    """(east, north) metres from one position to another, tangent plane.

    The exact companion of iceberg_physics.advect, so a displacement and the
    position it produces are the same quantity read two ways.
    """
    phi_mid = math.radians(0.5 * (lat0 + lat1))
    east = (EARTH_RADIUS_M * math.radians(_wrap_delta_deg(lon1 - lon0))
            * math.cos(phi_mid))
    north = EARTH_RADIUS_M * math.radians(lat1 - lat0)
    return east, north


def offset_position(lat: float, lon: float, east_m: float, north_m: float,
                    elapsed_s: float = HORIZON_S) -> tuple[float, float]:
    """Apply an east/north displacement, through iceberg_physics.advect."""
    return advect(lat, lon, east_m / elapsed_s, north_m / elapsed_s, elapsed_s)


def build_population(matches_csv: Path = MATCHES_CSV,
                     steps_csv: Path = STEPS_CSV,
                     physics_config: Path = PHYSICS_CONFIG) -> tuple[list[dict], dict]:
    """The clean 24 h rows, each with its physics prediction and its residual."""
    pcfg = load_physics(physics_config)
    rows, _ = load_rows(matches_csv, steps_csv)
    prev_vel = _previous_step_velocity(steps_csv)
    out, stats = [], Counter()
    for r in rows:
        if r["elapsed_days"] != 1:
            continue
        if r["interpolation_quality"] != QUALITY_CLEAN:
            stats["excluded_interpolated_endpoint"] += 1
            continue
        if not r["has_environment"]:
            stats["excluded_missing_current_or_wind"] += 1
            continue
        p = predict_physics(inputs_for(r), pcfg)
        obs_m = great_circle_m(r["start_latitude"], r["start_longitude"],
                               r["observed_end_latitude"],
                               r["observed_end_longitude"])
        oe, on = displacement_components(
            r["start_latitude"], r["start_longitude"],
            r["observed_end_latitude"], r["observed_end_longitude"])
        pe, pn = displacement_components(
            r["start_latitude"], r["start_longitude"], p.latitude, p.longitude)
        basis = local_basis(r["start_latitude"], r["start_longitude"])
        ce, cn = to_geographic(r["current_u_3976"], r["current_v_3976"], basis)
        pv = prev_vel.get((r["iceberg_id"], r["position_source"],
                           r["start_date"]))
        out.append({
            **{k: r[k] for k in ("iceberg_id", "position_source", "start_date",
                                 "end_date", "start_latitude",
                                 "start_longitude", "observed_end_latitude",
                                 "observed_end_longitude", "elapsed_seconds")},
            "observed_m": obs_m,
            "moved": obs_m > 0.0,
            "physics_pred_latitude": p.latitude,
            "physics_pred_longitude": p.longitude,
            "physics_disp_east_m": pe, "physics_disp_north_m": pn,
            "residual_east_m": oe - pe, "residual_north_m": on - pn,
            "current_east_mps": ce, "current_north_mps": cn,
            "wind_east_mps": r["wind_u10_east"],
            "wind_north_mps": r["wind_v10_north"],
            "current_speed_mps": math.hypot(ce, cn),
            "wind_speed_mps": math.hypot(r["wind_u10_east"],
                                         r["wind_v10_north"]),
            "latitude_deg": r["start_latitude"],
            "longitude_deg": r["start_longitude"],
            "prev_velocity": pv,
        })
        stats["moved" if obs_m > 0.0 else "zero_displacement"] += 1
    if not out:
        raise ResidualError("no clean 24 h rows with both environmental inputs")
    return out, dict(sorted(stats.items()))


def chronological_split(pop: list[dict], cfg: ResidualConfig) -> dict:
    """Date boundaries from the MOVED rows, applied to every clean row."""
    s = cfg.raw["split"]
    f_tr = float(s["train_fraction_of_moved"])
    f_va = float(s["validation_fraction_of_moved"])
    moved = sorted((r for r in pop if r["moved"]), key=lambda r: r["start_date"])
    if len(moved) < 10:
        raise ResidualError(f"only {len(moved)} moved rows; no split is meaningful")
    d_tr = moved[int(len(moved) * f_tr)]["start_date"]
    d_va = moved[int(len(moved) * (f_tr + f_va))]["start_date"]
    for r in pop:
        r["split"] = ("train" if r["start_date"] < d_tr else
                      "validation" if r["start_date"] < d_va else "test")
    counts = {}
    for name in SPLITS:
        rows = [r for r in pop if r["split"] == name]
        mv = [r for r in rows if r["moved"]]
        dates = sorted(r["start_date"] for r in rows)
        counts[name] = {
            "rows_all_clean": len(rows), "rows_moved": len(mv),
            "rows_zero_displacement": len(rows) - len(mv),
            "date_min": dates[0] if dates else None,
            "date_max": dates[-1] if dates else None,
            "rows_with_current_and_wind": len(rows),
            "rows_with_a_valid_observed_endpoint": sum(
                1 for r in rows if r["observed_end_latitude"] is not None),
            "rows_with_a_persistence_predecessor": sum(
                1 for r in rows if r["prev_velocity"] is not None),
        }
    return {"train_before": d_tr, "validation_before": d_va, "counts": counts}


def feature_matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    """Columns in the configured order. A missing value is an error, not a zero."""
    m = np.empty((len(rows), len(features)), dtype="float64")
    for i, r in enumerate(rows):
        for j, f in enumerate(features):
            v = r.get(f)
            if v is None or not math.isfinite(float(v)):
                raise ResidualError(
                    f"feature {f!r} is missing or non-finite for "
                    f"{r['iceberg_id']} {r['start_date']}; nothing is imputed")
            m[i, j] = float(v)
    return m


# ------------------------------------------------------------------ model
def _fit_pair(train: list[dict], features: list[str], params: dict,
              random_state: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = feature_matrix(train, features)
    models = {}
    for comp in ("east", "north"):
        y = np.array([r[f"residual_{comp}_m"] for r in train], dtype="float64")
        est = HistGradientBoostingRegressor(random_state=random_state, **params)
        est.fit(X, y)
        models[comp] = est
    return models


def predict_corrected(models: dict, rows: list[dict], features: list[str]
                      ) -> list[tuple[float, float, float, float]]:
    """(lat, lon, residual_east, residual_north) per row. No target is read."""
    X = feature_matrix(rows, features)
    re_ = models["east"].predict(X)
    rn_ = models["north"].predict(X)
    out = []
    for r, de, dn in zip(rows, re_, rn_):
        east = r["physics_disp_east_m"] + float(de)
        north = r["physics_disp_north_m"] + float(dn)
        lat, lon = offset_position(r["start_latitude"], r["start_longitude"],
                                   east, north, r["elapsed_seconds"])
        out.append((lat, lon, float(de), float(dn)))
    return out


def _endpoint_km(rows: list[dict], preds) -> list[float]:
    return [great_circle_m(lat, lon, r["observed_end_latitude"],
                           r["observed_end_longitude"]) / 1000.0
            for r, (lat, lon, _, _) in zip(rows, preds)]


def _metrics(errors_km: list[float]) -> dict:
    if not errors_km:
        return {"n": 0}
    s = sorted(errors_km)
    return {
        "n": len(s),
        "mean_error_km": round(statistics.fmean(s), 4),
        "median_error_km": round(statistics.median(s), 4),
        "rmse_error_km": round(math.sqrt(statistics.fmean([e * e for e in s])), 4),
        "p90_error_km": round(s[min(len(s) - 1,
                                    int(math.ceil(0.9 * len(s)) - 1))], 4),
        "max_error_km": round(s[-1], 4),
    }


def _physics_km(rows: list[dict]) -> list[float]:
    return [great_circle_m(r["physics_pred_latitude"], r["physics_pred_longitude"],
                           r["observed_end_latitude"],
                           r["observed_end_longitude"]) / 1000.0 for r in rows]


def _persistence_km(rows: list[dict]) -> tuple[list[float], int]:
    out, skipped = [], 0
    for r in rows:
        pv = r["prev_velocity"]
        if pv is None:
            skipped += 1
            continue
        lat, lon = predict_persistence(r["start_latitude"], r["start_longitude"],
                                       pv[0], pv[1], r["elapsed_seconds"])
        out.append(great_circle_m(lat, lon, r["observed_end_latitude"],
                                  r["observed_end_longitude"]) / 1000.0)
    return out, skipped


def _residual_diagnostics(rows: list[dict], preds) -> dict:
    se = [float(de) - r["residual_east_m"] for r, (_, _, de, _) in zip(rows, preds)]
    sn = [float(dn) - r["residual_north_m"] for r, (_, _, _, dn) in zip(rows, preds)]
    if not se:
        return {"n": 0}
    return {
        "n": len(se),
        "mean_signed_east_residual_error_m": round(statistics.fmean(se), 3),
        "mean_signed_north_residual_error_m": round(statistics.fmean(sn), 3),
        "residual_rmse_east_m": round(
            math.sqrt(statistics.fmean([v * v for v in se])), 3),
        "residual_rmse_north_m": round(
            math.sqrt(statistics.fmean([v * v for v in sn])), 3),
        "observed_mean_signed_east_residual_m": round(
            statistics.fmean([r["residual_east_m"] for r in rows]), 3),
        "observed_mean_signed_north_residual_m": round(
            statistics.fmean([r["residual_north_m"] for r in rows]), 3),
    }


def _importance(models: dict, rows: list[dict], features: list[str],
                random_state: int) -> dict:
    """Permutation importance on the VALIDATION rows. Never on test."""
    from sklearn.inspection import permutation_importance

    X = feature_matrix(rows, features)
    out = {}
    for comp in ("east", "north"):
        y = np.array([r[f"residual_{comp}_m"] for r in rows], dtype="float64")
        res = permutation_importance(models[comp], X, y, n_repeats=5,
                                     random_state=random_state, scoring=None)
        out[comp] = {f: round(float(v), 5) for f, v in
                     sorted(zip(features, res.importances_mean),
                            key=lambda kv: -kv[1])}
    return out


# ------------------------------------------------------------------- run
def run(matches_csv: Path = MATCHES_CSV, steps_csv: Path = STEPS_CSV,
        config: Path = CONFIG, physics_config: Path = PHYSICS_CONFIG,
        out_dir: Path = ICE_DIR) -> dict:
    cfg = load_config(config)
    pop, pop_stats = build_population(matches_csv, steps_csv, physics_config)
    split = chronological_split(pop, cfg)

    by = {s: [r for r in pop if r["split"] == s] for s in SPLITS}
    moved = {s: [r for r in by[s] if r["moved"]] for s in SPLITS}
    if len(moved["train"]) < cfg.min_training_rows:
        raise ResidualError(
            f"only {len(moved['train'])} moved training rows; below the "
            f"configured minimum {cfg.min_training_rows}")

    features = cfg.features
    no_phys = [f for f in features if f not in cfg.physics_features]

    # ---- selection on VALIDATION only --------------------------------------
    trials = []
    for params in cfg.grid:
        models = _fit_pair(moved["train"], features, params, cfg.random_state)
        preds = predict_corrected(models, moved["validation"], features)
        km = _endpoint_km(moved["validation"], preds)
        trials.append({"params": params, "validation": _metrics(km)})
    best = min(trials, key=lambda t: t["validation"]["mean_error_km"])
    models = _fit_pair(moved["train"], features, best["params"], cfg.random_state)
    models_no_phys = _fit_pair(moved["train"], no_phys, best["params"],
                               cfg.random_state)

    # ---- the one and only look at TEST -------------------------------------
    results = {}
    for label, rows in (("moved_only", moved["test"]), ("all_clean", by["test"])):
        preds = predict_corrected(models, rows, features)
        preds_np = predict_corrected(models_no_phys, rows, no_phys)
        pers, skipped = _persistence_km(rows)
        results[label] = {
            "n": len(rows),
            "constant_velocity_persistence": _metrics(pers),
            "persistence_rows_without_a_predecessor": skipped,
            "physics_baseline": _metrics(_physics_km(rows)),
            "physics_plus_ml_residual": _metrics(_endpoint_km(rows, preds)),
            "ablation_ml_without_physics_displacement_features":
                _metrics(_endpoint_km(rows, preds_np)),
            "residual_diagnostics": _residual_diagnostics(rows, preds),
        }

    report = {
        "model": "ML residual correction on the first-order physics drift baseline",
        "corrected_prediction": "physics_displacement + predicted_residual",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {"matches_csv": str(Path(matches_csv).relative_to(ROOT)),
                   "steps_csv": str(Path(steps_csv).relative_to(ROOT)),
                   "residual_config": str(Path(config).relative_to(ROOT)),
                   "physics_config": str(Path(physics_config).relative_to(ROOT))},
        "population": {"clean_24h_rows": len(pop), **pop_stats},
        "split": split,
        "features_used": features,
        "features_omitted": cfg.raw["features_deliberately_omitted"],
        "estimator": {
            "name": "HistGradientBoostingRegressor",
            "one_per_component": ["east", "north"],
            "random_state": cfg.random_state,
            "grid_searched": [t["params"] for t in trials],
            "grid_validation_scores": [
                {"params": t["params"],
                 "validation_mean_error_km": t["validation"]["mean_error_km"]}
                for t in trials],
            "selected_params": best["params"],
            "selected_on": "validation split only",
        },
        "validation_metrics_of_selected_model": best["validation"],
        "test_metrics": results,
        "ablation": {
            "a_physics_only": results["moved_only"]["physics_baseline"],
            "b_ml_residual_full_features":
                results["moved_only"]["physics_plus_ml_residual"],
            "c_ml_residual_without_physics_displacement_features":
                results["moved_only"][
                    "ablation_ml_without_physics_displacement_features"],
            "question": ("does the model correct the physics, or replace it? "
                         "If (c) matches (b), the physics displacement "
                         "features are carrying no information the rest of the "
                         "features do not already have."),
            "degeneracy_check": _ablation_is_degenerate(moved["train"], cfg),
        },
        "feature_importance_permutation_on_validation":
            _importance(models, moved["validation"], features, cfg.random_state),
        "out_of_horizon_48h": {
            "attempted": False,
            "reason": ("the target is a displacement over exactly 86400 s. "
                       "Applying it unscaled to a 48 h step would be wrong, and "
                       "scaling it by the elapsed-time ratio would be an "
                       "assumption this project has not measured. Only 6 clean "
                       "moved 48 h rows exist, so neither variant could be "
                       "told apart from noise."),
        },
        "leakage_checks": _leakage_checks(cfg, by, moved, features),
        "not_included": ["uncertainty cones", "iceberg risk rasters", "routing",
                         "POLARIS", "deep learning"],
        "improvement_claimed": False,
        "improvement_note": ("the measured test metrics above are reported as "
                             "they came out; no claim of improvement is made "
                             "beyond what they show"),
    }
    n = write_predictions(pop, models, features, out_dir / PRED_CSV.name)
    report["predictions_written"] = n
    (out_dir / EVAL_JSON.name).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _ablation_is_degenerate(rows: list[dict], cfg: ResidualConfig) -> dict:
    """Is a physics feature just a rescale of one that survives the ablation?

    With the shipped windage of 0.0 the drift IS the current, so
    physics_disp_east_m == 86400 * current_east_mps exactly. Dropping the
    physics features then removes nothing the model still has, the ablation
    reproduces the full model to the bit, and "identical results" would read
    as an informative null when it is arithmetic.
    """
    survivors = [f for f in cfg.features if f not in cfg.physics_features]
    out = {}
    for f in cfg.physics_features:
        a = np.array([float(r[f]) for r in rows])
        best, best_r, best_scale = None, 0.0, None
        for g in survivors:
            b = np.array([float(r[g]) for r in rows])
            if a.std() == 0 or b.std() == 0:
                continue
            r_ = abs(float(np.corrcoef(a, b)[0, 1]))
            if r_ > best_r:
                nz = b != 0
                best, best_r = g, r_
                best_scale = float(np.median(a[nz] / b[nz])) if nz.any() else None
        out[f] = {"most_correlated_surviving_feature": best,
                  "abs_pearson_r": round(best_r, 12),
                  "median_ratio": best_scale,
                  "is_a_rescale_of_it": best_r > 1 - 1e-9}
    degenerate = any(v["is_a_rescale_of_it"] for v in out.values())
    return {
        "physics_features_are_redundant": degenerate,
        "per_feature": out,
        "consequence": (
            "the ablation cannot test whether the model corrects the physics "
            "or replaces it: the dropped features are a constant multiple of "
            "features that remain, so both models see the same information and "
            "produce identical predictions. This is a property of the "
            "configured windage_coefficient = 0.0, which makes the drift "
            "exactly the ocean current. With a non-zero windage the physics "
            "displacement stops being collinear with the current and the "
            "ablation becomes informative."
            if degenerate else
            "the physics features carry information the others do not, so the "
            "ablation compares two genuinely different models"),
    }


def _leakage_checks(cfg: ResidualConfig, by: dict, moved: dict,
                    features: list[str]) -> dict:
    banned = cfg.raw["forbidden_feature_substrings"]
    tr = [r["start_date"] for r in by["train"]]
    va = [r["start_date"] for r in by["validation"]]
    te = [r["start_date"] for r in by["test"]]
    keys = {s: {(r["iceberg_id"], r["position_source"], r["start_date"])
                for r in by[s]} for s in SPLITS}
    return {
        "no_forbidden_substring_in_any_feature": not any(
            b in f for f in features for b in banned),
        "forbidden_substrings": banned,
        "max_train_date": max(tr), "min_validation_date": min(va),
        "max_validation_date": max(va), "min_test_date": min(te),
        "train_strictly_before_validation": max(tr) < min(va),
        "validation_strictly_before_test": max(va) < min(te),
        "splits_are_disjoint": (
            not (keys["train"] & keys["validation"])
            and not (keys["validation"] & keys["test"])
            and not (keys["train"] & keys["test"])),
        "fitted_rows": len(moved["train"]),
        "fitted_on_validation_or_test": False,
        "target_is_never_a_feature": True,
        "observed_endpoint_in_feature_list": [f for f in features
                                              if "observed" in f],
    }


def write_predictions(pop: list[dict], models: dict, features: list[str],
                      out: Path = PRED_CSV) -> int:
    preds = predict_corrected(models, pop, features)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PRED_FIELDS, lineterminator="\n")
        w.writeheader()
        for r, (lat, lon, de, dn) in zip(pop, preds):
            obs = (r["observed_end_latitude"], r["observed_end_longitude"])
            rec = {k: r.get(k, "") for k in PRED_FIELDS}
            rec["ml_pred_latitude"], rec["ml_pred_longitude"] = lat, lon
            rec["predicted_residual_east_m"] = de
            rec["predicted_residual_north_m"] = dn
            rec["ml_error_km"] = great_circle_m(lat, lon, *obs) / 1000.0
            rec["physics_error_km"] = great_circle_m(
                r["physics_pred_latitude"], r["physics_pred_longitude"],
                *obs) / 1000.0
            pv = r["prev_velocity"]
            if pv is not None:
                clat, clon = predict_persistence(
                    r["start_latitude"], r["start_longitude"], pv[0], pv[1],
                    r["elapsed_seconds"])
                rec["constant_velocity_pred_latitude"] = clat
                rec["constant_velocity_pred_longitude"] = clon
                rec["constant_velocity_error_km"] = great_circle_m(
                    clat, clon, *obs) / 1000.0
            else:
                for k in ("constant_velocity_pred_latitude",
                          "constant_velocity_pred_longitude",
                          "constant_velocity_error_km"):
                    rec[k] = ""
            w.writerow(rec)
    return len(pop)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="ML residual correction on the physics drift baseline. "
                    "No cones, no risk raster, no routing.")
    ap.add_argument("--matches", default=str(MATCHES_CSV))
    ap.add_argument("--steps", default=str(STEPS_CSV))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--physics-config", default=str(PHYSICS_CONFIG))
    ap.add_argument("--out-dir", default=str(ICE_DIR))
    ap.add_argument("--audit", action="store_true",
                    help="report the split counts and exit, writing nothing")
    args = ap.parse_args(argv)

    if args.audit:
        cfg = load_config(Path(args.config))
        pop, stats = build_population(Path(args.matches), Path(args.steps),
                                      Path(args.physics_config))
        print(json.dumps({"population": {"clean_24h_rows": len(pop), **stats},
                          "split": chronological_split(pop, cfg)}, indent=2))
        return 0

    report = run(Path(args.matches), Path(args.steps), Path(args.config),
                 Path(args.physics_config), Path(args.out_dir))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {Path(args.out_dir) / PRED_CSV.name}")
    print(f"wrote {Path(args.out_dir) / EVAL_JSON.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ResidualError as exc:
        print(f"\nCANNOT RUN THE RESIDUAL MODEL\n\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)

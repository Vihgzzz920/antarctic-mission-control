"""
Turn the trained HGB sea-ice model into a forecast raster on the routing grid.

    python -m src.models.forecast_sic_raster --origin 2025-01-08
    python -m src.models.forecast_sic_raster --origin 2025-12-15 --validate

WHAT THIS IS
  The missing stage between src/models/train_forecast_model.py, which fits
  features(t) -> SIC(t+1) and saves a joblib artifact, and anything that wants
  a FIELD rather than a score. Until now the model's output existed only as
  metrics; this writes it as a GeoTIFF on the same EPSG:3976 6.25 km grid the
  navigation system routes on, with the provenance that makes it auditable.

  One artifact per trained lead, and each field is produced by the model fitted
  for THAT lead (see TRAINED_LEAD_HOURS). Nothing here produces a +6 h or +12 h
  sea-ice forecast, and nothing here relabels one lead's field as another's --
  in the raster's tags, in the metrics, or in a printed report.

WHAT IT IS NOT
  Not real-time. Every origin this can be run for is a date already in the
  archive, so every product is a HINDCAST: the observation that validates it
  already exists. The label is written into the raster itself.

  Not a retraining, a new architecture, or a second feature definition. The
  feature columns are assembled BY NAME from build_forecast_dataset.FEATURE_NAMES,
  so a change there is followed here rather than silently diverging, and the
  clipping bounds come from train_forecast_model rather than being re-typed.

NO FUTURE INPUT, CHECKED RATHER THAN ASSERTED IN PROSE
  A forecast issued at origin T may read only what existed at T. The model's
  features are sic_t, current_u_t, current_v_t and current_speed_t -- all from
  the origin day -- and the target sic_{t+1} is never a feature. Every input
  file this module opens is recorded with the date parsed from its own name,
  and `assert_no_future_inputs` refuses the write if any of them is later than
  the origin. The valid-time observation is opened ONLY by the validation path,
  and only after the raster exists.

INVALID STAYS INVALID
  A cell is predicted only where every feature is finite under the project's
  own masking rule (preprocess.load_sic, which is also what the router loads
  through). Everywhere else the output is NaN. No cell is filled, interpolated,
  or promoted from missing to a concentration, and the model's output is
  clipped to the physical SIC bounds the training script already uses.

  Note the consequence of the archive's own metadata: the observed GeoTIFFs
  declare nodata=0, so load_sic masks zero cells. The valid area is therefore
  the ice-bearing part of the grid, not the whole ocean, exactly as the router
  already sees it. That is the shipped behaviour of the observation loader and
  is not changed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from src.data.preprocess import OBSERVED, RAW_DIR, load_sic
from src.models.build_forecast_dataset import (CURRENTS_DIR, FEATURE_NAMES,
                                               N_FEATURES)
from src.models.persistence import MIZ_HI, MIZ_LO, score
from src.models.train_forecast_model import (DEFAULT_MODEL, SIC_HI, SIC_LO,
                                             band_masks, model_path_for)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "sic_forecast"

#  Every lead the project has actually TRAINED a model for. A lead is in this
#  tuple only when its own artifact exists on disk; nothing here relabels one
#  lead's field as another's, and a lead with no artifact is refused rather
#  than served by its neighbour.
LEAD_HOURS = 24                       # the default, and the shipped lead
TRAINED_LEAD_HOURS = (24, 48)
MODEL_NAME = "HistGradientBoostingRegressor"
#  how the model is named in a one-line report, before its lead is appended
MODEL_NAME_SHORT = "HGB"
FORECAST_KIND = "FORECAST / HINDCAST VALIDATION"

#  NaN rather than a sentinel number: a reader that ignores the nodata tag gets
#  something that cannot be mistaken for a concentration.
NODATA = float("nan")
DTYPE = "float32"


#  the scikit-learn version that PICKLED the artifact, captured when it is
#  loaded. A mismatch with the running version does not stop the prediction --
#  it is recorded, so a reader can see it rather than discover it later.
_ARTIFACT_SKLEARN: dict = {}


class ForecastRasterError(RuntimeError):
    """The forecast raster cannot be produced or trusted as specified."""


class FutureInputError(ForecastRasterError):
    """An input postdates the forecast origin. That is leakage, not a forecast."""


# ------------------------------------------------------------------ naming
def model_label(lead_hours) -> str:
    """How a run of this model is named in a report: HGB +<lead>h.

    Derived from the VALIDATED lead the forecast carries -- never from a file
    name, and never from a default. A report that cannot state its lead does
    not get a lead in its label.
    """
    try:
        hours = int(lead_hours)
    except (TypeError, ValueError):
        raise ForecastRasterError(
            f"cannot label a run whose lead is {lead_hours!r}; the lead comes "
            f"from the forecast that was produced, so it is always known.")
    if hours < 1:
        raise ForecastRasterError(
            f"cannot label a run with a lead of {hours} h")
    return f"{MODEL_NAME_SHORT} +{hours}h"


def raster_name(origin: date, lead_hours: int = LEAD_HOURS) -> str:
    valid = origin + timedelta(hours=lead_hours)
    return (f"sic_forecast_hgb_origin{origin:%Y%m%d}_plus{lead_hours}h_"
            f"valid{valid:%Y%m%d}_3976.tif")


def raster_path(origin: date, out_dir: Path = OUT_DIR,
                lead_hours: int = LEAD_HOURS) -> Path:
    return Path(out_dir) / raster_name(origin, lead_hours)


def metrics_path(origin: date, out_dir: Path = OUT_DIR,
                 lead_hours: int = LEAD_HOURS) -> Path:
    return Path(out_dir) / (raster_name(origin, lead_hours)[:-4]
                            + "_hindcast_metrics.json")


def _repo_relative(path: Path) -> str:
    """Public metadata carries repository paths, never this machine's."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------- inputs and leakage
@dataclass(frozen=True)
class InputFile:
    """One file the prediction read, and the timestamp it belongs to."""

    role: str
    path: Path
    #  the instant this input describes, parsed from the archive's own naming
    timestamp: datetime

    def to_dict(self) -> dict:
        return {"role": self.role, "path": _repo_relative(self.path),
                "timestamp": self.timestamp.isoformat()}


def assert_no_future_inputs(inputs, origin_time: datetime) -> None:
    """Refuse a 'forecast' that read anything from after its own origin."""
    late = [i for i in inputs if i.timestamp > origin_time]
    if late:
        raise FutureInputError(
            "these inputs postdate the forecast origin "
            f"{origin_time.isoformat()} and would leak the future into it:\n  "
            + "\n  ".join(f"{i.role}: {_repo_relative(i.path)} "
                          f"({i.timestamp.isoformat()})" for i in late))


def _origin_datetime(origin: date) -> datetime:
    """Archive days carry no time of day; an origin is 00:00 UTC on its date,
    the same assumption the iceberg environment matching already records."""
    return datetime(origin.year, origin.month, origin.day)


# ------------------------------------------------------------- prediction
@dataclass(frozen=True)
class SICForecast:
    """One predicted field, with everything needed to audit it."""

    origin: date
    origin_time: datetime
    valid_time: datetime
    lead_hours: int
    sic: np.ndarray                  # float32 (H, W); NaN where not predicted
    valid_mask: np.ndarray           # bool (H, W)
    transform: tuple
    crs_epsg: int
    inputs: tuple
    model_path: Path
    model_sha256: str
    feature_names: tuple
    clipped_cells: int
    training_period: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return tuple(self.sic.shape)


def _feature_columns(sic: np.ndarray, u: np.ndarray, v: np.ndarray) -> dict:
    """The model's features, keyed by the names the dataset builder defined.

    Assembling by name rather than by position means this module cannot drift
    out of column order with build_forecast_dataset; an unknown name is a hard
    error rather than a guess.
    """
    columns = {
        "sic_t": sic,
        "current_u_t": u,
        "current_v_t": v,
        "current_speed_t": np.hypot(u, v),
    }
    unknown = [n for n in FEATURE_NAMES if n not in columns]
    if unknown:
        raise ForecastRasterError(
            f"the dataset defines feature(s) {unknown} that this module does "
            f"not know how to build for a single origin. Nothing is guessed: "
            f"add them here deliberately.")
    if len(FEATURE_NAMES) != N_FEATURES:
        raise ForecastRasterError("FEATURE_NAMES and N_FEATURES disagree")
    return columns


def load_model(model_path: Path = DEFAULT_MODEL):
    """The artifact the training script wrote. Nothing is retrained here."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ForecastRasterError(
            f"no trained model at {_repo_relative(model_path)}. Run "
            f"`python -m src.models.train_forecast_model` first; this module "
            f"does not train one.")
    try:
        import joblib
    except ImportError as exc:                                   # pragma: no cover
        raise ForecastRasterError(
            "joblib is required to load the trained model") from exc
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(model_path)
    for entry in caught:
        pickled = getattr(entry.message, "original_sklearn_version", None)
        if pickled:
            _ARTIFACT_SKLEARN[str(Path(model_path).resolve())] = str(pickled)
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != N_FEATURES:
        raise ForecastRasterError(
            f"the artifact expects {n_in} features but the dataset defines "
            f"{N_FEATURES} ({', '.join(FEATURE_NAMES)}); refusing to feed it "
            f"columns it was not trained on.")
    return model


def _training_period(npz_path: Path = ROOT / "data" / "processed"
                     / "forecast_dataset.npz") -> dict:
    """The split ranges the dataset recorded, if the dataset is still there."""
    if not Path(npz_path).exists():
        return {}
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            out = {}
            for split in ("train", "val", "test"):
                key = f"{split}_base_range"
                if key in data.files:
                    lo, hi = (str(v) for v in data[key])
                    out[split] = {"base_from": lo, "base_to": hi}
            return out
    except Exception:                                            # noqa: BLE001
        return {}


def predict_sic(origin: date, *, lead_hours: int = LEAD_HOURS, model=None,
                model_path: Path | None = None,
                sic_dir: Path = RAW_DIR, currents_dir: Path = CURRENTS_DIR,
                ) -> SICForecast:
    """Predict SIC at origin + lead from what was available at the origin.

    The lead selects the ARTIFACT: a +48 h field comes from the +48 h model and
    from nothing else. `model_path` overrides that choice explicitly; it is
    never inferred from a file name.
    """
    lead_hours = int(lead_hours)
    if lead_hours not in TRAINED_LEAD_HOURS:
        raise ForecastRasterError(
            f"no model has been trained for a +{lead_hours} h lead. Trained "
            f"lead(s): {', '.join(f'+{h}h' for h in TRAINED_LEAD_HOURS)}. A "
            f"field for another lead is never relabelled to fill the gap.")
    if model_path is None:
        model_path = model_path_for(lead_hours)
    origin = origin if isinstance(origin, date) else date.fromisoformat(str(origin))
    origin_time = _origin_datetime(origin)
    sic_path = Path(sic_dir) / f"sic_{origin:%Y%m%d}.tif"
    cur_path = Path(currents_dir) / f"currents_{origin:%Y%m%d}.tif"
    for path in (sic_path, cur_path):
        if not path.exists():
            raise ForecastRasterError(
                f"missing input for origin {origin}: {_repo_relative(path)}. "
                f"Nothing is substituted for it.")
    if not Path(model_path).exists():
        raise ForecastRasterError(
            f"no +{lead_hours} h model artifact at "
            f"{_repo_relative(Path(model_path))}. Train it first; no other "
            f"lead's model is used in its place.")

    inputs = (InputFile("sic_t", sic_path, origin_time),
              InputFile("currents_t", cur_path, origin_time))
    #  before a single byte is read into the model
    assert_no_future_inputs(inputs, origin_time)

    day = load_sic(sic_path)                    # the project's own masking rule
    sic = np.asarray(day.sic, dtype="float64")
    with rasterio.open(cur_path) as src:
        if src.count < 2:
            raise ForecastRasterError(
                f"{cur_path.name} has {src.count} band(s), need 2 (u_x, u_y)")
        u = src.read(1).astype("float64")
        v = src.read(2).astype("float64")
        cur_transform = tuple(float(t) for t in list(src.transform)[:6])
        cur_epsg = src.crs.to_epsg() if src.crs else None

    if u.shape != sic.shape or v.shape != sic.shape:
        raise ForecastRasterError(
            f"grid mismatch at {origin}: currents {u.shape} vs SIC {sic.shape}; "
            f"nothing is resampled here.")
    if tuple(day.grid.transform) != cur_transform or day.grid.epsg != cur_epsg:
        raise ForecastRasterError(
            f"SIC and currents for {origin} are not on the same grid "
            f"({day.grid.epsg}/{tuple(day.grid.transform)} against "
            f"{cur_epsg}/{cur_transform})")

    columns = _feature_columns(sic, u, v)
    #  exactly the dataset builder's rule, minus the target it cannot know yet
    valid = np.ones(sic.shape, dtype=bool)
    for name in FEATURE_NAMES:
        valid &= np.isfinite(columns[name])

    out = np.full(sic.shape, np.nan, dtype="float64")
    clipped = 0
    if valid.any():
        model = model if model is not None else load_model(model_path)
        X = np.column_stack([columns[name][valid] for name in FEATURE_NAMES])
        pred = np.asarray(model.predict(X), dtype="float64")
        #  the training script's own bounds, applied the same way
        clipped = int(((pred < SIC_LO) | (pred > SIC_HI)).sum())
        np.clip(pred, SIC_LO, SIC_HI, out=pred)
        out[valid] = pred

    return SICForecast(
        origin=origin, origin_time=origin_time,
        valid_time=origin_time + timedelta(hours=lead_hours),
        lead_hours=lead_hours, sic=out.astype(DTYPE), valid_mask=valid,
        transform=tuple(day.grid.transform), crs_epsg=int(day.grid.epsg),
        inputs=inputs, model_path=Path(model_path),
        model_sha256=_sha256(Path(model_path)),
        feature_names=tuple(FEATURE_NAMES), clipped_cells=clipped,
        training_period=_training_period())


# ------------------------------------------------------------- provenance
def provenance(forecast: SICForecast) -> dict:
    """Everything a reader needs to decide whether to trust this raster."""
    try:
        import sklearn
        running = sklearn.__version__
    except Exception:                                            # noqa: BLE001
        running = "unknown"
    return {
        "kind": FORECAST_KIND,
        "is_operational_realtime_forecast": False,
        "forecast_origin": forecast.origin_time.isoformat(),
        "valid_time": forecast.valid_time.isoformat(),
        "lead_hours": forecast.lead_hours,
        "time_assumption": "archive days carry no time of day; the origin is "
                           "00:00 UTC on its date",
        "model_name": MODEL_NAME,
        "model_artifact": _repo_relative(forecast.model_path),
        "model_artifact_sha256": forecast.model_sha256,
        "sklearn_version_at_prediction": running,
        "sklearn_version_that_pickled_the_artifact":
            _ARTIFACT_SKLEARN.get(str(Path(forecast.model_path).resolve())),
        "sklearn_version_matches":
            _ARTIFACT_SKLEARN.get(str(Path(forecast.model_path).resolve()),
                                  running) == running,
        "feature_names": list(forecast.feature_names),
        "training_period_base_dates": forecast.training_period,
        "inputs": [i.to_dict() for i in forecast.inputs],
        "no_inputs_after_origin": True,
        "target_crs": f"EPSG:{forecast.crs_epsg}",
        "source_grid": "observed SIC archive, already on the routing grid",
        "grid_shape": [int(forecast.shape[0]), int(forecast.shape[1])],
        "transform": [float(v) for v in forecast.transform],
        "prediction_policy":
            "one prediction per cell where every feature is finite under "
            "preprocess.load_sic; no interpolation, no gap filling, no "
            "neighbourhood smoothing",
        "clipping": f"model predictions clipped to [{SIC_LO:g}, {SIC_HI:g}]",
        "clipped_cells": int(forecast.clipped_cells),
        "nodata_policy":
            "NaN where the inputs were not all finite. An invalid input cell "
            "stays invalid and is never promoted to a concentration",
        "nodata": "nan",
        "valid_cells": int(forecast.valid_mask.sum()),
        "grid_cells": int(forecast.valid_mask.size),
        "trained_leads": list(TRAINED_LEAD_HOURS),
        "lead_is_this_artifact_s_own_trained_lead": True,
        "shorter_leads_are_not_produced_by_this_model":
            f"this model was fitted for a {forecast.lead_hours // 24}-day step; "
            f"this file is not a 6 h or 12 h sea-ice forecast and must not be "
            f"read as one",
    }


# ------------------------------------------------------------------ write
def write_forecast_raster(forecast: SICForecast, out_dir: Path = OUT_DIR,
                          overwrite: bool = True) -> Path:
    """Write the GeoTIFF plus its sidecar provenance. Deterministic."""
    assert_no_future_inputs(forecast.inputs, forecast.origin_time)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = raster_path(forecast.origin, out_dir, forecast.lead_hours)
    if path.exists() and not overwrite:
        raise ForecastRasterError(f"{_repo_relative(path)} already exists")

    meta = provenance(forecast)
    profile = dict(driver="GTiff", height=forecast.shape[0],
                   width=forecast.shape[1], count=1, dtype=DTYPE,
                   crs=rasterio.crs.CRS.from_epsg(forecast.crs_epsg),
                   transform=Affine(*forecast.transform), nodata=NODATA,
                   compress="deflate")
    tags = {k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v))
            for k, v in meta.items()}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(forecast.sic, dtype=DTYPE), 1)
        dst.update_tags(**tags)
        dst.set_band_description(1, "predicted sea ice concentration (%)")
    (path.with_suffix(".json")).write_text(json.dumps(meta, indent=2) + "\n")
    return path


# --------------------------------------------------------------- validate
def validate_forecast_raster(path: Path, contract=OBSERVED,
                             transform=None) -> dict:
    """Refuse a raster that is not on the routing grid, rather than let a
    runtime reprojection paper over it later."""
    path = Path(path)
    problems: list[str] = []
    with rasterio.open(path) as src:
        epsg = src.crs.to_epsg() if src.crs else None
        got = tuple(float(v) for v in list(src.transform)[:6])
        band = src.read(1)
        nodata = src.nodata
        tags = src.tags()
        dtype = src.dtypes[0]
        height, width = int(src.height), int(src.width)

    if epsg != contract.epsg:
        problems.append(f"CRS EPSG:{epsg} != EPSG:{contract.epsg}")
    if height != contract.height or width != contract.width:
        problems.append(f"shape {(height, width)} != "
                        f"{(contract.height, contract.width)}")
    want = (tuple(float(v) for v in transform) if transform is not None
            else (contract.pixel_size, 0.0, None, 0.0, -contract.pixel_size, None))
    for i, (g, w) in enumerate(zip(got, want)):
        if w is None:
            continue
        if abs(g - w) > 1e-6:
            problems.append(f"transform[{i}] {g} != {w}")
    if dtype != DTYPE:
        problems.append(f"dtype {dtype} != {DTYPE}")
    if nodata is None or not np.isnan(nodata):
        problems.append(f"nodata {nodata!r} is not the declared NaN")

    finite = np.isfinite(band)
    if finite.any():
        lo, hi = float(band[finite].min()), float(band[finite].max())
        if lo < SIC_LO - 1e-6 or hi > SIC_HI + 1e-6:
            problems.append(f"values leave [{SIC_LO:g}, {SIC_HI:g}]: "
                            f"min {lo}, max {hi}")
    else:
        problems.append("no valid cell in the raster")
    for required in ("forecast_origin", "valid_time", "lead_hours",
                     "model_artifact_sha256", "nodata_policy", "kind"):
        if required not in tags:
            problems.append(f"missing provenance tag {required!r}")

    if problems:
        raise ForecastRasterError(
            f"{_repo_relative(path)} is not usable on the routing grid:\n  "
            + "\n  ".join(problems)
            + "\n  -> a mismatched forecast raster is rejected here rather "
              "than reprojected at runtime.")
    return {"path": _repo_relative(path), "epsg": epsg,
            "shape": [height, width], "transform": list(got),
            "valid_cells": int(finite.sum()),
            "value_range": [float(band[finite].min()), float(band[finite].max())],
            "nodata": "nan", "tags": tags}


# ------------------------------------------------------------- validation
def _scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    """persistence.score(), plus bias, plus the training script's own bands."""
    s = score(pred, truth)
    out = {"n": s.n, "mae": s.mae, "rmse": s.rmse, "r2": s.r2,
           "bias": float((pred - truth).mean()) if pred.size else float("nan"),
           "n_miz": s.n_miz, "mae_miz": s.mae_miz}
    for name, mask in band_masks(truth).items():
        out[f"n_{name}"] = int(mask.sum())
        out[f"mae_{name}"] = (float(np.abs(pred[mask] - truth[mask]).mean())
                              if mask.any() else float("nan"))
    return out


def hindcast_validation(forecast: SICForecast, sic_dir: Path = RAW_DIR) -> dict:
    """Score the prediction, and persistence, against the observation that the
    archive already holds for the valid time. Scored ONLY on cells where the
    prediction, persistence and the truth are all valid."""
    truth_day = forecast.valid_time.date()
    truth_path = Path(sic_dir) / f"sic_{truth_day:%Y%m%d}.tif"
    if not truth_path.exists():
        raise ForecastRasterError(
            f"no observation for the valid time {truth_day}: "
            f"{_repo_relative(truth_path)} is not in the archive, so this "
            f"forecast cannot be validated. Nothing is substituted.")

    truth = np.asarray(load_sic(truth_path).sic, dtype="float64")
    #  persistence, the project's published baseline: SIC_hat(t+1) = SIC(t)
    persisted = np.asarray(load_sic(forecast.inputs[0].path).sic, dtype="float64")
    pred = np.asarray(forecast.sic, dtype="float64")

    scored = (np.isfinite(pred) & np.isfinite(persisted) & np.isfinite(truth))
    y = truth[scored]
    return {
        "kind": FORECAST_KIND,
        "is_operational_realtime_forecast": False,
        "forecast_origin": forecast.origin_time.isoformat(),
        "valid_time": forecast.valid_time.isoformat(),
        "lead_hours": forecast.lead_hours,
        "model_label": model_label(forecast.lead_hours),
        "observation": _repo_relative(truth_path),
        "scored_cells": int(scored.sum()),
        "scoring_rule": "cells where the prediction, persistence and the "
                        "observation are all finite; NaN is never counted, "
                        "filled or read as zero",
        "band_definition": {"open": f"y < {MIZ_LO:g}",
                            "miz": f"{MIZ_LO:g} <= y < {MIZ_HI:g}",
                            "ice": f"y >= {MIZ_HI:g}",
                            "note": "bands use the TRUE target y, not sic_t"},
        "metrics": {
            "hgb": _scores(pred[scored], y),
            "persistence": _scores(persisted[scored], y),
        },
        "training_period_base_dates": forecast.training_period,
        "origin_is_inside_the_training_period": _in_training(forecast),
    }


def _in_training(forecast: SICForecast):
    window = (forecast.training_period or {}).get("train")
    if not window:
        return None
    return (window["base_from"] <= forecast.origin.isoformat()
            <= window["base_to"])


# -------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--origin", required=True,
                    help="forecast origin date, YYYY-MM-DD")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--lead-hours", type=int, default=LEAD_HOURS,
                    help=f"trained lead to produce; one of "
                         f"{', '.join(str(h) for h in TRAINED_LEAD_HOURS)}")
    ap.add_argument("--model", default=None,
                    help="override the artifact; defaults to the one trained "
                         "for --lead-hours")
    ap.add_argument("--validate", action="store_true",
                    help="also score it against the observed valid-time field")
    args = ap.parse_args(argv)

    origin = date.fromisoformat(args.origin)
    forecast = predict_sic(origin, lead_hours=args.lead_hours,
                           model_path=Path(args.model) if args.model else None)
    path = write_forecast_raster(forecast, Path(args.out_dir))
    checked = validate_forecast_raster(path, transform=forecast.transform)

    print("=" * 74)
    print(f"SIC FORECAST RASTER   {FORECAST_KIND}")
    print("=" * 74)
    print(f"  origin      {forecast.origin_time.isoformat()}")
    print(f"  valid       {forecast.valid_time.isoformat()}  "
          f"(+{forecast.lead_hours} h)")
    print(f"  raster      {checked['path']}")
    print(f"  grid        EPSG:{checked['epsg']}  {checked['shape']}")
    print(f"  valid cells {checked['valid_cells']:,} of "
          f"{forecast.valid_mask.size:,}")
    print(f"  range       {checked['value_range'][0]:.2f} .. "
          f"{checked['value_range'][1]:.2f}  (clipped cells "
          f"{forecast.clipped_cells:,})")

    if args.validate:
        report = hindcast_validation(forecast)
        out = metrics_path(origin, Path(args.out_dir), forecast.lead_hours)
        out.write_text(json.dumps(report, indent=2) + "\n")
        hgb, per = report["metrics"]["hgb"], report["metrics"]["persistence"]
        print(f"\n  scored on {report['scored_cells']:,} cells against "
              f"{report['observation']}")
        print(f"  {'':<14}{'MAE':>9}{'RMSE':>9}{'bias':>9}{'MAE MIZ':>10}")
        for name, m in ((report["model_label"], hgb), ("persistence", per)):
            print(f"  {name:<14}{m['mae']:>9.3f}{m['rmse']:>9.3f}"
                  f"{m['bias']:>9.3f}{m['mae_miz']:>10.3f}")
        if report["origin_is_inside_the_training_period"]:
            print("  NOTE: this origin is inside the model's TRAINING period; "
                  "these numbers are in-sample.")
        print(f"  metrics     {_repo_relative(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

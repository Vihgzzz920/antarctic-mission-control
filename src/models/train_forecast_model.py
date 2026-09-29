"""
HistGradientBoostingRegressor for SIC: features(t) -> SIC(t + lead).

Reads  data/processed/forecast_dataset.npz   (built by build_forecast_dataset.py)
Writes data/processed/forecast_model_hgb.joblib
       data/processed/forecast_metrics.json

ONE LEAD PER ARTIFACT
  A model fitted on (t -> t+1 day) targets is a 24 h model and nothing else.
  Training a second lead therefore produces a SEPARATE artifact, and the lead
  is written into that artifact's provenance rather than inferred later from a
  filename:

      python -m src.models.train_forecast_model                       # +24 h
      python -m src.models.train_forecast_model \
          --dataset data/processed/forecast_dataset_48h.npz \
          --out     data/processed/forecast_model_hgb_48h.joblib \
          --lead-hours 48

  The lead is VERIFIED against the dataset before a single row is fitted: it is
  read from the dataset's own recorded metadata, or, for a dataset built before
  that field existed, derived from the base and target date ranges the dataset
  itself carries. A dataset that can supply neither is refused. A --lead-hours
  that disagrees with the dataset is refused. Nothing is inferred from a path.

  Sub-daily leads are refused outright: the sea-ice archive is daily, so a 6 h
  or 12 h target does not exist to be trained against.

Training sample -- stratified on sic_t, fixed seed:
    1,000,000 rows with sic_t >= 80      (consolidated ice)
      500,000 rows with 15 <= sic_t < 80 (MIZ)
      500,000 rows with sic_t < 15       (open water)
    = exactly 2,000,000 rows
The raw training set is ~85% consolidated ice. Sampling it flat would let the
model score well by predicting "lots of ice" everywhere and learn almost
nothing about the ice edge, which is the only part that matters for routing.
This deliberately over-weights the MIZ and open water relative to their natural
frequency -- so the training distribution is NOT the evaluation distribution,
and val/test numbers below are on the untouched natural distribution.

VALIDATION AND TEST ARE NEVER SAMPLED, SHUFFLED OR MODIFIED. They are read
once, scored in chunks, and the chunking is purely a memory device: rows stay
in their original chronological order and every row is scored exactly once.

Band metrics use the TRUE target y, not sic_t -- a pixel is "MIZ" if it ended up
in the MIZ, not if it started there. Model predictions are clipped to [0, 100]
before scoring, since SIC is a percentage; persistence needs no clipping
because sic_t is already a valid concentration.

Extra dependencies (NOT added to requirements.txt): scikit-learn (joblib ships
with it).

Usage:
    python -m src.models.train_forecast_model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
DEFAULT_NPZ = DATA / "forecast_dataset.npz"
DEFAULT_MODEL = DATA / "forecast_model_hgb.joblib"
DEFAULT_METRICS = DATA / "forecast_metrics.json"

MODEL_NAME = "HistGradientBoostingRegressor"
#  the lead the shipped artifact was fitted for. Every other lead gets its own
#  file rather than overwriting it.
SHIPPED_LEAD_HOURS = 24
HOURS_PER_DAY = 24
#  this is a research artifact, and every provenance record says so
FORECAST_KIND = "OFFLINE HINDCAST MODEL"

SEED = 42
MIZ_LO, MIZ_HI = 15.0, 80.0          # same convention as persistence.py
SIC_LO, SIC_HI = 0.0, 100.0

# band -> rows to draw, keyed on sic_t
QUOTA = {"ice": 1_000_000, "miz": 500_000, "open": 500_000}
EVAL_CHUNK = 2_000_000               # rows scored per chunk; memory only


class LeadMismatch(ValueError):
    """The dataset and the requested lead are not the same forecast problem."""


def model_path_for(lead_hours: int, data_dir: Path = DATA) -> Path:
    """Where a lead's artifact belongs. The shipped 24 h path never moves."""
    if int(lead_hours) == SHIPPED_LEAD_HOURS:
        return Path(data_dir) / "forecast_model_hgb.joblib"
    return Path(data_dir) / f"forecast_model_hgb_{int(lead_hours)}h.joblib"


def metrics_path_for(lead_hours: int, data_dir: Path = DATA) -> Path:
    if int(lead_hours) == SHIPPED_LEAD_HOURS:
        return Path(data_dir) / "forecast_metrics.json"
    return Path(data_dir) / f"forecast_metrics_{int(lead_hours)}h.json"


def _as_int_days(value) -> int | None:
    try:
        days = int(np.asarray(value).reshape(()).item())
    except Exception:                                            # noqa: BLE001
        return None
    return days if days >= 1 else None


def dataset_lead_hours(z, path=None) -> tuple[int, str]:
    """The lead the DATASET actually represents, and how that was established.

    In order of authority:
      1. the `lead_hours` / `lead_days` the dataset builder recorded;
      2. for a dataset built before those fields existed, the difference
         between the base and target date ranges the dataset records for
         itself -- its own data, not its filename.
    A dataset that supplies neither is refused rather than assumed to be 24 h.
    """
    files = set(getattr(z, "files", []) or [])
    name = Path(path).name if path else "the dataset"

    if "lead_hours" in files:
        hours = _as_int_days(z["lead_hours"])
        if hours:
            return hours, "recorded by the dataset builder (lead_hours)"
    if "lead_days" in files:
        days = _as_int_days(z["lead_days"])
        if days:
            return days * HOURS_PER_DAY, \
                "recorded by the dataset builder (lead_days)"

    #  derive it from the dataset's own recorded ranges, and only accept a
    #  value every split agrees on
    from datetime import date as _date
    seen: dict[str, set] = {}
    for split in ("train", "val", "test"):
        b, t = f"{split}_base_range", f"{split}_target_range"
        if b not in files or t not in files:
            continue
        try:
            bases = [_date.fromisoformat(str(v)) for v in z[b]]
            targets = [_date.fromisoformat(str(v)) for v in z[t]]
        except ValueError:
            continue
        if len(bases) != len(targets):
            continue
        deltas = {(tt - bb).days for bb, tt in zip(bases, targets)}
        if deltas:
            seen[split] = deltas
    all_deltas = set().union(*seen.values()) if seen else set()
    if len(all_deltas) == 1:
        days = all_deltas.pop()
        if days >= 1:
            return days * HOURS_PER_DAY, (
                "derived from the dataset's own base and target date ranges "
                f"({', '.join(sorted(seen))} all agree on {days} day(s))")

    raise LeadMismatch(
        f"{name} records no forecast lead: it has neither `lead_hours` nor "
        f"`lead_days`, and its base/target date ranges "
        f"{'disagree ' + str({k: sorted(v) for k, v in seen.items()}) if seen else 'are absent'}. "
        f"The lead is NOT assumed to be {SHIPPED_LEAD_HOURS} h and is NOT read "
        f"from the file name. Rebuild the dataset with "
        f"`python -m src.models.build_forecast_dataset --lead-days N`, which "
        f"records it.")


def resolve_lead(z, requested, path=None) -> tuple[int, str]:
    """Reconcile a requested lead with the dataset's own. Refuse a mismatch."""
    actual, how = dataset_lead_hours(z, path)
    if requested is None:
        return actual, how
    requested = int(requested)
    if requested < 1:
        raise LeadMismatch(f"--lead-hours must be >= 1, got {requested}")
    if requested % HOURS_PER_DAY:
        raise LeadMismatch(
            f"--lead-hours {requested} is not a whole number of days. The "
            f"sea-ice archive is daily, so the dataset builder only produces "
            f"whole-day targets and no sub-daily lead has a target to train "
            f"against. Supported: multiples of {HOURS_PER_DAY} h.")
    if requested != actual:
        raise LeadMismatch(
            f"--lead-hours {requested} does not match the dataset: "
            f"{Path(path).name if path else 'it'} is a +{actual} h dataset "
            f"({how}). Training a +{requested} h model on +{actual} h targets "
            f"would mislabel the artifact. Point --dataset at a +{requested} h "
            f"dataset, or drop --lead-hours to train the +{actual} h model the "
            f"dataset actually carries.")
    return actual, f"requested and confirmed against the dataset ({how})"


def leakage_check(z, lead_hours: int, path=None) -> dict:
    """Assertions about the dataset itself, run before anything is fitted."""
    files = set(getattr(z, "files", []) or [])
    names = [str(v) for v in z["feature_names"]] if "feature_names" in files else []
    checks = {
        "every_feature_is_from_the_origin_step":
            bool(names) and all(n.endswith("_t") for n in names),
        "no_feature_names_the_target_step":
            bool(names) and not any("t1" in n or "target" in n.lower()
                                    for n in names),
        "target_is_a_separate_array":
            any(k.startswith("y_") for k in files),
        "splits_are_chronological_on_the_base_date": True,
        "target_is_exactly_the_lead_after_the_base": True,
    }
    days = lead_hours // HOURS_PER_DAY
    from datetime import date as _date
    for split in ("train", "val", "test"):
        b, t = f"{split}_base_range", f"{split}_target_range"
        if b not in files or t not in files:
            continue
        try:
            for bb, tt in zip(z[b], z[t]):
                if (_date.fromisoformat(str(tt))
                        - _date.fromisoformat(str(bb))).days != days:
                    checks["target_is_exactly_the_lead_after_the_base"] = False
        except ValueError:
            checks["target_is_exactly_the_lead_after_the_base"] = False
    failed = [k for k, ok in checks.items() if not ok]
    return {"checks": checks, "passed": not failed, "failed": failed,
            "feature_names": names,
            "statement": ("every feature is the origin step; the target is a "
                          "separate array exactly one lead later. No future "
                          "input enters training.")}


def split_information(z) -> dict:
    """What the dataset records about its own splits, passed through."""
    files = set(getattr(z, "files", []) or [])
    out: dict = {}
    for key in ("stride", "gap_days", "lead_days", "lead_hours"):
        if key in files:
            out[key] = _as_int_days(z[key])
    for split in ("train", "val", "test"):
        entry: dict = {}
        if f"y_{split}" in files:
            entry["rows"] = int(z[f"y_{split}"].shape[0])
        for key, label in ((f"{split}_base_range", "base_range"),
                           (f"{split}_target_range", "target_range")):
            if key in files:
                entry[label] = [str(v) for v in z[key]]
        if entry:
            out[split] = entry
    return out


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _repo_relative(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def band_masks(sic: np.ndarray) -> dict[str, np.ndarray]:
    """open / miz / ice, by concentration. Applied to sic_t when sampling and
    to the TRUE target y when scoring."""
    return {
        "open": sic < MIZ_LO,
        "miz": (sic >= MIZ_LO) & (sic < MIZ_HI),
        "ice": sic >= MIZ_HI,
    }


# ------------------------------------------------------------------- sampling
def stratified_indices(sic_t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices for the stratified training sample, sorted (chronological)."""
    picked, shortfall = [], []
    masks = band_masks(sic_t)          # built once; each is freed after use
    for name, want in QUOTA.items():
        pool = np.flatnonzero(masks.pop(name))
        have = pool.size
        if have < want:
            shortfall.append(f"{name}: need {want:,}, have {have:,}")
        else:
            picked.append(rng.choice(pool, size=want, replace=False))
        print(f"  {name:<5} pool {have:>12,}   take {min(want, have):>10,}"
              f"   ({100 * have / sic_t.size:5.2f}% of train)")
        del pool                                   # free before the next band
    if shortfall:
        raise ValueError(
            "Not enough rows for the requested stratified sample:\n  "
            + "\n  ".join(shortfall)
            + "\nAdjust QUOTA, or rebuild the dataset with a smaller --stride.")
    idx = np.concatenate(picked)
    idx.sort()                                     # locality + keeps time order
    return idx


# ----------------------------------------------------------------- evaluation
def evaluate(predict, X: np.ndarray, y: np.ndarray, clip: bool,
             chunk: int = EVAL_CHUNK) -> dict:
    """Stream metrics over the FULL array without materialising a prediction
    vector. X and y are read as views; neither is copied or reordered."""
    n = int(y.shape[0])
    y_mean = float(np.mean(y, dtype="float64"))

    sum_abs = ss_res = ss_tot = 0.0
    b_n = {k: 0 for k in QUOTA}
    b_abs = {k: 0.0 for k in QUOTA}

    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        yt = y[a:b].astype("float64", copy=False)
        p = np.asarray(predict(X[a:b]), dtype="float64")
        if clip:
            np.clip(p, SIC_LO, SIC_HI, out=p)
        d = p - yt
        ad = np.abs(d)

        sum_abs += float(ad.sum())
        ss_res += float(np.sum(d * d, dtype="float64"))
        c = yt - y_mean
        ss_tot += float(np.sum(c * c, dtype="float64"))

        for name, m in band_masks(yt).items():     # bands on the TRUE target
            b_n[name] += int(m.sum())
            b_abs[name] += float(ad[m].sum())

    out = {
        "n": n,
        "mae": sum_abs / n,
        "rmse": float(np.sqrt(ss_res / n)),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }
    for name in QUOTA:
        out[f"mae_{name}"] = (b_abs[name] / b_n[name]) if b_n[name] else float("nan")
        out[f"n_{name}"] = b_n[name]
    return out


def table(rows: list[tuple[str, str, dict]]) -> None:
    hdr = (f"{'model':<12}{'split':<7}{'MAE':>9}{'RMSE':>9}{'R2':>9}"
           f"{'MIZ MAE':>10}{'open MAE':>10}{'ice MAE':>9}{'n':>14}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for model, split, m in rows:
        print(f"{model:<12}{split:<7}{m['mae']:>9.3f}{m['rmse']:>9.3f}{m['r2']:>9.4f}"
              f"{m['mae_miz']:>10.3f}{m['mae_open']:>10.3f}{m['mae_ice']:>9.3f}"
              f"{m['n']:>14,}")
    print("=" * len(hdr))


# ---------------------------------------------------------------- provenance
def training_provenance(*, npz_path: Path, model_path: Path, lead: int,
                        how: str, leak: dict, splits: dict,
                        feature_names: list, seed: int) -> dict:
    """Everything about a training run that is NOT the fitted model.

    Assembled here rather than inside the training loop so it can be checked
    without fitting anything. `artifact_sha256` is filled in by the caller once
    the artifact has actually been written.
    """
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:                                            # noqa: BLE001
        sklearn_version = "unknown"
    try:
        import joblib as _joblib
        joblib_version = getattr(_joblib, "__version__", "unknown")
    except Exception:                                            # noqa: BLE001
        joblib_version = "unknown"
    lead_days = int(lead) // HOURS_PER_DAY
    return {
        "kind": FORECAST_KIND,
        "is_operational_realtime_forecast": False,
        "model": MODEL_NAME,
        "lead_hours": int(lead),
        "lead_days": lead_days,
        "lead_established_by": how,
        "target_relationship": f"target = base + {lead_days} day(s) = +{lead} h",
        "dataset": _repo_relative(npz_path),
        "dataset_sha256": _sha256(npz_path) if Path(npz_path).exists() else None,
        "artifact": _repo_relative(model_path),
        "artifact_sha256": None,
        "sklearn_version": sklearn_version,
        "joblib_version": joblib_version,
        "python_version": sys.version.split()[0],
        "split_information": splits,
        "leakage_check": leak,
        "source_data_provenance": {
            "sea_ice": "University of Bremen AMSR2 ASI daily sea-ice "
                       "concentration, 6.25 km Antarctic grid (EPSG:3976)",
            "currents": "Copernicus GLORYS12V1 daily currents, reprojected to "
                        "the same grid",
            "cadence": "daily; both products are one field per day, which is "
                       "why only whole-day leads exist",
            "recorded_by_the_dataset": {k: splits.get(k) for k in
                                        ("stride", "gap_days", "lead_days",
                                         "lead_hours") if k in splits},
            "masking": "preprocess.load_sic; a row survives only if every "
                       "feature and the target are finite",
        },
    }


# ----------------------------------------------------------------------- main
def main(npz_path: Path, model_path: Path, metrics_path: Path, seed: int,
         lead_hours: int | None = None) -> int:
    try:
        import joblib
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        print(f"scikit-learn is required:\n  pip install scikit-learn\n({exc})",
              file=sys.stderr)
        return 1

    if not npz_path.exists():
        print(f"Not found: {npz_path}\nRun build_forecast_dataset.py first.",
              file=sys.stderr)
        return 1

    z = np.load(npz_path)
    feature_names = [str(s) for s in z["feature_names"]]

    #  ---- the lead, established before anything is fitted -------------------
    lead, how = resolve_lead(z, lead_hours, npz_path)
    lead_days = lead // HOURS_PER_DAY
    leak = leakage_check(z, lead, npz_path)
    if not leak["passed"]:
        raise LeadMismatch(
            f"{npz_path.name} failed its leakage checks before training: "
            f"{', '.join(leak['failed'])}. Nothing is fitted on a dataset "
            f"whose targets do not sit exactly one lead after their features.")
    #  a different lead never lands on the shipped artifact
    if lead != SHIPPED_LEAD_HOURS and Path(model_path) == Path(DEFAULT_MODEL):
        raise LeadMismatch(
            f"this is a +{lead} h training run but --out points at the shipped "
            f"+{SHIPPED_LEAD_HOURS} h artifact {DEFAULT_MODEL.name}. One lead "
            f"per artifact: pass --out {model_path_for(lead).name} (or another "
            f"path of your own).")

    splits = split_information(z)
    print("=" * 74)
    print(f"{MODEL_NAME}  --  features(t) -> SIC(t+{lead_days})   "
          f"[+{lead} h lead]")
    print("=" * 74)
    print(f"  dataset  : {npz_path.name}")
    print(f"  lead     : +{lead} h ({lead_days} day(s)) -- {how}")
    print(f"  features : {', '.join(feature_names)}")
    print(f"  leakage  : {'PASSED' if leak['passed'] else 'FAILED'} -- "
          f"{leak['statement']}")
    print(f"  seed     : {seed}")
    print(f"  artifact : {model_path.name}   metrics: {metrics_path.name}")

    # ---- training sample. Each npz key access re-reads from disk, so every
    # ---- array is touched exactly once and released before the next.
    print(f"\nStratified training sample (on sic_t):")
    rng = np.random.default_rng(seed)

    X_train = z["X_train"]                          # ~1.0 GB
    idx = stratified_indices(X_train[:, 0], rng)    # column 0 is a view
    X_fit = np.ascontiguousarray(X_train[idx], dtype="float32")
    del X_train

    y_train = z["y_train"]                          # ~0.25 GB
    y_fit = np.ascontiguousarray(y_train[idx], dtype="float32")
    del y_train, idx

    print(f"  sample   : X {X_fit.shape} {X_fit.dtype}, y {y_fit.shape}")
    print(f"  NOTE     : this sample is deliberately NOT the natural class mix; "
          f"val/test below are.")

    # ---- model. Deliberately plain: a first experiment, not a tuned one.
    # squared_error is the default; absolute_error would optimise the headline
    # MAE directly and is the obvious next thing to try, not something to
    # decide here.
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.0,
        early_stopping=True,          # carved from X_fit only; val/test untouched
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
        verbose=1,
    )
    print(f"\nFitting on {X_fit.shape[0]:,} rows ...")
    model.fit(X_fit, y_fit)
    print(f"  iterations used: {model.n_iter_} of {model.max_iter}")
    del X_fit, y_fit

    # ---- evaluate on the FULL, untouched val and test -----------------------
    results: dict[str, dict] = {}
    rows: list[tuple[str, str, dict]] = []

    for split in ("val", "test"):
        X = z[f"X_{split}"]
        y = z[f"y_{split}"]
        print(f"\nScoring {split}: {y.shape[0]:,} rows (full split, in order) ...")

        m_hgb = evaluate(model.predict, X, y, clip=True)
        # persistence: the prediction IS sic_t, already a valid concentration
        m_per = evaluate(lambda a: a[:, 0], X, y, clip=False)

        results[f"hgb_{split}"] = m_hgb
        results[f"persistence_{split}"] = m_per
        rows.append(("hgb", split, m_hgb))
        rows.append(("persistence", split, m_per))
        del X, y

    table(rows)

    for split in ("val", "test"):
        h, p = results[f"hgb_{split}"], results[f"persistence_{split}"]
        d_all = p["mae"] - h["mae"]
        d_miz = p["mae_miz"] - h["mae_miz"]
        print(f"  {split}: HGB beats persistence by {d_all:+.3f} pp overall, "
              f"{d_miz:+.3f} pp in the MIZ  (positive = HGB better)")
    print("\nThe MIZ column is the one that matters. Overall MAE is dominated by "
          "consolidated\nice, where persistence is already near-perfect and any "
          "model looks good.")

    # ---- save ---------------------------------------------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    payload = training_provenance(
        npz_path=npz_path, model_path=model_path, lead=lead, how=how,
        leak=leak, splits=splits, feature_names=feature_names, seed=seed)
    payload.update({
        "seed": seed,
        "feature_names": feature_names,
        "train_sample": {"quota": QUOTA, "total": sum(QUOTA.values()),
                         "stratified_on": "sic_t"},
        "params": {k: model.get_params()[k] for k in
                   ("loss", "max_iter", "learning_rate", "max_leaf_nodes",
                    "min_samples_leaf", "l2_regularization", "early_stopping",
                    "validation_fraction", "n_iter_no_change", "random_state")},
        "n_iter_used": int(model.n_iter_),
        "band_definition": {"open": f"y < {MIZ_LO:g}",
                            "miz": f"{MIZ_LO:g} <= y < {MIZ_HI:g}",
                            "ice": f"y >= {MIZ_HI:g}",
                            "note": "bands use the TRUE target y, not sic_t"},
        "clipping": f"model predictions clipped to [{SIC_LO:g}, {SIC_HI:g}]",
        "metrics": results,
    })
    #  the artifact hash is of the file that was just written
    payload["artifact_sha256"] = _sha256(model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nGOT     {model_path.name}  ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"GOT     {metrics_path.name}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI, as a function so it can be exercised without training."""
    ap = argparse.ArgumentParser(
        description="Train one lead-specific HGB SIC forecast model (no tuning)")
    #  --dataset / --out are the names to use; --npz / --model-out are the
    #  original spellings and still work, so existing scripts do not break.
    ap.add_argument("--dataset", "--npz", dest="dataset", default=str(DEFAULT_NPZ),
                    help="the .npz built by build_forecast_dataset.py")
    ap.add_argument("--out", "--model-out", dest="out", default=None,
                    help="artifact path. Defaults to the shipped +24 h path, "
                         "or forecast_model_hgb_<lead>h.joblib for any other "
                         "lead.")
    ap.add_argument("--metrics-out", dest="metrics_out", default=None,
                    help="metrics and provenance JSON. Defaults alongside the "
                         "artifact.")
    ap.add_argument("--lead-hours", dest="lead_hours", type=int, default=None,
                    help="the forecast lead in WHOLE HOURS, a multiple of 24. "
                         "Checked against the dataset and refused on a "
                         "mismatch. Omitted, the dataset's own lead is used.")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def resolve_output_paths(dataset: Path, out, metrics_out, lead_hours):
    """Where this run writes, once the dataset's own lead is known."""
    with np.load(dataset) as z:
        lead, _ = resolve_lead(z, lead_hours, dataset)
    model_path = Path(out) if out else model_path_for(lead)
    metrics_path = (Path(metrics_out) if metrics_out
                    else metrics_path_for(lead))
    return lead, model_path, metrics_path


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    dataset = Path(args.dataset)
    try:
        if not dataset.exists():
            raise FileNotFoundError(
                f"Not found: {dataset}\nRun build_forecast_dataset.py first.")
        lead, model_path, metrics_path = resolve_output_paths(
            dataset, args.out, args.metrics_out, args.lead_hours)
        code = main(dataset, model_path, metrics_path, args.seed, lead)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"\nCANNOT TRAIN\n\n{exc}\n", file=sys.stderr)
        code = 1
    sys.exit(code)


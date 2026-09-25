"""
First HistGradientBoostingRegressor experiment: features(t) -> SIC(t+1).

Reads  data/processed/forecast_dataset.npz   (built by build_forecast_dataset.py)
Writes data/processed/forecast_model_hgb.joblib
       data/processed/forecast_metrics.json

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

SEED = 42
MIZ_LO, MIZ_HI = 15.0, 80.0          # same convention as persistence.py
SIC_LO, SIC_HI = 0.0, 100.0

# band -> rows to draw, keyed on sic_t
QUOTA = {"ice": 1_000_000, "miz": 500_000, "open": 500_000}
EVAL_CHUNK = 2_000_000               # rows scored per chunk; memory only


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


# ----------------------------------------------------------------------- main
def main(npz_path: Path, model_path: Path, metrics_path: Path, seed: int) -> int:
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

    print("=" * 74)
    print("HistGradientBoostingRegressor  --  features(t) -> SIC(t+1)")
    print("=" * 74)
    print(f"  dataset  : {npz_path.name}")
    print(f"  features : {', '.join(feature_names)}")
    print(f"  seed     : {seed}")

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
    payload = {
        "model": "HistGradientBoostingRegressor",
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
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nGOT     {model_path.name}  ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"GOT     {metrics_path.name}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train the first HGB SIC forecast model (no tuning)")
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--model-out", default=str(DEFAULT_MODEL))
    ap.add_argument("--metrics-out", default=str(DEFAULT_METRICS))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    try:
        code = main(Path(args.npz), Path(args.model_out),
                    Path(args.metrics_out), args.seed)
    except (ValueError, KeyError) as exc:
        print(f"\nCANNOT TRAIN\n\n{exc}\n", file=sys.stderr)
        code = 1
    sys.exit(code)


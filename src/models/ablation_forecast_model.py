"""
Ablation: do ocean-current features add predictive value beyond SIC alone?

Three approaches, one question:
    1. persistence   prediction = sic_t                      (no model)
    2. sic_only      HGB on [sic_t]
    3. sic_currents  HGB on [sic_t, current_u_t, current_v_t, current_speed_t]

The number that answers the question is NOT "HGB beats persistence" -- a model
given sic_t can beat persistence just by learning the mean day-to-day decay.
It is the sic_only -> sic_currents step. That isolates what the currents
contribute once SIC has already been used.

Everything that could bias the comparison is held fixed and imported from
train_forecast_model.py rather than restated: the same 2,000,000-row stratified
sample (seed 42, so both models see the IDENTICAL rows), the same band
definitions, the same streaming evaluator, the same clipping. Both models also
share random_state, so scikit-learn carves the same internal early-stopping
split from the training sample -- the only difference between runs 2 and 3 is
the feature columns.

VALIDATION AND TEST ARE NEVER SAMPLED, SHUFFLED OR MODIFIED. They are scored in
chunks purely as a memory device; rows keep their chronological order and each
is scored exactly once.

Writes data/processed/ablation_metrics.json. Trains nothing that is saved --
this is an experiment, not a replacement for train_forecast_model.py.

Usage:
    python -m src.models.ablation_forecast_model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.models.train_forecast_model import (
    DATA, DEFAULT_NPZ, DEFAULT_METRICS, EVAL_CHUNK, MIZ_HI, MIZ_LO, QUOTA,
    SEED, band_masks, evaluate, stratified_indices,
)

DEFAULT_OUT = DATA / "ablation_metrics.json"

# Mirrors train_forecast_model.py exactly (verbose differs: two fits, less noise).
# Cross-checked at runtime against forecast_metrics.json so the two cannot drift
# apart silently -- an ablation run with different hyperparameters than the model
# it is meant to explain would be worse than no ablation at all.
MODEL_PARAMS = dict(
    loss="squared_error",
    max_iter=200,
    learning_rate=0.1,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

# name -> column indices into X. Order matches FEATURE_NAMES in the dataset.
VARIANTS = {
    "sic_only": [0],
    "sic_currents": [0, 1, 2, 3],
}

METRIC_KEYS = ("mae", "rmse", "r2", "mae_miz", "mae_open", "mae_ice")


def check_params_match(metrics_path: Path, seed: int) -> None:
    """Warn loudly if train_forecast_model.py has been retuned since."""
    if not metrics_path.exists():
        print(f"  (no {metrics_path.name}; cannot cross-check hyperparameters)")
        return
    try:
        saved = json.loads(metrics_path.read_text()).get("params", {})
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARNING: could not read {metrics_path.name}: {exc}", file=sys.stderr)
        return
    mine = {**MODEL_PARAMS, "random_state": seed}
    diff = [f"{k}: ablation {mine[k]!r} vs saved {saved[k]!r}"
            for k in mine if k in saved and saved[k] != mine[k]]
    if diff:
        print("  WARNING: hyperparameters differ from the trained model:\n    "
              + "\n    ".join(diff)
              + "\n    This ablation no longer explains forecast_model_hgb.joblib.",
              file=sys.stderr)
    else:
        print(f"  hyperparameters match {metrics_path.name}")


def improvement(baseline: dict, candidate: dict, key: str = "mae_miz") -> float:
    """Positive means the candidate is BETTER, i.e. lower error."""
    return baseline[key] - candidate[key]


def table(rows: list[tuple[str, str, dict]]) -> None:
    hdr = (f"{'model':<14}{'split':<7}{'MAE':>9}{'RMSE':>9}{'R2':>9}"
           f"{'MIZ MAE':>10}{'open MAE':>10}{'ice MAE':>9}{'n':>14}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for model, split, m in rows:
        print(f"{model:<14}{split:<7}{m['mae']:>9.3f}{m['rmse']:>9.3f}{m['r2']:>9.4f}"
              f"{m['mae_miz']:>10.3f}{m['mae_open']:>10.3f}{m['mae_ice']:>9.3f}"
              f"{m['n']:>14,}")
    print("=" * len(hdr))


def main(npz_path: Path, out_path: Path, seed: int) -> int:
    try:
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

    print("=" * 76)
    print("ABLATION  --  does adding ocean currents beat SIC alone?")
    print("=" * 76)
    print(f"  dataset  : {npz_path.name}")
    print(f"  features : {', '.join(feature_names)}")
    print(f"  seed     : {seed}  (identical training rows for both variants)")
    check_params_match(DEFAULT_METRICS, seed)

    # ---- the one shared training sample --------------------------------------
    print(f"\nStratified training sample (on sic_t), same as train_forecast_model.py:")
    X_train = z["X_train"]
    idx = stratified_indices(X_train[:, 0], np.random.default_rng(seed))
    X_fit = np.ascontiguousarray(X_train[idx], dtype="float32")
    del X_train
    y_train = z["y_train"]
    y_fit = np.ascontiguousarray(y_train[idx], dtype="float32")
    del y_train, idx
    print(f"  sample   : X {X_fit.shape}, y {y_fit.shape}  "
          f"(total {sum(QUOTA.values()):,} rows)")

    # ---- fit both variants on exactly those rows -----------------------------
    models = {}
    for name, cols in VARIANTS.items():
        print(f"\nFitting {name}: columns {cols} "
              f"-> {[feature_names[c] for c in cols]}")
        m = HistGradientBoostingRegressor(random_state=seed, verbose=0, **MODEL_PARAMS)
        m.fit(np.ascontiguousarray(X_fit[:, cols]), y_fit)
        print(f"  iterations used: {m.n_iter_} of {MODEL_PARAMS['max_iter']}")
        models[name] = m
    del X_fit, y_fit

    # ---- score on the FULL, untouched val and test ---------------------------
    results: dict[str, dict] = {}
    rows: list[tuple[str, str, dict]] = []

    for split in ("val", "test"):
        X, y = z[f"X_{split}"], z[f"y_{split}"]
        print(f"\nScoring {split}: {y.shape[0]:,} rows (full split, in order) ...")

        # persistence: sic_t IS the prediction, already a valid concentration
        m_per = evaluate(lambda a: a[:, 0], X, y, clip=False, chunk=EVAL_CHUNK)
        results[f"persistence_{split}"] = m_per
        rows.append(("persistence", split, m_per))

        for name, cols in VARIANTS.items():
            mdl = models[name]
            pred = (lambda mdl=mdl, cols=cols:
                    lambda a: mdl.predict(np.ascontiguousarray(a[:, cols])))()
            r = evaluate(pred, X, y, clip=True, chunk=EVAL_CHUNK)
            results[f"{name}_{split}"] = r
            rows.append((name, split, r))
        del X, y

    table(rows)

    # ---- the three deltas the experiment exists to produce -------------------
    deltas: dict[str, dict] = {}
    print(f"\nMIZ MAE improvement (percentage points; POSITIVE = lower error = better)")
    print(f"  MIZ is {MIZ_LO:g} <= TRUE SIC < {MIZ_HI:g}\n")
    print(f"{'comparison':<42}{'val':>10}{'test':>10}")
    print("-" * 62)
    pairs = [
        ("sic_only_vs_persistence", "SIC-only HGB  vs  persistence", "persistence", "sic_only"),
        ("sic_currents_vs_sic_only", "SIC+currents  vs  SIC-only    <- THE ANSWER",
         "sic_only", "sic_currents"),
        ("sic_currents_vs_persistence", "SIC+currents  vs  persistence",
         "persistence", "sic_currents"),
    ]
    for key, label, base, cand in pairs:
        d = {s: improvement(results[f"{base}_{s}"], results[f"{cand}_{s}"])
             for s in ("val", "test")}
        deltas[key] = d
        print(f"{label:<42}{d['val']:>+10.3f}{d['test']:>+10.3f}")
    print("-" * 62)

    marg = deltas["sic_currents_vs_sic_only"]
    print(f"\nThe currents contribute {marg['val']:+.3f} pp on validation and "
          f"{marg['test']:+.3f} pp on test,\nover and above what sic_t alone already "
          f"provides. A small or negative number\nhere is a real finding, not a "
          f"failure: it would mean the current features are\nnot earning their place "
          f"in the model and the pipeline complexity they cost.")
    print(f"\nOne seed and one hyperparameter setting cannot establish statistical\n"
          f"significance. Treat these as a single observation, not a proven effect.")

    # ---- save ----------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "ablation: SIC-only vs SIC+currents vs persistence",
        "seed": seed,
        "feature_names": feature_names,
        "variants": {k: [feature_names[c] for c in v] for k, v in VARIANTS.items()},
        "train_sample": {"quota": QUOTA, "total": sum(QUOTA.values()),
                         "stratified_on": "sic_t",
                         "note": "identical rows for both variants"},
        "params": {**MODEL_PARAMS, "random_state": seed},
        "n_iter_used": {k: int(m.n_iter_) for k, m in models.items()},
        "band_definition": {"open": f"y < {MIZ_LO:g}",
                            "miz": f"{MIZ_LO:g} <= y < {MIZ_HI:g}",
                            "ice": f"y >= {MIZ_HI:g}",
                            "note": "bands use the TRUE target y, not sic_t"},
        "clipping": "HGB predictions clipped to [0, 100]; persistence not clipped",
        "metrics": results,
        "miz_mae_improvement": {
            "convention": "positive = lower MAE = better", **deltas},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nGOT     {out_path.name}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ablate ocean-current features against SIC alone (no tuning)")
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    try:
        code = main(Path(args.npz), Path(args.out), args.seed)
    except (ValueError, KeyError) as exc:
        print(f"\nCANNOT RUN THE ABLATION\n\n{exc}\n", file=sys.stderr)
        code = 1
    sys.exit(code)

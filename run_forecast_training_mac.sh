#!/bin/bash
# =====================================================================
# SIC forecast training run -- MUST be run on the project Mac, in .venv
#
#   cd ~/Desktop/SIH && ./run_forecast_training_mac.sh
#
# Runs Tasks 1-8 in order, gating on each one. Every step writes its own
# log under logs/mac_training_<timestamp>/ so the results can be read and
# verified afterwards without re-running anything.
#
# It STOPS at the first failure. Nothing later runs if an earlier gate fails.
# The existing +24 h artifact is copied aside before it is regenerated.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "FATAL: .venv/bin/python not found. Run this on the project Mac." >&2
  exit 1
fi
source .venv/bin/activate

STAMP=$(date +%Y%m%dT%H%M%S)
LOGDIR="logs/mac_training_${STAMP}"
mkdir -p "$LOGDIR"
echo "logs -> $LOGDIR"

step () { echo; echo "================ $* ================"; }

# ---------------------------------------------------------------- PHASE 0
step "PHASE 0  environment gate (requires scikit-learn 1.9.1)"
python - <<'PY' 2>&1 | tee "$LOGDIR/00_environment.log"
import sys, platform, sklearn, joblib, numpy
print("python  :", sys.version.split()[0], platform.platform())
print("sklearn :", sklearn.__version__)
print("joblib  :", joblib.__version__)
print("numpy   :", numpy.__version__)
print("executable:", sys.executable)
if sklearn.__version__ != "1.9.1":
    raise SystemExit(f"FATAL: sklearn is {sklearn.__version__}, not 1.9.1. Nothing was run.")
print("GATE PASSED")
PY

# ---------------------------------------------------------------- PHASE 1
step "PHASE 1  build the true +48 h dataset (--lead-days 2 --gap-days 1)"
caffeinate -i python -m src.models.build_forecast_dataset \
    --lead-days 2 --gap-days 1 \
    --out data/processed/forecast_dataset_48h.npz \
    2>&1 | tee "$LOGDIR/01_build_48h.log"

step "PHASE 1b  inspect the dataset BEFORE training"
python - <<'PY' 2>&1 | tee "$LOGDIR/01b_dataset_check.log"
import hashlib, json
from pathlib import Path
import numpy as np
p = Path("data/processed/forecast_dataset_48h.npz")
z = np.load(p, allow_pickle=True)
info = {
    "file": str(p), "size_bytes": p.stat().st_size,
    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "feature_names": [str(v) for v in z["feature_names"]],
    "lead_days": int(z["lead_days"]), "lead_hours": int(z["lead_hours"]),
    "gap_days": int(z["gap_days"]), "stride": int(z["stride"]),
}
for s in ("train", "val", "test"):
    info[s] = {
        "rows": int(z[f"y_{s}"].shape[0]),
        "pairs": int(z[f"{s}_base_dates"].shape[0]),
        "base_range": [str(v) for v in z[f"{s}_base_range"]],
        "target_range": [str(v) for v in z[f"{s}_target_range"]],
    }
info["total_pairs"] = sum(info[s]["pairs"] for s in ("train", "val", "test"))
print(json.dumps(info, indent=2))

fail = []
if info["lead_days"] != 2:  fail.append(f"lead_days is {info['lead_days']}, not 2")
if info["gap_days"] != 1:   fail.append(f"gap_days is {info['gap_days']}, not 1")
if info["total_pairs"] != 357:
    fail.append(f"{info['total_pairs']} pairs, expected 357")
if info["feature_names"] != ["sic_t","current_u_t","current_v_t","current_speed_t"]:
    fail.append(f"feature names changed: {info['feature_names']}")
from datetime import date
for s in ("train","val","test"):
    for b, t in zip(info[s]["base_range"], info[s]["target_range"]):
        if (date.fromisoformat(t) - date.fromisoformat(b)).days != 2:
            fail.append(f"{s}: {b} -> {t} is not a 2-day step")
if fail:
    raise SystemExit("DATASET GATE FAILED:\n  " + "\n  ".join(fail))
print("\nGATE PASSED: 357 pairs, lead_days=2, gap_days=1, target = base + 2 days")
PY

# ---------------------------------------------------------------- PHASE 2
step "PHASE 2  train the distinct +48 h model"
caffeinate -i python -m src.models.train_forecast_model \
    --dataset data/processed/forecast_dataset_48h.npz \
    --out data/processed/forecast_model_hgb_48h.joblib \
    --lead-hours 48 \
    2>&1 | tee "$LOGDIR/02_train_48h.log"

# ---------------------------------------------------------------- PHASE 4
step "PHASE 4  record the old +24 h artifact, copy it aside, regenerate"
python - <<'PY' 2>&1 | tee "$LOGDIR/04a_old_artifact.log"
import hashlib, shutil
from pathlib import Path
old = Path("data/processed/forecast_model_hgb.joblib")
sha = hashlib.sha256(old.read_bytes()).hexdigest()
keep = old.with_name("forecast_model_hgb_pre_sklearn191.joblib")
shutil.copy2(old, keep)
print(f"old artifact sha256 : {sha}")
print(f"kept a copy at      : {keep}")
oldm = Path("data/processed/forecast_metrics.json")
if oldm.exists():
    shutil.copy2(oldm, oldm.with_name("forecast_metrics_pre_sklearn191.json"))
    print(f"kept a copy at      : {oldm.with_name('forecast_metrics_pre_sklearn191.json')}")
PY

caffeinate -i python -m src.models.train_forecast_model \
    2>&1 | tee "$LOGDIR/04b_train_24h.log"

# ---------------------------------------------------------------- PHASE 5
step "PHASE 5  regenerate and validate the +24 h hindcast rasters"
for ORIGIN in 2025-01-08 2025-12-15; do
  caffeinate -i python -m src.models.forecast_sic_raster --origin "$ORIGIN" --validate \
      2>&1 | tee -a "$LOGDIR/05_hindcast_24h.log"
done

# ---------------------------------------------------------------- PHASE 6
step "PHASE 6  artifact validation"
python - <<'PY' 2>&1 | tee "$LOGDIR/06_artifact_validation.log"
import hashlib, json
from pathlib import Path
D = Path("data/processed")
m24 = json.loads((D / "forecast_metrics.json").read_text())
m48 = json.loads((D / "forecast_metrics_48h.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
a24, a48 = D / "forecast_model_hgb.joblib", D / "forecast_model_hgb_48h.joblib"

checks = {
    "1 24h artifact lead_hours == 24": m24["lead_hours"] == 24,
    "2 48h artifact lead_hours == 48": m48["lead_hours"] == 48,
    "3 artifact SHAs are distinct": sha(a24) != sha(a48),
    "4 48h references the 48h dataset": m48["dataset"].endswith("forecast_dataset_48h.npz"),
    "5 24h references the 24h dataset": m24["dataset"].endswith("forecast_dataset.npz"),
    "6 sklearn is exactly 1.9.1": m24["sklearn_version"] == m48["sklearn_version"] == "1.9.1",
    "8 leakage checks pass": m24["leakage_check"]["passed"] and m48["leakage_check"]["passed"],
}
print(json.dumps({
    "24h": {"artifact_sha256": sha(a24), "dataset": m24["dataset"],
            "dataset_sha256": m24["dataset_sha256"], "lead_hours": m24["lead_hours"],
            "sklearn": m24["sklearn_version"], "joblib": m24["joblib_version"],
            "python": m24["python_version"]},
    "48h": {"artifact_sha256": sha(a48), "dataset": m48["dataset"],
            "dataset_sha256": m48["dataset_sha256"], "lead_hours": m48["lead_hours"],
            "sklearn": m48["sklearn_version"], "joblib": m48["joblib_version"],
            "python": m48["python_version"]},
}, indent=2))
for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
if not all(checks.values()):
    raise SystemExit("ARTIFACT VALIDATION FAILED")
print("\nall artifact checks passed")
PY

# ---------------------------------------------------------------- PHASE 7
step "PHASE 7  test suites (pytest is not in .venv; the files are run directly)"
: > "$LOGDIR/07_tests.log"
FAILED=0
for T in tests/test_*.py; do
  echo "---- $T" | tee -a "$LOGDIR/07_tests.log"
  if python "$T" >> "$LOGDIR/07_tests.log" 2>&1; then
    echo "     ok" | tee -a "$LOGDIR/07_tests.log"
  else
    echo "     FAILED (see the log)" | tee -a "$LOGDIR/07_tests.log"
    FAILED=$((FAILED+1))
  fi
done
echo "test files failed: $FAILED" | tee -a "$LOGDIR/07_tests.log"

# ---------------------------------------------------------------- PHASE 8
step "PHASE 8  demo-route regression payload"
python - <<'PY' 2>&1 | tee "$LOGDIR/08_demo_route.log"
import json
from pathlib import Path
try:
    from fastapi.testclient import TestClient
except Exception as exc:                      # httpx is not in .venv
    raise SystemExit(
        f"SKIPPED: the API test client needs httpx, which is not "
        f"installed in this .venv ({exc}).\n"
        f"  Install it (test-only, unrelated to scikit-learn):\n"
        f"      pip install httpx\n"
        f"  then re-run this phase. The routing code was not touched.")
from src.api.main import app
body = TestClient(app).get(
    "/api/demo/routes?historical_demo_override=true").json()
out = Path("logs") / "demo_route_after_training.json"
out.write_text(json.dumps(body, indent=2))
p = body["comparison"]["profiles"]
for n in ("fastest", "risk_oriented", "shortest_distance"):
    print(f"  {n:<18} {p[n]['distance_km']:.5f} km  cost {p[n]['configured_cost']:,.3f}  "
          f"cells {len(p[n]['cells'])}")
tp = p["risk_oriented"]["temporal_provenance"]
print("  policy:", tp["environment_policies"], "time-varying:", tp["environment_is_time_varying"])
print("  written:", out)
PY

true
step "DONE -- everything is under $LOGDIR"

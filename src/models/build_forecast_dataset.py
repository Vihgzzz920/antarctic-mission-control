"""
Build the N-day-ahead SIC forecasting dataset: day t predicts SIC on day t+N.

The default is N = 1 (a 24 h lead), which is what the shipped model was
trained on and what --lead-days leaves untouched when it is not given. Longer
leads are supported because the archive genuinely contains their targets --
361 usable (t, t+2) pairs, for instance -- and a lead must have a real target
before any model may claim it. Sub-daily leads are NOT supported at any
setting: both sea-ice sources in this project are daily composites, so there is
no 6 h or 12 h target to build, and --lead-days therefore takes whole days.

Rows are (pixel, day) pairs. Per row:
    features  sic_t, current_u_t, current_v_t, current_speed_t
    target    sic_t1

Inputs (already on the identical EPSG:3976 6.25 km grid, 1328 x 1264):
    data/raw/sic_YYYYMMDD.tif                    via preprocess.load_sic
    data/processed/currents/currents_YYYYMMDD.tif  band 1 = u_x, band 2 = u_y

Rules, matching the rest of the pipeline:
  * only TRUE consecutive days are used: (target - base).days == 1. A missing
    archive day such as 2025-06-08 silently removes two pairs (07->08 and
    08->09) rather than producing a mislabelled 2-day forecast.
  * SIC masking is delegated to preprocess.load_sic, so 0.0 stays as VALID open
    water and only >100 / <0 / non-finite are dropped. Nothing is interpolated.
  * a row survives only if ALL FOUR features and the target are finite. Because
    currents exist only where GLORYS had data, this intersection is smaller than
    either product's own valid area.

Chronological split on the BASE date t, no shuffling:
    train  t <  2025-10-01
    val    2025-10-01 <= t <= 2025-11-30
    test   2025-12-01 <= t <= 2025-12-31
Base dates outside every window are dropped and reported.

BOUNDARY NOTE, so it is not discovered later: splitting on t means the last
train pair's TARGET day is the first val pair's BASE day (2025-09-30 -> 10-01,
then 10-01 -> 10-02). One day's SIC field therefore appears as a target in
train and as a feature in val. That is one day out of ~270 and is the standard
reading of "split on the base date", but it is not a strictly disjoint split.
Pass --gap-days 1 to drop the straddling pair entirely if you need that.

MEMORY. A full-stride year is on the order of 10^8 rows, which is several GB.
The build therefore runs in two passes: pass 1 counts the surviving rows per
split, prints the exact size, and stops if it exceeds --max-gb; pass 2
allocates those arrays once and fills them in place. That avoids the
list-of-arrays-then-concatenate pattern, which peaks at twice the final size.
Use --stride to subsample the grid (--stride 2 keeps every 2nd row and column,
so about a quarter of the rows).

Usage:
    python -m src.models.build_forecast_dataset
    python -m src.models.build_forecast_dataset --stride 2
    python -m src.models.build_forecast_dataset --max-gb 6
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio

from src.data.preprocess import RAW_DIR, load_sic

ROOT = Path(__file__).resolve().parents[2]
CURRENTS_DIR = ROOT / "data" / "processed" / "currents"
OUT_PATH = ROOT / "data" / "processed" / "forecast_dataset.npz"

#  the archive's cadence: one field per day, so a lead is a whole number of
#  days. This is a property of the DATA, not a configuration choice.
SOURCE_CADENCE_DAYS = 1

FEATURE_NAMES = ["sic_t", "current_u_t", "current_v_t", "current_speed_t"]
N_FEATURES = len(FEATURE_NAMES)

SPLITS = {
    "train": (date(1900, 1, 1), date(2025, 9, 30)),
    "val":   (date(2025, 10, 1), date(2025, 11, 30)),
    "test":  (date(2025, 12, 1), date(2025, 12, 31)),
}


def split_of(base: date) -> str | None:
    for name, (lo, hi) in SPLITS.items():
        if lo <= base <= hi:
            return name
    return None


def _day_from_stem(stem: str, prefix: str) -> date | None:
    try:
        return datetime.strptime(stem[len(prefix):], "%Y%m%d").date()
    except ValueError:
        return None


def list_pairs(sic_dir: Path, cur_dir: Path, gap_days: int,
               lead_days: int = SOURCE_CADENCE_DAYS) -> list[tuple[date, date, str]]:
    """Every (base, target, split) with a true `lead_days` step and all three
    files present. A missing archive day removes the pairs that would have
    spanned it rather than producing a mislabelled longer forecast."""
    if not isinstance(lead_days, int) or lead_days < 1:
        raise ValueError(f"lead_days must be a whole number of days >= 1, got "
                         f"{lead_days!r}. The sea-ice archive is daily, so a "
                         f"sub-daily lead has no target to train against.")
    sic_days = sorted(d for d in (_day_from_stem(p.stem, "sic_")
                                  for p in sic_dir.glob("sic_*.tif")) if d)
    have_sic = set(sic_days)
    have_cur = {d for d in (_day_from_stem(p.stem, "currents_")
                            for p in cur_dir.glob("currents_*.tif")) if d}

    pairs: list[tuple[date, date, str]] = []
    for base in sic_days:
        target = base + timedelta(days=lead_days)
        if target not in have_sic or base not in have_cur:
            continue                      # missing archive day, or no currents for t
        s = split_of(base)
        if s is None:
            continue
        pairs.append((base, target, s))

    if gap_days:
        # Drop pairs whose TARGET lands in a later split, so no field is both a
        # target in one split and a feature in the next.
        pairs = [(b, t, s) for b, t, s in pairs if split_of(t) == s]
    return pairs


def load_day(base: date, target: date, sic_dir: Path, cur_dir: Path, stride: int):
    #  features come from `base` only; `target` is opened for the LABEL alone
    """Return the five strided 2-D fields for one pair, or None if a file is bad."""
    sic_t = load_sic(sic_dir / f"sic_{base:%Y%m%d}.tif").sic[::stride, ::stride]
    sic_t1 = load_sic(sic_dir / f"sic_{target:%Y%m%d}.tif").sic[::stride, ::stride]
    with rasterio.open(cur_dir / f"currents_{base:%Y%m%d}.tif") as src:
        if src.count < 2:
            raise ValueError(
                f"currents_{base:%Y%m%d}.tif has {src.count} band(s), need 2")
        u = src.read(1)[::stride, ::stride]
        v = src.read(2)[::stride, ::stride]
    if u.shape != sic_t.shape:
        raise ValueError(
            f"grid mismatch on {base:%Y-%m-%d}: currents {u.shape} vs SIC {sic_t.shape}")
    return sic_t, u, v, sic_t1


def valid_mask(sic_t, u, v, sic_t1) -> np.ndarray:
    """All four features and the target must be finite. SIC 0.0 is valid."""
    return (np.isfinite(sic_t) & np.isfinite(u) & np.isfinite(v) & np.isfinite(sic_t1))


def build(sic_dir: Path, cur_dir: Path, out_path: Path,
          stride: int, max_gb: float, gap_days: int,
          lead_days: int = SOURCE_CADENCE_DAYS) -> int:
    pairs = list_pairs(sic_dir, cur_dir, gap_days, lead_days)
    if not pairs:
        print(f"No usable (t, t+1) pairs found.\n"
              f"  SIC      : {sic_dir}\n  currents : {cur_dir}", file=sys.stderr)
        return 1

    print("=" * 74)
    print(f"{lead_days}-DAY-AHEAD SIC FORECAST DATASET   features(t) -> "
          f"sic(t+{lead_days})   [+{lead_days * 24} h lead]")
    print("=" * 74)
    print(f"  features   : {', '.join(FEATURE_NAMES)}")
    print(f"  target     : sic_t1")
    print(f"  pairs      : {len(pairs)} true {lead_days}-day-step pairs")
    print(f"  stride     : {stride}" + ("" if stride == 1 else "  (grid subsampled)"))
    if gap_days:
        print(f"  gap-days   : {gap_days}  (pairs straddling a split boundary dropped)")

    # ---- pass 1: count surviving rows, so pass 2 can allocate exactly once ----
    counts = {s: 0 for s in SPLITS}
    per_day: dict[str, list[int]] = {s: [] for s in SPLITS}
    dates_by_split: dict[str, list[date]] = {s: [] for s in SPLITS}

    print(f"\nPass 1/2: counting valid rows over {len(pairs)} pair(s)...")
    for i, (base, target, s) in enumerate(pairs, 1):
        n = int(valid_mask(*load_day(base, target, sic_dir, cur_dir, stride)).sum())
        counts[s] += n
        per_day[s].append(n)
        dates_by_split[s].append(base)
        if i % 50 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)} pairs scanned")

    total_rows = sum(counts.values())
    nbytes = total_rows * (N_FEATURES + 1) * 4          # float32 X and y
    print(f"\n{'split':<8}{'pairs':>8}{'rows':>16}{'size':>12}")
    print("-" * 44)
    for s in ("train", "val", "test"):
        gb = counts[s] * (N_FEATURES + 1) * 4 / 1e9
        print(f"{s:<8}{len(per_day[s]):>8}{counts[s]:>16,}{gb:>11.2f}G")
    print("-" * 44)
    print(f"{'TOTAL':<8}{len(pairs):>8}{total_rows:>16,}{nbytes / 1e9:>11.2f}G")

    if total_rows == 0:
        print("\nNo valid rows. Check that SIC and currents overlap spatially.",
              file=sys.stderr)
        return 1
    if nbytes / 1e9 > max_gb:
        print(f"\nABORTING BEFORE ALLOCATION: {nbytes / 1e9:.2f} GB exceeds "
              f"--max-gb {max_gb:g}.\n"
              f"  Re-run with --stride 2 (about 1/4 the rows) or --stride 3 "
              f"(about 1/9),\n  or raise --max-gb if you really have the memory.",
              file=sys.stderr)
        return 1
    if any(counts[s] == 0 for s in SPLITS):
        empty = [s for s in SPLITS if counts[s] == 0]
        print(f"\nWARNING: empty split(s): {', '.join(empty)}", file=sys.stderr)

    # ---- pass 2: allocate once, fill in place -------------------------------
    X = {s: np.empty((counts[s], N_FEATURES), dtype="float32") for s in SPLITS}
    y = {s: np.empty(counts[s], dtype="float32") for s in SPLITS}
    cursor = {s: 0 for s in SPLITS}

    print(f"\nPass 2/2: filling {total_rows:,} rows...")
    for i, (base, target, s) in enumerate(pairs, 1):
        sic_t, u, v, sic_t1 = load_day(base, target, sic_dir, cur_dir, stride)
        m = valid_mask(sic_t, u, v, sic_t1)
        n = int(m.sum())
        if n == 0:
            continue
        a, b = cursor[s], cursor[s] + n
        ui, vi = u[m], v[m]
        X[s][a:b, 0] = sic_t[m]
        X[s][a:b, 1] = ui
        X[s][a:b, 2] = vi
        X[s][a:b, 3] = np.hypot(ui, vi)         # current_speed_t
        y[s][a:b] = sic_t1[m]
        cursor[s] = b
        if i % 50 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)} pairs written")

    for s in SPLITS:
        assert cursor[s] == counts[s], f"{s}: wrote {cursor[s]} of {counts[s]}"

    # ---- save ---------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": np.array(FEATURE_NAMES),
        "stride": np.array(stride),
        "gap_days": np.array(gap_days),
        "lead_days": np.array(lead_days),
        "lead_hours": np.array(lead_days * 24),
    }
    for s in SPLITS:
        payload[f"X_{s}"] = X[s]
        payload[f"y_{s}"] = y[s]
        # Rows are appended in chronological order, so base_dates + rows_per_day
        # recover the row -> day mapping exactly without a per-row index column.
        payload[f"{s}_base_dates"] = np.array(
            [str(d) for d in dates_by_split[s]], dtype="U10")
        payload[f"{s}_rows_per_day"] = np.array(per_day[s], dtype="int64")
        lo, hi = SPLITS[s]
        payload[f"{s}_base_range"] = np.array(
            [str(min(dates_by_split[s])), str(max(dates_by_split[s]))], dtype="U10"
        ) if dates_by_split[s] else np.array(["", ""], dtype="U10")
        payload[f"{s}_target_range"] = np.array(
            [str(min(dates_by_split[s]) + timedelta(days=lead_days)),
             str(max(dates_by_split[s]) + timedelta(days=lead_days))], dtype="U10"
        ) if dates_by_split[s] else np.array(["", ""], dtype="U10")

    print(f"\nWriting {out_path} ...")
    np.savez(out_path, **payload)          # uncompressed: fast, and float32 noise
                                           # compresses poorly anyway
    print(f"\nGOT     {out_path.name}  ({out_path.stat().st_size / 1e9:.2f} GB)")
    for s in ("train", "val", "test"):
        if counts[s]:
            print(f"  X_{s:<6} {X[s].shape}   base {payload[f'{s}_base_range'][0]}"
                  f" .. {payload[f'{s}_base_range'][1]}"
                  f"   target {payload[f'{s}_target_range'][0]}"
                  f" .. {payload[f'{s}_target_range'][1]}")
    print("\nNo shuffling, no interpolation, no model. Splits are chronological "
          "on the base date t.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build the 1-day-ahead SIC forecast dataset (no training)")
    ap.add_argument("--sic-dir", default=str(RAW_DIR))
    ap.add_argument("--currents-dir", default=str(CURRENTS_DIR))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth grid row and column (default 1 = all)")
    ap.add_argument("--max-gb", type=float, default=4.0,
                    help="abort before allocating if the dataset exceeds this")
    ap.add_argument("--gap-days", type=int, default=0, choices=(0, 1),
                    help="1 drops pairs whose target falls in the next split")
    ap.add_argument("--lead-days", type=int, default=SOURCE_CADENCE_DAYS,
                    help="forecast lead in WHOLE DAYS (default 1 = +24 h). "
                         "The archive is daily; sub-daily leads have no target.")
    args = ap.parse_args()

    if args.stride < 1:
        ap.error("--stride must be >= 1")
    if args.lead_days < 1:
        ap.error("--lead-days must be a whole number of days >= 1")
    try:
        code = build(Path(args.sic_dir), Path(args.currents_dir), Path(args.out),
                     args.stride, args.max_gb, args.gap_days, args.lead_days)
    except (ValueError, FileNotFoundError) as exc:
        print(f"\nCANNOT BUILD THE DATASET\n\n{exc}\n", file=sys.stderr)
        code = 1
    sys.exit(code)

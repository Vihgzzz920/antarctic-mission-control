"""
Step 3 -- persistence baseline for sea-ice concentration.

    SIC_hat(t+1) = SIC(t)

This is the number every later forecast has to beat. It is not a strawman:
sea ice is highly autocorrelated day to day, so persistence is genuinely hard
to beat at 24 h lead, and a model that only matches it has added nothing.

Scoring rules:
  * chronological only -- day t predicts day t+1. No shuffling, no random split.
  * ONLY true 24 h steps are scored. If the archive is missing a day, or the
    stack jumps across seasons, the adjacent pair is skipped entirely -- it is
    not a 1-day forecast and scoring it would understate persistence's skill.
  * a pixel is scored only where BOTH the prediction and the truth are finite.
    NaN is missing (land, pole hole, swath gap) and is never counted, never
    filled, and never treated as 0.
  * 0.0 IS scored. Open water is data.
  * "overall" pools every scored pixel from every pair into one calculation,
    rather than averaging per-pair metrics, so pairs with more valid pixels
    carry proportionally more weight.

Usage:
    python -m src.models.persistence
    python -m src.models.persistence data/raw/sic_A.tif data/raw/sic_B.tif ...
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date

import numpy as np

from src.data.preprocess import GridMismatch, SICStack, load_sic_series

MIZ_LO, MIZ_HI = 15.0, 80.0     # marginal ice zone, by the observed truth


@dataclass(frozen=True)
class Score:
    """Metrics over one set of scored pixels."""

    n: int
    mae: float
    rmse: float
    r2: float
    n_miz: int
    mae_miz: float


def score(pred: np.ndarray, truth: np.ndarray) -> Score:
    """Score two 1-D arrays of co-located, already-valid values."""
    if pred.size == 0:
        nan = float("nan")
        return Score(n=0, mae=nan, rmse=nan, r2=nan, n_miz=0, mae_miz=nan)

    err = pred - truth
    abs_err = np.abs(err)
    ss_res = float(np.dot(err, err))
    # R2 is measured against the variance of the TRUTH on these same pixels.
    ss_tot = float(((truth - truth.mean()) ** 2).sum())

    miz = (truth >= MIZ_LO) & (truth < MIZ_HI)
    return Score(
        n=int(pred.size),
        mae=float(abs_err.mean()),
        rmse=float(np.sqrt(ss_res / pred.size)),
        r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        n_miz=int(miz.sum()),
        mae_miz=float(abs_err[miz].mean()) if miz.any() else float("nan"),
    )


@dataclass(frozen=True)
class Pair:
    """One day-to-day persistence forecast and how it did."""

    base: date | None       # day t, the prediction
    target: date | None     # day t+1, the truth
    gap_days: int | None    # should be 1; anything else is not a 24 h forecast
    score: Score


def evaluate(stack: SICStack) -> tuple[list[Pair], Score]:
    """Score persistence on every consecutive pair in the stack, oldest first."""
    if len(stack) < 2:
        raise ValueError(
            f"Persistence needs at least 2 days; the stack has {len(stack)}.\n"
            f"Run:  python -m src.data.fetch_sic --days 5"
        )

    pairs: list[Pair] = []
    pooled_pred: list[np.ndarray] = []
    pooled_truth: list[np.ndarray] = []

    for i in range(len(stack) - 1):
        base, target = stack.days[i], stack.days[i + 1]
        gap = (target - base).days if (base and target) else None
        if gap != 1:
            continue        # missing day, season jump, or undated file -- not a 24 h step

        pred_field, truth_field = stack.sic[i], stack.sic[i + 1]
        both = np.isfinite(pred_field) & np.isfinite(truth_field)
        p, t = pred_field[both], truth_field[both]

        pairs.append(Pair(base=base, target=target, gap_days=gap, score=score(p, t)))

        pooled_pred.append(p)
        pooled_truth.append(t)

    if not pairs:
        raise ValueError(
            f"No consecutive-day pairs in {len(stack)} file(s): every adjacent pair "
            f"is separated by more or less than 1 day, so there is nothing to score."
        )

    overall = score(np.concatenate(pooled_pred), np.concatenate(pooled_truth))
    return pairs, overall


def _label(p: Pair) -> str:
    b = p.base.isoformat() if p.base else "--"
    t = p.target.strftime("%m-%d") if p.target else "--"
    return f"{b} -> {t}"


def main(argv: list[str]) -> int:
    try:
        stack = load_sic_series(argv or None)
        pairs, overall = evaluate(stack)
    except (GridMismatch, FileNotFoundError, ValueError) as exc:
        print(f"\nCANNOT RUN THE PERSISTENCE BASELINE\n\n{exc}\n", file=sys.stderr)
        return 1

    print("=" * 78)
    print("PERSISTENCE BASELINE   SIC_hat(t+1) = SIC(t)")
    skipped = (len(stack) - 1) - len(pairs)
    print(f"{len(stack)} day(s) on {stack.grid.crs} {stack.grid.shape}"
          f"  ->  {len(pairs)} forecast pair(s)")
    if skipped:
        print(f"{skipped} adjacent pair(s) skipped: calendar gap is not exactly 1 day")
    print("=" * 78)

    head = f"{'pair':<22}{'dd':>3}{'scored px':>12}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'MIZ MAE':>9}{'MIZ px':>10}"
    print(head)
    print("-" * 78)
    for p in pairs:
        s = p.score
        flag = "" if p.gap_days == 1 else "  <- NOT a 24 h step"
        print(f"{_label(p):<22}{str(p.gap_days or '?'):>3}{s.n:>12,}"
              f"{s.mae:>8.2f}{s.rmse:>8.2f}{s.r2:>8.3f}{s.mae_miz:>9.2f}{s.n_miz:>10,}{flag}")

    print("-" * 78)
    print(f"{'POOLED':<22}{'':>3}{overall.n:>12,}"
          f"{overall.mae:>8.2f}{overall.rmse:>8.2f}{overall.r2:>8.3f}"
          f"{overall.mae_miz:>9.2f}{overall.n_miz:>10,}")

    print("\nUnits are percent concentration. MAE/RMSE lower is better; R2 higher.")
    print("MIZ MAE is the same error restricted to pixels where the TRUTH is")
    print(f"{MIZ_LO:.0f}-{MIZ_HI:.0f}% ice -- the marginal ice zone, where the ice edge actually")
    print("moves and where a ship's route is actually decided.")
    print("\nHOW TO READ R2 HERE: it is measured against the variance of the truth")
    print("over the whole scored field, which spans open water to solid pack. That")
    print("variance is enormous, so R2 will sit close to 1.00 even for a forecast")
    print("that is useless at the ice edge. Treat MIZ MAE as the number to beat.")
    print(f"\nSMOKE TEST ONLY: {len(pairs)} pair(s) is a pipeline check, not a validation")
    print("study. No conclusion about forecast skill should be drawn from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

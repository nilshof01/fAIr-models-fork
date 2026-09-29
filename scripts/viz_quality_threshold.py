"""Find the natural threshold cliff for the pred_feats quality classifier.

Loads cached features + labels, runs the same 5-fold XGBoost CV as
train_quality_classifier.py (identical config), collects OOF probabilities,
then plots the score distributions for good vs bad chips and marks:
  - the valley (minimum density between the two modes)
  - the threshold that achieves the user-specified bad_caught target

Usage:
    python scripts/viz_quality_threshold.py \
        --out label-cleanup/run2/quality_xgb \
        --target-bad-caught 0.95
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_valley(oof_probs: np.ndarray, y: np.ndarray,
                n_bins: int = 100) -> float:
    """Return threshold at minimum density between the two modes.

    Builds a KDE-style histogram on the combined score distribution,
    then finds the local minimum in the range where both good and bad
    chips are present (i.e. the overlap region).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    hist, _ = np.histogram(oof_probs, bins=bins, density=True)

    # overlap region: where both classes have non-trivial density
    g_hist, _ = np.histogram(oof_probs[y == 1], bins=bins, density=True)
    b_hist, _ = np.histogram(oof_probs[y == 0], bins=bins, density=True)
    overlap = (g_hist > 0) & (b_hist > 0)

    if not overlap.any():
        return float(centers[hist.argmin()])

    # valley = minimum total density in the overlap region
    hist_overlap = hist.copy()
    hist_overlap[~overlap] = np.inf
    return float(centers[hist_overlap.argmin()])


def oof_stats(oof_probs: np.ndarray, y: np.ndarray,
              thr: float) -> dict:
    accept     = oof_probs >= thr
    tp         = int(( accept & (y == 1)).sum())
    fp         = int(( accept & (y == 0)).sum())
    fn         = int((~accept & (y == 1)).sum())
    tn         = int((~accept & (y == 0)).sum())
    n_bad      = int((y == 0).sum())
    return dict(
        accept     = tp + fp,
        precision  = tp / max(tp + fp, 1),
        recall     = tp / max(tp + fn, 1),
        bad_caught = tn / max(n_bad, 1),
    )


def run_cv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, va in skf.split(X, y):
        n_pos = int((y[tr] == 1).sum())
        n_neg = int((y[tr] == 0).sum())
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        clf.fit(X[tr], y[tr].astype(int))
        oof[va] = clf.predict_proba(X[va])[:, 1].astype(np.float32)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="label-cleanup/run2/quality_xgb",
                    help="directory with cached pred_feats.npy / labels.npy / proj_ids.json")
    ap.add_argument("--target-bad-caught", type=float, default=0.95,
                    help="minimum fraction of bad chips to reject (default 0.95)")
    ap.add_argument("--plot-out", default=None,
                    help="save plot to this path instead of showing interactively")
    a = ap.parse_args()

    out = Path(a.out)
    pf       = np.load(out / "pred_feats.npy")
    y        = np.load(out / "labels.npy")
    proj_ids = json.loads((out / "proj_ids.json").read_text())

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_quality_classifier import build_proj_vocab, encode_proj_ids, N_TOP_PROJECTS
    proj_vocab = build_proj_vocab(proj_ids, top_n=N_TOP_PROJECTS)
    proj_feats = encode_proj_ids(proj_ids, proj_vocab)
    X = np.hstack([pf, proj_feats])

    print(f"Running 5-fold CV to collect OOF probabilities ...")
    oof = run_cv(X, y.astype(int))
    np.save(out / "oof_probs.npy", oof)
    print(f"Saved OOF probs -> {out / 'oof_probs.npy'}")

    # ── Find valley and business threshold ────────────────────────────────────
    valley_thr = find_valley(oof, y)
    v = oof_stats(oof, y, valley_thr)
    print(f"\nNatural valley at thr={valley_thr:.3f}")
    print(f"  accept={v['accept']}  prec={v['precision']:.1%}  "
          f"recall={v['recall']:.1%}  bad_caught={v['bad_caught']:.1%}")

    # find lowest threshold that hits the bad_caught target
    thresholds = np.arange(0.01, 1.00, 0.01)
    target_thr = None
    for t in thresholds[::-1]:   # start low, stop as soon as target met
        if oof_stats(oof, y, t)["bad_caught"] >= a.target_bad_caught:
            target_thr = t
        else:
            break
    if target_thr is None:
        target_thr = 0.99

    b = oof_stats(oof, y, target_thr)
    print(f"\nLowest threshold meeting {a.target_bad_caught:.0%} bad_caught: "
          f"thr={target_thr:.2f}")
    print(f"  accept={b['accept']}  prec={b['precision']:.1%}  "
          f"recall={b['recall']:.1%}  bad_caught={b['bad_caught']:.1%}")

    # ── Print fine-grained table ───────────────────────────────────────────────
    print(f"\n-- Fine-grained OOF threshold table (0.80–0.99) --")
    print(f"  {'thr':>5}  {'accept':>7}  {'precision':>10}  {'recall':>8}  "
          f"{'bad_caught':>11}")
    for t in np.arange(0.80, 1.00, 0.01):
        t = round(t, 2)
        s = oof_stats(oof, y, t)
        marker = " ← valley" if abs(t - valley_thr) < 0.005 else \
                 " ← target" if abs(t - target_thr) < 0.005 else ""
        print(f"  {t:>5.2f}  {s['accept']:>7}  {s['precision']:>10.1%}  "
              f"{s['recall']:>8.1%}  {s['bad_caught']:>11.1%}{marker}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    import os
    import matplotlib
    # TkAgg needs a display; pods and servers have none, and falling back to
    # Agg there is better than dying after the scoring work is already done
    matplotlib.use("TkAgg" if (not a.plot_out and os.environ.get("DISPLAY"))
                   else "Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    bins = np.linspace(0, 1, 80)
    good_scores = oof[y == 1]
    bad_scores  = oof[y == 0]
    ax1.hist(good_scores, bins=bins, alpha=0.55, color="#2166ac",
             density=True, label=f"good (n={len(good_scores):,})")
    ax1.hist(bad_scores,  bins=bins, alpha=0.55, color="#d6604d",
             density=True, label=f"bad  (n={len(bad_scores):,})")
    ax1.axvline(valley_thr, color="#4dac26", lw=2, ls="--",
                label=f"valley  thr={valley_thr:.2f}")
    ax1.axvline(target_thr, color="#b8860b", lw=2, ls=":",
                label=f"{a.target_bad_caught:.0%} bad_caught  thr={target_thr:.2f}")
    ax1.set(xlabel="OOF quality score", ylabel="density",
            title="Score distributions: good vs bad chips")
    ax1.legend(frameon=False, fontsize=9)

    thrs = np.arange(0.01, 0.995, 0.005)
    stats = [oof_stats(oof, y, t) for t in thrs]
    ax2.plot(thrs, [s["bad_caught"] for s in stats], color="#d6604d",
             lw=2, label="bad_caught (fraction bad rejected)")
    ax2.plot(thrs, [s["recall"] for s in stats], color="#2166ac",
             lw=2, label="recall (fraction good accepted)")
    ax2.axhline(a.target_bad_caught, color="#b8860b", lw=1.2, ls=":",
                label=f"{a.target_bad_caught:.0%} bad_caught target")
    ax2.axvline(valley_thr, color="#4dac26", lw=1.5, ls="--",
                label=f"valley  thr={valley_thr:.2f}")
    ax2.axvline(target_thr, color="#b8860b", lw=1.5, ls=":",
                label=f"target  thr={target_thr:.2f}")
    ax2.set(xlabel="threshold", ylabel="fraction",
            title="Bad_caught vs recall trade-off",
            xlim=(0.0, 1.0), ylim=(0.0, 1.05))
    ax2.legend(frameon=False, fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Quality classifier threshold analysis (OOF)", y=1.01,
                 fontsize=12)
    fig.tight_layout()

    out_path = a.plot_out or str(out / "threshold_analysis.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

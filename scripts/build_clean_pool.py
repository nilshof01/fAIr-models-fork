"""Build a cleaned training-pool CSV from quality classifier scores.

Combines:
  1. Reviewed "keep" chips with OOF quality score >= threshold
  2. Unreviewed chips with quality score >= threshold (if mlp_scores.npy present)

Output pool.csv has the columns make_cv_folds.py needs:
  tile_id, dataset_row, project_name, num_buildings, country, source

Usage:
    python scripts/build_clean_pool.py \
        --score-dir label-cleanup/run2/quality_xgb \
        --threshold 0.95 \
        --out data/clean_pool/pool.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", default="label-cleanup/run2/quality_xgb",
                    help="directory with oof_probs.npy, rows.npy, labels.npy "
                         "(and optionally mlp_scores.npy, mlp_row_ids.npy)")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--out", default="data/clean_pool/pool.csv")
    ap.add_argument("--decisions-repo",
                    default="nilsho01/vhr-buildings-review-decisions")
    ap.add_argument("--source-repo",
                    default="hotosm/vhr-building-segmentation")
    ap.add_argument("--no-unreviewed", action="store_true",
                    help="skip unreviewed chips even if mlp_scores.npy is present")
    ap.add_argument("--benchmark", default="data/benchmark_v1/benchmark_chips.csv",
                    help="benchmark CSV; those tile_ids are excluded from the pool")
    a = ap.parse_args()

    score_dir = Path(a.score_dir)
    out_path  = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    # ── 1. Reviewed chips — vhr-buildings-review-decisions is the source of truth
    print(f"\nLoading reviewed pool from {a.decisions_repo} ...")
    rev = load_dataset(a.decisions_repo, split="train").to_pandas()
    k   = rev[rev.usable].copy()
    reviewed_df = k[["tile_id", "dataset_row", "project_name",
                      "num_buildings", "country"]].copy()
    reviewed_df["source"] = "reviewed"
    print(f"  usable chips in review decisions: {len(reviewed_df):,}")

    # ── 2. Unreviewed chips ───────────────────────────────────────────────────
    mlp_scores_path  = score_dir / "mlp_scores.npy"
    mlp_row_ids_path = score_dir / "mlp_row_ids.npy"
    unreviewed_df    = pd.DataFrame()

    if not a.no_unreviewed and mlp_scores_path.exists() and mlp_row_ids_path.exists():
        mlp_scores  = np.load(mlp_scores_path)
        mlp_row_ids = np.load(mlp_row_ids_path)

        unrev_mask      = mlp_scores >= a.threshold
        kept_unrev_rows = mlp_row_ids[unrev_mask].tolist()
        print(f"\nUnreviewed pool:")
        print(f"  total unreviewed chips    : {len(mlp_scores):,}")
        print(f"  passing thr={a.threshold:.2f}        : {len(kept_unrev_rows):,}  "
              f"({100*len(kept_unrev_rows)/max(len(mlp_scores),1):.1f}%)")

        print(f"  loading metadata from {a.source_repo} ...")
        src = load_dataset(a.source_repo, split="train")
        unrev_rows = []
        for dr in kept_unrev_rows:
            r = src[int(dr)]
            unrev_rows.append({
                "tile_id":       r["tile_id"],
                "dataset_row":   int(dr),
                "project_name":  str(r.get("project_id", "")),
                "num_buildings": int(r.get("n_instances", 0)),
                "country":       str(r.get("country", "")),
                "source":        "unreviewed",
            })
        unreviewed_df = pd.DataFrame(unrev_rows)
        print(f"  kept {len(unreviewed_df):,} unreviewed chips")
    else:
        if not a.no_unreviewed:
            print(f"\nNo mlp_scores.npy found in {score_dir} — skipping unreviewed chips.")
            print("Run train_quality_classifier.py --score-remaining first to include them.")

    # ── 3. Combine and strip benchmark chips ─────────────────────────────────
    pool = pd.concat([reviewed_df, unreviewed_df], ignore_index=True)
    pool = pool.drop_duplicates(subset="tile_id")

    bench_path = Path(a.benchmark)
    if bench_path.exists():
        bench_tiles = set(pd.read_csv(bench_path).tile_id)
        before = len(pool)
        pool = pool[~pool.tile_id.isin(bench_tiles)]
        removed = before - len(pool)
        if removed:
            print(f"\nRemoved {removed} benchmark chips from pool "
                  f"(must never appear in training data)")
    else:
        print(f"\nWarning: benchmark CSV not found at {bench_path} — "
              f"benchmark chips may be included in pool")

    pool.to_csv(out_path, index=False)

    print(f"\n{'='*50}")
    print(f"Total pool: {len(pool):,} chips")
    print(f"  reviewed  : {(pool.source == 'reviewed').sum():,}")
    if len(unreviewed_df):
        print(f"  unreviewed: {(pool.source == 'unreviewed').sum():,}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

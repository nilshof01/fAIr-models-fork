"""Cross-validation folds over the training pool, with a FIXED external test set.

The benchmark (nilsho01/vhr-buildings-benchmark-v1, 400 chips) is the test set
for every fold, so it is never re-drawn and results stay comparable across
experiments. Cross-validation varies only the validation slice, which is what
measures run-to-run variance; varying the test set would confound variance
with test-set composition.

Benchmark chips are removed from the pool entirely, so nothing leaks.
Validation is carved by 4x4 spatial block so adjacent tiles cannot straddle it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "nilsho01/vhr-buildings-review-decisions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark_v1/benchmark_chips.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/cv_folds")
    ap.add_argument("--pool-csv", default=None,
                    help="pre-filtered pool CSV (from build_clean_pool.py); "
                         "if omitted, reads from HF review-decisions dataset")
    a = ap.parse_args()

    bench = set(pd.read_csv(a.benchmark).tile_id)
    if a.pool_csv:
        k = pd.read_csv(a.pool_csv)
        pool = k[~k.tile_id.isin(bench)].copy()
        print(f"Using pre-filtered pool from {a.pool_csv}: {len(pool):,} chips")
    else:
        from datasets import load_dataset
        rev = load_dataset(REPO, split="train").to_pandas()
        k = rev[rev.usable].copy()
        pool = k[~k.tile_id.isin(bench)].copy()

    blk = pool.tile_id.str.extract(r"OAM-(\d+)-(\d+)")
    pool["block"] = (pool.project_name.astype(str) + "|"
                     + (blk[0].astype(float) // a.block).astype("Int64").astype(str) + "_"
                     + (blk[1].astype(float) // a.block).astype("Int64").astype(str))

    rng = np.random.default_rng(a.seed)
    blocks = pool.block.dropna().unique()
    blocks = blocks[rng.permutation(len(blocks))]
    # Deal into 1/val_frac bins, not into `folds` bins. With 5 folds a
    # fold-per-bin split would make validation 20%; the ask is 10%, so each
    # fold takes one bin of ten and the other nine go to train.
    n_bins = max(a.folds, int(round(1 / a.val_frac)))
    # deal blocks round-robin into `folds` validation slices, balancing on
    # BUILDING count rather than chip count so no fold's val is all bare ground
    # Two passes. Balancing on buildings alone sends every zero-building block
    # to whichever fold is behind, and since ~45% of chips are verified-empty
    # one fold ends up holding half the pool. So: deal populated blocks by
    # building count, then empty blocks by chip count.
    bstat = pool.groupby("block").num_buildings.agg(["size", "sum"])
    assign = {}
    got_b = np.zeros(n_bins)
    for b in sorted([x for x in blocks if bstat.loc[x, "sum"] > 0],
                    key=lambda x: -bstat.loc[x, "sum"]):
        i = int(np.argmin(got_b))
        assign[b] = i
        got_b[i] += bstat.loc[b, "sum"]
    got_c = np.array([sum(bstat.loc[b, "size"] for b, f in assign.items() if f == i)
                      for i in range(n_bins)], float)
    for b in sorted([x for x in blocks if bstat.loc[x, "sum"] == 0],
                    key=lambda x: -bstat.loc[x, "size"]):
        i = int(np.argmin(got_c))
        assign[b] = i
        got_c[i] += bstat.loc[b, "size"]
    pool["val_bin"] = pool.block.map(assign)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = {"benchmark_test_chips": len(bench), "pool_chips": len(pool),
            "folds": a.folds, "val_frac_target": a.val_frac, "bins": n_bins,
            "block": a.block, "seed": a.seed, "per_fold": {}}
    print(f"pool {len(pool):,} chips (benchmark's {len(bench)} removed)\n")
    print(f"{'fold':>4} {'train':>7} {'val':>6} {'val %':>6} {'train bldgs':>12} {'val bldgs':>10}")
    for f in range(a.folds):
        va = pool[pool.val_bin == f]
        tr = pool[pool.val_bin != f]
        (out / f"fold{f}_train_tiles.txt").write_text("\n".join(tr.tile_id))
        (out / f"fold{f}_val_tiles.txt").write_text("\n".join(va.tile_id))
        meta["per_fold"][f] = {
            "train_chips": len(tr), "val_chips": len(va),
            "train_buildings": int(tr.num_buildings.sum()),
            "val_buildings": int(va.num_buildings.sum()),
            "train_empty": int((tr.num_buildings == 0).sum()),
            "val_empty": int((va.num_buildings == 0).sum()),
        }
        print(f"{f:>4} {len(tr):7,} {len(va):6,} {100*len(va)/len(pool):5.1f}% "
              f"{int(tr.num_buildings.sum()):12,} {int(va.num_buildings.sum()):10,}")
    (out / "benchmark_test_tiles.txt").write_text("\n".join(sorted(bench)))

    # Also emit the cv5.csv shape that train.py already reads, so
    # there is one split format rather than two. Test is the benchmark in every
    # fold, so it gets its own bin that fold_roles never selects as val.
    rows = pool[["tile_id"]].copy()
    rows["assignment"] = "fold_" + pool.val_bin.astype(int).astype(str)
    # Build benchmark rows with metadata from the benchmark CSV directly,
    # since benchmark chips are excluded from the pool and would get NaN otherwise.
    bench_df = pd.read_csv(a.benchmark).set_index("tile_id")
    bench_rows = pd.DataFrame({"tile_id": sorted(bench), "assignment": "benchmark"})
    for c in ("country", "project_name", "num_buildings"):
        bench_rows[c] = bench_rows.tile_id.map(bench_df[c])
    spec = pd.concat([rows, bench_rows], ignore_index=True)
    meta_cols = k.set_index("tile_id")
    for c in ("country", "project_name", "num_buildings"):
        # fill pool chips from pool metadata; benchmark chips already populated above
        spec[c] = spec[c].where(spec.assignment == "benchmark",
                                spec.tile_id.map(meta_cols[c]))
    spec["is_empty"] = spec.num_buildings.fillna(0) == 0
    # split_report.py and pool_data.py both key on n_instances; the review
    # dataset calls the same quantity num_buildings. Emit both rather than
    # renaming, so either consumer works.
    spec["n_instances"] = spec.num_buildings.fillna(0).astype(int)
    spec["block"] = spec.tile_id.map(pool.set_index("tile_id").block)
    spec.to_csv(out / "cv_folds.csv", index=False)
    print(f"wrote {out}/cv_folds.csv in the cv5 format "
          f"({n_bins} val bins + a fixed 'benchmark' bin)")
    meta["notes"] = [
        "Test is ALWAYS the fixed benchmark; only validation rotates.",
        "Benchmark chips are excluded from the pool, so train and val never see them.",
        "Validation blocks are dealt to balance BUILDING count, not chip count.",
    ]
    (out / "cv_folds.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}/ (test = {len(bench)} benchmark chips, fixed)")


if __name__ == "__main__":
    main()

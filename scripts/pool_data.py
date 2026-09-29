"""Turn a split file into the three HF datasets a training run needs.

The pool on HF carries no meaningful split column - whatever splits it ships
are ignored. Roles come from the split file: test is the chosen fold, val is
the next fold round the ring, train is everything else that is not the
holdout. The holdout is never returned by a training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from pool_splits import cap_split, fold_roles


class PoolDataSource:
    def __init__(self, dataset, split_file, fold, folds=None,
                 max_test=0, max_val=0, empty_frac_train=None, seed=1337):
        from datasets import concatenate_datasets, load_dataset

        self.split = pd.read_csv(split_file)
        bins = sorted(b for b in self.split.assignment.unique()
                      if str(b).startswith("fold_"))
        self.folds = folds or len(bins)
        if not 0 <= fold < self.folds:
            raise SystemExit(f"--fold must be in 0..{self.folds-1}")
        self.split["role"] = fold_roles(self.split.assignment, fold, self.folds)

        ds = load_dataset(dataset)
        pool = concatenate_datasets(list(ds.values()))
        seen, order = {}, []
        for i, t in enumerate(pool["tile_id"]):
            if t not in seen:                       # the pool repeats one tile_id
                seen[t] = i
                order.append(t)
        self.pool = pool
        self._row = seen

        # Benchmark chips (assignment="benchmark") live in a separate HF dataset and
        # may not be in the training pool; only non-benchmark chips are required.
        bench_tiles = set(self.split[self.split.assignment == "benchmark"].tile_id)
        missing = set(self.split.tile_id) - bench_tiles - set(seen)
        if missing:
            raise SystemExit(f"{len(missing)} chips in the split file are absent "
                             f"from {dataset} - rebuild the split file")
        missing_bench = bench_tiles - set(seen)
        if missing_bench:
            print(f"[data] {len(missing_bench)} benchmark chips not in {dataset} "
                  f"— training eval uses only those present")

        self.frames = {}
        for role, limit in (("train", 0), ("val", max_val), ("test", max_test)):
            g = self.split[self.split.role == role]
            if role == "train" and empty_frac_train is not None:
                g = self._rebalance_empty(g, empty_frac_train, seed)
            if limit:
                before = len(g)
                g = cap_split(g, limit, seed)
                print(f"[data] {role} capped {before} -> {len(g)} chips "
                      f"(populated/empty mix preserved)")
            self.frames[role] = g

    @staticmethod
    def _rebalance_empty(g, frac, seed):
        """Drop empty chips until they are `frac` of the training split."""
        pop = g[~g.is_empty]
        emp = g[g.is_empty]
        want = int(round(frac * len(pop) / max(1e-9, 1 - frac)))
        if want >= len(emp):
            return g
        rng = np.random.default_rng(seed)
        emp = emp.iloc[rng.permutation(len(emp))[:want]]
        print(f"[data] train empty chips {len(g)-len(pop)} -> {len(emp)} "
              f"({100*len(emp)/max(len(pop)+len(emp),1):.1f}% of the split)")
        return pd.concat([pop, emp]).sort_values("tile_id")

    def hf(self, role):
        # Skip chips absent from the pool (benchmark chips not in training dataset)
        idx = [self._row[t] for t in self.frames[role].tile_id if t in self._row]
        return self.pool.select(idx)

    @property
    def roles_frame(self):
        """Every chip with the role it plays in THIS run, for reporting."""
        used = pd.concat([f.assign(role=r) for r, f in self.frames.items()])
        rest = self.split[~self.split.tile_id.isin(used.tile_id)].copy()
        rest["role"] = np.where(rest.role == "holdout", "holdout", "unused")
        return pd.concat([used, rest], ignore_index=True)

"""Derive train/val/test folds from one undivided pool of labelled chips.

The pool is a single HF dataset with no split column; every split is produced
here from (criteria, proportions, seed) and written to a CSV that is committed
alongside the code, so a result traces to (pool commit + split file) rather
than to a seed a later edit could silently change.

Design points that the data forced:
  * Spatial blocks, not chips, are the assignment unit. Neighbouring tiles at
    zoom 19 share buildings across their borders and identical illumination.
  * Assignment is balanced, not random. Over 300 random 5-fold deals the
    richest fold carried 2.1x the building mass of the poorest and the empty
    share swung 13 points - fold variance would have been composition, not
    model.
  * Blocks are placed largest-first: the single biggest block holds ~9% of all
    instances, and a fold cannot absorb it if it arrives last.
  * Assignments are sticky. Re-running after more labelling keeps every block
    where it was and only places the new ones, so past numbers stay valid.

Layout: one bin per fold plus an optional `holdout` bin, which is set aside
before cross-validation and read once at the very end.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = "hotosm/vhr-building-segmentation"
BLOCK = 16
COUNTRY_WEIGHT = 0.25


class Pool:
    """Every labelled chip, with the metadata the splitter balances on."""

    def __init__(self, dataset, block=BLOCK):
        from datasets import load_dataset
        ds = load_dataset(dataset)
        keep = ("tile_id", "dataset_row", "n_instances", "project_id",
                "density_bucket")
        frames = []
        for name, split in ds.items():
            cols = [c for c in split.column_names if c in keep]
            frames.append(split.remove_columns(
                [c for c in split.column_names if c not in cols]).to_pandas())
        df = pd.concat(frames, ignore_index=True).drop_duplicates("tile_id")

        up = load_dataset(UPSTREAM, split="train")
        upm = up.remove_columns([c for c in up.column_names if c not in
            ("tile_id", "project_name", "country", "tile_x", "tile_y")
            ]).to_pandas().drop_duplicates("tile_id")
        df = df.merge(upm, on="tile_id", how="left")
        missing = int(df.project_name.isna().sum())
        if missing:
            df = df.dropna(subset=["project_name"])
            print(f"[pool] dropped {missing} chips absent from {UPSTREAM}")

        df["block"] = (df.project_name + "|" + (df.tile_x // block).astype(str)
                       + "_" + (df.tile_y // block).astype(str))
        df["is_empty"] = df.n_instances == 0
        self.df = df
        self.dataset = dataset

    @property
    def blocks(self):
        g = self.df.groupby("block")
        b = g.agg(chips=("tile_id", "size"), instances=("n_instances", "sum"),
                  n_empty=("is_empty", "sum"), country=("country", "first"))
        # `empty`, `pop` and `gt` are DataFrame methods - never name a column
        # after one, the attribute access silently returns the method instead
        assert not (set(b.columns) & set(dir(pd.DataFrame))), "column shadows a DataFrame attribute"
        return b

    def describe(self):
        d = self.df
        return (f"{len(d):,} chips | {int((~d.is_empty).sum()):,} populated "
                f"({100*(~d.is_empty).mean():.1f}%) | "
                f"{int(d.n_instances.sum()):,} instances | "
                f"{d.block.nunique():,} blocks | {d.country.nunique()} countries "
                f"| {d.project_name.nunique()} projects")


class BalancedSplitter:
    """Greedy largest-first assignment of blocks to bins.

    Each block goes to the bin whose composition is worst-served by NOT taking
    it: the bin minimising the total squared relative deviation from target on
    chips, building instances, empty chips, and per-country chip counts.
    """

    METRICS = ("chips", "instances", "n_empty")

    def __init__(self, bins, weights, seed=1337):
        self.bins = list(bins)
        self.weights = np.asarray(weights, float)
        self.weights = self.weights / self.weights.sum()
        self.seed = seed

    def _cost(self, totals, targets, countries, c_targets):
        cost = 0.0
        for m in self.METRICS:
            t = np.maximum(targets[m], 1e-9)
            cost += float(np.sum(((totals[m] - t) / t) ** 2))
        for c, tgt in c_targets.items():
            t = np.maximum(tgt, 1e-9)
            cost += COUNTRY_WEIGHT * float(np.sum(((countries[c] - t) / t) ** 2))
        return cost

    def assign(self, blocks, fixed=None):
        """`fixed` maps block -> bin for blocks already placed (sticky)."""
        fixed = dict(fixed or {})
        n = len(self.bins)
        totals = {m: np.zeros(n) for m in self.METRICS}
        counts = {c: np.zeros(n) for c in blocks.country.unique()}

        grand = {"chips": blocks.chips.sum(),
                 "instances": blocks.instances.sum(),
                 "n_empty": blocks.n_empty.sum()}
        targets = {m: self.weights * grand[m] for m in self.METRICS}
        c_targets = {c: self.weights * g.chips.sum()
                     for c, g in blocks.groupby("country")}

        idx = {b: i for i, b in enumerate(self.bins)}
        for blk, row in blocks.iterrows():          # seed existing assignments
            if blk in fixed:
                i = idx[fixed[blk]]
                for m in self.METRICS:
                    totals[m][i] += row[m]
                counts[row.country][i] += row.chips

        todo = blocks[~blocks.index.isin(fixed)]
        rng = np.random.default_rng(self.seed)
        todo = todo.iloc[rng.permutation(len(todo))]           # seed breaks ties
        todo = todo.sort_values("instances", kind="stable", ascending=False)

        out = dict(fixed)
        for blk, row in todo.iterrows():
            best, best_cost = None, np.inf
            for i, name in enumerate(self.bins):
                for m in self.METRICS:
                    totals[m][i] += row[m]
                counts[row.country][i] += row.chips
                c = self._cost(totals, targets, counts, c_targets)
                for m in self.METRICS:
                    totals[m][i] -= row[m]
                counts[row.country][i] -= row.chips
                if c < best_cost:
                    best, best_cost = i, c
            for m in self.METRICS:
                totals[m][best] += row[m]
            counts[row.country][best] += row.chips
            out[blk] = self.bins[best]
        return out


def bin_layout(folds, holdout_frac):
    """Bin names and their target share of the pool."""
    names, weights = [], []
    if holdout_frac > 0:
        names.append("holdout")
        weights.append(holdout_frac)
    share = (1.0 - holdout_frac) / folds
    for i in range(folds):
        names.append(f"fold_{i}")
        weights.append(share)
    return names, weights


def fold_roles(assignment_col, fold, folds):
    """Roles for one fold.

    Two layouts are supported. If the split file carries a 'benchmark' bin,
    that IS the test set for every fold and only validation rotates - the
    benchmark never moves, so cross-fold spread measures run-to-run variance
    rather than test-set composition. Otherwise the older ring layout applies:
    test = this fold, val = the next one round.
    """
    role = pd.Series("train", index=assignment_col.index)
    if (assignment_col == "benchmark").any():
        role[assignment_col == "benchmark"] = "test"
        role[assignment_col == f"fold_{fold}"] = "val"
        return role
    role[assignment_col == f"fold_{fold}"] = "test"
    role[assignment_col == f"fold_{(fold + 1) % folds}"] = "val"
    role[assignment_col == "holdout"] = "holdout"
    return role


def load_splits(path):
    return pd.read_csv(path)


def cap_split(df, limit, seed):
    """Deterministically shrink a split, preserving its populated/empty mix."""
    if not limit or len(df) <= limit:
        return df
    rng = np.random.default_rng(seed)
    frac = limit / len(df)
    parts = []
    for _, g in df.groupby("is_empty"):
        k = int(round(frac * len(g)))
        parts.append(g.iloc[rng.permutation(len(g))[:k]])
    out = pd.concat(parts)
    return out.sort_values("tile_id")

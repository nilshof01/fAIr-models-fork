"""Merge both review campaigns into one decisions table and publish it.

The verdicts are the part that exists nowhere else. The tidied datasets carry
only what survived filtering, so they cannot say which labels were REJECTED -
and the rejections are the evidence that a large share of the upstream labels
are unusable. Imagery is not duplicated here; every row keys into
hotosm/vhr-building-segmentation by tile_id.

Merge rule: union of both campaigns, deduplicated on tile_id, Supabase winning
a conflict as the later campaign. Nothing is collapsed - the original verdict
from each source is kept in its own column so the merge is reversible.
"""
from __future__ import annotations

import os
import argparse
import sqlite3
from pathlib import Path

import pandas as pd

LOCAL_DB = Path(os.environ.get("VHR_REVIEW_DB", DEFAULT_REVIEW_DB))
SUPABASE = "nilsho01/vhr-buildings-supabase-labels"
UPSTREAM = "hotosm/vhr-building-segmentation"


def build():
    from datasets import load_dataset

    con = sqlite3.connect(LOCAL_DB)
    loc = pd.read_sql("select tile_id, decision, ts from decisions", con)
    loc = loc.sort_values("ts").drop_duplicates("tile_id", keep="last")
    fixes = set(pd.read_sql("select tile_id from fixes", con).tile_id)
    loc = loc.rename(columns={"decision": "decision_local", "ts": "ts_local"})

    sb = load_dataset(SUPABASE, split="train")
    sb = sb.remove_columns([c for c in sb.column_names
                            if c not in ("tile_id", "dataset_row", "decision")]
                           ).to_pandas().drop_duplicates("tile_id")
    sb = sb.rename(columns={"decision": "decision_supabase"})

    df = pd.merge(sb, loc, on="tile_id", how="outer")
    df["source"] = ["both" if (a and b) else ("supabase" if a else "local")
                    for a, b in zip(df.decision_supabase.notna(),
                                    df.decision_local.notna())]
    # later campaign wins; a hand-drawn mask means the label was repaired, so
    # the chip is usable even though it was first flagged
    df["decision"] = df.decision_supabase.fillna(df.decision_local)
    df["has_hand_drawn_mask"] = df.tile_id.isin(fixes)
    df.loc[df.has_hand_drawn_mask & (df.decision != "keep"), "decision"] = "keep"
    df["usable"] = df.decision == "keep"

    up = load_dataset(UPSTREAM, split="train")
    meta = up.remove_columns([c for c in up.column_names if c not in
        ("tile_id", "num_buildings", "project_name", "country")]
        ).to_pandas().drop_duplicates("tile_id")
    df = df.merge(meta, on="tile_id", how="left")
    df["num_buildings"] = df.num_buildings.fillna(0).astype(int)
    df["has_buildings"] = df.num_buildings >= 1

    cols = ["tile_id", "dataset_row", "decision", "usable", "has_buildings",
            "num_buildings", "project_name", "country", "source",
            "decision_local", "decision_supabase", "has_hand_drawn_mask"]
    return df[cols].sort_values("tile_id").reset_index(drop=True)


CARD = """---
license: cc-by-4.0
task_categories: [image-segmentation]
tags: [remote-sensing, buildings, label-quality, data-curation]
---

# VHR Buildings — manual review decisions

Per-chip human verdicts on the labels of
[`hotosm/vhr-building-segmentation`](https://huggingface.co/datasets/hotosm/vhr-building-segmentation),
from two review campaigns by a single reviewer.

# local review-app SQLite; override with VHR_REVIEW_DB. Only needed to
# REBUILD the pool from raw decisions - the published HF datasets are
# the normal source and need no local database.
DEFAULT_REVIEW_DB = "data/review/decisions.sqlite"

**No imagery here.** Every row keys into the upstream dataset by `tile_id`;
this table adds only the judgement. The point is the *rejections*: derived
datasets carry what survived filtering and therefore cannot say which labels
were thrown out.

## Verdicts

{counts}

## Columns

| column | meaning |
|---|---|
| `tile_id` | key into `hotosm/vhr-building-segmentation` (split `train`) |
| `dataset_row` | row index in that split |
| `decision` | `keep`, `needs_fix` or `drop` — the resolved verdict |
| `usable` | `decision == "keep"` |
| `has_buildings`, `num_buildings` | from upstream metadata |
| `project_name`, `country` | upstream mapping campaign |
| `source` | `local`, `supabase` or `both` |
| `decision_local`, `decision_supabase` | the raw verdict from each campaign, so the merge is reversible |
| `has_hand_drawn_mask` | a corrected mask was drawn; the chip counts as usable |

## How it was built

Union of two campaigns, deduplicated on `tile_id`, Supabase winning a conflict
as the later one. Both raw verdicts are retained.

- **local** — Streamlit app, chips ordered by farthest-point sampling over
  DINOv2-small embeddings. 2026-07-29 → 2026-08-04.
- **supabase** — hosted app, chips ordered by descending building density.

## Caveats

**Not a random sample.** One campaign ordered chips by embedding diversity and
the other by building density; neither is uniform over the dataset, and dense
chips carry more label problems than sparse ones. The rejection rate here is
the rate *in what was reviewed*, not an estimate for the dataset as a whole.
A uniformly random sample would be needed for that.

**One reviewer, no second opinion**, so inter-annotator agreement is unknown.

**`needs_fix` and `drop` are both "not usable as-is"** and are not reliably
distinguished — treat them as one category.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nilsho01/vhr-buildings-review-decisions")
    ap.add_argument("--public", action="store_true",
                    help="default is private, matching the other repos")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    df = build()
    g = df.decision.value_counts()
    kept = df[df.usable]
    lines = [f"| verdict | chips | share |", "|---|---|---|"]
    for k in ("keep", "needs_fix", "drop"):
        lines.append(f"| `{k}` | {g.get(k,0):,} | {100*g.get(k,0)/len(df):.1f}% |")
    lines += [f"| **total reviewed** | **{len(df):,}** | |", "",
              f"Of the {len(kept):,} kept chips, {int(kept.has_buildings.sum()):,} "
              f"({100*kept.has_buildings.mean():.1f}%) contain buildings and "
              f"{int((~kept.has_buildings).sum()):,} are verified empty, "
              f"together holding {int(kept.num_buildings.sum()):,} buildings.",
              "",
              f"Rejected as unusable: **{g.get('needs_fix',0)+g.get('drop',0):,} "
              f"chips, {100*(g.get('needs_fix',0)+g.get('drop',0))/len(df):.1f}%**."]
    card = CARD.format(counts="\n".join(lines))

    print(df.head(4).to_string(), "\n")
    print("\n".join(lines))
    if a.dry_run:
        print("\nDRY RUN - nothing uploaded")
        return

    from datasets import Dataset
    from huggingface_hub import HfApi
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(a.repo, private=not a.public)
    HfApi().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=a.repo, repo_type="dataset")
    print(f"\npushed {len(df):,} rows -> https://huggingface.co/datasets/{a.repo}"
          f"  ({'public' if a.public else 'private'})")


if __name__ == "__main__":
    main()

"""Push the quality-filtered pool to HuggingFace.

Reads data/clean_pool/pool.csv (from build_clean_pool.py), fetches images +
masks from hotosm/vhr-building-segmentation by dataset_row, and pushes a
smaller clean dataset to HF.

Usage:
    python scripts/push_clean_pool.py \
        --pool-csv data/clean_pool/pool.csv \
        --repo nilsho01/vhr-buildings-clean-pool-v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, Features
from datasets import Image as HFImage
from datasets import Value, load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-csv", default="data/clean_pool/pool.csv")
    ap.add_argument("--repo", default="nilsho01/vhr-buildings-clean-pool-v1")
    ap.add_argument("--verified-source", default="nilsho01/vhr-buildings-verified-v1",
                    help="source for reviewed chips (has correct images + masks)")
    ap.add_argument("--unreviewed-source", default="hotosm/vhr-building-segmentation",
                    help="source for unreviewed chips (looked up by dataset_row position)")
    ap.add_argument("--private", action="store_true", default=True)
    a = ap.parse_args()

    pool = pd.read_csv(a.pool_csv)
    print(f"Pool: {len(pool):,} chips  ({pool.source.value_counts().to_dict()})")

    print(f"Loading verified source {a.verified_source} ...")
    ver = load_dataset(a.verified_source, split="train")
    ver_by_tile = {t: i for i, t in enumerate(ver["tile_id"])}
    print(f"  {len(ver):,} chips")

    print(f"Loading unreviewed source {a.unreviewed_source} ...")
    unrev = load_dataset(a.unreviewed_source, split="train")
    print(f"  {len(unrev):,} chips")

    rows = []
    n_verified = n_unreviewed = n_missing = 0
    for _, r in tqdm(pool.iterrows(), total=len(pool), desc="assembling"):
        tid = r["tile_id"]
        src_label = str(r.get("source", "reviewed"))

        if src_label == "reviewed":
            if tid not in ver_by_tile:
                n_missing += 1
                continue
            chip = ver[ver_by_tile[tid]]
            n_verified += 1
        else:
            if pd.isna(r["dataset_row"]):
                n_missing += 1
                continue
            dr = int(r["dataset_row"])
            if dr >= len(unrev):
                n_missing += 1
                continue
            chip = unrev[dr]
            n_unreviewed += 1

        rows.append({
            "tile_id":       tid,
            "dataset_row":   int(r["dataset_row"]) if not pd.isna(r["dataset_row"]) else -1,
            "image":         chip["image"],
            "mask":          chip["mask"],
            "project_name":  str(r.get("project_name", "")),
            "num_buildings": int(r.get("num_buildings", 0)),
            "country":       str(r.get("country", "")),
            "source":        src_label,
        })

    print(f"  from verified: {n_verified:,}  from unreviewed: {n_unreviewed:,}  "
          f"not found: {n_missing:,}")

    print(f"\nBuilding HF dataset from {len(rows):,} chips ...")
    ds = (Dataset.from_list(rows)
          .cast_column("image", HFImage())
          .cast_column("mask",  HFImage()))

    print(f"Pushing to {a.repo} ...")
    ds.push_to_hub(a.repo, private=a.private)
    print(f"Done — {len(ds):,} chips at https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()

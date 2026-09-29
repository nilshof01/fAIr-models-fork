"""One dataset holding every manually verified chip, with the correct mask.

The training pool was spread across three places - vhr-buildings-tidied-v1
(4,270 chips), the Supabase export (4,692) and the local review SQLite - and
none holds all 4,701 verified chips, so any split file drawn from the full
review inevitably references chips the trainer cannot load.

Mask provenance, in precedence order:
  hand_drawn  a corrected mask was drawn in the review app - the only right one
  reviewed    the image/mask pair as exported from the hosted app
  upstream    the original OSM mask. Valid here BECAUSE every chip is
              decision=keep, i.e. the reviewer inspected that mask and found it
              correct. Recorded per chip rather than assumed.
"""
from __future__ import annotations

import os
import argparse
import io
import sqlite3
from pathlib import Path

# local review-app SQLite; override with VHR_REVIEW_DB. Only needed to
# REBUILD the pool from raw decisions - the published HF datasets are
# the normal source and need no local database.
DEFAULT_REVIEW_DB = "data/review/decisions.sqlite"

LOCAL_DB = Path(os.environ.get("VHR_REVIEW_DB", DEFAULT_REVIEW_DB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nilsho01/vhr-buildings-verified-v1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from datasets import Dataset, Image as HFImage, load_dataset
    from PIL import Image as PILImage

    rev = load_dataset("nilsho01/vhr-buildings-review-decisions", split="train").to_pandas()
    kept = rev[rev.usable].reset_index(drop=True)
    print(f"verified kept chips: {len(kept):,}")

    sb = load_dataset("nilsho01/vhr-buildings-supabase-labels", split="train")
    sb_at = {t: i for i, t in enumerate(sb["tile_id"])}
    up = load_dataset("hotosm/vhr-building-segmentation", split="train")
    up_at = {t: i for i, t in enumerate(up["tile_id"])}
    drawn = {t: bytes(m) for t, m in
             sqlite3.connect(LOCAL_DB).execute("select tile_id, mask_png from fixes")}

    rows, missing = [], []
    for r in kept.itertuples():
        t = r.tile_id
        if t in sb_at:
            rec, src = sb[sb_at[t]], "reviewed"
        elif t in up_at:
            rec, src = up[up_at[t]], "upstream"
        else:
            missing.append(t)
            continue
        mask = rec["mask"]
        if t in drawn:
            mask = PILImage.open(io.BytesIO(drawn[t]))
            src = "hand_drawn"
        rows.append({"tile_id": t, "image": rec["image"], "mask": mask,
                     "num_buildings": int(r.num_buildings),
                     "country": r.country, "project_name": r.project_name,
                     "mask_source": src})
    if missing:
        print(f"WARNING: {len(missing)} verified chips have no image anywhere")

    counts = {}
    for x in rows:
        counts[x["mask_source"]] = counts.get(x["mask_source"], 0) + 1
    print(f"assembled {len(rows):,} chips | mask sources {counts}")
    if a.dry_run:
        return
    ds = (Dataset.from_list(rows).cast_column("image", HFImage())
          .cast_column("mask", HFImage()))
    ds.push_to_hub(a.repo, private=True)
    print(f"pushed -> https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()

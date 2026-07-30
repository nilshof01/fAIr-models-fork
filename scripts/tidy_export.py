"""Export vhr-building-segmentation chips to disk for label review.

Writes <out>/<split>/images/<key>.png and <out>/<split>/masks/<key>.png with
key = <tile_id>__<project_id>. tile_id alone is NOT unique (108 duplicate
rows in train, from overlapping projects), so the composite key is the one
identifier the whole tidy toolchain uses. Masks are 0/255 PNG.

Review flow: never delete or edit files in this export. Record decisions in
a manifest directory instead (consumed by build_tidied_dataset.py):
  <manifest>/drops_<split>.txt              one key per line, row removed
  <manifest>/masks_fixed/<split>/<key>.png  replacement mask (nonzero=building)
"""
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

SRC = "hotosm/vhr-building-segmentation"
HF_SPLIT = {"train": "train", "validation": "validation", "test": "test"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=list(HF_SPLIT))
    p.add_argument("--out", default="data/tidy_workspace")
    p.add_argument("--keys", default=None,
                   help="Optional file with one key per line; export only these")
    args = p.parse_args()

    ds = load_dataset(SRC, split=HF_SPLIT[args.split])
    keys = [f"{t}__{p_}" for t, p_ in zip(ds["tile_id"], ds["project_id"])]
    wanted = set(Path(args.keys).read_text().split()) if args.keys else None

    img_dir = Path(args.out) / args.split / "images"
    mask_dir = Path(args.out) / args.split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for i, key in enumerate(keys):
        if wanted is not None and key not in wanted:
            continue
        row = ds[i]
        row["image"].convert("RGB").save(img_dir / f"{key}.png")
        m = (np.asarray(row["mask"].convert("L")) > 0).astype(np.uint8) * 255
        Image.fromarray(m).save(mask_dir / f"{key}.png")
        n += 1
        if n % 2000 == 0:
            print(f"{n} exported", flush=True)
    print(f"done: {n} chips -> {Path(args.out) / args.split}")


if __name__ == "__main__":
    main()

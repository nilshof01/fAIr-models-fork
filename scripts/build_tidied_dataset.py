"""Build a tidied variant of hotosm/vhr-building-segmentation from a manifest.

Manifest directory (every part optional; keys = <tile_id>__<project_id>,
see tidy_export.py):
  <manifest>/drops_train.txt / drops_validation.txt / drops_test.txt
  <manifest>/masks_fixed/<split>/<key>.png   replacement masks (nonzero=building)

By default also drops the train rows whose tile_id appears in the
validation split (5 exact-tile leaks in the source dataset; --keep-leaked
disables). Untouched rows and the full schema pass through unchanged, so
the result is a drop-in dataset_repo for both base trainers. The source
norm_stats.json is carried over.

Output: --save-dir (datasets.load_from_disk) and/or --push-to (HF dataset
repo, created private). Training via HotBuildingDataModule requires a hub
repo id, so --push-to is the path that needs no further code changes.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

SRC = "hotosm/vhr-building-segmentation"
SPLITS = ["train", "validation", "test"]


class SplitTidier:
    def __init__(self, split, manifest, leaked_tile_ids):
        self.split = split
        drops_file = manifest / f"drops_{split}.txt"
        self.drops = set(drops_file.read_text().split()) if drops_file.exists() else set()
        fixed_dir = manifest / "masks_fixed" / split
        self.fixed = {f.stem: f for f in fixed_dir.glob("*.png")} if fixed_dir.is_dir() else {}
        self.leaked = leaked_tile_ids if split == "train" else set()
        self.kept_keys = []

    def run(self, ds):
        keys = [f"{t}__{p}" for t, p in zip(ds["tile_id"], ds["project_id"])]
        tile_ids = ds["tile_id"]
        keep = [i for i, k in enumerate(keys)
                if k not in self.drops and tile_ids[i] not in self.leaked]
        n_leak = sum(1 for t in tile_ids if t in self.leaked)
        ds = ds.select(keep)
        self.kept_keys = [keys[i] for i in keep]
        n_fixed = 0
        if self.fixed:
            ds = ds.map(self._fix_mask, with_indices=True)
            n_fixed = sum(1 for k in self.kept_keys if k in self.fixed)
        print(f"{self.split}: kept {len(ds)}  dropped {len(keys) - len(ds)} "
              f"(manifest {len(self.drops & set(keys))}, leaked {n_leak})  "
              f"masks fixed {n_fixed}")
        unused = self.drops - set(keys)
        if unused:
            print(f"  WARNING: {len(unused)} drop keys not found in {self.split}: "
                  f"{sorted(unused)[:5]}...")
        return ds

    def _fix_mask(self, ex, idx):
        f = self.fixed.get(self.kept_keys[idx])
        if f is not None:
            m = (np.asarray(Image.open(f).convert("L")) > 0).astype(np.uint8) * 255
            ex["mask"] = Image.fromarray(m)
        return ex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--push-to", default=None,
                   help="HF dataset repo id, e.g. nilsho01/vhr-building-segmentation-tidied")
    p.add_argument("--keep-leaked", action="store_true")
    p.add_argument("--public", action="store_true",
                   help="Push as PUBLIC (default: private).")
    args = p.parse_args()
    if not (args.save_dir or args.push_to):
        p.error("need --save-dir and/or --push-to")
    manifest = Path(args.manifest)

    leaked = set()
    if not args.keep_leaked:
        val_ids = set(load_dataset(SRC, split="validation")["tile_id"])
        leaked = val_ids & set(load_dataset(SRC, split="train")["tile_id"])
        print(f"train tile_ids also in validation (dropped from train): {len(leaked)}")

    out = DatasetDict()
    for split in SPLITS:
        tidier = SplitTidier(split, manifest, leaked)
        out[split] = tidier.run(load_dataset(SRC, split=split))

    stats_path = hf_hub_download(SRC, "norm_stats.json", repo_type="dataset")
    if args.save_dir:
        out.save_to_disk(args.save_dir)
        shutil.copy(stats_path, Path(args.save_dir) / "norm_stats.json")
        print(f"saved: {args.save_dir}")
    if args.push_to:
        out.push_to_hub(args.push_to, private=not args.public)
        HfApi().upload_file(path_or_fileobj=stats_path, path_in_repo="norm_stats.json",
                            repo_id=args.push_to, repo_type="dataset")
        print(f"pushed: https://huggingface.co/datasets/{args.push_to}")


if __name__ == "__main__":
    main()

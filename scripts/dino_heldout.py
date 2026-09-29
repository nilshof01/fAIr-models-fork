"""Reconstruct which chips the shipped DINOv3 model never trained on.

models/dinov3s_buildings/stac-item.json documents the split exactly - spatial
blocks of 4x4 OAM tiles, seed 42, 20% held out - and dinov3_hot/dataset.py
implements it deterministically. So the val side is recoverable, and any chip
in it is clean for that model.

That matters for comparing against a model trained on a hand-filtered subset
of the same dataset: chips in DINO's val AND outside your own train/val are
clean for both, which is the only honest ground to compare on.

Uncertainty worth stating: the card says '~37k chips' while the upstream train
split holds 57,890 (38,113 with at least one building). --pool populated
assumes it trained on populated chips only, which matches that number; --pool
all assumes otherwise. The two give different val sets, so the assumption is
exposed rather than buried.

    python scripts/dino_heldout.py --pool populated --out data/dino_val_tiles.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

OAM_TILE_RE = re.compile(r"OAM-(\d+)-(\d+)-(\d+)")


def spatial_split(chip_names, val_ratio, seed, block_size=4):
    """Verbatim port of dinov3_hot.dataset.spatial_split."""
    matched, unmatched = {}, []
    for name in chip_names:
        m = OAM_TILE_RE.match(name)
        if m is None:
            unmatched.append(name)
            continue
        matched.setdefault((int(m.group(1)) // block_size,
                            int(m.group(2)) // block_size), []).append(name)
    rng = np.random.default_rng(seed)
    blocks = sorted(matched.keys())
    rng.shuffle(blocks)
    n_total = sum(len(v) for v in matched.values()) + len(unmatched)
    n_val_target = max(1, int(n_total * val_ratio))
    val, train = [], []
    for b in blocks:
        (val if len(val) < n_val_target else train).extend(matched[b])
    if unmatched:
        leftover = sorted(unmatched)
        rng.shuffle(leftover)
        rem = max(0, n_val_target - len(val))
        val.extend(leftover[:rem]); train.extend(leftover[rem:])
    return sorted(train), sorted(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="populated", choices=["populated", "all"])
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--out", default="data/dino_val_tiles.txt")
    ap.add_argument("--mine", default=None,
                    help="a tile list of your own (train or test) to intersect")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("hotosm/vhr-building-segmentation", split="train")
    meta = ds.remove_columns([c for c in ds.column_names
                              if c not in ("tile_id", "num_buildings")]).to_pandas()
    pool = meta[meta.num_buildings >= 1] if a.pool == "populated" else meta
    names = sorted(pool.tile_id.unique())
    print(f"pool '{a.pool}': {len(names):,} chips "
          f"(card says the model saw ~37k)")

    train, val = spatial_split(names, a.val_ratio, a.seed, a.block_size)
    print(f"reconstructed split: train {len(train):,} | val {len(val):,} "
          f"({100*len(val)/max(len(names),1):.1f}%)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(val))
    print(f"wrote {len(val):,} tile ids DINO never trained on -> {a.out}")

    if a.mine:
        mine = set(Path(a.mine).read_text().split())
        clean = mine & set(val)
        print(f"\nyour list {Path(a.mine).name}: {len(mine):,} chips")
        print(f"  of those, clean for DINO too: {len(clean):,} "
              f"({100*len(clean)/max(len(mine),1):.1f}%)")
        p = Path(a.out).with_name("clean_for_both.txt")
        p.write_text("\n".join(sorted(clean)))
        print(f"  wrote -> {p}")


if __name__ == "__main__":
    main()

"""A fixed, manually verified benchmark for building instance segmentation.

fAIr has no such thing: every published number is measured against labels of
unknown quality, on splits that differ between models. This is 400 chips the
reviewer checked by hand, stratified so the axes that actually drive failure
are each represented, and frozen so future models are comparable.

Stratified on DENSITY (the dominant axis) x REGION (the second), 100 chips per
density band:

  empty    the only way to measure false positives on bare ground
  1-9      sparse, where small buildings get missed
  10-49    moderate
  50-149   dense, where instance merging dominates

150+ is omitted deliberately: only 63 such chips exist and all are Ghana, so
"very dense" and "Ghana" would be inseparable in any result.

Selection within a cell: one chip per 4x4 spatial block, so no two test chips
are direct neighbours, and preference for chips lying in BOTH candidate
reconstructions of the shipped DINOv3 model's held-out validation split - which
costs nothing and keeps that comparison open, without the benchmark depending
on it.

Two details that a first attempt got wrong. Bands are filled SCARCEST FIRST:
dense chips cluster into few blocks (474 chips across 194 4x4 blocks against
2,124 empty chips across 1,133), so filling the common bands first leaves the
dense band nothing. And the block is 4x4, not 16x16 - at 16x16 the dense band
has only 82 distinct blocks and cannot reach 100 chips at all.
"""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# local review-app SQLite; override with VHR_REVIEW_DB. Only needed to
# REBUILD the pool from raw decisions - the published HF datasets are
# the normal source and need no local database.
DEFAULT_REVIEW_DB = "data/review/decisions.sqlite"

BANDS = [("empty", 0, 0), ("1-9", 1, 9), ("10-49", 10, 49), ("50-149", 50, 149)]
AFRICA = ["Sierra Leone", "Malawi", "Ghana", "Mozambique", "Tanzania",
          "Uganda", "Liberia", "Kenya", "Nigeria"]


def region(c):
    return "Myanmar" if c == "Myanmar" else ("Africa" if c in AFRICA else "Other")


def dino_clean(tile_ids):
    """Chips outside the shipped model's val split under BOTH plausible pools.

    Which chips it trained on is not recorded; two readings of its card both
    land near '~37k' and give val sets overlapping only 22%. Taking chips
    outside neither, i.e. in the intersection of the two held-out sets, is
    clean under either reading.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dino_heldout import spatial_split
    from datasets import load_dataset
    ds = load_dataset("hotosm/vhr-building-segmentation", split="train")
    m = ds.remove_columns([c for c in ds.column_names
                           if c not in ("tile_id", "num_buildings")]).to_pandas()
    _, a = spatial_split(sorted(m[m.num_buildings >= 1].tile_id.unique()), 0.2, 42, 4)
    _, b = spatial_split(sorted(m.tile_id.unique()), 0.2, 42, 4)
    return set(a) & set(b)


def pick(cell, n, blocks_used, prefer, rng):
    """One chip per spatial block; preferred chips first, then the rest."""
    cell = cell.copy()
    cell["pref"] = cell.tile_id.isin(prefer)
    cell = cell.iloc[rng.permutation(len(cell))]
    cell = cell.sort_values("pref", ascending=False, kind="stable")
    out = []
    for _, r in cell.iterrows():
        if len(out) >= n:
            break
        if r.block in blocks_used:
            continue
        blocks_used.add(r.block)
        out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=100)
    ap.add_argument("--per-band-empty", type=int, default=None,
                    help="override chip count for the empty band only "
                         "(default: same as --per-band)")
    ap.add_argument("--block", type=int, default=4,
                    help="tiles per side of the separation block")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--repo", default="nilsho01/vhr-buildings-benchmark-v1")
    ap.add_argument("--out", default="data/benchmark_v1")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    a.per_band_empty = a.per_band_empty if a.per_band_empty is not None else a.per_band

    from datasets import load_dataset
    rev = load_dataset("nilsho01/vhr-buildings-review-decisions", split="train").to_pandas()
    k = rev[rev.usable].copy()
    blk = k.tile_id.str.extract(r"OAM-(\d+)-(\d+)")
    k["block"] = (k.project_name.astype(str) + "|"
                  + (blk[0].astype(float) // a.block).astype("Int64").astype(str) + "_"
                  + (blk[1].astype(float) // a.block).astype("Int64").astype(str))
    k["region"] = k.country.map(region)

    prefer = dino_clean(k.tile_id)
    print(f"chips clean for the shipped DINO under both readings: "
          f"{len(set(k.tile_id) & prefer):,} of {len(k):,}\n")

    rng = np.random.default_rng(a.seed)
    blocks_used, chosen = set(), []
    # scarcest band first: dense chips cluster into few blocks, so filling the
    # common bands first would leave the dense band with nothing to pick from
    order = sorted(BANDS, key=lambda b: len(k[k.num_buildings.between(b[1], b[2])]))
    for name, lo, hi in order:
        cell = k[k.num_buildings.between(lo, hi)]
        n_band = a.per_band_empty if name == "empty" else a.per_band
        # region quota, capped by what exists, remainder filled from the largest
        want = {r: n_band // 3 for r in ("Myanmar", "Africa", "Other")}
        got = []
        for r, q in want.items():
            sub = cell[cell.region == r]
            got.append(pick(sub, min(q, len(sub)), blocks_used, prefer, rng))
        have = pd.concat(got) if got else pd.DataFrame()
        short = n_band - len(have)
        if short > 0:
            rest = cell[~cell.tile_id.isin(have.tile_id)]
            have = pd.concat([have, pick(rest, short, blocks_used, prefer, rng)])
        have = have.assign(band=name)
        chosen.append(have)
        by_r = have.region.value_counts().to_dict()
        print(f"{name:8s} {len(have):4d} chips  {by_r}")

    bench = pd.concat(chosen).reset_index(drop=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cols = ["tile_id", "dataset_row", "band", "region", "country",
            "project_name", "num_buildings", "block"]
    bench[cols].to_csv(out / "benchmark_chips.csv", index=False)
    (out / "benchmark_tiles.txt").write_text("\n".join(bench.tile_id))

    meta = {
        "name": "vhr-buildings-benchmark-v1", "chips": len(bench),
        "per_band": a.per_band, "seed": a.seed,
        "bands": {n: int((bench.band == n).sum()) for n, _, _ in BANDS},
        "regions": bench.region.value_counts().to_dict(),
        "countries": int(bench.country.nunique()),
        "buildings": int(bench.num_buildings.sum()),
        "dino_clean_chips": int(bench.tile_id.isin(prefer).sum()),
        "blocks": int(bench.block.nunique()),
    }
    (out / "benchmark.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(bench)} chips | {int(bench.num_buildings.sum()):,} buildings | "
          f"{bench.country.nunique()} countries | "
          f"{int(bench.tile_id.isin(prefer).sum())} clean for DINO under both readings")
    print(f"one chip per spatial block: {bench.block.nunique() == len(bench)}")
    print(f"\nwrote {out}/")
    if a.push:
        push(bench, a.repo, meta)


def push(bench, repo, meta):
    """Assemble images and masks, preferring a reviewed copy.

    Every chip here is decision=keep, which means the ORIGINAL OSM mask was
    inspected and found correct - so the upstream mask is the verified label
    for them. Nine carry a hand-drawn correction instead, and for those the
    reviewed copy is the only right one. Source is recorded per chip rather
    than assumed.
    """
    import sqlite3
    from datasets import Dataset, Features, Image as HFImage, Value, load_dataset

    sb = load_dataset("nilsho01/vhr-buildings-supabase-labels", split="train")
    sb_at = {t: i for i, t in enumerate(sb["tile_id"])}
    up = load_dataset("hotosm/vhr-building-segmentation", split="train")
    up_at = {t: i for i, t in enumerate(up["tile_id"])}

    con = sqlite3.connect(os.environ.get("VHR_REVIEW_DB", DEFAULT_REVIEW_DB))
    drawn = {t: bytes(m) for t, m in
             con.execute("select tile_id, mask_png from fixes")}

    rows = []
    for _, r in bench.iterrows():
        t = r.tile_id
        if t in sb_at:
            rec, src_name = sb[sb_at[t]], "reviewed"
            img, mask = rec["image"], rec["mask"]
        elif t in up_at:
            rec, src_name = up[up_at[t]], "upstream"
            img, mask = rec["image"], rec["mask"]
        else:
            print(f"  skipping {t}: no image anywhere")
            continue
        if t in drawn:                       # hand-drawn correction wins
            import io
            from PIL import Image as PILImage
            mask = PILImage.open(io.BytesIO(drawn[t]))
            src_name = "hand_drawn"
        rows.append({"tile_id": t, "image": img, "mask": mask,
                     "band": r.band, "region": r.region, "country": r.country,
                     "project_name": r.project_name,
                     "num_buildings": int(r.num_buildings),
                     "mask_source": src_name})

    ds = Dataset.from_list(rows).cast_column("image", HFImage()).cast_column("mask", HFImage())
    counts = {}
    for x in rows:
        counts[x["mask_source"]] = counts.get(x["mask_source"], 0) + 1
    print(f"  mask sources: {counts}")
    ds.push_to_hub(repo, private=True)
    print(f"pushed {len(ds)} chips -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()

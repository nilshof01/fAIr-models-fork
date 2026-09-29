"""Every fold of both architectures on the fixed 400-chip benchmark, one protocol.

The comparison it replaces was one UNet fold against one ViT fold at different
mask thresholds. The ViT gained ~5 PQ from threshold tuning and the UNet had
never been swept, so the gap was partly an operating-point artefact.

Three columns, because "tune the threshold" and "fix the threshold" are both
defensible and they are not the same number:

  fixed       threshold 0.5 for everything. No tuning anywhere, so no way for
              one architecture to be quietly advantaged. The conservative read.
  val-picked  threshold chosen on each run's OWN validation split, then applied
              unchanged to the benchmark. Honest - selection never sees the test
              chips - and it is what a deployed model would actually do. This is
              the headline.
  oracle      best achievable on the benchmark itself. NOT an operating point:
              it is selected on the thing it is scored on. Reported only as a
              ceiling, and the gap to val-picked measures how transferable the
              threshold choice is.

Core threshold is held at 0.5 throughout - sweeping it was already measured flat
(PQ 27.10/27.21/27.13 at 0.5/0.7/0.85), so it is not a free parameter here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from instance_metrics import InstanceScores
from pool_data import PoolDataSource
from instance_decode import MIN_PX, grow

GRID = np.round(np.arange(0.25, 0.91, 0.05), 2)   # wide: the ViT pins high
CORE_THR = 0.5


class Run:
    """One checkpoint, scored over the threshold grid on a given role."""

    def __init__(self, path, dev):
        self.path = Path(path)
        self.name = self.path.name
        m = re.search(r"_f(\d+)(?:_|$)", self.name)   # cv_vits_f0, cv_f0_effb4_dist3
        if not m:
            raise SystemExit(f"cannot read a fold number from {self.name}")
        self.fold = int(m.group(1))
        from vit_adapter import VitRun, is_vit_run
        self.is_vit = is_vit_run(self.path)
        if self.is_vit:
            v = VitRun(self.path, dev)
            self.predict, self.repo = v.predict, v.cfg.dataset_repo
            self.arch = "ViT-S"
        else:
            from eval_features import NORM_MEAN, NORM_STD, build_model
            cfg = json.loads((self.path / "config.json").read_text())
            model = build_model(cfg, self.path / "best.pth", dev).eval().to(dev)
            self.repo = cfg["args"]["dataset"]
            self.arch = cfg["args"].get("arch", "unet")

            def predict(rgb, _m=model, _d=dev):
                x = (rgb.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
                with torch.no_grad():
                    return torch.sigmoid(_m(torch.from_numpy(x).permute(2, 0, 1)[None]
                                            .to(_d))[0]).cpu().numpy()
            self.predict = predict

    def sweep(self, ds, desc):
        """PQ/SQ/RQ at every threshold in GRID, one forward pass per chip.

        Empty chips (no GT buildings) are excluded from PQ but tracked
        separately: fp_empty_inst = total predicted instances on empty chips,
        fp_empty_chips = fraction of empty chips with >= 1 prediction.
        """
        acc = {float(t): InstanceScores(min_px=MIN_PX) for t in GRID}
        # per threshold: [n_empty_chips, n_empty_with_pred, n_empty_inst]
        empty = {float(t): [0, 0, 0] for t in GRID}
        for i in tqdm(range(len(ds)), desc=desc, leave=False):
            r = ds[i]
            gt = np.asarray(r["mask"].convert("L")) > 127
            prob = self.predict(np.asarray(r["image"].convert("RGB")))
            if not gt.any():
                for t in GRID:
                    pm = prob[0] > t
                    seeds = ndimage.label((prob[1] > CORE_THR) & pm)[0] \
                        if prob.shape[0] > 1 else ndimage.label(pm)[0]
                    n_pred = len(np.unique(grow(seeds, pm))) - 1  # exclude bg
                    e = empty[float(t)]
                    e[0] += 1
                    e[1] += int(n_pred > 0)
                    e[2] += n_pred
                continue
            gl = ndimage.label(gt)[0]
            for t in GRID:
                pm = prob[0] > t
                seeds = ndimage.label((prob[1] > CORE_THR) & pm)[0] \
                    if prob.shape[0] > 1 else ndimage.label(pm)[0]
                acc[float(t)].add(grow(seeds, pm), gl, prob[0])
        out = {}
        for t, s in acc.items():
            p = s.panoptic(0.5)
            n_e, with_pred, n_inst = empty[t]
            out[t] = dict(PQ=100 * p["PQ"], SQ=100 * p["SQ"], RQ=100 * p["RQ"],
                          det=100 * p["tp"] / max(s.n_gt, 1),
                          pred_true=100 * s.n_pred / max(s.n_gt, 1),
                          fp_empty_chips=100 * with_pred / max(n_e, 1),
                          fp_empty_inst=n_inst,
                          n_empty=n_e)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split-file", default="data/cv_folds/cv_folds.csv")
    ap.add_argument("--out-dir", default="data/review_charts")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset as _lds
    _img_src_cache = {}

    def _attach_images(split, img_repo="hotosm/vhr-building-segmentation"):
        """If split has no image column, fetch images by dataset_row."""
        if "image" in split.column_names:
            return split
        if img_repo not in _img_src_cache:
            print(f"  loading images from {img_repo} ...")
            _img_src_cache[img_repo] = _lds(img_repo, split="train")
        src_img = _img_src_cache[img_repo]
        split = split.filter(lambda x: x["dataset_row"] is not None)
        rows = [int(r) for r in split["dataset_row"]]
        return src_img.select(rows)

    BENCH_REPO = "nilsho01/vhr-buildings-benchmark-v1"
    print(f"Loading benchmark from {BENCH_REPO} ...")
    _img_src_cache[BENCH_REPO] = _lds(BENCH_REPO, split="train")

    results = []
    for rp in a.runs:
        r = Run(rp, dev)
        src = PoolDataSource(r.repo, a.split_file, r.fold)
        val  = _attach_images(src.hf("val"))
        # Always use the full fixed benchmark regardless of what is in the pool
        test = _img_src_cache[BENCH_REPO]
        print(f"\n{r.name}  [{r.arch}, fold {r.fold}]  "
              f"val {len(val)} chips, benchmark {len(test)} chips")
        sv = r.sweep(val, "val")
        st = r.sweep(test, "test")
        picked = max(sv, key=lambda t: sv[t]["PQ"])
        oracle = max(st, key=lambda t: st[t]["PQ"])
        results.append(dict(run=r.name, arch=r.arch, fold=r.fold,
                            val=sv, test=st, thr_val_picked=picked,
                            thr_oracle=oracle, n_val=len(val), n_test=len(test)))
        print(f"  fixed 0.50  PQ {st[0.5]['PQ']:5.2f}  SQ {st[0.5]['SQ']:5.2f}  "
              f"RQ {st[0.5]['RQ']:5.2f}")
        print(f"  val-picked {picked:.2f}  PQ {st[picked]['PQ']:5.2f}  "
              f"SQ {st[picked]['SQ']:5.2f}  RQ {st[picked]['RQ']:5.2f}")
        print(f"  oracle     {oracle:.2f}  PQ {st[oracle]['PQ']:5.2f}  "
              f"(ceiling, selected on the test set)")

    (out / "benchmark_folds.json").write_text(json.dumps(
        dict(grid=[float(t) for t in GRID], core_thr=CORE_THR,
             split_file=a.split_file, runs=results), indent=1))
    print(f"\nwrote {out/'benchmark_folds.json'}")
    report(results, out)


def report(results, out):
    """Mean +/- spread per architecture under each protocol."""
    print(f"\n{'='*78}\nBENCHMARK, 400 fixed chips, all folds\n{'='*78}")
    for prot in ("fixed", "val-picked", "oracle"):
        print(f"\n{prot}")
        print(f"{'arch':8s} {'n':>3} {'thr':>10} {'PQ':>14} {'SQ':>14} {'RQ':>14}")
        for arch in sorted({r["arch"] for r in results}):
            rs = [r for r in results if r["arch"] == arch]
            thr = [0.5 if prot == "fixed" else
                   r["thr_val_picked"] if prot == "val-picked" else r["thr_oracle"]
                   for r in rs]
            v = {k: np.array([r["test"][t][k] for r, t in zip(rs, thr)])
                 for k in ("PQ", "SQ", "RQ")}
            ts = f"{thr[0]:.2f}" if len(set(thr)) == 1 else \
                 f"{min(thr):.2f}-{max(thr):.2f}"
            print(f"{arch:8s} {len(rs):3d} {ts:>10} " + " ".join(
                f"{v[k].mean():6.2f}±{v[k].std(ddof=1) if len(rs)>1 else 0:<5.2f}"
                for k in ("PQ", "SQ", "RQ")))
    print(f"\nper fold, val-picked\n{'run':22s} {'thr':>5} {'PQ':>7} {'SQ':>7} "
          f"{'RQ':>7} {'det@.5':>8} {'pred/true':>10} {'FP@empty':>10}")
    for r in sorted(results, key=lambda x: (x["arch"], x["fold"])):
        t = r["thr_val_picked"]; m = r["test"][t]
        print(f"{r['run']:22s} {t:5.2f} {m['PQ']:7.2f} {m['SQ']:7.2f} "
              f"{m['RQ']:7.2f} {m['det']:7.1f}% {m['pred_true']:9.0f}% "
              f"{m.get('fp_empty_chips', float('nan')):9.1f}%")
    chart(results, out)


def chart(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"ViT-S": "#2166ac", "smp_unet": "#d6604d"}   # blue/orange: CVD-safe pair
    archs = sorted({r["arch"] for r in results})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    for a_ in archs:
        rs = sorted([r for r in results if r["arch"] == a_], key=lambda x: x["fold"])
        c = C.get(a_, "#666666")
        x = [r["fold"] for r in rs]
        y = [r["test"][r["thr_val_picked"]]["PQ"] for r in rs]
        ax[0].plot(x, y, "o", color=c, ms=9, label=a_)
        ax[0].axhline(np.mean(y), color=c, lw=2, ls="--", alpha=.55)
        ax[0].annotate(f"{np.mean(y):.1f}", (max(x) + .15, np.mean(y)),
                       color=c, va="center", fontsize=10, weight="bold")
        g = [float(t) for t in sorted(rs[0]["test"])]
        m = np.array([[r["test"][t]["PQ"] for t in g] for r in rs])
        ax[1].plot(g, m.mean(0), "-", color=c, lw=2, label=a_)
        if len(rs) > 1:
            ax[1].fill_between(g, m.mean(0) - m.std(0), m.mean(0) + m.std(0),
                               color=c, alpha=.16, lw=0)
        ax[2].scatter([r["test"][r["thr_val_picked"]]["RQ"] for r in rs],
                      [r["test"][r["thr_val_picked"]]["SQ"] for r in rs],
                      color=c, s=80, label=a_)

    ax[0].set(xlabel="fold", ylabel="PQ", title="PQ per fold (val-picked threshold)")
    ax[0].set_xticks(sorted({r["fold"] for r in results}))
    ax[1].set(xlabel="mask threshold", ylabel="PQ",
              title="threshold sensitivity (mean ± 1 sd)")
    ax[2].set(xlabel="RQ (object F1)", ylabel="SQ (mask IoU of matches)",
              title="what each architecture is limited by")
    for b in ax:
        b.grid(alpha=.25, lw=.6); b.set_axisbelow(True)
        b.legend(frameon=False, fontsize=9)
        for side in ("top", "right"):
            b.spines[side].set_visible(False)
    fig.suptitle("400-chip fixed benchmark — all folds, one protocol", y=1.0,
                 fontsize=12)
    fig.tight_layout()
    p = out / "4_benchmark_folds.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

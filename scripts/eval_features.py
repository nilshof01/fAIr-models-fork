"""Where does a model fail, and what property of the chip explains it?

Scores every chip of one split, pairs the score with descriptors of the chip
(effective resolution, exposure, colour, label geometry), and reports
performance sliced by each descriptor. The point is to turn "the model is at
0.63" into "the model is at 0.45 on the blurriest fifth of chips and 0.72 on
the sharpest".

Reads a run directory written by train.py, so the architecture and
the split come from its config.json rather than being retyped:

    python scripts/eval_features.py --run data/runs/cv5_f0
    python scripts/eval_features.py --run data/runs/cv5_f0 --role holdout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

import segmentation_models_pytorch as smp
from image_features import FEATURES, ChipFeatures
from instance_metrics import InstanceScores

ROOT = Path(__file__).resolve().parent.parent
NORM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_INSTANCE_PX = 30


class Scorer:
    """Per-chip pixel scores, instance matching, and boundary-vs-interior IoU.

    A 3-channel checkpoint (mask, core, contour) is decoded by watershed from
    the predicted core; anything else falls back to connected components on the
    mask. Pixel metrics always describe the mask channel either way, so the two
    remain directly comparable.
    """

    def __init__(self, model, device, threshold, decoder=None, raw_from=None):
        self.m = model.eval().to(device)
        self.dev = device
        self.thr = threshold
        self.decoder = decoder
        self.raw_from = raw_from

    def predict(self, rgb):
        """Returns (C, H, W) sigmoid probabilities; channel 0 is the mask."""
        x = (rgb.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.dev)
        with torch.no_grad():
            lg = self.m(x)[0]
            if self.raw_from is None:
                return torch.sigmoid(lg).cpu().numpy()
            # a flow field is a vector, not a probability: squashing it to
            # 0..1 would destroy the sign and the decoder would integrate
            # garbage. Only the mask channel gets a sigmoid.
            out = torch.sigmoid(lg).cpu().numpy()
            out[self.raw_from:] = lg[self.raw_from:].cpu().numpy()
            return out

    @staticmethod
    def classify_fp(pred, gt, halo=4):
        """Split false positives into populations that mean different things.

        Eyeballing the crops shows FP blobs are not one thing:
          rim      a collar around a building the model DID find - the label
                   polygon and the predicted extent disagree by a few pixels.
                   A localisation wobble, usually a labelling artefact, and it
                   inflates FP counts out of all proportion to its importance.
          bridge   a blob touching two or more separate labelled buildings:
                   this IS the instance-merging failure, not a spurious find.
          isolated no label within `halo` px - either a genuine hallucination
                   or an unlabelled building. These are the ones worth
                   adjudicating against Google/Microsoft.
        """
        near = ndimage.binary_dilation(gt, iterations=halo)
        gl, gn = ndimage.label(gt)
        lab, n = ndimage.label(pred & ~gt)
        out = {"rim": 0, "bridge": 0, "isolated": 0,
               "rim_px": 0, "bridge_px": 0, "isolated_px": 0}
        for i in range(1, n + 1):
            m = lab == i
            a = int(m.sum())
            if a < MIN_INSTANCE_PX:
                continue
            touched = set(np.unique(gl[ndimage.binary_dilation(m, iterations=halo)])) - {0}
            kind = "isolated" if not touched else ("bridge" if len(touched) > 1 else "rim")
            out[kind] += 1
            out[f"{kind}_px"] += a
        return out

    @staticmethod
    def _instances(binary):
        lab, n = ndimage.label(binary)
        keep = [i for i in range(1, n + 1) if (lab == i).sum() >= MIN_INSTANCE_PX]
        return lab, keep

    def _pred_instances(self, prob):
        """Decode with whatever head this checkpoint has, else blobs."""
        if self.decoder is not None and prob.shape[0] >= 2:
            lab = self.decoder(prob)
            return lab, [i for i in np.unique(lab) if i > 0]
        return self._instances(prob[0] > self.thr)

    def instance_match(self, pred, gt, ious=(0.5, 0.75, 0.8), pred_lab=None):
        """Detection recall at several IoU thresholds, plus merges and splits.

        Reporting at one threshold hides the distinction the crops make
        obvious: a building found with a slightly wrong outline and a building
        missed entirely are both "not detected" at IoU 0.8 but only one is a
        real failure. Quoting 0.5 and 0.8 together separates them.
        """
        pl, pk = pred_lab if pred_lab is not None else self._instances(pred)
        gl, gk = self._instances(gt)
        if not gk:
            return {"gt_instances": 0, "merged": 0, "split": 0,
                    "pred_instances": len(pk),
                    **{f"detected_{t}": 0 for t in ious}}
        detected = {t: 0 for t in ious}
        merged = split = 0
        for g in gk:
            gm = gl == g
            best = max((((pl == p) & gm).sum() / ((pl == p) | gm).sum()
                        for p in pk if (pl == p)[gm].sum() > 0), default=0.0)
            for t in ious:
                if best >= t:
                    detected[t] += 1
            hits = [p for p in pk if (pl == p)[gm].sum() >= 0.5 * gm.sum()]
            if len(hits) >= 2:
                split += 1
        for p in pk:
            pm = pl == p
            covered = sum(1 for g in gk if (gl == g)[pm].sum() >= 0.5 * (gl == g).sum())
            if covered >= 2:
                merged += 1
        return {"gt_instances": len(gk), "merged": merged, "split": split,
                "pred_instances": len(pk),
                **{f"detected_{t}": detected[t] for t in ious}}

    def score(self, rgb, gt, ignore_band=0):
        """`ignore_band` excludes a k-pixel collar around every label boundary
        from the pixel metrics. Preferred over eroding the labels: shrinking
        the ground truth biases it, while excluding the band simply declines to
        score where the polygon and the roof edge are known to disagree."""
        prob = self.predict(rgb)
        pred = prob[0] > self.thr
        valid = np.ones_like(gt, bool)
        if ignore_band and gt.any():
            valid = ~(ndimage.binary_dilation(gt, iterations=ignore_band) &
                      ~ndimage.binary_erosion(gt, iterations=ignore_band))
        tp = int((pred & gt & valid).sum()); fp = int((pred & ~gt & valid).sum())
        fn = int((~pred & gt & valid).sum())
        rec = {"tp": tp, "fp": fp, "fn": fn,
               "iou": tp / max(tp + fp + fn, 1), "mean_prob": float(prob.mean()),
               "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
        if gt.any():
            band = ndimage.binary_dilation(gt, iterations=3) & \
                   ~ndimage.binary_erosion(gt, iterations=3)
            interior = ndimage.binary_erosion(gt, iterations=3)
            for name, m in (("boundary", band), ("interior", interior)):
                if m.sum():
                    t = int((pred & gt & m).sum())
                    rec[f"{name}_iou"] = t / max(int(((pred | gt) & m).sum()), 1)
        rec.update(self.instance_match(pred, gt, pred_lab=self._pred_instances(prob)))
        rec.update(self.classify_fp(pred, gt))
        return rec


def build_model(cfg, ckpt, device):
    a = cfg["args"]
    arch = a.get("arch", "smp_unet")
    n_out = {"instance": 3, "embedding": 1 + a.get("embed_dim", 8),
             "center": 2, "flow": 3, "dist": 2, "dist3": 3}.get(a.get("heads"), 1)
    if a.get("frame_field"):
        n_out += 4          # c0.re, c0.im, c2.re, c2.im - not probabilities,
                            # ignored by the decoder, but they are in the weights
    if arch == "smp_unet":
        m = smp.Unet(encoder_name=a.get("encoder", "resnet34"),
                     encoder_weights=None, in_channels=3, classes=n_out)
    else:
        sys.path.insert(0, str(ROOT))
        from models.custom_unet import CustomUNet
        m = CustomUNet(classes=n_out, base_filters=a["base_filters"], depth=a["depth"],
                       kernel_size=a["kernel_size"], n_conv=a["n_conv"],
                       norm=a["norm"], dropout=a["dropout"],
                       up_mode=a["up_mode"], width_mult=a["width_mult"],
                       max_filters=a["max_filters"])
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return m


def quintile_table(df, col, metric="iou", n=5):
    x = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if x.nunique() < n:
        return None
    q = pd.qcut(df[col].rank(method="first"), n, labels=[f"Q{i+1}" for i in range(n)])
    g = df.groupby(q, observed=True).agg(
        chips=("iou", "size"), lo=(col, "min"), hi=(col, "max"),
        score=(metric, "mean"), instances=("gt_instances", "sum"),
        detect_rate=("detected_0.5", "sum"))
    g["detect_rate"] = (g.detect_rate / g.instances.replace(0, np.nan)).round(3)
    g["score"] = g.score.round(4)
    g[["lo", "hi"]] = g[["lo", "hi"]].round(4)
    return g


def eval_vit(run, a):
    """Score a dinov3_hot run with the UNet's decoder and metrics."""
    from instance_decode import WatershedDecoder
    from pool_data import PoolDataSource
    from vit_adapter import VitRun
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = VitRun(run, dev)
    out = Path(a.out or run / f"features_{a.role}")
    out.mkdir(parents=True, exist_ok=True)

    split_file = a.split_file or "data/cv_folds/cv_folds.csv"
    src = PoolDataSource(vit.cfg.dataset_repo, split_file, a.fold)
    ds = src.hf(a.role)
    print(f"[eval] ViT {run.name}: {len(ds)} chips ({a.role}), "
          f"decoded by watershed like the UNet "
          f"(mask thr {a.threshold}, core thr {a.core_thr})")

    dec = WatershedDecoder(mask_thr=a.threshold, core_thr=a.core_thr,
                           min_instance_px=MIN_INSTANCE_PX,
                           contour_weight=a.contour_weight)
    inst = InstanceScores(min_px=MIN_INSTANCE_PX)
    px = np.zeros(3)
    for i in tqdm(range(len(ds)), leave=False):
        r = ds[i]
        rgb = np.asarray(r["image"].convert("RGB"))
        gt = np.asarray(r["mask"].convert("L")) > 127
        prob = vit.predict(rgb)
        lab = dec(prob)
        inst.add(lab, ndimage.label(gt)[0], prob[0])
        pred = prob[0] > a.threshold
        px += [(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()]
    tp, fp, fn = px
    lines = [f"== {run.name} / {a.role} ==",
             f"checkpoint {vit.ckpt.name}", "",
             f"pixel F1  {200*tp/max(2*tp+fp+fn,1):.2f}",
             f"pixel IoU {100*tp/max(tp+fp+fn,1):.2f}", "",
             inst.report()]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text)
    print(text)
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run dir with config.json + best.pth")
    ap.add_argument("--role", default="test", choices=["test", "val", "train", "holdout"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="mask threshold. Default: the value the checkpoint was "
                         "SELECTED at, read from best_threshold.json in the run "
                         "dir, so grading matches selection. Falls back to 0.5")
    ap.add_argument("--peak-thr", type=float, default=0.3,
                    help="centre head: minimum heatmap value for a peak")
    ap.add_argument("--min-distance", type=int, default=2,
                    help="centre head: minimum px between two peaks")
    ap.add_argument("--flow-steps", type=int, default=40,
                    help="flow head: integration steps")
    ap.add_argument("--seed-mode", default="threshold",
                    choices=["threshold", "hmaxima"],
                    help="hmaxima seeds from regional maxima of the core "
                         "channel instead of a global threshold, so touching "
                         "cores still give two seeds")
    ap.add_argument("--h", type=float, default=0.15,
                    help="minimum prominence a core peak needs to be a seed")
    ap.add_argument("--seed-smooth", type=float, default=1.0)
    ap.add_argument("--seed-floor", type=float, default=0.2)
    ap.add_argument("--contour-weight", type=float, default=2.0,
                    help="how strongly predicted contours act as ridges in the "
                         "watershed; lower it if the model over-segments")
    ap.add_argument("--core-thr", type=float, default=0.5,
                    help="threshold on the predicted core channel; the core is "
                         "a smaller, harder target than the mask and usually "
                         "wants a lower value")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ignore-band", type=int, default=2,
                    help="exclude a k-pixel collar around every label boundary "
                         "from the pixel metrics, reported ALONGSIDE the strict "
                         "score. 0 disables the second number. Default 2: on "
                         "this data a 2px collar is 25%% of label pixels and "
                         "recovers +5.3 F1, while 4px costs 44%% of label pixels "
                         "for +1.5 more and 6px swallows a third of the "
                         "buildings whole")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split-file", default=None,
                    help="needed for a ViT run, whose config does not record it")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    run = Path(a.run)
    from vit_adapter import VitRun, is_vit_run
    if a.threshold is None and is_vit_run(run):
        a.threshold = 0.5
    if is_vit_run(run):
        # A dinov3_hot run: different layout (config.yaml + ckpts/) and a
        # different channel order, but the same three channels, so the same
        # decoder and the same instance metrics apply. That is what makes the
        # ViT and UNet numbers comparable rather than one IoU vs one PQ.
        return eval_vit(run, a)
    cfg = json.loads((run / "config.json").read_text())
    if a.threshold is None:
        bt = run / "best_threshold.json"
        a.threshold = (json.loads(bt.read_text())["threshold"]
                       if bt.exists() else 0.5)
        print(f"[eval] threshold {a.threshold:g} "
              + ("(the value this checkpoint was selected at)" if bt.exists()
                 else "(default; this run recorded none)"))
    out = Path(a.out or run / f"features_{a.role}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = cfg["args"]
    if not args.get("split_file"):
        raise SystemExit("this run did not use --split-file; nothing to slice")
    from pool_data import PoolDataSource
    from pool_splits import fold_roles
    src = PoolDataSource(args["dataset"], args["split_file"], args["fold"])
    if a.role == "holdout":
        split = pd.read_csv(args["split_file"])
        split["role"] = fold_roles(split.assignment, args["fold"], src.folds)
        src.frames["holdout"] = split[split.role == "holdout"]
    ds = src.hf(a.role)
    meta = src.frames[a.role].reset_index(drop=True)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
        meta = meta.iloc[:len(ds)]
    print(f"[eval] {len(ds)} chips ({a.role}) from fold {args['fold']}")

    model = build_model(cfg, a.checkpoint or run / "best.pth", dev)
    decoder = None
    heads = cfg["args"].get("heads")
    if heads in ("instance", "dist", "dist3"):
        from instance_decode import WatershedDecoder
        decoder = WatershedDecoder(mask_thr=a.threshold, core_thr=a.core_thr,
                                   min_instance_px=MIN_INSTANCE_PX,
                                   contour_weight=a.contour_weight,
                                   seed_mode=a.seed_mode, h=a.h,
                                   smooth=a.seed_smooth, floor=a.seed_floor)
        print(f"[eval] instance head: watershed from the predicted core "
              f"(mask thr {a.threshold}, core thr {a.core_thr})")
    elif heads == "center":
        from seed_decode import CentreDecoder
        decoder = CentreDecoder(mask_thr=a.threshold, peak_thr=a.peak_thr,
                                min_distance=a.min_distance,
                                min_instance_px=MIN_INSTANCE_PX)
        print(f"[eval] centre head: peaks of the heatmap become markers "
              f"(mask thr {a.threshold}, peak thr {a.peak_thr}, "
              f"min distance {a.min_distance}px)")
    elif heads == "flow":
        from seed_decode import FlowDecoder
        decoder = FlowDecoder(mask_thr=a.threshold, steps=a.flow_steps,
                              min_instance_px=MIN_INSTANCE_PX)
        print(f"[eval] flow head: integrating the field {a.flow_steps} steps")
    scorer = Scorer(model, dev, a.threshold, decoder,
                    raw_from=1 if heads == "flow" else None)

    inst = InstanceScores(min_px=MIN_INSTANCE_PX)
    rows = []
    for i in tqdm(range(len(ds)), leave=False):
        r = ds[i]
        rgb = np.asarray(r["image"].convert("RGB"))
        gt = np.asarray(r["mask"].convert("L")) > 127
        prob = scorer.predict(rgb)
        pl, _ = scorer._pred_instances(prob)
        inst.add(pl, ndimage.label(gt)[0], prob[0])
        rec = {"tile_id": r["tile_id"]}
        rec.update(ChipFeatures(rgb, gt).all())
        rec.update(scorer.score(rgb, gt))
        if a.ignore_band:
            b = scorer.score(rgb, gt, a.ignore_band)
            rec.update({f"band_{k}": v for k, v in b.items()
                        if k in ("tp", "fp", "fn")})
        rows.append(rec)
    df = pd.DataFrame(rows).merge(
        meta[["tile_id", "country", "project_name", "n_instances"]], on="tile_id")
    df.to_csv(out / "per_chip.csv", index=False)

    lines = [f"== {run.name} / {a.role} / thr {a.threshold} ==",
             f"{len(df)} chips, {int(df.gt_instances.sum())} instances", ""]
    def pooled(tp, fp, fn):
        return (200 * tp / max(2 * tp + fp + fn, 1), 100 * tp / max(tp + fp + fn, 1))

    f1, iou = pooled(df.tp.sum(), df.fp.sum(), df.fn.sum())
    lines += [f"{'':22s} {'F1':>7s} {'IoU':>7s}",
              f"{'strict':22s} {f1:7.2f} {iou:7.2f}"]
    if a.ignore_band and "band_tp" in df:
        bf1, biou = pooled(df.band_tp.sum(), df.band_fp.sum(), df.band_fn.sum())
        lines += [f"{f'ignoring {a.ignore_band}px boundary':22s} {bf1:7.2f} {biou:7.2f}",
                  f"{'label-geometry cost':22s} {bf1-f1:+7.2f} {biou-iou:+7.2f}"]
    lines += ["",
              "Two numbers on purpose. Strict is the headline. The second declines to",
              "score a collar around every label, where the polygon and the roof edge",
              "are known to disagree - the difference is how much of the gap is label",
              "geometry rather than model capability. Quoting only the second flatters",
              "the model; quoting only the first blames it for the labels.",
              f"mean per-chip IoU (populated) {100*df[df.gt_instances>0].iou.mean():.2f}", ""]
    ni = max(df.gt_instances.sum(), 1)
    lines += ["-- instance level --"]
    for t in (0.5, 0.75, 0.8):
        c = f"detected_{t}"
        if c in df:
            lines.append(f"detection recall @IoU{t:<5} {100*df[c].sum()/ni:5.1f}%")
    lines += ["", "-- object-level (the right unit for this problem) --",
              inst.report(), "",
              "PQ = SQ x RQ. SQ is the mask quality of the objects you matched,",
              "RQ an F1 over objects. High SQ with low RQ means good outlines on",
              "the wrong object count - the signature of merging.", "",
              f"buildings merged into one  {int(df.merged.sum())}",
              f"buildings split into many  {int(df.split.sum())}",
              f"predicted vs actual count  {int(df.pred_instances.sum())} vs "
              f"{int(df.gt_instances.sum())}", ""]
    tot_fp_px = df[["rim_px", "bridge_px", "isolated_px"]].sum().sum()
    lines += ["-- what the false positives actually ARE --"]
    for k in ("rim", "bridge", "isolated"):
        lines.append(f"{k:9s} {int(df[k].sum()):6,} blobs   "
                     f"{int(df[k+'_px'].sum()):9,} px  "
                     f"({100*df[k+'_px'].sum()/max(tot_fp_px,1):5.1f}% of FP area)")
    lines += ["   rim      = a collar on a building the model DID find "
              "(localisation, largely a label artefact)",
              "   bridge   = one blob spanning two labelled buildings "
              "(the instance-merging failure)",
              "   isolated = nothing labelled nearby - hallucination OR an "
              "unlabelled building; adjudicate with scripts/adjudicate_fp.py", ""]
    if "boundary_iou" in df:
        lines += ["-- where the errors sit --",
                  f"IoU within 3px of the label boundary {100*df.boundary_iou.mean():.1f}",
                  f"IoU in the building interior         {100*df.interior_iou.mean():.1f}", ""]

    pop = df[df.gt_instances > 0]
    lines.append("-- per-chip IoU by descriptor quintile (populated chips only) --")
    lines.append("   NOTE: these slices are marginal, not causal. Descriptors are")
    lines.append("   correlated with each other and with density - the sharpest chips")
    lines.append("   here are drone imagery, which is also the densest - so a spread")
    lines.append("   along one axis may belong to another. Use them to find where to")
    lines.append("   look, then confirm with a controlled comparison.")
    for f in FEATURES:
        if f not in pop:
            continue
        t = quintile_table(pop, f)
        if t is None:
            continue
        span = t.score.max() - t.score.min()
        lines += [f"\n{f}   (Q1 low -> Q5 high; spread {span:.3f} IoU)",
                  t.to_string()]
    lines.append("\n-- by project (>=10 chips) --")
    g = df.groupby("project_name").agg(chips=("iou", "size"),
        instances=("gt_instances", "sum"), iou=("iou", "mean"),
        edge_width=("edge_width", "median"), hf=("hf_ratio", "median"))
    g = g[g.chips >= 10].sort_values("iou")
    g.index = [p[:48] for p in g.index]
    lines.append(g.round(3).to_string())

    text = "\n".join(lines)
    (out / "report.txt").write_text(text)
    print(text)
    print(f"\nwrote {out}/report.txt and per_chip.csv")


if __name__ == "__main__":
    main()

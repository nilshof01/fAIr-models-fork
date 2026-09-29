"""Train a UNet building segmentation model on the tidied HF dataset.

Expects the dataset to have "train", "val", "test" splits with columns:
  image (PIL RGB), mask (PIL L), density_bucket (int 0-4)

Usage
-----
  # Production run: efficientnet-b4 encoder, dist3 head (S6 baseline)
  python scripts/train.py \
      --dataset nilsho01/vhr-buildings-review-decisions \
      --image-source hotosm/vhr-building-segmentation \
      --split-file data/cv_folds/cv_folds.csv --fold 0 \
      --encoder efficientnet-b4 --heads dist3 \
      --out checkpoints/unet_eb4_dist3_f0 \
      --epochs 80 --batch 16 --lr-enc 1e-4 --lr-dec 1e-3

  # Evaluate the best checkpoint against the test split:
  python scripts/train.py --eval-only --out checkpoints/unet_eb4_dist3_f0

  # A UNet you designed yourself, random weights, no pretraining:
  python scripts/train.py --arch custom_unet \
      --base-filters 32 --depth 4 --kernel-size 3 --n-conv 2

  # The same student, stabilised by a pretrained EfficientNet-B4 teacher:
  python scripts/train.py --arch distill_unet \
      --base-filters 32 --depth 4 --feat-weight 1.0

  # Cross-validation over one undivided pool (splits from scripts/make_cv_folds.py):
  python scripts/train.py --split-file data/cv_folds/cv_folds.csv --fold 0

Every run writes config.json, split_stats.csv/txt, split_by_country.csv and
split_composition.png into --out, so the composition behind a number is
recoverable without re-deriving anything.
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise SystemExit("pip install segmentation-models-pytorch")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    raise SystemExit("pip install albumentations")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


def build_train_aug(size: int = 256) -> A.Compose:
    return A.Compose([
        # Scale 0.8–1.2× then crop/pad back to size
        A.RandomScale(scale_limit=0.2, p=0.8),
        A.PadIfNeeded(min_height=size, min_width=size,
                      border_mode=0),
        A.RandomCrop(height=size, width=size),
        # Flips and 90° rotations — free augmentation for satellite imagery
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Colour: brightness/contrast + hue/saturation shift
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                             val_shift_limit=15, p=0.5),
        # Geometric distortion — varied building shapes
        A.ElasticTransform(alpha=60, sigma=6, p=0.3),
        # Coarse dropout on image only (mask untouched — building still there)
        A.CoarseDropout(num_holes_range=(2, 8),
                        hole_height_range=(16, 48),
                        hole_width_range=(16, 48),
                        fill=0, p=0.3),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


def build_val_aug(size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegDataset(Dataset):
    """Returns (image, target, weight).

    In semantic mode target is the 1-channel mask and weight is all ones. In
    instance mode target is (mask, core, contour) and weight is the separation
    map. Both are derived from the AUGMENTED mask, never augmented themselves:
    rotating a contour band or an eroded core changes its width.
    """

    def __init__(self, hf_split, train: bool = True, size: int = 256,
                 targets=None, semantic_only: bool = False, no_aug: bool = False):
        self.ds  = hf_split
        self.aug = build_train_aug(size) if (train and not no_aug) else build_val_aug(size)
        self.targets = targets
        self.semantic_only = semantic_only

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s       = self.ds[idx]
        img_np  = np.asarray(s["image"].convert("RGB"))
        mask_np = (np.asarray(s["mask"].convert("L")) > 0).astype(np.uint8)
        result  = self.aug(image=img_np, mask=mask_np)
        m = result["mask"].numpy().astype(np.uint8)
        if self.targets is None:
            return (result["image"], torch.from_numpy(m[None]).float(),
                    torch.ones(1, *m.shape))
        if self.semantic_only:
            _, w = self.targets(m)
            return (result["image"], torch.from_numpy(m[None]).float(),
                    torch.from_numpy(w[None]))
        if self.targets == "embedding":
            from scipy import ndimage
            inst, _ = ndimage.label(m)
            return (result["image"], torch.from_numpy(m[None]).float(),
                    torch.from_numpy(inst.astype(np.int64)))
        out = self.targets(m)
        if isinstance(out, tuple):
            ch, w = out
            return result["image"], torch.from_numpy(ch), torch.from_numpy(w[None])
        return (result["image"], torch.from_numpy(out),
                torch.ones(1, *m.shape))


# ---------------------------------------------------------------------------
# Loss: BCE + Dice
# ---------------------------------------------------------------------------

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = pred.sigmoid().flatten(1)
    t = target.flatten(1)
    return 1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)


def embedding_loss(logits, mask, inst, criterion, embed_weight):
    """BCE+Dice on channel 0, discriminative loss on the rest."""
    base = seg_loss(logits[:, :1], mask)
    return base + embed_weight * criterion(logits[:, 1:].float(), inst)


def dist_loss(logits: torch.Tensor, targets: torch.Tensor,
              weight: torch.Tensor | None = None,
              seg_weight: float = 1.0) -> torch.Tensor:
    """Composite loss for the dist / dist3 heads.

    Ch0  mask: BCE + Dice + separation weight map
    Ch1  normalised EDT: MSE only over building pixels (mask==1), so the vast
         background region (where both prediction and target are ~0) does not
         drown the within-building shape signal
    Ch2  instance boundary (dist3 only): BCE + Dice — a thin positive band at
         shared walls, trained like a second binary segmentation target
    """
    mask_l = seg_loss(logits[:, :1], targets[:, :1], weight)
    inside = targets[:, :1]
    mse = ((logits[:, 1:2].sigmoid() - targets[:, 1:2]) ** 2 * inside).sum() / \
          (inside.sum() + 1e-6)
    # the three terms are on wildly different scales - a masked MSE over a
    # normalised EDT sits near 0.05 while BCE+Dice sits near 1 - so summing
    # them 1:1:1 lets the auxiliary channels dominate. seg_weight restores the
    # mask as the primary objective.
    loss = seg_weight * mask_l + mse
    if logits.shape[1] > 2:
        loss = loss + seg_loss(logits[:, 2:3], targets[:, 2:3])
    return loss


TVERSKY_BETA = 0.5          # 0.5 == Dice; below 0.5 penalises false positives


def tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                 beta: float = 0.5, eps: float = 1.0) -> torch.Tensor:
    """Dice generalised so false positives and negatives can cost differently.

    tp / (tp + beta*fn + (1-beta)*fp).  beta = 0.5 is exactly Dice, which
    weights them equally and therefore rewards recall - a model that paints
    generously is not punished for it. beta < 0.5 makes a false positive more
    expensive than a miss.

    Measured context before reaching for this: the predicted boundary already
    sits a median 1px INSIDE the label, so the over-prediction is not a uniform
    collar. 27% of false-positive pixels lie more than 10px from any labelled
    building - that far-field excess is what this can reach. It cannot fix
    boundary placement variance (p10 -6px, p90 +9px), and pushing the boundary
    further inward costs recall.
    """
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    tp = (p * target).sum(dims)
    fp = (p * (1 - target)).sum(dims)
    fn = ((1 - p) * target).sum(dims)
    t = (tp + eps) / (tp + beta * fn + (1 - beta) * fp + eps)
    return (1 - t).mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor,
             weight: torch.Tensor | None = None,
             beta: float | None = None) -> torch.Tensor:
    """BCE + Dice on every channel, with the separation weight applied to the
    mask channel only.

    The weight belongs on the mask: it exists to make background pixels wedged
    between two buildings expensive to flood, and those pixels are background in
    the mask channel. Applying it to the core or contour channel would just
    re-weight their own borders, which the channels already encode.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        bce = bce.clone()
        bce[:, :1] = bce[:, :1] * weight
    b = TVERSKY_BETA if beta is None else beta
    region = (dice_loss(logits, target).mean() if b == 0.5
              else tversky_loss(logits, target, b))
    return bce.mean(dim=[1, 2, 3]).mean() + region


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             threshold: float = 0.5, decoder=None, max_instance_chips: int = 0
             ) -> dict[str, float]:
    """Pixel metrics always; PQ/SQ/RQ too when a decoder is given.

    Pixel quality and instance quality move in OPPOSITE directions on this
    task - measured on fold 0, the semantic model scores pixel F1 83.7 with PQ
    21.8 while the instance head scores 80.2 with PQ 29.0. Selecting a
    checkpoint on pixel loss therefore selects against separation, which is
    why PQ belongs here and not only in a separate evaluation script.
    """
    model.eval()
    tp = fp = fn = total_loss = 0.0
    inst = None
    if decoder is not None:
        import sys as _s
        from pathlib import Path as _P
        _s.path.insert(0, str(_P(__file__).resolve().parent))
        from instance_metrics import InstanceScores
        inst = InstanceScores(min_px=30)
        seen = 0
    for imgs, masks, wts in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(imgs)
        total_loss += seg_loss(logits[:, :1], masks[:, :1]).item() * len(imgs)
        # metrics always describe the building mask, channel 0, so semantic and
        # instance runs stay directly comparable
        pred = (logits[:, :1].sigmoid() > threshold).float()
        gt = masks[:, :1]
        tp += (pred * gt).sum().item()
        fp += (pred * (1 - gt)).sum().item()
        fn += ((1 - pred) * gt).sum().item()
        if inst is not None and (not max_instance_chips or seen < max_instance_chips):
            from scipy import ndimage
            prob = logits.float().sigmoid().cpu().numpy()
            gtn = gt.cpu().numpy()[:, 0] > 0.5
            for b in range(len(prob)):
                if not gtn[b].any():
                    continue
                inst.add(decoder(prob[b]), ndimage.label(gtn[b])[0], prob[b, 0])
                seen += 1
    n = len(loader.dataset)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    out = {"loss": total_loss / n if n else 0.0, "f1": f1, "iou": iou,
           "precision": precision, "recall": recall}
    if inst is not None and inst.n_gt:
        p = inst.panoptic(0.5)
        out.update(pq=p["PQ"], sq=p["SQ"], rq=p["RQ"],
                   pred_instances=inst.n_pred, gt_instances=inst.n_gt)
    return out


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

class ModelBundle:
    """Holds the module that trains and the module that is scored and saved.

    They differ for distillation: the teacher is part of the training graph but
    must never reach the checkpoint or the metrics.
    """

    def __init__(self, train_model, infer_model, param_groups, description):
        self.train_model = train_model
        self.infer_model = infer_model
        self.param_groups = param_groups
        self.description = description

    @classmethod
    def build(cls, args, device):
        n_out = {"instance": 3, "embedding": 1 + args.embed_dim,
                 "center": 2, "flow": 3, "dist": 2, "dist3": 3}.get(args.heads, 1)
        if args.frame_field:
            n_out += 4
        if args.arch == "smp_unet":
            kw = {}
            if args.decoder_channels:
                kw["decoder_channels"] = tuple(args.decoder_channels)
            net = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                           in_channels=3, classes=n_out, **kw).to(device)
            groups = [
                {"params": net.encoder.parameters(), "lr": args.lr_enc},
                {"params": list(net.decoder.parameters()) +
                           list(net.segmentation_head.parameters()), "lr": args.lr_dec},
            ]
            dc = tuple(args.decoder_channels) if args.decoder_channels else "default"
            return cls(net, net, groups,
                       f"smp.Unet({args.encoder}, imagenet, {n_out}ch, "
                       f"{args.heads}, decoder={dc})")

        from models.custom_unet import CustomUNet
        student = CustomUNet(
            in_channels=3, classes=n_out, base_filters=args.base_filters,
            depth=args.depth, kernel_size=args.kernel_size, n_conv=args.n_conv,
            norm=args.norm, dropout=args.dropout, up_mode=args.up_mode,
            width_mult=args.width_mult, max_filters=args.max_filters)

        if args.arch == "custom_unet":
            student = student.to(device)
            # every weight is random here, so one learning rate for the lot
            groups = [{"params": student.parameters(), "lr": args.lr_dec}]
            return cls(student, student, groups, student.describe())

        from models.distill_unet import DistillUNet
        net = DistillUNet(student, encoder_name=args.encoder,
                          teacher_ckpt=args.teacher_ckpt,
                          feat_weight=args.feat_weight, kd_weight=args.kd_weight,
                          temperature=args.kd_temperature,
                          input_size=args.size).to(device)
        groups = [{"params": list(net.student.parameters()) +
                             list(net.adapters.parameters()), "lr": args.lr_dec}]
        return cls(net, net.student, groups, net.describe())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",       default="nilsho01/vhr-buildings-tidied-v1")
    ap.add_argument("--image-source",  default=None,
                    help="source dataset for images/masks when --dataset is metadata-only "
                         "(defaults to hotosm/vhr-building-segmentation)")
    ap.add_argument("--val-frac",    type=float, default=0.10,
                    help="fraction of pool for val (used when dataset has no val split)")
    ap.add_argument("--test-frac",   type=float, default=0.10,
                    help="fraction of pool for test (used when dataset has no test split)")
    ap.add_argument("--max-empty",     type=int,   default=None,
                    help="max empty chips in train (absolute; use --empty-ratio for relative)")
    ap.add_argument("--empty-ratio",   type=float, default=None,
                    help="empty chips per labeled chip in train (overrides --max-empty)")
    ap.add_argument("--max-empty-val", type=int,   default=0,
                    help="max empty chips in val split (default 0 = labeled only)")
    ap.add_argument("--max-empty-test",type=int,   default=0,
                    help="max empty chips in test split (default 0 = labeled only)")
    ap.add_argument("--out",       default="checkpoints/unet_v1")
    ap.add_argument("--encoder",   default="efficientnet-b4")
    ap.add_argument("--epochs",    type=int,   default=80)
    ap.add_argument("--batch",     type=int,   default=16)
    ap.add_argument("--lr-enc",    type=float, default=1e-4,
                    help="Encoder learning rate (pretrained weights)")
    ap.add_argument("--lr-dec",    type=float, default=1e-3,
                    help="Decoder learning rate")
    ap.add_argument("--size",      type=int,   default=256)
    ap.add_argument("--no-aug",    action="store_true",
                    help="disable training augmentation (resize+normalize only, "
                         "same pipeline as val). Useful for overfitting diagnostics.")
    ap.add_argument("--workers",   type=int,   default=4)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="mask threshold. With --threshold-sweep this is only "
                         "the fallback; the selected one is written to "
                         "best_threshold.json and used by eval_features")
    ap.add_argument("--threshold-sweep", type=float, nargs="+",
                    default=[0.2, 0.25, 0.3, 0.4, 0.45, 0.5],
                    help="thresholds tried each validation pass; the checkpoint "
                         "is selected at whichever is best, and that value is "
                         "saved with it. Without this the model is CHOSEN at one "
                         "operating point and GRADED at another - worth 5 PQ "
                         "points on the ViT, so not a rounding error. Range "
                         "covers 0.20-0.50: S6 folds 3/4 selected 0.25 and "
                         "folds 0-2 selected 0.45. Pass a single value to "
                         "disable the sweep")
    ap.add_argument("--ema-decay", type=float, default=0.99,
                    help="EMA decay for model weights (0 = disabled)")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop when the selection metric has not improved for "
                         "this many epochs (0 = run all epochs). Counted on the "
                         "SAME metric the checkpoint is chosen by, so it cannot "
                         "stop on loss while selecting on PQ")
    ap.add_argument("--select-on", default="auto",
                    choices=["auto", "loss", "pq", "f1"],
                    help="what the best checkpoint is chosen by. 'auto' uses PQ "
                         "for any head that decodes instances and loss "
                         "otherwise. Selecting on loss picks against separation: "
                         "on fold 0 the semantic model scores pixel F1 83.7 / PQ "
                         "21.8 and the instance head 80.2 / 29.0, so the two "
                         "objectives disagree in sign")
    ap.add_argument("--val-instance-chips", type=int, default=300,
                    help="cap chips used for PQ during validation; instance "
                         "matching is the slow part (0 = all)")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training; evaluate best checkpoint on test split")

    sp = ap.add_argument_group("splits")
    sp.add_argument("--split-file", default=None,
                    help="split CSV from scripts/make_splits.py; the dataset's "
                         "own split column is ignored when this is given")
    sp.add_argument("--fold", type=int, default=0,
                    help="which fold is the test set; val is the next one round")
    sp.add_argument("--max-test", type=int, default=0,
                    help="cap the test split at N chips (0 = no cap); the "
                         "populated/empty mix is preserved")
    sp.add_argument("--max-val", type=int, default=0,
                    help="cap the val split at N chips (0 = no cap)")
    sp.add_argument("--empty-frac-train", type=float, default=None,
                    help="target share of empty chips in TRAIN (0-1). Default "
                         "keeps the pool's natural rate. Never applied to val "
                         "or test: changing the empty share there moves the "
                         "metric for reasons unrelated to the model")
    sp.add_argument("--split-seed", type=int, default=1337,
                    help="seed for capping and empty-chip subsampling")

    h = ap.add_argument_group("instance head (--heads instance)")
    h.add_argument("--heads", default="semantic",
                   choices=["semantic", "instance", "embedding", "center", "flow",
                            "dist", "dist3"],
                   help="semantic: one mask channel (simple baseline). "
                        "dist3 (recommended for production): mask + per-instance "
                        "normalised EDT + shared-boundary ridge channel; the "
                        "boundary acts as an impassable wall in the watershed "
                        "flood, fixing buildings that share a wall. S6 UNet "
                        "baseline (efficientnet-b4 + dist3) scores PQ 37.65 on "
                        "the fixed benchmark. "
                        "dist: same without the boundary channel. "
                        "instance/center/flow/embedding: older multi-head modes.")
    h.add_argument("--sigma-scale", type=float, default=0.125,
                   help="centre peak width as a fraction of sqrt(area)")
    h.add_argument("--sigma-min", type=float, default=1.0)
    h.add_argument("--sigma-max", type=float, default=4.0)
    h.add_argument("--marker-weight", type=float, default=1.0,
                   help="weight on the centre or flow term")
    h.add_argument("--embed-dim", type=int, default=8)
    h.add_argument("--delta-v", type=float, default=0.5,
                   help="pixels are free within this radius of their instance mean")
    h.add_argument("--delta-d", type=float, default=1.5,
                   help="instance means are pushed to 2x this apart; must exceed "
                        "delta-v or clusters overlap")
    h.add_argument("--embed-weight", type=float, default=1.0)
    h.add_argument("--seg-weight", type=float, default=1.0,
                   help="weight on the MASK term of the dist/dist3 loss. Its "
                        "three terms are on different scales (masked MSE ~0.05 "
                        "vs BCE+Dice ~1), so 1:1:1 lets the auxiliary channels "
                        "dominate the primary objective")
    h.add_argument("--boundary-weight", type=float, default=0.0,
                   help="up-weight the loss near every label boundary, inside "
                        "and out. BCE+Dice barely notices a rounded corner, so "
                        "nothing pushes the model toward the straight edges the "
                        "rasterised OSM polygons actually have. 3-5 is a "
                        "reasonable start; 0 disables")
    h.add_argument("--boundary-sigma", type=float, default=5.0,
                   help="px falloff of that bonus")
    h.add_argument("--frame-field", action="store_true",
                   help="add a Girard et al. 2021 frame field head: 4 extra "
                        "channels encoding TWO wall directions per pixel, so "
                        "corners are representable, trained to align with the "
                        "mask gradient and stay smooth. Enables polygonisation "
                        "that snaps edges to real wall angles")
    h.add_argument("--ff-align", type=float, default=1.0)
    h.add_argument("--ff-align90", type=float, default=0.2)
    h.add_argument("--ff-smooth", type=float, default=0.05)
    h.add_argument("--core-erosion", type=int, default=3,
                   help="px each building is shrunk by to make a seed; must "
                        "exceed half the typical gap between touching roofs")
    h.add_argument("--contour-width", type=int, default=2)
    h.add_argument("--sep-weight", type=float, default=10.0,
                   help="height of the separation bonus on background pixels "
                        "between two instances (0 disables it)")
    h.add_argument("--sep-sigma", type=float, default=8.0,
                   help="px width of that bonus; ~half a typical alley")
    h.add_argument("--tversky-beta", type=float, default=0.5,
                   help="region loss asymmetry: 0.5 is Dice (false positives "
                        "and misses cost the same), below 0.5 makes a false "
                        "positive more expensive. 0.3 is a usual first step")
    h.add_argument("--ib-dilate", type=int, default=2,
                   help="px each instance is dilated when computing the shared "
                        "boundary channel (dist3 only); 2 marks a 4px-wide band "
                        "at every shared wall")

    g = ap.add_argument_group("architecture")
    g.add_argument("--decoder-channels", type=int, nargs="+", default=None,
                   help="smp.Unet decoder widths, coarse to fine. Default "
                        "(256 128 64 32 16). Measured motivation: giving the "
                        "model a perfect mask lifts SQ 74.3 -> 94.1, so ~20 "
                        "points of mask quality are unrealised - this tests "
                        "whether decoder capacity is what reaches them")
    g.add_argument("--arch", default="smp_unet",
                   choices=["smp_unet", "custom_unet", "distill_unet"],
                   help="smp_unet: pretrained encoder (default). "
                        "custom_unet: your own UNet, random weights. "
                        "distill_unet: the same, stabilised by a frozen teacher")
    g.add_argument("--base-filters", type=int, default=32,
                   help="channels at the finest stage")
    g.add_argument("--depth", type=int, default=4,
                   help="encoder stages, i.e. strides 1..2^(depth-1)")
    g.add_argument("--kernel-size", type=int, default=3)
    g.add_argument("--n-conv", type=int, default=2, help="convs per block")
    g.add_argument("--norm", default="batch",
                   choices=["batch", "group", "instance", "none"])
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--up-mode", default="transpose",
                   choices=["transpose", "bilinear"])
    g.add_argument("--width-mult", type=float, default=2.0,
                   help="channel growth per stage")
    g.add_argument("--max-filters", type=int, default=512)

    d = ap.add_argument_group("distillation (--arch distill_unet)")
    d.add_argument("--teacher-ckpt", default=None,
                   help="a teacher actually trained on this task; without it "
                        "the teacher is ImageNet-only and just the encoder "
                        "features are distilled")
    d.add_argument("--feat-weight", type=float, default=1.0,
                   help="weight on the encoder hint loss")
    d.add_argument("--kd-weight", type=float, default=0.0,
                   help="weight on output-level KD; requires --teacher-ckpt")
    d.add_argument("--kd-temperature", type=float, default=2.0)
    return ap.parse_args()


def save_config(args, out_dir, extra=None):
    import json
    import subprocess
    cfg = {"args": vars(args), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        cfg["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    except Exception:
        cfg["git_commit"] = None
    if extra:
        cfg.update(extra)
    (Path(out_dir) / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return cfg


def main():
    args = parse_args()
    # the region loss is reached from several call sites (dist_loss, the plain
    # semantic path, the val loss); setting it once here keeps training and
    # validation on the same objective rather than threading beta through each.
    global TVERSKY_BETA
    TVERSKY_BETA = args.tversky_beta
    if args.tversky_beta != 0.5:
        print(f"region loss: Tversky beta={args.tversky_beta} "
              f"(a false positive costs {(1-args.tversky_beta)/args.tversky_beta:.2f}x a miss)")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    bundle = ModelBundle.build(args, device)
    model, core = bundle.train_model, bundle.infer_model
    print(f"arch: {bundle.description}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.dataset} …")
    split_meta = None
    if args.split_file:
        from pool_data import PoolDataSource
        from split_report import SplitStats
        src = PoolDataSource(args.dataset, args.split_file, args.fold,
                             max_test=args.max_test, max_val=args.max_val,
                             empty_frac_train=args.empty_frac_train,
                             seed=args.split_seed)
        ds = {r: src.hf(r) for r in ("train", "val", "test")}

        # if the dataset has no image column (metadata-only), fetch images
        # from the source dataset by dataset_row index
        if "image" not in next(iter(ds.values())).column_names:
            img_src = args.image_source or "hotosm/vhr-building-segmentation"
            print(f"  no image column — loading images from {img_src} ...")
            _src_img = load_dataset(img_src, split="train")
            _merged = {}
            for _role, _split in ds.items():
                _split = _split.filter(lambda x: x["dataset_row"] is not None)
                _rows   = [int(r) for r in _split["dataset_row"]]
                _sub    = _src_img.select(_rows)
                for _col in ("n_instances", "num_buildings", "decision"):
                    if _col in _split.column_names and _col not in _sub.column_names:
                        _sub = _sub.add_column(_col, _split[_col])
                _merged[_role] = _sub
            ds = _merged

        stats = SplitStats(src.roles_frame)
        title = (f"fold {args.fold} of {src.folds} - {Path(args.split_file).name}")
        print()
        print(stats.text(title))
        print()
        stats.save(out_dir)
        print(f"[report] {stats.plot(out_dir, title=title)}")
        split_meta = {"split_file": args.split_file, "fold": args.fold,
                      "folds": src.folds,
                      "composition": stats.table.to_dict("records")}
    else:
        from datasets import DatasetDict, concatenate_datasets as _cat
        raw = load_dataset(args.dataset)

        # 1. combine all HF splits into one pool
        pool = _cat([raw[s] for s in raw.keys()])

        # filter to kept chips only when decision column is present
        if "decision" in pool.column_names:
            before = len(pool)
            pool = pool.filter(lambda x: x["decision"] == "keep")
            print(f"  filtered decision==keep: {before} → {len(pool)} chips")

        # normalise num_buildings → n_instances
        if "num_buildings" in pool.column_names and "n_instances" not in pool.column_names:
            pool = pool.rename_column("num_buildings", "n_instances")

        # if no image/mask columns, fetch them from the source dataset by dataset_row
        if "image" not in pool.column_names:
            img_src = args.image_source or "hotosm/vhr-building-segmentation"
            print(f"  no image column — loading images from {img_src} ...")
            pool    = pool.filter(lambda x: x["dataset_row"] is not None)
            src     = load_dataset(img_src, split="train")
            rows    = [int(r) for r in pool["dataset_row"]]
            src_sub = src.select(rows)
            # merge: add metadata cols onto src_sub (which already has the image schema)
            for col in ("n_instances", "decision"):
                if col in pool.column_names and col not in src_sub.column_names:
                    src_sub = src_sub.add_column(col, pool[col])
            pool = src_sub

        # 2. cap empty chips
        if "n_instances" in pool.column_names:
            labeled   = pool.filter(lambda x: x["n_instances"] > 0)
            empty_all = pool.filter(lambda x: x["n_instances"] == 0)
            if args.empty_ratio is not None:
                cap = int(len(labeled) * args.empty_ratio)
            else:
                cap = args.max_empty if args.max_empty is not None else len(empty_all)
            if cap < len(empty_all):
                empty_all = empty_all.select(range(cap))
            pool = _cat([labeled, empty_all])
        pool = pool.shuffle(seed=42)
        print(f"  pool: {len(pool)} chips")

        # 3. split into train / val / test (skip if fracs are 0)
        holdout = args.val_frac + args.test_frac
        if holdout > 0:
            tv = pool.train_test_split(test_size=holdout, seed=42)
            if args.test_frac > 0:
                vt = tv["test"].train_test_split(
                    test_size=args.test_frac / holdout, seed=42)
                ds = DatasetDict({"train": tv["train"], "val": vt["train"], "test": vt["test"]})
            else:
                ds = DatasetDict({"train": tv["train"], "val": tv["test"]})
        else:
            ds = DatasetDict({"train": pool})
        splits_str = "  ".join(f"{k}={len(v)}" for k, v in ds.items())
        print(f"  {splits_str}")
        if holdout > 0:
            print("  NOTE: splits not checked for spatial leakage — prefer --split-file.")
    save_config(args, out_dir, {"split": split_meta,
                                "arch_description": bundle.description})

    if args.eval_only:
        ckpt = out_dir / "best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        core.load_state_dict(torch.load(ckpt, map_location=device))
        tg = None
        if args.heads == "instance":
            from instance_targets import InstanceTargets
            tg = InstanceTargets(core_erosion=args.core_erosion,
                                 contour_width=args.contour_width,
                                 sep_w0=args.sep_weight, sep_sigma=args.sep_sigma)
        elif args.heads in ("dist", "dist3"):
            from instance_targets import DistTransformTargets
            tg = DistTransformTargets(sep_w0=args.sep_weight,
                                      sep_sigma=args.sep_sigma,
                                      bnd_w0=args.boundary_weight,
                                      bnd_sigma=args.boundary_sigma,
                                      instance_boundary=(args.heads == "dist3"),
                                      ib_dilate=args.ib_dilate)
        test_loader = DataLoader(SegDataset(ds["test"], train=False, size=args.size,
                                            targets=tg),
                                 batch_size=args.batch, num_workers=args.workers)
        metrics = evaluate(core, test_loader, device, args.threshold)
        print(f"\nTest results ({ckpt}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return

    targets = None
    if args.heads in ("dist", "dist3"):
        from instance_targets import DistTransformTargets
        targets = DistTransformTargets(sep_w0=args.sep_weight,
                                       sep_sigma=args.sep_sigma,
                                       bnd_w0=args.boundary_weight,
                                       bnd_sigma=args.boundary_sigma,
                                       instance_boundary=(args.heads == "dist3"),
                                       ib_dilate=args.ib_dilate)
        print(f"{args.heads} head: normalised per-instance EDT"
              + (" + instance boundary ridge" if args.heads == "dist3" else "")
              + f", sep_w0={args.sep_weight} sep_sigma={args.sep_sigma}px")
    elif args.boundary_weight > 0 and args.heads == "semantic":
        from instance_targets import InstanceTargets
        targets = InstanceTargets(core_erosion=args.core_erosion,
                                  contour_width=args.contour_width,
                                  sep_w0=0.0, bnd_w0=args.boundary_weight,
                                  bnd_sigma=args.boundary_sigma)
        print(f"boundary weighting: w0={args.boundary_weight} "
              f"sigma={args.boundary_sigma}px (semantic head keeps 1 channel; "
              f"only the weight map changes)")
    if args.heads in ("center", "flow"):
        from seed_targets import CentreTargets, FlowTargets
        targets = (CentreTargets(args.sigma_scale, args.sigma_min, args.sigma_max)
                   if args.heads == "center" else FlowTargets())
        print(f"{args.heads} head: "
              + (f"sigma {args.sigma_scale}*sqrt(area) clipped to "
                 f"[{args.sigma_min}, {args.sigma_max}]" if args.heads == "center"
                 else "unit vectors toward each building's innermost point")
              + f", weight {args.marker_weight}")
    elif args.heads == "embedding":
        targets = "embedding"
        print(f"embedding head: dim {args.embed_dim}, delta_v {args.delta_v}, "
              f"delta_d {args.delta_d}, weight {args.embed_weight}")
    elif args.heads == "instance":
        from instance_targets import InstanceTargets
        targets = InstanceTargets(core_erosion=args.core_erosion,
                                  contour_width=args.contour_width,
                                  sep_w0=args.sep_weight, sep_sigma=args.sep_sigma,
                                  bnd_w0=args.boundary_weight,
                                  bnd_sigma=args.boundary_sigma)
        print(f"instance head: core erosion {args.core_erosion}px, contour "
              f"{args.contour_width}px, separation weight w0={args.sep_weight} "
              f"sigma={args.sep_sigma}px")

    train_loader = DataLoader(
        SegDataset(ds["train"], train=True,  size=args.size, targets=targets,
                   semantic_only=args.heads == "semantic", no_aug=args.no_aug),
        batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(
        SegDataset(ds["val"],   train=False, size=args.size, targets=targets,
                   semantic_only=args.heads == "semantic"),
        batch_size=args.batch, num_workers=args.workers, pin_memory=True) if "val" in ds else None

    # ── Optimizer ────────────────────────────────────────────────────────────
    ff = None
    if args.frame_field:
        from frame_field import FrameFieldLoss
        ff = FrameFieldLoss(args.ff_align, args.ff_align90, args.ff_smooth)
        print(f"frame field: align {args.ff_align} align90 {args.ff_align90} "
              f"smooth {args.ff_smooth} (4 extra channels)")

    marker = None
    if args.heads == "center":
        from seed_decode import CentreLoss
        marker = CentreLoss()
    elif args.heads == "flow":
        from seed_decode import FlowLoss
        marker = FlowLoss()

    disc = None
    if args.heads == "embedding":
        from embedding_loss import DiscriminativeLoss
        disc = DiscriminativeLoss(delta_v=args.delta_v, delta_d=args.delta_d)

    optimizer = torch.optim.AdamW(bundle.param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── EMA ──────────────────────────────────────────────────────────────────
    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(core).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    def update_ema(model: nn.Module, ema: nn.Module, decay: float) -> None:
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)
            # Buffers (BatchNorm running stats) are copied, not averaged: an
            # EMA of the weights paired with the *initial* running stats makes
            # activations explode, which with fp16 autocast shows up as a NaN
            # validation loss and no checkpoint ever being saved.
            for b_ema, b in zip(ema.buffers(), model.buffers()):
                b_ema.data.copy_(b.data)

    # ── Train loop ────────────────────────────────────────────────────────────
    val_decoder = None
    if args.heads in ("instance", "dist", "dist3"):
        from instance_decode import WatershedDecoder
        val_decoder = WatershedDecoder(mask_thr=args.threshold, core_thr=0.5,
                                       min_instance_px=30)
    elif args.heads == "center":
        from seed_decode import CentreDecoder
        val_decoder = CentreDecoder(mask_thr=args.threshold)
    elif args.heads == "flow":
        from seed_decode import FlowDecoder
        val_decoder = FlowDecoder(mask_thr=args.threshold)

    select = args.select_on
    if select == "auto":
        select = "pq" if val_decoder is not None else "loss"
    print(f"selecting the best checkpoint on: {select}"
          + ("  (higher is better)" if select != "loss" else "  (lower is better)"))
    best_score = float("inf") if select == "loss" else -float("inf")
    stale = 0

    for epoch in range(1, args.epochs + 1):
        # -- train --
        model.train()
        train_loss = 0.0
        for imgs, masks, wts in tqdm(train_loader, leave=False,
                                     desc=f"epoch {epoch}/{args.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            wts = wts.to(device)
            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(imgs)
                n_seg = logits.shape[1] - (4 if ff is not None else 0)
                if args.heads == "dist":
                    loss = dist_loss(logits[:, :n_seg], masks, wts,
                                     seg_weight=args.seg_weight)
                elif marker is not None:
                    loss = seg_loss(logits[:, :1], masks[:, :1])
                    if args.heads == "center":
                        loss = loss + args.marker_weight * marker(
                            logits[:, 1:2].float(), masks[:, 1:2].float())
                    else:
                        loss = loss + args.marker_weight * marker(
                            logits[:, 1:3].float(), masks[:, 1:3].float(),
                            masks[:, :1].float())
                elif disc is not None:
                    loss = embedding_loss(logits[:, :n_seg], masks, wts.to(device),
                                          disc, args.embed_weight)
                else:
                    loss = seg_loss(logits[:, :n_seg], masks, wts)
                if ff is not None:
                    loss = loss + ff(logits[:, n_seg:].float(),
                                     masks[:, :1].float())
                extra  = getattr(model, "distill_term", None)
                if extra is not None:
                    loss = loss + extra
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                update_ema(core, ema_model, args.ema_decay)
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # -- val (skip if no val split) --
        eval_model = ema_model if ema_model is not None else core
        if val_loader is not None:
            grid = args.threshold_sweep if (val_decoder is not None and
                                            len(args.threshold_sweep) > 1) \
                else [args.threshold]
            best_thr, val_metrics = args.threshold, None
            for thr in grid:
                if val_decoder is not None:
                    val_decoder.mask_thr = thr
                m = evaluate(eval_model, val_loader, device, thr,
                             decoder=val_decoder,
                             max_instance_chips=args.val_instance_chips)
                key = m.get(select, m["loss"])
                cur = val_metrics.get(select, val_metrics["loss"]) if val_metrics else None
                take = (val_metrics is None
                        or (key < cur if select == "loss" else key > cur))
                if take:
                    best_thr, val_metrics = thr, m
            line = (f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_f1={val_metrics['f1']:.4f}  "
                    f"val_iou={val_metrics['iou']:.4f}")
            if "pq" in val_metrics:
                line += (f"  PQ={100*val_metrics['pq']:.2f}"
                         f" (SQ {100*val_metrics['sq']:.1f}"
                         f" RQ {100*val_metrics['rq']:.1f})")
            if len(grid) > 1:
                line += f" @thr{best_thr:g}"
            print(line + (" [ema]" if ema_model is not None else ""))
            score = val_metrics.get(select, val_metrics["loss"])
            better = score < best_score if select == "loss" else score > best_score
            if better:
                best_score = score
                stale = 0
                torch.save(eval_model.state_dict(), out_dir / "best.pth")
                import json as _json
                (out_dir / "best_threshold.json").write_text(_json.dumps(
                    {"threshold": best_thr, "select_on": select,
                     "score": float(best_score), "epoch": epoch}))
                print(f"          -> saved best ({select}="
                      f"{best_score:.4f} @thr {best_thr:g})")
            else:
                stale += 1
                if args.patience and stale >= args.patience:
                    print(f"          -> early stop: no {select} improvement "
                          f"for {stale} epochs (best {best_score:.4f})")
                    break
        else:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}"
                  + (" [ema]" if ema_model is not None else ""))
            torch.save(eval_model.state_dict(), out_dir / "best.pth")

    save_model = ema_model if ema_model is not None else core
    torch.save(save_model.state_dict(), out_dir / "last.pth")

    # ── Final test evaluation ─────────────────────────────────────────────────
    core.load_state_dict(torch.load(out_dir / "best.pth", map_location=device))
    if "test" in ds:
        test_loader = DataLoader(SegDataset(ds["test"], train=False, size=args.size,
                                            targets=targets,
                                            semantic_only=args.heads == "semantic"),
                                 batch_size=args.batch, num_workers=args.workers)
        test_metrics = evaluate(core, test_loader, device, args.threshold)
        print(f"\nTest (best checkpoint):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
    print(f"\nDone. Checkpoints -> {out_dir}")


if __name__ == "__main__":
    main()

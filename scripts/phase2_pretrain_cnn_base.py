"""Phase 2: pre-train a building-base ResNet34-UNet on HOT's global dataset.

Gives the CNN what the shipped DINOv3 model has: a task-matched base decoder
(the shipped CNN base is a tree-crown model). Mirrors the ViT base-training
design: encoder frozen (ImageNet), decoder + 2-class head from random init on
hotosm/vhr-building-segmentation (57,890 train / 7,237 val chips).

Recipe: production CNN loss (2-class CE) and preprocessing (/255, enforced by
pre-writing unit norm stats), wd 1e-4, cosine schedule, batch 32, grad clip
1.0. lr 1e-3 rather than the production fine-tune 1e-4: the H5 sweep shows
1e-3 is optimal for training a decoder from scratch (documented deviation).
Best val-CE checkpoint is saved to data/base_ckpts/unet_bldg_base.pth.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas  # noqa: F401
import sklearn.utils  # noqa: F401
import sklearn.base   # noqa: F401

import torch
import segmentation_models_pytorch as smp
from dinov3_hot.data import HotBuildingDataModule

ROOT = Path(__file__).resolve().parent.parent

_p = argparse.ArgumentParser()
_p.add_argument("--dataset-repo", default="hotosm/vhr-building-segmentation")
_p.add_argument("--data-root", default=str(ROOT / "data" / "hot_building"),
                help="Use a fresh dir per dataset repo (norm stats are cached here)")
_p.add_argument("--out", default=str(ROOT / "data" / "base_ckpts" / "unet_bldg_base.pth"))
_p.add_argument("--data-pct", type=float, default=100.0,
                help="Percent of train rows (shuffled, seed 1337) - for size-matched controls")
_p.add_argument("--freeze-encoder", default=True, action=argparse.BooleanOptionalAction,
                help="--no-freeze-encoder trains the ResNet34 end to end "
                     "(encoder gets --encoder-lr, default lr/10)")
_p.add_argument("--encoder-lr", type=float, default=None)
_p.add_argument("--loss", default="ce", choices=["ce", "gce_tversky"])
_p.add_argument("--gce-q", type=float, default=0.7)
_p.add_argument("--tversky-alpha", type=float, default=0.3)
_p.add_argument("--tversky-beta", type=float, default=0.7)
_p.add_argument("--augment", default="basic", choices=["basic", "geo", "strong"],
                help="basic = flips only (datamodule default); geo adds rotation "
                     "and zoom-crop; strong additionally adds color jitter")
_p.add_argument("--mosaic", action="store_true",
                help="2x2 mosaic augmentation: stitch 4 batch samples, random 256 crop "
                     "(applied to half the samples per batch)")
_p.add_argument("--ema-decay", type=float, default=0.0,
                help="EMA of weights; val + checkpoint use the EMA model. 0 disables")
_p.add_argument("--encoder", default="resnet34",
                choices=["resnet18", "resnet34", "efficientnet-b0"])
_p.add_argument("--patience", type=int, default=5)
_p.add_argument("--focal-gamma", type=float, default=0.0,
                help="Focal-Tversky exponent (1-TI)^gamma; 0 = plain Tversky term")
_p.add_argument("--init-ckpt", default=None,
                help="Warm-start from this state dict (e.g. noisy-pretrain -> clean-finetune)")
_p.add_argument("--lr", type=float, default=1e-3,
                help="Decoder/head learning rate (encoder gets --encoder-lr or lr/10)")
_p.add_argument("--blurpool", action="store_true",
                help="Zhang anti-aliasing retrofit on the encoder (resnet18/34 only)")
_p.add_argument("--decoder-silu", action="store_true",
                help="Swap decoder ReLU activations for SiLU (decoder trains from scratch)")
_p.add_argument("--heads", default="semantic", choices=["semantic", "instance"],
                help="instance = 3 output channels (mask, boundary, signed distance), "
                     "same head semantics as the shipped DINOv3 stack so its "
                     "watershed instance separation applies unchanged")
_args = _p.parse_args()
DATASET_REPO = _args.dataset_repo
DATA_ROOT = Path(_args.data_root)
OUT = Path(_args.out)

LR = _args.lr
WEIGHT_DECAY = 1e-4
BATCH = 32
MAX_EPOCHS = 50
CLIP = 1.0


def train_transform(level):
    """geo = flips + rotation + zoom-crop; strong adds photometric jitter.
    v2 is type-aware: geometric ops use nearest on the Mask, ColorJitter
    touches the Image only. Rotation corners are filled with background; the
    zoom-crop usually removes them. Norm stays unit stats (production /255)."""
    from torchvision.transforms import v2

    ops = [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(15, fill=0),
        v2.RandomResizedCrop(256, scale=(0.7, 1.0), ratio=(1.0, 1.0), antialias=True),
    ]
    if level == "strong":
        ops.append(v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15))
    return v2.Compose([
        *ops,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])


def mosaic_batch(x, y):
    """2x2 mosaic on ~half the batch: four samples stitched to a 512 canvas,
    then a random 256 crop. Preserves scale; run after normalisation."""
    b = x.shape[0]
    if b < 4:
        return x, y
    out_x, out_y = x.clone(), y.clone()
    for i in range(b):
        if torch.rand(()) > 0.5:
            continue
        j = torch.randperm(b)[:4]
        cx = torch.cat([torch.cat([x[j[0]], x[j[1]]], dim=2),
                        torch.cat([x[j[2]], x[j[3]]], dim=2)], dim=1)
        cy = torch.cat([torch.cat([y[j[0]], y[j[1]]], dim=1),
                        torch.cat([y[j[2]], y[j[3]]], dim=1)], dim=0)
        top = int(torch.randint(0, 257, ()))
        left = int(torch.randint(0, 257, ()))
        out_x[i] = cx[:, top:top + 256, left:left + 256]
        out_y[i] = cy[top:top + 256, left:left + 256]
    return out_x, out_y


class Ema:
    """Exponential moving average of all state-dict tensors; float tensors are
    averaged, integer buffers (BN counters) copied."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def state_dict(self, model):
        ref = model.state_dict()
        return {k: s.to(ref[k].dtype) for k, s in self.shadow.items()}


def gce_tversky_loss(logits, y):
    """Generalized cross-entropy (Zhang & Sabuncu 2018, L_q with q=--gce-q;
    noise-robust: down-weights confident disagreements with the label) plus
    a Tversky loss on the building class (alpha penalises FP, beta FN)."""
    p = torch.softmax(logits.float(), dim=1)
    p_y = p.gather(1, y.unsqueeze(1)).squeeze(1).clamp_min(1e-7)
    gce = ((1.0 - p_y.pow(_args.gce_q)) / _args.gce_q).mean()
    p1, yf = p[:, 1], (y == 1).float()
    tp = (p1 * yf).sum()
    fp = (p1 * (1.0 - yf)).sum()
    fn = ((1.0 - p1) * yf).sum()
    tversky = 1.0 - (tp + 1.0) / (tp + _args.tversky_alpha * fp + _args.tversky_beta * fn + 1.0)
    if _args.focal_gamma > 0:
        tversky = tversky.pow(_args.focal_gamma)
    return gce + tversky


def criterion(logits, y):
    if _args.loss == "gce_tversky":
        return gce_tversky_loss(logits, y)
    return torch.nn.functional.cross_entropy(logits, y)


def gce_tversky_binary(logit, y):
    """Binary-logit variant of the GCE+Tversky loss, for the mask head."""
    p1 = torch.sigmoid(logit.float())
    p_y = torch.where(y > 0.5, p1, 1.0 - p1).clamp_min(1e-7)
    gce = ((1.0 - p_y.pow(_args.gce_q)) / _args.gce_q).mean()
    yf = (y > 0.5).float()
    tp = (p1 * yf).sum()
    fp = (p1 * (1.0 - yf)).sum()
    fn = ((1.0 - p1) * yf).sum()
    tversky = 1.0 - (tp + 1.0) / (tp + _args.tversky_alpha * fp + _args.tversky_beta * fn + 1.0)
    if _args.focal_gamma > 0:
        tversky = tversky.pow(_args.focal_gamma)
    return gce + tversky


# Head weights follow the shipped ViT base's HPO values, rounded.
BOUNDARY_W = 0.27
DISTANCE_W = 0.47


def instance_criterion(logits, batch, device):
    """mask (GCE+Tversky) + boundary (BCE) + signed distance (tanh + MSE)."""
    y = (batch["mask"] > 0.5).float().to(device)
    b = batch["boundary"].float().to(device)
    d = batch["distance"].float().to(device)
    lm, lb = logits[:, 0], logits[:, 1]
    ld = torch.tanh(logits[:, 2])
    return (gce_tversky_binary(lm, y)
            + BOUNDARY_W * torch.nn.functional.binary_cross_entropy_with_logits(lb, b)
            + DISTANCE_W * torch.nn.functional.mse_loss(ld, d))


def main():
    torch.manual_seed(1337)
    device = torch.device("cuda")

    # Production CNN preprocessing is images/255 only: pre-write unit norm
    # stats so the datamodule's Normalize becomes a no-op after ToDtype scaling.
    # (Run AFTER the ViT base pretrain, which needs the real hot_global stats.)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "norm_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}))

    dm = HotBuildingDataModule(
        repo_id=DATASET_REPO, root=DATA_ROOT,
        img_size=256, boundary_width=2, distance_clip=15.0,
        drop_null_images=True, data_pct=_args.data_pct,
        batch_size=BATCH, eval_batch_size=BATCH,
        num_workers=8, pin_memory=True, persistent_workers=True, seed=1337)
    dm.setup()
    if _args.augment in ("geo", "strong"):
        dm._train_tf = train_transform(_args.augment)
        print(f"augment={_args.augment}"
              + (" + mosaic" if _args.mosaic else ""))

    n_out = 3 if _args.heads == "instance" else 2
    assert not (_args.heads == "instance" and _args.mosaic), \
        "mosaic operates on (image, mask) only and would desync boundary/distance targets"
    m = smp.create_model("unet", encoder_name=_args.encoder,
                         encoder_weights="imagenet", classes=n_out, in_channels=3)
    if _args.blurpool:
        assert _args.encoder in ("resnet18", "resnet34"), "blurpool retrofit is BasicBlock-only"
        from model_mods import apply_blurpool
        apply_blurpool(m.encoder)
        print("blurpool retrofit applied to encoder")
    if _args.decoder_silu:
        from model_mods import swap_relu_silu
        swap_relu_silu(m.decoder)
        print("decoder activations swapped to SiLU")
    if _args.init_ckpt:
        state = torch.load(_args.init_ckpt, map_location="cpu", weights_only=True)
        m.load_state_dict(state)
        print(f"warm-started from {_args.init_ckpt}")
    m.to(device)

    if _args.freeze_encoder:
        for p in m.encoder.parameters():
            p.requires_grad = False
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        enc_lr = _args.encoder_lr if _args.encoder_lr is not None else LR * 0.1
        head = [p for n, p in m.named_parameters() if not n.startswith("encoder.")]
        opt = torch.optim.AdamW(
            [{"params": m.encoder.parameters(), "lr": enc_lr},
             {"params": head, "lr": LR}],
            weight_decay=WEIGHT_DECAY)
        print(f"encoder unfrozen: encoder lr={enc_lr:g}, decoder/head lr={LR:g}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler()
    ema = Ema(m, _args.ema_decay) if _args.ema_decay > 0 else None

    best, bad = float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        m.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for batch in dm.train_dataloader():
            x = batch["image"].to(device, non_blocking=True)
            y = (batch["mask"] > 0.5).long().to(device, non_blocking=True)
            if _args.mosaic:
                x, y = mosaic_batch(x, y)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = m(x)
                loss = (instance_criterion(logits, batch, device)
                        if _args.heads == "instance" else criterion(logits, y))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(m)
            tot += loss.item(); n += 1
        sched.step()

        m.eval()
        if ema is not None:
            raw_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            m.load_state_dict(ema.state_dict(m))
        vt, vn = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for batch in dm.val_dataloader():
                x = batch["image"].to(device)
                y = (batch["mask"] > 0.5).long().to(device)
                logits = m(x)
                vloss = (instance_criterion(logits, batch, device)
                         if _args.heads == "instance" else criterion(logits, y))
                vt += vloss.item(); vn += 1
        val_loss = vt / max(1, vn)
        print(f"epoch {epoch}/{MAX_EPOCHS}  train_loss={tot/max(1,n):.4f}  "
              f"val_loss={val_loss:.4f}  ({(time.perf_counter()-t0)/60:.1f} min)",
              flush=True)

        stop = False
        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), OUT)
        else:
            bad += 1
            stop = bad >= _args.patience
        if ema is not None:
            m.load_state_dict(raw_state)
        if stop:
            print("early stop")
            break

    print(f"best val_ce={best:.4f}  saved: {OUT}")


if __name__ == "__main__":
    main()

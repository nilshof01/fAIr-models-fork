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
_args = _p.parse_args()
DATASET_REPO = _args.dataset_repo
DATA_ROOT = Path(_args.data_root)
OUT = Path(_args.out)

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 32
MAX_EPOCHS = 50
PATIENCE = 5
CLIP = 1.0


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
        drop_null_images=True, data_pct=100.0,
        batch_size=BATCH, eval_batch_size=BATCH,
        num_workers=8, pin_memory=True, persistent_workers=True, seed=1337)
    dm.setup()

    m = smp.create_model("unet", encoder_name="resnet34",
                         encoder_weights="imagenet", classes=2, in_channels=3)
    for p in m.encoder.parameters():
        p.requires_grad = False
    m.to(device)

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler()

    best, bad = float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        m.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for batch in dm.train_dataloader():
            x = batch["image"].to(device, non_blocking=True)
            y = (batch["mask"] > 0.5).long().to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(m(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); n += 1
        sched.step()

        m.eval()
        vt, vn = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for batch in dm.val_dataloader():
                x = batch["image"].to(device)
                y = (batch["mask"] > 0.5).long().to(device)
                vt += torch.nn.functional.cross_entropy(m(x), y).item(); vn += 1
        val_loss = vt / max(1, vn)
        print(f"epoch {epoch}/{MAX_EPOCHS}  train_ce={tot/max(1,n):.4f}  "
              f"val_ce={val_loss:.4f}  ({(time.perf_counter()-t0)/60:.1f} min)",
              flush=True)

        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), OUT)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stop")
                break

    print(f"best val_ce={best:.4f}  saved: {OUT}")


if __name__ == "__main__":
    main()

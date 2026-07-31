"""Zero-shot eval of the UNet base checkpoints on hotosm/vhr-building-segmentation.

Scores the shipped tree-crown base and the phase-2 building base on the
dataset's own validation and test splits: pixel accuracy, per-class IoU,
building F1 at the production operating point (argmax), plus a 17-point
threshold curve on sigmoid(logit_bldg - logit_bg) for context.

Production CNN preprocessing is /255 only: data/hot_building/norm_stats.json
must hold unit stats so the datamodule Normalize is a no-op (it does after
the phase-2 CNN pretrain; verified before running).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas  # noqa: F401
import sklearn.utils  # noqa: F401
import sklearn.base   # noqa: F401

import numpy as np
import torch
import segmentation_models_pytorch as smp
from dinov3_hot.data import HotBuildingDataModule

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "hot_building"
OUT_JSON = ROOT / "data" / "base_eval_dataset_val.json"
CKPTS = {
    "tree_crown_shipped": ROOT / "data" / "base_ckpts" / "unet_base.ckpt",
    "building_phase2": ROOT / "data" / "base_ckpts" / "unet_bldg_base.pth",
    "building_v0_1500_curated": ROOT / "data" / "base_ckpts" / "unet_bldg_v0_1500.pth",
    "building_rand1445_control": ROOT / "data" / "base_ckpts" / "unet_bldg_rand1445.pth",
}
GRID = np.linspace(0.1, 0.9, 17)
BATCH = 64


def build(ckpt_path, device):
    m = smp.create_model("unet", encoder_name="resnet34", encoder_weights=None,
                         classes=2, in_channels=3)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(state, strict=False)
    print(f"  loaded {ckpt_path.name}: {len(missing)} missing, "
          f"{len(unexpected)} unexpected keys", flush=True)
    return m.to(device).eval()


def evaluate(model, loader, device):
    n_thr = len(GRID)
    thr = torch.tensor(GRID, device=device, dtype=torch.float32).view(-1, 1, 1, 1)
    inter = torch.zeros(2, device=device)
    union = torch.zeros(2, device=device)
    correct = torch.zeros((), device=device)
    tp = torch.zeros((), device=device)
    fp = torch.zeros((), device=device)
    fn = torch.zeros((), device=device)
    ttp = torch.zeros(n_thr, device=device)
    tfp = torch.zeros(n_thr, device=device)
    tfn = torch.zeros(n_thr, device=device)
    total = n_img = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = (batch["mask"] > 0.5).long().to(device, non_blocking=True)
            if y.dim() == 4:
                y = y.squeeze(1)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum()
            total += y.numel()
            for c in (0, 1):
                inter[c] += ((pred == c) & (y == c)).sum()
                union[c] += ((pred == c) | (y == c)).sum()
            tp += ((pred == 1) & (y == 1)).sum()
            fp += ((pred == 1) & (y == 0)).sum()
            fn += ((pred == 0) & (y == 1)).sum()

            prob = torch.sigmoid(logits[:, 1] - logits[:, 0]).unsqueeze(0)
            yb = (y == 1).unsqueeze(0)
            pt = prob > thr
            ttp += (pt & yb).sum(dim=(1, 2, 3)).float()
            tfp += (pt & ~yb).sum(dim=(1, 2, 3)).float()
            tfn += (~pt & yb).sum(dim=(1, 2, 3)).float()
            n_img += x.shape[0]

    curve = (2 * ttp / (2 * ttp + tfp + tfn).clamp(min=1)).cpu().numpy()
    iou = (inter / union.clamp(min=1)).cpu().numpy()
    return {
        "n_chips": n_img,
        "accuracy": (correct / max(total, 1)).item(),
        "iou_background": float(iou[0]),
        "iou_building": float(iou[1]),
        "mean_iou": float(iou.mean()),
        "f1_building_argmax": (2 * tp / (2 * tp + fp + fn).clamp(min=1)).item(),
        "curve_thresholds": GRID.tolist(),
        "curve_f1": curve.tolist(),
        "f1_best_threshold": float(curve.max()),
        "best_threshold": float(GRID[int(curve.argmax())]),
    }


def main():
    torch.manual_seed(1337)
    device = torch.device("cuda")
    stats = json.loads((DATA_ROOT / "norm_stats.json").read_text())
    assert stats["mean"] == [0.0, 0.0, 0.0] and stats["std"] == [1.0, 1.0, 1.0], \
        f"norm_stats.json must be unit stats for CNN eval, got {stats}"

    dm = HotBuildingDataModule(
        repo_id="hotosm/vhr-building-segmentation", root=DATA_ROOT,
        img_size=256, boundary_width=2, distance_clip=15.0,
        drop_null_images=True, data_pct=100.0,
        batch_size=BATCH, eval_batch_size=BATCH,
        num_workers=8, pin_memory=True, persistent_workers=False, seed=1337)
    dm.setup(stage="validate")
    dm.setup(stage="test")
    loaders = {"validation": dm.val_dataloader(), "test": dm.test_dataloader()}

    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for name, ckpt in CKPTS.items():
        print(f"### {name} ###", flush=True)
        model = build(ckpt, device)
        for split, loader in loaders.items():
            if results.get(name, {}).get(split):
                print(f"  skip {split} (already done)", flush=True)
                continue
            t0 = time.perf_counter()
            r = evaluate(model, loader, device)
            results.setdefault(name, {})[split] = r
            OUT_JSON.write_text(json.dumps(results, indent=2))
            print(f"  {split}: n={r['n_chips']}  acc={r['accuracy']*100:.2f}  "
                  f"IoU_bldg={r['iou_building']*100:.2f}  mIoU={r['mean_iou']*100:.2f}  "
                  f"F1_argmax={r['f1_building_argmax']*100:.2f}  "
                  f"F1_best={r['f1_best_threshold']*100:.2f}@{r['best_threshold']:.2f}  "
                  f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
        del model
        torch.cuda.empty_cache()

    print("done", flush=True)


if __name__ == "__main__":
    main()

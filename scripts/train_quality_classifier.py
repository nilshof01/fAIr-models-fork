"""Train a label-quality classifier on top of the frozen EfficientNet-B4 encoder.

Pipeline
--------
1. Fetch decisions from Supabase  (keep=1 / everything else=0)
2. Load each chip from HF dataset by dataset_row
3. Run through frozen UNet → compute 9 pred-quality features
4. One-hot encode project_id (top-N projects)
5. Train XGBoost with cross-validation
6. Score all remaining unreviewed chips → mlp_scores.npy / mlp_row_ids.npy

Usage
-----
  # Full pipeline
  python scripts/train_quality_classifier.py \
      --checkpoint "checkpoints/effb4_dist3_v2/best.pth" \
      --encoder efficientnet-b4 \
      --source hotosm/vhr-building-segmentation \
      --out label-cleanup/run2/quality_mlp

  # Skip feature extraction if already done
  python scripts/train_quality_classifier.py \
      --checkpoint "checkpoints/effb4_dist3_v2/best.pth" \
      --encoder efficientnet-b4 \
      --source hotosm/vhr-building-segmentation \
      --out label-cleanup/run2/quality_mlp \
      --skip-extract

  # Also score unreviewed chips
  python scripts/train_quality_classifier.py ... --score-remaining
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    raise SystemExit("pip install albumentations")

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise SystemExit("pip install segmentation-models-pytorch")

NORM_MEAN      = [0.485, 0.456, 0.406]
NORM_STD       = [0.229, 0.224, 0.225]
IMG_SIZE       = 256
N_PRED_FEATS   = 9    # iou, prec, rec, gt_cov, pred_cov,
                      # boundary_grad, gt_comp_ratio, rgb_var_inside, pred_gt_comp_ratio
N_TOP_PROJECTS = 50   # one-hot columns; remainder → "other" bucket


DECISIONS_REPO = "nilsho01/vhr-buildings-review-decisions"


def fetch_decisions() -> dict[int, int]:
    """Return {dataset_row: label} where label=1 (keep) or 0 (bad).

    Loads directly from the HF review-decisions dataset so all ~6k labels
    are used, not just those synced to Supabase.
    """
    import pandas as pd
    from datasets import load_dataset

    print(f"Loading decisions from {DECISIONS_REPO} ...")
    rev = load_dataset(DECISIONS_REPO, split="train").to_pandas()

    labels = {}
    for _, r in rev.iterrows():
        if pd.isna(r.get("dataset_row")):
            continue
        labels[int(r["dataset_row"])] = 1 if r["usable"] else 0

    good = sum(v for v in labels.values())
    bad  = len(labels) - good
    print(f"Loaded {len(labels)} decisions — good={good}  bad={bad}")
    return labels


def merge_hf_keeps(labels: dict[int, int], extra_hf: str) -> dict[int, int]:
    """Add labeled chips from extra_hf as label=1, using their dataset_row field.

    Supabase decisions take precedence — if a chip already has an explicit
    decision in labels, we don't override it.  Empty chips (n_instances == 0)
    are skipped since they carry no label-quality signal.
    """
    from datasets import concatenate_datasets

    print(f"\nLoading extra keeps from {extra_hf} ...")
    extra_raw = load_dataset(extra_hf)
    extra_ds  = concatenate_datasets([extra_raw[s] for s in extra_raw.keys()])
    print(f"  {len(extra_ds)} chips total in extra dataset")

    added = skipped_existing = skipped_empty = 0
    for row in extra_ds:
        if int(row.get("n_instances", 0)) == 0:
            skipped_empty += 1
            continue
        row_idx = int(row["dataset_row"])
        if row_idx in labels:
            skipped_existing += 1
            continue
        labels[row_idx] = 1
        added += 1

    print(f"  Added {added} new keep labels  "
          f"(skipped {skipped_existing} already-decided, "
          f"{skipped_empty} empty chips)")
    good = sum(v for v in labels.values())
    bad  = len(labels) - good
    print(f"  Total after merge: {len(labels)} decisions — good={good}  bad={bad}")
    return labels


# ── Extra label-quality features ──────────────────────────────────────────────

def _extra_features(rgb_np: np.ndarray, mask_np: np.ndarray,
                    pred: np.ndarray, n_instances: int) -> list[float]:
    """Return [boundary_grad, gt_comp_ratio, rgb_var_inside, pred_gt_comp_ratio].

    boundary_grad     : mean Sobel gradient at GT mask edge — good labels sit on
                        sharp image boundaries; bad ones don't
    gt_comp_ratio     : n_gt_components / n_instances — >>1 means fragmented mask
    rgb_var_inside    : pixel variance inside GT mask — buildings have texture;
                        vegetation/ground are uniform
    pred_gt_comp_ratio: n_pred_components / n_gt_components — instance count
                        disagreement between model and GT
    """
    from scipy import ndimage

    # 1. Sobel gradient at GT mask boundary
    if mask_np.any():
        gray     = rgb_np.mean(axis=2).astype(np.float32)
        sx       = ndimage.sobel(gray, axis=0)
        sy       = ndimage.sobel(gray, axis=1)
        grad     = np.sqrt(sx ** 2 + sy ** 2)
        eroded   = ndimage.binary_erosion(mask_np)
        boundary = mask_np & ~eroded
        boundary_grad = float(grad[boundary].mean()) if boundary.any() else 0.0
    else:
        boundary_grad = 0.0

    # 2. GT component count vs annotated instance count
    _, n_gt_comp   = ndimage.label(mask_np)
    gt_comp_ratio  = n_gt_comp / max(int(n_instances), 1)

    # 3. RGB variance inside GT mask
    rgb_var_inside = float(rgb_np[mask_np].astype(np.float32).var()) \
                     if mask_np.any() else 0.0

    # 4. Pred vs GT component count ratio
    _, n_pred_comp      = ndimage.label(pred)
    pred_gt_comp_ratio  = n_pred_comp / max(n_gt_comp, 1)

    return [boundary_grad, gt_comp_ratio, rgb_var_inside, pred_gt_comp_ratio]


# ── Project-ID encoding ───────────────────────────────────────────────────────

def build_proj_vocab(proj_ids: list[str], top_n: int = N_TOP_PROJECTS) -> dict[str, int]:
    """Return {project_id: column_index} for the top_n most common projects."""
    counts = Counter(str(p) for p in proj_ids)
    top    = [p for p, _ in counts.most_common(top_n)]
    return {p: i for i, p in enumerate(top)}


def encode_proj_ids(proj_ids: list[str], vocab: dict[str, int]) -> np.ndarray:
    """One-hot [N, len(vocab)+1]; last column is 'other'."""
    n_cols = len(vocab) + 1
    feats  = np.zeros((len(proj_ids), n_cols), dtype=np.float32)
    for i, pid in enumerate(proj_ids):
        col = vocab.get(str(pid), len(vocab))   # unknown → "other"
        feats[i, col] = 1.0
    return feats


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(args, device):
    """Return full frozen UNet (encoder + decoder) in eval mode."""
    n_out = {"dist": 2, "dist3": 3, "instance": 3,
             "center": 2, "flow": 3}.get(args.heads, 1)
    m = smp.Unet(encoder_name=args.encoder, encoder_weights=None,
                 in_channels=3, classes=n_out)
    m.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                 weights_only=True))
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def extract_features(model, dataset_rows: list[int], hf_ds,
                     device, threshold: float = 0.5
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (pred_feats [N, N_PRED_FEATS], project_ids [N]).

    Uses a forward hook on model.encoder to drive the forward pass; the hook
    result is discarded — only logits (for pred features) are used.
    """
    norm = A.Compose([A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()])

    _enc_out: dict = {}
    def _hook(_m, _inp, out):
        _enc_out["last"] = out[-1]
    handle = model.encoder.register_forward_hook(_hook)

    pred_feats, proj_ids, n_instances_list = [], [], []
    try:
        for row in tqdm(dataset_rows, desc="extracting"):
            sample      = hf_ds[int(row)]
            n_instances = int(sample.get("n_instances", 0))
            proj_ids.append(str(sample.get("project_id", "unknown")))
            n_instances_list.append(n_instances)

            rgb_np  = np.asarray(sample["image"].convert("RGB")
                                 .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            mask_np = (np.asarray(sample["mask"].convert("L")
                                  .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 0)
            img_t   = norm(image=rgb_np, mask=mask_np.astype(np.uint8))["image"]

            logits = model(img_t.unsqueeze(0).to(device))

            pred = (logits.sigmoid().cpu().numpy()[0, 0] > threshold)
            tp = int((pred &  mask_np).sum())
            fp = int((pred & ~mask_np).sum())
            fn = int((~pred & mask_np).sum())
            pred_feats.append([
                tp / max(tp + fp + fn, 1),   # iou
                tp / max(tp + fp, 1),          # precision
                tp / max(tp + fn, 1),          # recall
                float(mask_np.mean()),         # gt_cov
                float(pred.mean()),            # pred_cov
                *_extra_features(rgb_np, mask_np, pred, n_instances),
            ])
    finally:
        handle.remove()

    return (np.array(pred_feats, dtype=np.float32),
            proj_ids,
            np.array(n_instances_list, dtype=np.int32))


# ── XGBoost CV ───────────────────────────────────────────────────────────────

def cross_validate(X: np.ndarray, y: np.ndarray, n_folds: int = 5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from xgboost import XGBClassifier

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"\n-- {n_folds}-fold stratified CV (XGBoost) "
          f"(good={int(y.sum())}  bad={int((y==0).sum())}) ----------")
    print(f"  {'fold':<6}  {'AUC':>6}  {'F1':>6}  {'P':>6}  {'R':>6}  "
          f"{'n_val':>6}  {'good_val':>8}  {'bad_val':>7}")

    aucs, f1s, ps, rs = [], [], [], []
    oof_probs = np.zeros(len(y), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        n_pos = int((y[tr_idx] == 1).sum())
        n_neg = int((y[tr_idx] == 0).sum())
        clf   = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        clf.fit(X[tr_idx], y[tr_idx].astype(int))
        probs = clf.predict_proba(X[va_idx])[:, 1].astype(np.float32)

        oof_probs[va_idx] = probs
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y[va_idx], probs)
        f1  = f1_score(y[va_idx], preds, zero_division=0)
        p   = precision_score(y[va_idx], preds, zero_division=0)
        r   = recall_score(y[va_idx], preds, zero_division=0)
        aucs.append(auc); f1s.append(f1); ps.append(p); rs.append(r)

        n_good = int(y[va_idx].sum())
        n_bad  = len(va_idx) - n_good
        print(f"  {fold+1:<6}  {auc:.4f}  {f1:.4f}  {p:.4f}  {r:.4f}  "
              f"{len(va_idx):>6}  {n_good:>8}  {n_bad:>7}")

    print(f"  {'mean':<6}  {np.mean(aucs):.4f}  {np.mean(f1s):.4f}  "
          f"{np.mean(ps):.4f}  {np.mean(rs):.4f}")
    print(f"  {'std':<6}  {np.std(aucs):.4f}  {np.std(f1s):.4f}  "
          f"{np.std(ps):.4f}  {np.std(rs):.4f}")

    # ── OOF precision / recall at threshold ──────────────────────────────────
    n_bad = int((y == 0).sum())
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print("\n-- OOF precision / recall at threshold -------------------------")
    print(f"  {'thr':>5}  {'accept':>7}  {'precision':>10}  {'recall':>8}  {'bad_caught':>11}")
    print(f"  {'':>5}  {'':>7}  {'(of accept)':>10}  {'(of good)':>8}  {'(of bad)':>11}")
    for thr in thresholds:
        accept     = oof_probs >= thr
        tp         = int(( accept & (y == 1)).sum())
        fp         = int(( accept & (y == 0)).sum())
        fn         = int((~accept & (y == 1)).sum())
        tn         = int((~accept & (y == 0)).sum())
        prec       = tp / max(tp + fp, 1)
        rec        = tp / max(tp + fn, 1)
        bad_caught = tn / max(n_bad, 1)
        print(f"  {thr:>5.2f}  {tp+fp:>7}  {prec:>10.1%}  {rec:>8.1%}  {bad_caught:>11.1%}")

    return np.mean(aucs)


def train_final(X: np.ndarray, y: np.ndarray):
    """Fit XGBoost on full data; returns clf (no scaler needed)."""
    from xgboost import XGBClassifier

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    clf   = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    print("\nTraining final XGBoost on all data ...")
    clf.fit(X, y.astype(int))
    return clf


# ── Score remaining chips ─────────────────────────────────────────────────────

@torch.no_grad()
def score_remaining(clf, proj_vocab: dict[str, int],
                    model, hf_ds, reviewed_rows: set[int],
                    device, out_dir: Path, threshold: float = 0.5) -> None:
    all_rows   = list(range(len(hf_ds)))
    unreviewed = [r for r in all_rows if r not in reviewed_rows]
    print(f"\nScoring {len(unreviewed)} unreviewed chips ...")

    norm = A.Compose([A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()])

    _enc_out: dict = {}
    def _hook(_m, _inp, out):
        _enc_out["last"] = out[-1]
    handle = model.encoder.register_forward_hook(_hook)

    scores, rows_out = [], []
    try:
        for row in tqdm(unreviewed, desc="scoring"):
            sample      = hf_ds[int(row)]
            n_instances = int(sample.get("n_instances", 0))
            proj_id     = str(sample.get("project_id", "unknown"))

            rgb_np  = np.asarray(sample["image"].convert("RGB")
                                 .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            mask_np = (np.asarray(sample["mask"].convert("L")
                                  .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 0)
            img_t   = norm(image=rgb_np, mask=mask_np.astype(np.uint8))["image"]

            logits = model(img_t.unsqueeze(0).to(device))
            pred   = (logits.sigmoid().cpu().numpy()[0, 0] > threshold)

            tp = int((pred &  mask_np).sum())
            fp = int((pred & ~mask_np).sum())
            fn = int((~pred & mask_np).sum())

            pf = np.array([
                tp / max(tp + fp + fn, 1),
                tp / max(tp + fp, 1),
                tp / max(tp + fn, 1),
                float(mask_np.mean()),
                float(pred.mean()),
                *_extra_features(rgb_np, mask_np, pred, n_instances),
            ], dtype=np.float32)

            proj_oh  = encode_proj_ids([proj_id], proj_vocab)[0]
            feat_vec = np.concatenate([pf, proj_oh]).reshape(1, -1)
            score    = clf.predict_proba(feat_vec)[0, 1]
            scores.append(score)
            rows_out.append(row)
    finally:
        handle.remove()

    scores   = np.array(scores,   dtype=np.float32)
    rows_out = np.array(rows_out, dtype=np.int64)

    np.save(out_dir / "mlp_scores.npy",  scores)
    np.save(out_dir / "mlp_row_ids.npy", rows_out)

    thresholds = [0.9, 0.8, 0.7, 0.5]
    print("\n-- Score distribution ------------------------------------------")
    for thr in thresholds:
        n = int((scores >= thr).sum())
        print(f"  score >= {thr:.1f} : {n:6d} chips  ({100*n/len(scores):.1f}%)")

    print(f"\nSaved mlp_scores.npy + mlp_row_ids.npy -> {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    # ── Decisions ──────────────────────────────────────────────────────────────
    labels = fetch_decisions()
    if args.hf_extra_keeps:
        labels = merge_hf_keeps(labels, args.hf_extra_keeps)
    reviewed_rows = set(labels.keys())

    # ── Feature extraction / cache ─────────────────────────────────────────────
    pf_path      = out_dir / "pred_feats.npy"
    lbl_path     = out_dir / "labels.npy"
    row_path     = out_dir / "rows.npy"
    proj_id_path = out_dir / "proj_ids.json"
    ninst_path   = out_dir / "n_instances.npy"

    def _cache_valid():
        if not (pf_path.exists() and proj_id_path.exists() and row_path.exists()):
            return False, "missing files"
        pf_cached   = np.load(pf_path)
        rows_cached = np.load(row_path)
        if pf_cached.shape[1] != N_PRED_FEATS:
            return False, f"expected {N_PRED_FEATS} pred features, got {pf_cached.shape[1]}"
        cached_set  = set(rows_cached.tolist())
        label_set   = set(labels.keys())
        if cached_set != label_set:
            return False, (f"label set changed: cache has {len(cached_set)}, "
                           f"now have {len(label_set)}")
        return True, "ok"

    cache_ok, cache_reason = _cache_valid() if args.skip_extract else (False, "skip_extract not set")
    if args.skip_extract and not cache_ok:
        print(f"  Cache invalid ({cache_reason}) — re-extracting.")

    if cache_ok:
        print(f"Loading cached features from {out_dir} ...")
        pf         = np.load(pf_path)
        y          = np.load(lbl_path)
        rows       = np.load(row_path)
        proj_ids   = json.loads(proj_id_path.read_text())
        n_inst_arr = np.load(ninst_path) if ninst_path.exists() else np.zeros(len(rows), dtype=np.int32)
    else:
        print(f"\nLoading {args.source} ...")
        hf_ds = load_dataset(args.source, split=args.split)
        print(f"  {len(hf_ds)} chips")

        model = build_model(args, device)
        print(f"Loaded model from {args.checkpoint}")

        sorted_rows           = sorted(labels.keys())
        pf, proj_ids, n_inst_arr = extract_features(model, sorted_rows, hf_ds, device,
                                                     threshold=args.pred_threshold)
        y    = np.array([labels[r] for r in sorted_rows], dtype=np.float32)
        rows = np.array(sorted_rows, dtype=np.int64)

        np.save(pf_path,    pf)
        np.save(lbl_path,   y)
        np.save(row_path,   rows)
        np.save(ninst_path, n_inst_arr)
        proj_id_path.write_text(json.dumps(proj_ids))
        print(f"Saved features pf={pf.shape} + {len(proj_ids)} project IDs -> {out_dir}")

    # ── Project encoding ───────────────────────────────────────────────────────
    proj_vocab = build_proj_vocab(proj_ids, top_n=args.n_proj_features)
    proj_feats = encode_proj_ids(proj_ids, proj_vocab)
    print(f"Project vocab: {len(proj_vocab)} top projects + 1 'other' "
          f"({proj_feats.shape[1]} one-hot columns)")

    X = np.hstack([pf, proj_feats])
    print(f"\nFeature matrix: {X.shape}  "
          f"(pred={pf.shape[1]}  proj_oh={proj_feats.shape[1]})  "
          f"good={int(y.sum())}  bad={int((y==0).sum())}")

    # ── Cross-validate ─────────────────────────────────────────────────────────
    cross_validate(X, y.astype(int))

    # ── Train final model ──────────────────────────────────────────────────────
    clf = train_final(X, y.astype(int))

    import pickle
    model_path = out_dir / "quality_xgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"clf": clf, "proj_vocab": proj_vocab,
                     "n_pred_feats": N_PRED_FEATS}, f)
    print(f"Saved XGBoost model -> {model_path}")

    # ── Score remaining chips ──────────────────────────────────────────────────
    if args.score_remaining:
        if "hf_ds" not in dir():
            print(f"\nLoading {args.source} for scoring ...")
            hf_ds = load_dataset(args.source, split=args.split)
        if "model" not in dir():
            model = build_model(args, device)

        score_remaining(clf, proj_vocab, model,
                        hf_ds, reviewed_rows, device, out_dir,
                        threshold=args.pred_threshold)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",        required=True)
    ap.add_argument("--encoder",           default="efficientnet-b4")
    ap.add_argument("--heads",             default="dist3")
    ap.add_argument("--source",            default="hotosm/vhr-building-segmentation")
    ap.add_argument("--split",             default="train")
    ap.add_argument("--out",               default="label-cleanup/run2/quality_mlp")
    ap.add_argument("--device",            default=None)
    ap.add_argument("--pred-threshold",    type=float, default=0.5,
                    help="threshold for binarising model predictions when computing pred features")
    ap.add_argument("--n-proj-features",   type=int,   default=N_TOP_PROJECTS,
                    help="top-N project IDs to encode as one-hot; rest → 'other'")
    ap.add_argument("--hf-extra-keeps",    default=None,
                    metavar="REPO",
                    help="HF dataset repo whose chips are all treated as keep=1 "
                         "(matched to --source via tile_id); merged before extraction")
    ap.add_argument("--skip-extract",      action="store_true",
                    help="reuse cached pred_feats.npy + proj_ids.json if present")
    ap.add_argument("--score-remaining",   action="store_true",
                    help="score all unreviewed chips after training")
    return ap.parse_args()


if __name__ == "__main__":
    try:
        from sklearn.model_selection import StratifiedKFold as _  # noqa: F401
        from xgboost import XGBClassifier as _                     # noqa: F401
    except ImportError:
        raise SystemExit("pip install scikit-learn xgboost")
    main(parse_args())

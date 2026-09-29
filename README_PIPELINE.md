# Building instance segmentation — reproducible pipeline

This branch adds an instance-segmentation pipeline for building footprints on top
of fAIr's existing training code: a reviewed-label pool, a fixed benchmark, a
label-quality classifier that filters unreviewed chips, and an evaluation
protocol that reports Panoptic Quality rather than pixel F1.

`NOTES.md` holds the research record — why each choice was made and what was
measured. This file is the operational guide: what to run, in what order.

---

## Why Panoptic Quality

Pixel F1 is the wrong unit for this task. A model that merges every terrace into
one blob scores **84 pixel F1 while recovering a fifth of the buildings** —
measured, not hypothetical. PQ decomposes as `SQ x RQ`:

- **SQ** — mean IoU of matched instances. Are the outlines good?
- **RQ** — `TP / (TP + ½FP + ½FN)` over objects. Did you find the right things?

The split is the diagnostic. SQ 74 with RQ 51 means adequate outlines and broken
object recovery, which is a merging problem; the mirror case needs a completely
different fix. Report both, plus pixel F1 as a secondary for comparability with
the rest of the literature.

One caveat worth carrying: **RQ is robust to imprecise labels and SQ is not.**
Matching happens at IoU >= 0.5, which a two-pixel boundary wobble almost never
crosses, while SQ is IoU-based and inherits that noise directly.

---

## Setup

```bash
pip install -r requirements.txt segmentation-models-pytorch albumentations \
            datasets huggingface_hub xgboost scipy scikit-image tqdm
export HF_TOKEN=hf_...
```

Steps 1–7 read only the published HF datasets. `build_verified_pool.py`,
`build_benchmark.py` and `publish_review_decisions.py` additionally read the
review app's local SQLite, which is only needed to rebuild the pool from raw
decisions — point `VHR_REVIEW_DB` at it if you do.

On a pod, `HF_HUB_ENABLE_HF_TRANSFER=1` has produced corrupted encoder-weight
downloads (`SafetensorError: header too small`). Set it to `0` if that appears.

## Datasets

| Role | HF repo |
|---|---|
| Source images + OSM masks | `hotosm/vhr-building-segmentation` |
| Review decisions (metadata only) | `nilsho01/vhr-buildings-review-decisions` |
| Reviewed pool (images + masks) | `nilsho01/vhr-buildings-verified-v1` |
| Fixed benchmark, 400 chips | `nilsho01/vhr-buildings-benchmark-v1` |
| Auto-labelled pool | `nilsho01/vhr-buildings-automatic-annotation` |

The review-decisions repo carries **no image or mask columns**. Any script that
needs pixels must join to `hotosm/vhr-building-segmentation` on `dataset_row`.
Forgetting this has silently broken two scripts already.

---

## Pipeline

### 0 — Splits

`data/cv_folds/cv_folds.csv` and `data/benchmark_v1/` are committed, so steps 1–7
run without rebuilding them. Regenerate only if the pool changes:

```bash
python scripts/build_verified_pool.py
python scripts/build_benchmark.py
python scripts/make_cv_folds.py --val-frac 0.15 --out data/cv_folds
```

Splits are by **4x4 spatial tile block, never per chip**. Adjacent OAM tiles share
buildings across their border and the same imagery capture, so a per-chip split
puts near-duplicates on both sides and inflates every number.

### 1 — Train the segmentation model on reviewed labels

```bash
python scripts/train.py \
    --split-file data/cv_folds/cv_folds.csv --fold 0 \
    --dataset nilsho01/vhr-buildings-verified-v1 \
    --arch smp_unet --encoder efficientnet-b4 --heads dist3 \
    --boundary-weight 4 --sep-weight 10 --sep-sigma 4 --ib-dilate 2 \
    --core-erosion 3 --seg-weight 40 \
    --epochs 40 --batch 8 --workers 4 --seed 42 --size 256 \
    --threshold-sweep 0.3 0.4 0.45 0.5 0.55 --select-on pq \
    --val-instance-chips 300 \
    --out data/runs/cv_f0_effb4_dist3
```

`--heads dist3` predicts three channels: the **mask**, a **per-instance
normalised distance transform**, and an **instance-boundary band**. The distance
channel is what separates touching buildings — connected components on the mask
alone merges about 2.3 buildings per blob. Instances come from marker-controlled
watershed, and watershed emits exactly one instance per marker, so the seed
channel is a hard ceiling on recall.

### 2 — Score unreviewed chips for label quality

```bash
python scripts/train_quality_classifier.py \
    --checkpoint data/runs/cv_f0_effb4_dist3/best.pth \
    --encoder efficientnet-b4 --heads dist3 \
    --out label-cleanup/run2/quality_xgb \
    --skip-extract --score-remaining
```

XGBoost on features describing **how the model disagrees with the label** — IoU,
predicted/true object-count ratio, centroid offset, missed buildings. Image
embeddings were tested against this and lost: a DINOv2 or task-trained encoder
embedding reaches AUROC 0.77, *below* five columns of free metadata, because
whether a label is wrong is a property of the label and an image-only feature
cannot see it.

Use predictions from a model that did **not** train on the chip being scored, or
the classifier learns "was this in training" instead of "is this label wrong".

```bash
python scripts/viz_quality_threshold.py \
    --out label-cleanup/run2/quality_xgb --target-bad-caught 0.95
```

### 3–4 — Build and publish the clean pool

```bash
python scripts/build_clean_pool.py \
    --score-dir label-cleanup/run2/quality_xgb \
    --threshold 0.95 --out data/clean_pool/pool.csv

python scripts/push_clean_pool.py \
    --pool-csv data/clean_pool/pool.csv \
    --repo nilsho01/vhr-buildings-automatic-annotation
```

`build_clean_pool.py` strips benchmark chips before writing, and keeps every
human-verified chip regardless of classifier score (`--trust-reviewed`, on by
default).

### 5 — Rebuild folds over the clean pool

```bash
python scripts/make_cv_folds.py \
    --pool-csv data/clean_pool/pool.csv \
    --val-frac 0.15 --out data/cv_folds_clean
```

### 6 — Train five folds

```bash
for FOLD in 0 1 2 3 4; do
  python scripts/train.py \
    --dataset nilsho01/vhr-buildings-automatic-annotation \
    --split-file data/cv_folds_clean/cv_folds.csv --fold $FOLD \
    --arch smp_unet --encoder efficientnet-b4 --heads dist3 \
    --boundary-weight 4 --sep-weight 10 --sep-sigma 4 --ib-dilate 2 \
    --core-erosion 3 --seg-weight 40 \
    --epochs 40 --batch 8 --workers 4 --seed 42 --size 256 \
    --threshold-sweep 0.3 0.4 0.45 0.5 0.55 --select-on pq \
    --out data/runs/cv_clean_f${FOLD}_effb4_dist3
done
```

### 7 — Benchmark

```bash
python scripts/benchmark_folds.py \
    --runs data/runs/cv_clean_f{0,1,2,3,4}_effb4_dist3 \
    --split-file data/cv_folds_clean/cv_folds.csv \
    --out-dir data/review_charts_clean
```

---

## Evaluation protocol

`benchmark_folds.py` scores every checkpoint on the same 400 chips under three
columns, because "tune the threshold" and "fix the threshold" are both defensible
and are not the same number:

| column | meaning |
|---|---|
| **fixed** | threshold 0.50 everywhere. No tuning, so neither architecture is quietly advantaged. |
| **val-picked** | threshold chosen on each run's own validation split, applied unchanged. Selection never sees the test chips. **This is the headline.** |
| **oracle** | best achievable on the benchmark itself. **Not an operating point** — it is selected on what it is scored on. Reported only as a ceiling; the gap to val-picked measures how well the choice transfers. |

The benchmark is 400 chips: 100 each at 0, 1–9, 10–49 and 50–149 buildings, one
chip per 4x4 block, and it is test for every fold. Never train or select on it.

## Baseline results

Five folds each, val-picked threshold, 400-chip benchmark:

| | PQ | SQ | RQ |
|---|---|---|---|
| UNet efficientnet-b4, dist3 | **37.65 ± 1.29** | 73.72 ± 0.85 | 51.06 ± 1.22 |
| ViT-S (dinov3_hot) | 27.50 ± 2.48 | 70.91 ± 1.02 | 38.78 ± 3.35 |

The gap is object recovery, not outlines — SQ differs by 2.8, RQ by 12.3. The ViT
predicts 51–67% of the true object count against the UNet's 91–93%, and wants
thresholds of 0.70–0.85 where the UNet peaks near 0.45.

**State this caveat with the result:** the shipped DINOv3's *pretraining* pool
overlaps this benchmark, so contamination runs in the ViT's favour. Everything is
one seed per fold, so the spread measures data variance, not training variance.

## Known limitations

- **Labels bound the measurement.** Excluding a 2px collar around each label
  moves pixel F1 by +5.3 — half a metre at this resolution, inside what a
  hand-drawn OSM footprint can promise. A perfect model scores ~95, not 100.
- **The benchmark has label debt.** 91 of its 300 populated chips contain at
  least one building that the model finds and Google Open Buildings or Microsoft
  footprints confirm, but OSM does not label.
- **Empty chips.** PQ on a chip with no ground truth was previously dropped
  entirely, hiding hallucinations on 25% of the benchmark. `instance_metrics.py`
  now counts them and reports `empty_fp` separately from populated-chip false
  positives. The effect here is ~0.04 PQ, but it would not stay small.

Diagnostic and analysis scripts — false-positive adjudication, boundary-offset
measurement, seeding sweeps, rectangularity — are deliberately not in this branch.
They live in the study repository.

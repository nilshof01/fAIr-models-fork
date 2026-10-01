# unet-effb4-buildings

Building **instance** segmentation. U-Net, EfficientNet-B4 encoder (ImageNet),
20.2 M parameters.

## What is different about it

The other building models here perform semantic segmentation: one output
channel, thresholded into a mask. Two buildings sharing a wall form one
connected component and are reported as one building. Taking connected
components of such a mask merges roughly 2.3 buildings per blob on dense chips.

This model emits three channels and resolves them into separate instances:

| channel | content |
|---|---|
| 0 | mask — building / not building |
| 1 | per-instance normalised distance transform; one peak per building |
| 2 | boundary band at walls shared by two instances |

Instances come from marker-controlled watershed: connected components of
channel 1 seed the flood, which spreads inside the mask of channel 0. Watershed
emits exactly one instance per marker, so the markers rather than the mask
determine the object count.

`postprocess` returns a binary mask so the model satisfies the serving contract
used by the other models in this repository. `postprocess_instances` returns the
label image and is what the model is for.

## Results

Mean ± sd over five cross-validation folds, threshold selected on each fold's
own validation split.

**`nilsho01/vhr-buildings-benchmark-v1`** — 400 chips, labels manually reviewed,
stratified by building density and region.

| model | PQ | SQ | RQ | pixel F1 | pixel IoU |
|---|---|---|---|---|---|
| this model | **47.79 ± 0.23** | 75.44 ± 0.17 | 63.35 ± 0.27 | 84.43 ± 0.11 | 73.05 ± 0.16 |
| dinov3s-buildings (shipped) | 33.92 | 72.08 | 47.06 | 83.03 | 70.98 |

All 400 benchmark chips lie in the `train` split of
`hotosm/vhr-building-segmentation`, so the shipped model trained on every chip it
is scored on while this one did not. The comparison therefore favours the
baseline.

**`hotosm/vhr-building-segmentation` `test`** — 7,236 chips, raw OSM labels,
trained on by neither model.

| metric | this model | dinov3s-buildings |
|---|---|---|
| PQ | **30.02 ± 0.26** | 27.81 |
| SQ | **78.45 ± 0.29** | 76.82 |
| RQ | **38.28 ± 0.30** | 36.20 |
| pixel F1 | 60.46 ± 0.48 | **61.50** |
| pixel IoU | 43.33 ± 0.49 | **44.41** |

Ahead on the instance metrics, behind on the pixel metrics by about one point.
The shipped model predicts 75% of the true object count against this model's
116%: a model that under-predicts is rewarded by an area-weighted metric and
penalised by an object-weighted one.

## Known limitation

On the `test` split, **37.4% ± 2.3 of chips containing no buildings receive at
least one spurious detection**, against 8.3% for the shipped model (15.8–19.8%
on the reviewed benchmark). The rate rose as training data was added. Its effect
on the metrics is small — removing every such false positive would add about 0.7
pixel IoU — but it is visible to a user. The likely cause is geographic coverage:
the training pool is approximately 61% Myanmar, while the `validation` and `test`
splits of vhr-buildings contain none.

## Why panoptic quality

Pixel F1 and IoU cannot distinguish a correctly separated building from several
merged into one. A single predicted blob covering a terrace of four buildings
scores a pixel IoU of 86 and a PQ of 0; the same four separated but traced one
pixel too wide score a pixel IoU of 86 and a PQ of 86. Pixel metrics weight by
area and panoptic quality by object, and with a median building of about 256 px
most objects carry very little area.

Of the two components, recognition quality is the more robust where labels are
imprecise: it matches instances at an IoU of 0.5, which a boundary error of one
or two pixels rarely crosses, whereas segmentation quality is itself an IoU.

## Inference

```python
from models.unet_effb4_buildings import pipeline

x      = pipeline.preprocess("chip.tif")
logits = session.run(None, {session.get_inputs()[0].name: x})[0]

labels = pipeline.postprocess_instances(logits)   # 0 = background, 1..N
mask   = pipeline.postprocess(logits)             # binary, contract-compatible
```

Ship `best_threshold.json` with a checkpoint: folds selected thresholds between
0.40 and 0.50 and the operating point is not interchangeable between runs.
`pipeline.load_inference_params()` reads it.

## Training data and attribution

Imagery and base labels: `hotosm/vhr-building-segmentation` (OpenAerialMap +
OpenStreetMap, ODbL). The training pool was filtered with a classifier whose
label-debt evidence draws on Google Open Buildings (CC-BY 4.0 / ODbL) and
Microsoft Building Footprints (ODbL). Attribution review for those sources is
outstanding.

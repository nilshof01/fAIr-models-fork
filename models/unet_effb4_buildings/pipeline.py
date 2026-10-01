"""unet-effb4-buildings: U-Net + EfficientNet-B4 for building INSTANCE segmentation.

Differs from the other building models in this repository in what it emits. A
semantic model returns one channel and a threshold turns it into a mask; two
buildings sharing a wall become one connected component and are reported as one
building. This model emits three channels and resolves them into separate
instances:

    channel 0  mask            building / not building
    channel 1  normalised EDT  per-instance Euclidean distance transform, 1.0 at
                               each building's deepest interior point and 0 at
                               its walls - one peak per building
    channel 2  instance boundary band at shared walls

Instances come from marker-controlled watershed: connected components of the
distance channel seed the flood, which then spreads inside the mask. Watershed
emits exactly one instance per marker, so the markers - not the mask - set the
object count. Taking connected components of the mask alone merges roughly 2.3
buildings per blob on dense chips.

`postprocess` returns a binary mask to satisfy the serving contract used by the
other models here. `postprocess_instances` returns the label image and is what
makes this model worth adding; a caller that wants footprints should use it.
"""

import json
from pathlib import Path
from typing import Any

MODEL_NAME = "unet-effb4-buildings"
ENCODER_NAME = "efficientnet-b4"
MODEL_INPUT_SIZE = 256
N_CHANNELS = 3

# ImageNet statistics - the encoder is ImageNet-pretrained, unlike the OAM-TCD
# (tree-crown) initialisation used by models/unet_segmentation.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Selected on validation during cross validation, never on a test set. Folds
# chose 0.40-0.50; the operating point is not interchangeable between runs, so a
# checkpoint should ship with its own value.
DEFAULT_INFERENCE_PARAMS: dict[str, Any] = {
    "confidence_threshold": 0.50,   # channel 0, mask
    "core_threshold": 0.50,         # channel 1, seeds
    "min_instance_px": 30,          # drop specks after flooding
}


def preprocess(image_path: Any) -> Any:
    """RGB chip to a normalised NCHW float32 tensor at the model's input size."""
    import numpy as np
    import rasterio

    with rasterio.open(image_path) as src:
        rgb = src.read(
            [1, 2, 3], out_shape=(3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        ).astype(np.float32) / 255.0
    mean = np.asarray(NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(NORM_STD, dtype=np.float32).reshape(3, 1, 1)
    return ((rgb - mean) / std)[np.newaxis, ...].astype(np.float32)


def postprocess(
    logits: Any,
    confidence_threshold: float = DEFAULT_INFERENCE_PARAMS["confidence_threshold"],
) -> Any:
    """Binary building mask from channel 0, for the shared serving contract.

    This discards the instance information the model was trained to produce. Use
    `postprocess_instances` where separate footprints are wanted.
    """
    import numpy as np

    probability = 1.0 / (1.0 + np.exp(-np.asarray(logits)[:, 0]))
    return (probability >= confidence_threshold).astype(np.uint8)


def postprocess_instances(
    logits: Any,
    confidence_threshold: float = DEFAULT_INFERENCE_PARAMS["confidence_threshold"],
    core_threshold: float = DEFAULT_INFERENCE_PARAMS["core_threshold"],
    min_instance_px: int = DEFAULT_INFERENCE_PARAMS["min_instance_px"],
) -> Any:
    """Separate buildings: an int32 label image, 0 = background, 1..N = instances.

    The flood surface is the negative distance transform of the predicted mask,
    so a blob's interior is a valley and its rim high ground. Water rising from
    each seed spreads until neighbouring floods meet, and that meeting line is
    the cut between two buildings. The mask bounds the flood, so water never
    leaves the predicted building area.
    """
    import numpy as np
    from scipy import ndimage
    from skimage.segmentation import watershed

    probability = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    if probability.ndim == 4:
        probability = probability[0]

    mask = probability[0] >= confidence_threshold
    if not mask.any():
        return np.zeros(mask.shape, np.int32)

    markers = ndimage.label((probability[1] >= core_threshold) & mask)[0]
    if markers.max() == 0:
        return np.zeros(mask.shape, np.int32)

    labels = watershed(
        -ndimage.distance_transform_edt(mask), markers=markers, mask=mask
    )
    counts = np.bincount(labels.ravel())
    drop = np.flatnonzero(counts < min_instance_px)
    return np.where(np.isin(labels, drop[drop > 0]), 0, labels).astype(np.int32)


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run the ONNX session over a directory of chips and return instance counts.

    Returns one entry per chip with the number of separate buildings found and
    their pixel areas, which is the quantity this model exists to provide.
    """
    import numpy as np

    from fair.utils.data import resolve_directory

    cfg = {**DEFAULT_INFERENCE_PARAMS, **(params or {})}
    root = Path(resolve_directory(input_images))
    input_name = session.get_inputs()[0].name

    results: dict[str, Any] = {}
    for chip in sorted(p for p in root.rglob("*") if p.suffix.lower() in
                       {".tif", ".tiff", ".png", ".jpg", ".jpeg"}):
        logits = session.run(None, {input_name: preprocess(chip)})[0]
        labels = postprocess_instances(
            logits,
            confidence_threshold=cfg["confidence_threshold"],
            core_threshold=cfg["core_threshold"],
            min_instance_px=cfg["min_instance_px"],
        )
        ids, areas = np.unique(labels, return_counts=True)
        keep = ids > 0
        results[chip.name] = {
            "n_buildings": int(keep.sum()),
            "areas_px": areas[keep].astype(int).tolist(),
        }
    return results


def build_model(num_classes: int = N_CHANNELS, encoder_weights: str | None = None):
    """The architecture, so a checkpoint can be loaded without the training code."""
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


def export_onnx(checkpoint_path: str, output_path: str) -> str:
    """Export a trained checkpoint to ONNX at the model's fixed input size."""
    import torch

    model = build_model()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()

    dummy = torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return output_path


def load_inference_params(checkpoint_dir: str) -> dict[str, Any]:
    """Read the threshold a checkpoint was validated at, if it shipped one.

    Folds of the same model selected thresholds between 0.40 and 0.50, so using
    one checkpoint's value with another's weights costs accuracy.
    """
    path = Path(checkpoint_dir) / "best_threshold.json"
    params = dict(DEFAULT_INFERENCE_PARAMS)
    if path.exists():
        params["confidence_threshold"] = float(json.loads(path.read_text())["threshold"])
    return params

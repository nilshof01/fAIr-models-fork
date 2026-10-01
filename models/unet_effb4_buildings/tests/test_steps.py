"""Contract and behaviour tests for unet-effb4-buildings."""

import json
from pathlib import Path

import numpy as np
import pytest

from models.unet_effb4_buildings import pipeline

MODEL_DIR = Path(__file__).resolve().parent.parent


def test_stac_item_declares_instance_segmentation() -> None:
    item = json.loads((MODEL_DIR / "stac-item.json").read_text())
    props = item["properties"]
    assert props["mlm:name"] == pipeline.MODEL_NAME
    assert "instance-segmentation" in props["mlm:tasks"]
    # the instance path is the reason this model exists; a reader following the
    # STAC item must be sent to it rather than to the binary mask
    assert props["mlm:output"][0]["post_processing_function"].endswith("postprocess_instances")
    assert props["mlm:output"][0]["result"]["shape"][1] == pipeline.N_CHANNELS


def test_postprocess_returns_a_binary_mask(touching_pair_logits) -> None:
    """The shared serving contract: uint8, 0/1, one plane per batch item."""
    mask = pipeline.postprocess(touching_pair_logits)
    assert mask.dtype == np.uint8
    assert mask.shape == touching_pair_logits.shape[0:1] + touching_pair_logits.shape[2:]
    assert set(np.unique(mask)) <= {0, 1}
    assert mask.sum() > 0


def test_watershed_separates_what_the_mask_merges(touching_pair_logits) -> None:
    """The behaviour the model is for.

    The mask channel is a single rectangle covering both buildings, so connected
    components finds one object. The distance channel has two peaks, so the
    watershed finds two.
    """
    from scipy import ndimage

    merged = pipeline.postprocess(touching_pair_logits)[0].astype(bool)
    assert ndimage.label(merged)[1] == 1, "fixture should present a single blob"

    labels = pipeline.postprocess_instances(touching_pair_logits)
    assert labels.dtype == np.int32
    assert int(labels.max()) == 2, "watershed did not separate the two buildings"
    assert (labels > 0).sum() == pytest.approx(merged.sum(), rel=0.02)


def test_small_specks_are_dropped(touching_pair_logits) -> None:
    kept = pipeline.postprocess_instances(touching_pair_logits, min_instance_px=30)
    gone = pipeline.postprocess_instances(touching_pair_logits, min_instance_px=10**6)
    assert int(kept.max()) == 2
    assert int(gone.max()) == 0


def test_empty_prediction_yields_no_instances() -> None:
    logits = np.full((1, pipeline.N_CHANNELS, 64, 64), -8.0, np.float32)
    assert pipeline.postprocess_instances(logits).max() == 0
    assert pipeline.postprocess(logits).sum() == 0


def test_preprocess_shape_and_normalisation(toy_chips) -> None:
    chip = sorted(toy_chips["chips"].glob("*.tif"))[0]
    x = pipeline.preprocess(chip)
    assert x.shape == (1, 3, pipeline.MODEL_INPUT_SIZE, pipeline.MODEL_INPUT_SIZE)
    assert x.dtype == np.float32
    assert -5 < float(x.mean()) < 5


def test_threshold_is_read_from_the_checkpoint(tmp_path) -> None:
    """Folds select 0.40-0.50; using one checkpoint's value with another's
    weights costs accuracy, so the value travels with the checkpoint."""
    (tmp_path / "best_threshold.json").write_text(json.dumps({"threshold": 0.42}))
    assert pipeline.load_inference_params(tmp_path)["confidence_threshold"] == 0.42
    assert (
        pipeline.load_inference_params(tmp_path / "absent")["confidence_threshold"]
        == pipeline.DEFAULT_INFERENCE_PARAMS["confidence_threshold"]
    )

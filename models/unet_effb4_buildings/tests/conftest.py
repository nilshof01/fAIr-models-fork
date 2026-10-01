"""Toy chips for the unet-effb4-buildings tests.

Deliberately includes a chip with two buildings sharing a wall. That is the case
the model exists to handle and the case a semantic model cannot: connected
components of a mask reports the pair as one building.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIP_SIZE = 256
BASE_LON, BASE_LAT, STEP = 85.5, 27.6, 0.001


def _write(path: Path, rgb: np.ndarray, lon: float, lat: float) -> None:
    transform = from_bounds(lon, lat, lon + STEP, lat + STEP, CHIP_SIZE, CHIP_SIZE)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=CHIP_SIZE,
        height=CHIP_SIZE,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(rgb)


def create_toy_data(root: Path) -> dict[str, Path]:
    chips = root / "chips"
    chips.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(3):
        rgb = rng.integers(60, 120, (3, CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)
        rgb[:, 80:140, 40:100] = 220  # one building
        if i == 0:
            rgb[:, 80:140, 104:164] = 220  # a second, 4 px away
        _write(chips / f"OAM-{i:04d}-0000-0000.tif", rgb, BASE_LON + i * STEP, BASE_LAT)
    return {"chips": chips}


@pytest.fixture
def toy_chips(tmp_path: Path) -> dict[str, Path]:
    return create_toy_data(tmp_path)


@pytest.fixture
def touching_pair_logits() -> Any:
    """Logits for two buildings whose predicted MASK merges them into one blob.

    The mask channel is a single rectangle; the distance channel carries two
    peaks. A semantic reading returns one building, the watershed returns two.
    """
    h = 128
    mask = np.zeros((h, h), np.float32)
    mask[40:88, 20:108] = 1.0
    dist = np.zeros((h, h), np.float32)
    yy, xx = np.ogrid[:h, :h]
    for cx in (44, 84):
        dist = np.maximum(dist, np.clip(1 - np.hypot(yy - 64, xx - cx) / 26, 0, 1))

    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p * 0.94 + 0.03, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).astype(np.float32)

    return np.stack([logit(mask), logit(dist), logit(np.zeros((h, h), np.float32))])[None]

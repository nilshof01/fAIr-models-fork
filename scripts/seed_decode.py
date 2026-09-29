"""Losses and decoders for the centre-heatmap and flow-field heads."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


class CentreLoss(torch.nn.Module):
    """CenterNet's penalty-reduced focal loss.

    Plain BCE is hopeless on a heatmap that is ~99.5% zeros - the model
    converges to predicting nothing. This down-weights easy background and,
    via (1-y)^beta, forgives predictions near a true centre, where being
    slightly off is not really wrong.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.a, self.b = alpha, beta

    def forward(self, logit, target):
        """Each term is a mean over its own population, not a sum over all
        pixels divided by the positives.

        CenterNet normalises by the positive count, which is right when the
        heatmap is the whole task. Here it has to sit beside a mask loss, and
        that normalisation puts it around 300 against the mask's ~1 - so the
        mask is effectively ignored and val F1 stays at zero. Averaging each
        term separately keeps both O(1) and preserves what the focal weights
        are for: hard positives still dominate the positive term, easy
        background still vanishes from the negative one.
        """
        p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
        pos = target >= 0.999
        neg = ~pos
        pos_loss = -((1 - p) ** self.a) * torch.log(p) * pos
        neg_loss = -((1 - target) ** self.b) * (p ** self.a) * torch.log(1 - p) * neg
        return (pos_loss.sum() / pos.sum().clamp_min(1)
                + neg_loss.sum() / neg.sum().clamp_min(1))


class FlowLoss(torch.nn.Module):
    """MSE on the vector field, building pixels only.

    Scored outside the buildings too and the loss is dominated by the trivial
    'predict zero' region.
    """

    def forward(self, pred, target, mask):
        w = mask.clamp(0, 1)
        return (((pred - target) ** 2).sum(1, keepdim=True) * w).sum() / w.sum().clamp_min(1)


class CentreDecoder:
    """Peaks of the heatmap become markers; watershed floods them in the mask."""

    def __init__(self, mask_thr=0.3, peak_thr=0.3, min_distance=2,
                 min_instance_px=30):
        self.mask_thr, self.peak_thr = mask_thr, peak_thr
        self.min_distance, self.min_px = min_distance, min_instance_px

    def __call__(self, prob):
        mask = prob[0] > self.mask_thr
        if not mask.any():
            return np.zeros(mask.shape, np.int32)
        heat = prob[1]
        coords = peak_local_max(heat, min_distance=self.min_distance,
                                threshold_abs=self.peak_thr, labels=mask)
        markers = np.zeros(mask.shape, np.int32)
        for k, (y, x) in enumerate(coords, 1):
            markers[y, x] = k
        if not markers.any():
            markers, _ = ndimage.label(mask)
        lab = watershed(-ndimage.distance_transform_edt(mask),
                        markers=markers, mask=mask)
        return drop_small(lab, self.min_px)


class FlowDecoder:
    """Follow the field; pixels landing in the same place are one building."""

    def __init__(self, mask_thr=0.3, steps=40, min_instance_px=30):
        self.mask_thr, self.steps, self.min_px = mask_thr, steps, min_instance_px

    def __call__(self, prob):
        mask = prob[0] > self.mask_thr
        if not mask.any():
            return np.zeros(mask.shape, np.int32)
        fy, fx = prob[1], prob[2]
        ys, xs = np.where(mask)
        py, px = ys.astype(np.float32), xs.astype(np.float32)
        H, W = mask.shape
        for _ in range(self.steps):
            iy = np.clip(np.round(py).astype(int), 0, H - 1)
            ix = np.clip(np.round(px).astype(int), 0, W - 1)
            py = np.clip(py + fy[iy, ix], 0, H - 1)
            px = np.clip(px + fx[iy, ix], 0, W - 1)
        # convergence points, on a 1px grid; connected blobs of arrivals are
        # one building, which is what makes this robust to touching objects
        sink = np.zeros(mask.shape, np.int32)
        np.add.at(sink, (np.round(py).astype(int), np.round(px).astype(int)), 1)
        basins, _ = ndimage.label(sink > 0)
        lab = np.zeros(mask.shape, np.int32)
        lab[ys, xs] = basins[np.round(py).astype(int), np.round(px).astype(int)]
        return drop_small(lab, self.min_px)


def drop_small(lab, min_px):
    c = np.bincount(lab.ravel())
    kill = np.flatnonzero(c < min_px)
    kill = kill[kill > 0]
    return np.where(np.isin(lab, kill), 0, lab).astype(np.int32)

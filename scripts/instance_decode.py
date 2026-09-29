"""Turn a 3-channel prediction (mask, core, contour) into separated instances.

Connected components on the mask alone merge every pair of touching buildings.
Seeding from the predicted core and flooding back out inside the mask keeps
them apart - the standard recipe for touching-object segmentation, and on this
data an oracle-seeded version of exactly this reaches 69% detection recall
where connected components give 21%.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


class WatershedDecoder:
    """mask_thr / core_thr are separate on purpose: the core is a smaller,
    harder target, so it usually wants a lower threshold than the mask."""

    def __init__(self, mask_thr=0.5, core_thr=0.5, min_instance_px=30,
                 contour_weight=2.0, seed_mode="threshold", h=0.15,
                 smooth=1.0, floor=0.2):
        """contour_weight raises predicted contours as ridges in the flood
        surface. It was set when contours were blurry; boundary weighting makes
        them sharp, and too strong a ridge stops the flood at features INSIDE a
        building, which shows up as over-segmentation. 0 disables it."""
        self.mask_thr = mask_thr
        self.core_thr = core_thr
        self.min_px = min_instance_px
        self.contour_weight = contour_weight
        self.seed_mode = seed_mode
        self.h = h
        self.smooth = smooth
        self.floor = floor

    def __call__(self, prob):
        """prob: (3, H, W) of sigmoid outputs. Returns an int label image."""
        mask = prob[0] > self.mask_thr
        if not mask.any():
            return np.zeros(mask.shape, np.int32)
        seeds, n = ndimage.label(self._seeds(prob, mask))
        if n == 0:                       # nothing seeded: fall back to blobs
            seeds, n = ndimage.label(mask)
            if n == 0:
                return np.zeros(mask.shape, np.int32)
        # flood downhill from the seeds. The surface is the distance transform
        # of the mask, pushed up at predicted contours so instance borders act
        # as ridges the flood does not cross.
        surface = -ndimage.distance_transform_edt(mask).astype(np.float32)
        if self.contour_weight and prob.shape[0] > 2:
            surface = surface + self.contour_weight * prob[2].astype(np.float32)
        lab = watershed(surface, markers=seeds, mask=mask)
        return self._drop_small(lab)

    def _seeds(self, prob, mask):
        """Where do the markers come from?

        'threshold' takes prob[1] > core_thr. Looking at the core channel on
        dense chips shows why that caps the result: neighbouring buildings'
        cores TOUCH in the probability map, so one global threshold fuses them
        into a single marker and the two buildings become one instance. Seeds
        then run ~70% of the true count on dense chips while over-seeding
        sparse ones - which is why every threshold sweep came back flat, the
        two errors cancelling in the aggregate.

        'hmaxima' seeds from regional maxima with a minimum prominence h
        instead. Each building's own peak survives whether or not its
        neighbour's core touches it, so the seeding adapts to the scene rather
        than to a number chosen globally.
        """
        core = prob[1]
        if self.seed_mode == "threshold":
            return (core > self.core_thr) & mask
        c = ndimage.gaussian_filter(core.astype(np.float32), self.smooth) \
            if self.smooth else core.astype(np.float32)
        peaks = h_maxima(c, self.h) > 0
        return peaks & (core > self.floor) & mask

    def _drop_small(self, lab):
        if self.min_px <= 1:
            return lab.astype(np.int32)
        counts = np.bincount(lab.ravel())
        kill = np.flatnonzero(counts < self.min_px)
        kill = kill[kill > 0]
        if len(kill):
            lab = np.where(np.isin(lab, kill), 0, lab)
        return lab.astype(np.int32)


class DistTransformDecoder:
    """Decode a (mask, normalised-EDT) head into separated instance labels.

    Seeds are the regional maxima of the predicted distance map with a minimum
    prominence h — each building contributes exactly one peak, so touching
    buildings never produce a fused marker even when their predicted masks
    overlap completely.

    The watershed surface is -dt (downhill from peaks), which mirrors the
    training target: predicted confidence is highest at building centres and
    falls off toward walls, so the flood naturally stops at the shared
    boundary between two neighbours.
    """

    def __init__(self, mask_thr=0.5, min_instance_px=30,
                 h=0.10, smooth=1.0, floor=0.05, ridge_weight=3.0):
        self.mask_thr     = mask_thr
        self.min_px       = min_instance_px
        self.h            = h
        self.smooth       = smooth
        self.floor        = floor
        self.ridge_weight = ridge_weight

    def __call__(self, prob):
        """prob: (2, H, W) or (3, H, W) of sigmoid outputs.

        Ch0  mask probability
        Ch1  normalised EDT — seeds come from its local maxima
        Ch2  instance boundary probability (optional) — added as a ridge to
             the watershed surface so touching buildings are not flooded together
        """
        mask = prob[0] > self.mask_thr
        if not mask.any():
            return np.zeros(mask.shape, np.int32)

        dt = prob[1].astype(np.float32)
        sm = (ndimage.gaussian_filter(dt, self.smooth) if self.smooth else dt)
        peaks = h_maxima(sm, self.h) > 0
        seeds, n = ndimage.label(peaks & (dt > self.floor) & mask)
        if n == 0:
            seeds, n = ndimage.label(mask)
            if n == 0:
                return np.zeros(mask.shape, np.int32)

        surface = -dt
        if prob.shape[0] > 2 and self.ridge_weight > 0:
            # instance boundary predictions raise the surface, making shared
            # walls impassable ridges that stop the watershed flood
            surface = surface + self.ridge_weight * prob[2].astype(np.float32)
        lab = watershed(surface, markers=seeds, mask=mask)
        return WatershedDecoder(min_instance_px=self.min_px)._drop_small(lab)


def connected_components(prob, thr=0.5, min_instance_px=30):
    """The baseline this replaces: blobs of the mask channel."""
    lab, _ = ndimage.label(prob[0] > thr if prob.ndim == 3 else prob > thr)
    return WatershedDecoder(min_instance_px=min_instance_px)._drop_small(lab)

MIN_PX = 30          # instances smaller than this are dropped as noise


def grow(markers, mask, min_px=MIN_PX):
    """Flood each marker outwards inside `mask` and drop the specks.

    Watershed emits exactly one instance per marker, so the marker set is a hard
    ceiling on how many objects can be recovered. The surface is the negative
    distance transform of the mask: the middle of a blob is a valley, its rim
    high ground, and water rising from each seed meets its neighbour along the
    cut. Used by benchmark_folds.py as the reference decoder - deliberately the
    plainest one, with no contour ridge, so a benchmark number never depends on
    a decoding option.
    """
    if not mask.any() or markers.max() == 0:
        return np.zeros(mask.shape, np.int32)
    lab = watershed(-ndimage.distance_transform_edt(mask), markers=markers,
                    mask=mask)
    counts = np.bincount(lab.ravel())
    kill = np.flatnonzero(counts < min_px)
    return np.where(np.isin(lab, kill[kill > 0]), 0, lab).astype(np.int32)

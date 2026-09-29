"""Marker representations that do not depend on eroding the building.

The interior (eroded-mask) core channel fails in two opposite ways, measured
on fold 0: 29% of buildings share a marker with a neighbour because their
interiors touch, and 23% have no marker at all - 64% of the SMALL ones -
because eroding a 10x10 shack leaves almost nothing to learn. One erosion
value cannot serve both, which is why every decode sweep cancelled out.

Both representations here sidestep that. Neither shrinks the building, so
neither has a size regime it fails in:

  CENTRE  a Gaussian bump at each building's innermost point. A dot is the
          same target whether the building is 10x10 or 40x40, and two
          buildings have two centres by definition, so markers can neither
          vanish nor fuse.
  FLOW    a unit vector at every building pixel pointing at its own centre.
          Pixels that flow to the same place are one building - identity is
          carried by the field rather than recovered from a shape.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def inner_point(inst):
    """Deepest point of the instance, not its arithmetic centroid.

    For an L-shaped building or one round a courtyard the centroid can fall
    outside the footprint; the distance transform's maximum is always inside.
    """
    d = ndimage.distance_transform_edt(inst)
    return np.unravel_index(int(np.argmax(d)), d.shape)


class CentreTargets:
    """mask + a centre heatmap.

    sigma scales with the building so a warehouse gets a broader peak than a
    shack, but never below `sigma_min` - a literal single pixel is close to
    unlearnable against ~65k background pixels.
    """

    def __init__(self, sigma_scale=0.125, sigma_min=1.0, sigma_max=4.0):
        self.k, self.lo, self.hi = sigma_scale, sigma_min, sigma_max

    def sigma_for(self, area):
        return float(np.clip(self.k * np.sqrt(area), self.lo, self.hi))

    def __call__(self, mask):
        m = mask.astype(bool)
        heat = np.zeros(m.shape, np.float32)
        lab, n = ndimage.label(m)
        yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
        for i in range(1, n + 1):
            inst = lab == i
            cy, cx = inner_point(inst)
            s = self.sigma_for(int(inst.sum()))
            g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s)))
            np.maximum(heat, g, out=heat)      # overlapping peaks keep the max
        return np.stack([m.astype(np.float32), heat])


class FlowTargets:
    """mask + a 2-channel unit vector field pointing at each building's centre.

    Limitation worth knowing: the vector points straight at the centre, so for
    a strongly concave footprint part of the path can leave the building.
    CellPose solves this with heat diffusion inside each object, which is
    correct but needs an iterative solve per instance - too slow for the
    dataloader, and most buildings here are near-convex.
    """

    def __call__(self, mask):
        m = mask.astype(bool)
        fy = np.zeros(m.shape, np.float32)
        fx = np.zeros(m.shape, np.float32)
        lab, n = ndimage.label(m)
        for i in range(1, n + 1):
            inst = lab == i
            cy, cx = inner_point(inst)
            ys, xs = np.where(inst)
            vy, vx = cy - ys, cx - xs
            norm = np.maximum(np.hypot(vy, vx), 1e-6)
            fy[ys, xs] = vy / norm
            fx[ys, xs] = vx / norm
        return np.stack([m.astype(np.float32), fy, fx])

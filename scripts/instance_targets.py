"""Derive instance-aware training targets and a separation weight map from a
binary building mask.

A single binary channel cannot express "these two roofs are different
buildings": where two buildings touch, the optimal binary mask genuinely is
one merged blob, so the network is right to produce it. Measured on fold 0,
that costs the model almost everything at instance level - its predicted
pixels support ~69% detection recall if split with oracle seeds, but
connected components on its own output give 21%.

Two mechanisms fix that, and both live here:

  CORE      each building eroded by `core_erosion` px, so neighbours separate.
            Connected components of the predicted core become watershed seeds.
  CONTOUR   a thin band on each building's own outline, which sharpens edges.
  WEIGHTS   the separation weight map from the original U-Net paper: background
            pixels lying in the narrow gap between two different instances cost
            more to get wrong. These are exactly the street and alley pixels the
            model currently floods.

Targets are derived AFTER augmentation, from the augmented mask - deriving them
first and then rotating or scaling would change the contour width and erode
distances inconsistently.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class InstanceTargets:
    """Turn one binary mask into (mask, core, contour) plus a weight map.

    core_erosion   px to shrink each building by; must exceed half the typical
                   gap between touching roofs or they stay merged
    contour_width  px band drawn on each building's outline
    sep_w0/sigma   height and width of the separation bonus. The original paper
                   uses w0=10, sigma=5 px; at 0.25 m/px with a median building
                   of ~16x16 px, sigma 3-5 covers a real alley.
    """

    def __init__(self, core_erosion=3, contour_width=2, sep_w0=10.0,
                 sep_sigma=5.0, gap_radius=None, bnd_w0=0.0, bnd_sigma=2.0):
        self.core_erosion = core_erosion
        self.contour_width = contour_width
        self.sep_w0 = sep_w0
        self.sep_sigma = sep_sigma
        self.gap_radius = gap_radius or int(np.ceil(3 * sep_sigma))
        self.bnd_w0 = bnd_w0
        self.bnd_sigma = bnd_sigma

    def __call__(self, mask):
        m = mask.astype(bool)
        lab, n = ndimage.label(m)
        core = np.zeros_like(m)
        contour = np.zeros_like(m)
        if n:
            # erode per instance: a global erosion would merge neighbours first
            for i in range(1, n + 1):
                inst = lab == i
                c = ndimage.binary_erosion(inst, iterations=self.core_erosion)
                if not c.any():                      # tiny building: keep a seed
                    c = ndimage.binary_erosion(inst, iterations=1)
                    if not c.any():
                        c = inst
                core |= c
                contour |= inst & ~ndimage.binary_erosion(
                    inst, iterations=self.contour_width)
        return (np.stack([m, core, contour]).astype(np.float32),
                self.weights(lab, n))

    def weights(self, lab, n):
        """1.0 everywhere, plus a bonus on background pixels wedged between two
        different instances.

        One distance transform, not one per instance: the EDT's `indices`
        output gives each background pixel its NEAREST instance, i.e. a Voronoi
        partition. A pixel sits in a gap when instances disagree within a small
        neighbourhood of it - that is a Voronoi boundary - and it is close to
        buildings on both sides.
        """
        w = np.ones(lab.shape, np.float32)
        w += self._boundary_bonus(lab > 0)
        if n < 2:
            return w
        bg = lab == 0
        dist, idx = ndimage.distance_transform_edt(bg, return_distances=True,
                                                   return_indices=True)
        nearest = lab[tuple(idx)]                    # Voronoi cell id per pixel
        k = 3
        hi = ndimage.maximum_filter(nearest, size=k)
        lo = ndimage.minimum_filter(np.where(nearest > 0, nearest, 10 ** 6), size=k)
        gap = bg & (hi != lo) & (dist <= self.gap_radius)
        w[gap] += self.sep_w0 * np.exp(-(dist[gap] ** 2) /
                                       (2.0 * self.sep_sigma ** 2))
        return w

    def _boundary_bonus(self, m):
        """Make the outline expensive, on BOTH sides of it.

        Plain BCE+Dice barely notices a rounded corner - smoothing a 16x16
        building costs a fraction of a point of Dice - so nothing pushes the
        model toward the straight edges the rasterised OSM polygons actually
        have. This puts the loss where the shape is decided: weight falls off
        as exp(-d^2 / 2 sigma^2) with d the distance to the nearest label
        boundary, inside and outside alike. The separation bonus is different
        and complementary - it only touches background BETWEEN two buildings.
        """
        if self.bnd_w0 <= 0 or not m.any():
            return np.zeros(m.shape, np.float32)
        inner = m & ~ndimage.binary_erosion(m)
        edge = inner | (ndimage.binary_dilation(m) & ~m)
        d = ndimage.distance_transform_edt(~edge)
        return (self.bnd_w0 * np.exp(-(d ** 2) /
                                     (2.0 * self.bnd_sigma ** 2))).astype(np.float32)


class DistTransformTargets:
    """Two- or three-channel target: (mask, normalised EDT [, instance boundary]).

    Ch0  binary building mask
    Ch1  per-instance normalised EDT: 0.0 at every wall, 1.0 at the furthest
         interior point.  Local maxima → one watershed seed per building.
    Ch2  instance boundary (only when instance_boundary=True): a thin band
         marking every pixel where two DIFFERENT building instances are within
         `ib_dilate` pixels of each other.  At inference this becomes a ridge
         in the watershed surface that the flood cannot cross, separating
         buildings that share a wall or are separated by only 1-2 px of alley.

    The same separation/boundary weight maps as InstanceTargets are attached.
    """

    def __init__(self, sep_w0: float = 10.0, sep_sigma: float = 5.0,
                 bnd_w0: float = 0.0, bnd_sigma: float = 2.0,
                 instance_boundary: bool = False, ib_dilate: int = 2):
        self._aux = InstanceTargets(sep_w0=sep_w0, sep_sigma=sep_sigma,
                                    bnd_w0=bnd_w0, bnd_sigma=bnd_sigma)
        self.instance_boundary = instance_boundary
        self.ib_dilate = ib_dilate

    def __call__(self, mask: np.ndarray):
        m = mask.astype(bool)
        lab, n = ndimage.label(m)
        dt = np.zeros(m.shape, np.float32)
        for i in range(1, n + 1):
            inst = lab == i
            d = ndimage.distance_transform_edt(inst)
            dmax = float(d.max())
            if dmax > 0:
                dt[inst] = d[inst] / dmax
        _, w = self._aux(mask)
        channels = [m.astype(np.float32), dt]
        if self.instance_boundary:
            channels.append(self._instance_boundary(lab, n))
        return np.stack(channels), w

    def _instance_boundary(self, lab: np.ndarray, n: int) -> np.ndarray:
        """1.0 where dilations of two different instances overlap.

        Each building is dilated by ib_dilate pixels; a pixel that falls inside
        the dilation of instance A and also inside the dilation of instance B
        (A ≠ B) lies on the shared boundary.  Two buildings sharing a wall or
        separated by a 1-2 px alley always produce a non-empty boundary band.
        """
        if n < 2:
            return np.zeros(lab.shape, np.float32)
        # accumulated dilation map: records which instance first claimed each px
        claimed = np.zeros(lab.shape, np.int32)
        boundary = np.zeros(lab.shape, np.float32)
        for i in range(1, n + 1):
            dil = ndimage.binary_dilation(lab == i, iterations=self.ib_dilate)
            # pixel already claimed by a DIFFERENT instance → it's a boundary
            boundary[(claimed > 0) & (claimed != i) & dil] = 1.0
            claimed[dil & (claimed == 0)] = i
        return boundary

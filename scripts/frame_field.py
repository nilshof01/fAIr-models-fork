"""Frame Field Learning (Girard et al., CVPR 2021) for polygonal buildings.

The problem it solves: BCE+Dice is nearly indifferent to shape. Rounding the
corners of a 16x16 building costs a fraction of a point of Dice, so nothing
pushes the network toward the straight edges the rasterised OSM polygons
actually have, and the output is blobs.

A frame field gives it somewhere to put that knowledge. At every pixel the
network predicts TWO directions (not one), which is what a building corner
needs - a corner is where two wall directions meet, and a single direction
field cannot represent it without a discontinuity.

The two directions {u, -u, v, -v} are encoded as the roots of

    f(z) = z^4 + c2 z^2 + c0

with c0, c2 complex, so the head emits 4 real channels. This encoding is
smooth through corners and rotation-equivariant, which a pair of angles is
not.

Losses:
  align    on boundary pixels the mask's gradient direction must be a root of
           f, i.e. |f(z_grad)| = 0 - the field must contain the wall direction
  align90  the direction perpendicular to the gradient must NOT be a root,
           which stops the field collapsing to something that fits everything
  smooth   the field varies slowly, so walls stay straight between corners
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def complex_mul(a, b):
    """(B,2,H,W) complex tensors as [real, imag]."""
    return torch.stack([a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1],
                        a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]], 1)


def poly_f(z, c0, c2):
    """f(z) = z^4 + c2 z^2 + c0, all complex (B,2,H,W)."""
    z2 = complex_mul(z, z)
    z4 = complex_mul(z2, z2)
    return z4 + complex_mul(c2, z2) + c0


class FrameFieldLoss(torch.nn.Module):
    """align + align90 + smoothness, evaluated only where there is a boundary."""

    def __init__(self, w_align=1.0, w_align90=0.2, w_smooth=0.05):
        super().__init__()
        self.wa, self.w90, self.ws = w_align, w_align90, w_smooth

    @staticmethod
    def mask_gradient(mask):
        """Unit gradient of the (soft) mask: normal to the wall. (B,2,H,W)."""
        k = torch.tensor([[-1., 0., 1.]], device=mask.device, dtype=mask.dtype)
        gx = F.conv2d(mask, k.view(1, 1, 1, 3), padding=(0, 1))
        gy = F.conv2d(mask, k.t().view(1, 1, 3, 1), padding=(1, 0))
        g = torch.cat([gx, gy], 1)
        n = torch.norm(g, dim=1, keepdim=True).clamp_min(1e-6)
        return g / n, n

    def forward(self, field, mask):
        """field (B,4,H,W) = [c0.re, c0.im, c2.re, c2.im]; mask (B,1,H,W) in 0..1."""
        c0, c2 = field[:, :2], field[:, 2:]
        g, gnorm = self.mask_gradient(mask)
        w = (gnorm > 0.1).float()                     # boundary pixels only
        denom = w.sum().clamp_min(1.0)

        align = (poly_f(g, c0, c2) ** 2).sum(1, keepdim=True)
        l_align = (align * w).sum() / denom

        gperp = torch.stack([-g[:, 1], g[:, 0]], 1)   # along the wall
        a90 = (poly_f(gperp, c0, c2) ** 2).sum(1, keepdim=True)
        # reward the perpendicular NOT being a root, hinged so it cannot run away
        l_align90 = (F.relu(1.0 - a90) * w).sum() / denom

        d = (field[:, :, 1:, :] - field[:, :, :-1, :]).pow(2).mean() + \
            (field[:, :, :, 1:] - field[:, :, :, :-1]).pow(2).mean()
        return self.wa * l_align + self.w90 * l_align90 + self.ws * d


class FieldPolygoniser:
    """Simplify each instance's contour, snapping edges to the frame field.

    Douglas-Peucker alone gives straight edges at arbitrary angles. Snapping
    the dominant direction to the field - which was trained to hold the wall
    directions - is what recovers right angles.
    """

    def __init__(self, epsilon=1.5, snap_deg=12.0):
        self.eps = epsilon
        self.snap = np.deg2rad(snap_deg)

    @staticmethod
    def field_directions(field, m):
        """Mean wall direction inside one instance, from the field's roots."""
        c0 = field[0][m].mean() + 1j * field[1][m].mean()
        c2 = field[2][m].mean() + 1j * field[3][m].mean()
        roots = np.roots([1, 0, c2, 0, c0])
        return np.angle(roots)

    def __call__(self, lab, field=None):
        import cv2
        out = []
        for i in np.unique(lab):
            if i == 0:
                continue
            m = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            poly = cv2.approxPolyDP(c, self.eps, True)[:, 0, :].astype(np.float64)
            if field is not None and len(poly) >= 4:
                poly = self._snap(poly, self.field_directions(field, lab == i))
            out.append(poly)
        return out

    def _snap(self, poly, dirs):
        """Rotate each edge onto the nearest field direction, within snap_deg."""
        p = poly.copy()
        for k in range(len(p)):
            a, b = p[k], p[(k + 1) % len(p)]
            v = b - a
            ang = np.arctan2(v[1], v[0])
            cand = np.concatenate([dirs, dirs + np.pi])
            diff = np.angle(np.exp(1j * (cand - ang)))
            j = np.argmin(np.abs(diff))
            if abs(diff[j]) < self.snap:
                L = np.linalg.norm(v)
                mid = (a + b) / 2
                d = np.array([np.cos(cand[j]), np.sin(cand[j])])
                p[k], p[(k + 1) % len(p)] = mid - d * L / 2, mid + d * L / 2
        return p

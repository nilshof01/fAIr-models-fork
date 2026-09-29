"""Per-chip image and label descriptors for weakness analysis.

Sharpness deserves a note. Every chip is nominally zoom 19 (0.23-0.30 m/px),
but effective resolution varies enormously because the pool mixes satellite
and drone campaigns - measured across 400 chips the high-frequency energy
ratio spans 17x, and the drone projects sit at the sharp end exactly as
physics predicts.

The obvious statistic - mean absolute pixel-to-pixel difference - does NOT
measure that. It correlates +0.49 with building count: a dense city has more
edge energy than bare ground at identical sharpness, so it is half content.
The spectral measures below are near-independent of content (|r| <= 0.10)
while still explaining 39-58% of their variance by project, i.e. they track
the imagery source rather than what happens to be in frame.

Limit: all three saturate under extreme blur (beyond about sigma 2 on a
256-px chip almost no signal survives and the estimates turn around). They are
reliable across the range real imagery occupies, not as absolute blur meters.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class ChipFeatures:
    """Descriptors for one RGB chip, optionally with its label mask."""

    def __init__(self, rgb, mask=None):
        self.rgb = rgb.astype(np.float32) / 255.0
        self.gray = self.rgb.mean(2)
        self.mask = mask

    # ---- effective resolution -------------------------------------------
    def _spectrum(self):
        g = self.gray - self.gray.mean()
        F = np.fft.fftshift(np.abs(np.fft.fft2(g)) ** 2)
        n = min(F.shape)
        c = n // 2
        y, x = np.ogrid[:F.shape[0], :F.shape[1]]
        r = np.sqrt((y - F.shape[0] // 2) ** 2 + (x - F.shape[1] // 2) ** 2)
        return F, r, c

    def hf_ratio(self):
        """Share of spectral energy above half-Nyquist. Low = blurred or
        upsampled from a coarser source."""
        F, r, c = self._spectrum()
        return float(F[r > 0.5 * c].sum() / max(F.sum(), 1e-9))

    def spectral_slope(self):
        """Log-log falloff of the radially averaged spectrum. Steeper (more
        negative) = less fine detail."""
        F, r, c = self._spectrum()
        rb = r.astype(int).ravel()
        prof = np.bincount(rb, F.ravel()) / np.maximum(np.bincount(rb), 1)
        k = np.arange(1, c)[3:]
        if len(k) < 4:
            return float("nan")
        return float(np.polyfit(np.log(k), np.log(prof[1:c][3:] + 1e-12), 1)[0])

    def edge_width(self):
        """Approximate edge transition width in pixels: for a ramp of width w,
        gradient ~ 1/w and Laplacian ~ 1/w^2, so their ratio ~ w. Measured only
        AT the strongest edges, so it is close to content-independent.

        HIGHER = blurrier. Satellite chips here sit near 11 px, drone chips
        near 3 px, at identical nominal 0.23-0.30 m/px."""
        gx = ndimage.sobel(self.gray, 1)
        gy = ndimage.sobel(self.gray, 0)
        mag = np.hypot(gx, gy)
        lap = np.abs(ndimage.laplace(self.gray))
        strong = mag >= max(np.percentile(mag, 99), 1e-6)
        if strong.sum() < 8:
            return float("nan")
        return float(mag[strong].mean() / max(lap[strong].mean(), 1e-9))


    # ---- exposure and colour --------------------------------------------
    def brightness(self):
        return float(self.gray.mean())

    def contrast(self):
        return float(self.gray.std())

    def saturation(self):
        mx = self.rgb.max(2)
        mn = self.rgb.min(2)
        return float(((mx - mn) / np.maximum(mx, 1e-6)).mean())

    def green_excess(self):
        """Vegetation proxy: the tree-crown false-positive hypothesis."""
        r, g, b = self.rgb[..., 0], self.rgb[..., 1], self.rgb[..., 2]
        return float((2 * g - r - b).mean())

    def clipped(self):
        """Share of pixels crushed to black or blown to white."""
        return float(((self.gray < 0.02) | (self.gray > 0.98)).mean())

    # ---- label geometry --------------------------------------------------
    def label_stats(self):
        if self.mask is None:
            return {}
        lab, n = ndimage.label(self.mask)
        if n == 0:
            return {"n_components": 0, "median_area": 0.0, "building_frac": 0.0,
                    "crowding": 0.0}
        areas = np.bincount(lab.ravel())[1:]
        dil = ndimage.binary_dilation(self.mask, iterations=3)
        touching = ndimage.label(dil)[1]
        return {"n_components": int(n),
                "median_area": float(np.median(areas)),
                "building_frac": float(self.mask.mean()),
                # < 1 means separate buildings merge when dilated: crowded
                "crowding": float(touching / max(n, 1))}

    def all(self):
        d = {"hf_ratio": self.hf_ratio(), "spectral_slope": self.spectral_slope(),
             "edge_width": self.edge_width(), "brightness": self.brightness(),
             "contrast": self.contrast(), "saturation": self.saturation(),
             "green_excess": self.green_excess(), "clipped": self.clipped()}
        d.update(self.label_stats())
        return d


FEATURES = ["hf_ratio", "spectral_slope", "edge_width", "brightness",
            "contrast", "saturation", "green_excess", "clipped",
            "n_components", "median_area", "building_frac", "crowding"]

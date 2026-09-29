"""Per-pixel instance embeddings with the discriminative loss.

De Brabandere, Neven & Van Gool (2017). The network emits a D-dimensional
vector per pixel; the loss pulls every pixel of one building toward that
building's mean embedding and pushes different buildings' means apart. Two
merged roofs are then penalised at OBJECT scale - their pixels failed to
separate in embedding space - without needing proposals, Hungarian matching,
or a non-differentiable watershed in the graph.

That last point is why this is the natural next rung above mask+core+contour:
a matched loss on watershed output cannot train, because watershed is not
differentiable. This can.

  L = alpha * L_var + beta * L_dist + gamma * L_reg

  L_var   hinged pull, free inside delta_v of the mean
  L_dist  hinged push, active until two means are 2*delta_d apart
  L_reg   weak pull toward the origin so embeddings stay bounded

delta_v < delta_d is what makes clustering unambiguous: every pixel sits
within delta_v of its own mean while means are 2*delta_d apart, so a ball of
radius delta_v around any pixel contains only its own instance.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class DiscriminativeLoss(torch.nn.Module):
    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=1e-3):
        super().__init__()
        self.dv, self.dd = delta_v, delta_d
        self.a, self.b, self.g = alpha, beta, gamma

    def forward(self, emb, inst):
        """emb (B, D, H, W) embeddings; inst (B, H, W) int instance ids, 0 = bg."""
        total = emb.new_zeros(())
        n = 0
        for e, y in zip(emb, inst):
            ids = torch.unique(y)
            ids = ids[ids > 0]
            if len(ids) == 0:
                continue
            e = e.flatten(1)                       # D, HW
            y = y.flatten()                        # HW
            means = []
            l_var = e.new_zeros(())
            for i in ids:
                m = y == i
                pix = e[:, m]                      # D, N
                mu = pix.mean(1, keepdim=True)
                means.append(mu)
                d = torch.norm(pix - mu, dim=0)
                l_var = l_var + (F.relu(d - self.dv) ** 2).mean()
            l_var = l_var / len(ids)

            mu = torch.cat(means, 1)               # D, C
            l_dist = e.new_zeros(())
            if len(ids) > 1:
                diff = mu[:, :, None] - mu[:, None, :]
                dist = torch.norm(diff, dim=0)
                off = ~torch.eye(len(ids), dtype=torch.bool, device=e.device)
                l_dist = (F.relu(2 * self.dd - dist[off]) ** 2).mean()
            l_reg = torch.norm(mu, dim=0).mean()
            total = total + self.a * l_var + self.b * l_dist + self.g * l_reg
            n += 1
        return total / max(n, 1)


class EmbeddingClusterer:
    """Assign mask pixels to instances by clustering the embedding.

    Greedy variant from the paper: take an unassigned pixel, gather everything
    within delta_v of it, recentre, repeat to convergence, emit as one
    instance. Cheap and deterministic if the seed order is deterministic.
    """

    def __init__(self, delta_v=0.5, min_px=30, max_instances=200, iters=6):
        self.dv, self.min_px, self.max_n, self.iters = delta_v, min_px, max_instances, iters

    def __call__(self, emb, mask):
        """emb (D, H, W) numpy; mask (H, W) bool. Returns an int label image."""
        out = np.zeros(mask.shape, np.int32)
        idx = np.flatnonzero(mask.ravel())
        if not len(idx):
            return out
        X = emb.reshape(emb.shape[0], -1)[:, idx].T          # N, D
        unassigned = np.ones(len(idx), bool)
        label = 1
        while unassigned.any() and label <= self.max_n:
            start = X[unassigned][0]
            centre = start
            for _ in range(self.iters):
                d = np.linalg.norm(X - centre, axis=1)
                near = (d < self.dv) & unassigned
                if not near.any():
                    break
                centre = X[near].mean(0)
            d = np.linalg.norm(X - centre, axis=1)
            members = (d < self.dv) & unassigned
            if members.sum() >= self.min_px:
                out.ravel()[idx[members]] = label
                label += 1
            elif members.sum() == 0:
                break
            unassigned &= ~members
        return out

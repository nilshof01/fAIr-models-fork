"""Load a dinov3_hot checkpoint and present it like the UNet models.

Both families emit three channels, but in different orders:

    UNet --heads dist3   [mask, normalised EDT (interior), instance boundary]
    dinov3_hot           [mask, boundary, signed distance (tanh)]

So the ViT's channel 2 plays the role the UNet's channel 1 does - high in
building interiors, which is what seeds a watershed - and its channel 1 is the
ridge. Reordering to [mask, interior, boundary] lets the SAME decoder and the
same instance metrics run over both, which is the only way the numbers compare.

The signed distance is tanh-ranged [-1, 1]; it is mapped to [0, 1] so a single
core threshold means the same thing for either model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class VitRun:
    """A dinov3_hot run directory, with the same surface as a UNet run."""

    def __init__(self, run_dir, device):
        from dinov3_hot.config import load_config
        from dinov3_hot.train import build_model
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import timm_dinov3_backbone  # noqa: F401  registers the backbones

        run = Path(run_dir)
        self.cfg = load_config(str(run / "config.yaml"))
        ckpts = sorted((run / "ckpts").glob("best-*.ckpt"))
        if not ckpts:
            ckpts = sorted((run / "ckpts").glob("*.ckpt"))
        if not ckpts:
            raise SystemExit(f"no checkpoint under {run}/ckpts")
        self.ckpt = ckpts[-1]
        lit = build_model(self.cfg)
        state = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        lit.load_state_dict(state["state_dict"])
        self.model = lit.eval().to(device)
        self.device = device
        self.mean, self.std = self._norm()

    def _norm(self):
        from dinov3_hot.data import load_norm_stats
        m, s = load_norm_stats(self.cfg.dataset_repo, self.cfg.data_root)
        return np.array(m, np.float32), np.array(s, np.float32)

    @torch.no_grad()
    def predict(self, rgb):
        """(3, H, W) as [mask, interior, boundary] in 0..1, matching dist3."""
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        out = self.model(t)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits[0]
        mask = torch.sigmoid(logits[0])
        boundary = torch.sigmoid(logits[1])
        interior = (torch.tanh(logits[2]) + 1.0) / 2.0   # [-1,1] -> [0,1]
        return torch.stack([mask, interior, boundary]).float().cpu().numpy()


def is_vit_run(run_dir):
    run = Path(run_dir)
    return (run / "config.yaml").exists() and (run / "ckpts").is_dir()

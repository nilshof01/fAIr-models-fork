"""Register timm-hosted DINOv3 backbones under terratorch keys.

Meta released the satellite pretraining (SAT-493M) only for ViT-L/16 and
ViT-7B/16 - there is no ViT-S SAT checkpoint - and those weights are
distributed through timm/HF rather than the gated Meta download that
terratorch's own wrapper expects. This module exposes them through the same
registry key mechanism `dinov3_hot.model.DinoV3UperNet` already uses, so the
shipped trainer, loss and evaluation run unchanged with a swapped backbone.

The facade mimics the two members DinoV3UperNet touches: `embed_dim` and
`get_intermediate_layers(...)` with Meta's keyword names (timm calls the
prefix-token flag `return_prefix_tokens`).
"""
import timm
import torch
from torch import nn
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

VARIANTS = {
    "dinov3_vitl16_sat493m": "vit_large_patch16_dinov3.sat493m",
    "dinov3_vitl16_lvd1689m": "vit_large_patch16_dinov3.lvd1689m",
}


class TimmDinoV3(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.embed_dim = self.model.embed_dim
        self.blocks = self.model.blocks

    def get_intermediate_layers(self, x, n, reshape=True, norm=True,
                                return_class_token=False):
        # timm implements DINOv3 as `Eva`, which exposes `forward_intermediates`
        # (NCHW already reshaped, prefix tokens dropped) instead of Meta's
        # `get_intermediate_layers`.
        return self.model.forward_intermediates(
            x, indices=list(n), norm=norm,
            output_fmt="NCHW" if reshape else "NLC",
            return_prefix_tokens=return_class_token,
            intermediates_only=True)


class _Wrapper(nn.Module):
    """terratorch's dinov3 wrapper exposes the model as `.dinov3`."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.dinov3 = inner


def _make(model_name):
    def build(ckpt_path=None, **kwargs):   # ckpt_path unused: timm holds weights
        return _Wrapper(TimmDinoV3(model_name))
    return build


for _key, _name in VARIANTS.items():
    _fn = _make(_name)
    _fn.__name__ = _key
    TERRATORCH_BACKBONE_REGISTRY.register(_fn)

"""Post-build model surgeries, shared by the trainer and both eval scripts.

Encoder specs are strings like "resnet34", "resnet34+blurpool",
"resnet34+silu" or combinations; build_unet() applies the mods in a fixed
order so trainer and eval always construct identical module trees (BlurPool
adds buffer keys and SiLU changes the forward, so reconstruction must match
the checkpoint exactly).
"""
import torch.nn as nn
import segmentation_models_pytorch as smp


def apply_blurpool(enc):
    """Zhang (2019) anti-aliasing retrofit for torchvision-style ResNet
    encoders (BasicBlock only, i.e. resnet18/34): every stride-2 op becomes
    stride 1 followed by a fixed BlurPool. Conv weight shapes are unchanged,
    so ImageNet initialisation is preserved."""
    from timm.layers import BlurPool2d

    enc.maxpool = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        BlurPool2d(enc.conv1.out_channels, stride=2),
    )
    for layer in (enc.layer2, enc.layer3, enc.layer4):
        b = layer[0]
        if b.conv1.stride == (2, 2):
            b.conv1.stride = (1, 1)
            b.conv2 = nn.Sequential(BlurPool2d(b.conv1.out_channels, stride=2), b.conv2)
        if b.downsample is not None and b.downsample[0].stride == (2, 2):
            ds_conv = b.downsample[0]
            ds_conv.stride = (1, 1)
            b.downsample = nn.Sequential(
                BlurPool2d(ds_conv.in_channels, stride=2), *b.downsample)


def swap_relu_silu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.SiLU(inplace=True))
        else:
            swap_relu_silu(child)


def build_unet(encoder_spec, encoder_weights=None):
    base, *mods = encoder_spec.split("+")
    m = smp.create_model("unet", encoder_name=base, encoder_weights=encoder_weights,
                         classes=2, in_channels=3)
    if "blurpool" in mods:
        apply_blurpool(m.encoder)
    if "silu" in mods:
        swap_relu_silu(m.decoder)
    return m

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
import ltr.models.backbone as backbones

from util.misc import NestedTensor

from ltr.models.neck.position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensors):

        xs = self.body(tensors)

        out = {}

        for name, x in xs.items():
            out[name] = x
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 name,
                 output_layers,
                 pretrained,
                 frozen_layers):
        backbone = getattr(backbones, name)(
            output_layers=output_layers, pretrained=pretrained, frozen_layers=frozen_layers)
        if name in ('resnet18', 'resnet34'):
            num_channels = 256
        elif name in ('convnext_tiny'):
            num_channels = 384
        elif name in ('lightrack_backbone_M'):
            num_channels = 96
        else:
            num_channels = 1024
        super().__init__(backbone, num_channels)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensors):
        xs = self[0](tensors)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))
        return out, pos

    def track(self, tensors):
        xs = self[0](tensors)
        out = []
        for name, x in xs.items():
            out.append(x)
        return out



def build_backbone(settings, backbone_pretrained=True, frozen_backbone_layers=()):
    position_embedding = build_position_encoding(settings)
    if not hasattr(settings, 'backbone'):
        settings.backbone = 'resnet50'
    # backbone = Backbone(name=settings.backbone, output_layers=['conv1','layer1','layer2','layer3'], pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    backbone = Backbone(name=settings.backbone, output_layers=['layer3'], pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

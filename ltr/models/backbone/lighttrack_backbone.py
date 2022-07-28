import numpy as np
import random
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.backbone.lighttrack import build_subnet

def lightrack_backbone_M(output_layers=None, pretrained=False, **kwargs):
    path_backbone = [[0], [4, 5], [0, 2, 5, 1], [4, 0, 4, 4], [5, 2, 1, 0], [4, 2, 5, 4], [0]]
    path_ops = (3, 2)
    model = build_subnet(path_backbone=path_backbone, ops=path_ops)
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'lighttrack/LightTrackM.pth')
    if pretrained:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k[0:8] == 'features':
                name = k[9:]
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    return model




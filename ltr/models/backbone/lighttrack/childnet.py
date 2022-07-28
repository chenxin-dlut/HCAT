import warnings

warnings.filterwarnings('ignore')

import os
import logging
import torch
from collections import OrderedDict
from .model import _gen_childnet


def build_subnet(path_backbone, ops=None):
    arch_list = path_backbone

    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25',
         'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25',
         'ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25',
         'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25'],
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
         'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25'],
        # stage 5, 14x14in
        ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25',
         'ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'],
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c320_se0.25'],
    ]

    model = _gen_childnet(
        arch_list,
        arch_def,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0,
        global_pool='avg',
        bn_momentum=None,
        bn_eps=None,
        pool_bn=False,
        zero_gamma=False,
        ops=ops)

    return model



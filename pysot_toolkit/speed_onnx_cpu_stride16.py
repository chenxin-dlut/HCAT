import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import torch
import time
import onnx
import onnxruntime
import numpy as np
import torch.nn as nn
from ltr.models.tracking.hcat import MLP
from ltr.models.backbone.transt_backbone import build_backbone,Backbone
from ltr.models.neck.featurefusion_network import build_featurefusion_network
import ltr.admin.settings as ws_settings
from ltr.models.neck.position_encoding import build_position_encoding


def getdata(b=1,x=256,z=8):
    x = torch.randn(1,3,x,x)
    feature = torch.randn(1,384,z,z)
    # feature = torch.randn(1,1024,z,z)
    # feature = torch.randn(1,256,z,z)
    # feature = torch.randn(1,96,z,z)
    pos = torch.randn(1,256,z,z)
    return x, feature, pos

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def inference_speed_track():
    T_w = 300  # warmup
    T_t = 5000  # test
    ort_session = onnxruntime.InferenceSession("models/com_cov.onnx_test",providers=['CPUExecutionProvider'])
    with torch.no_grad():
        x, feature, pos = getdata()
        ort_inputs = {'x':to_numpy(x),
                      'feature_template':to_numpy(feature),
                      'pos_template':to_numpy(pos)}
        for i in range(T_w):
            nxp = ort_session.run(None, ort_inputs)

        onnx_s = time.time()
        for i in range(T_t):
            nxp = ort_session.run(None, ort_inputs)
        onnx_e = time.time()
        onnxt = onnx_e - onnx_s
    print('The tracking process inference speed of onnx model: %.2f FPS' % (T_t / onnxt))

if __name__ == "__main__":
    inference_speed_track()
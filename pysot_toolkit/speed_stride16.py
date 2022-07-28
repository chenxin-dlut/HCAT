import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import torch
import time
import numpy as np
import torch.nn as nn
from ltr.models.tracking.hcat import MLP
from ltr.models.backbone.transt_backbone import build_backbone,Backbone
from ltr.models.neck.featurefusion_network import build_featurefusion_network
import ltr.admin.settings as ws_settings
from ltr.models.neck.position_encoding import build_position_encoding
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


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

class TransT(nn.Module):
    def __init__(self, backbone, featurefusion_network, num_classes):
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(16, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, search,feature_template,pos_template):
        feature_search, pos_search = self.backbone(search)
        src_search = feature_search[-1]
        src_template = feature_template
        # hs, torch_s, torch_e = self.featurefusion_network(self.input_proj(src_template), self.input_proj(src_search), pos_template, pos_search[-1])
        hs = self.featurefusion_network(self.input_proj(src_template), self.input_proj(src_search), pos_template, pos_search[-1], self.query_embed.weight)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # return out, torch_s, torch_e
        return out

def inference_speed_track(net):
    T_w = 100  # warmup
    T_t = 1000 # test
    with torch.no_grad():
        x, feature, pos = getdata()
        x_cuda,feature_cuda, pos_cuda = x.cuda(),feature.cuda(),pos.cuda()
        for i in range(T_w):
            oup = net(x_cuda,feature_cuda, pos_cuda)

        torch_s = time.time()
        for i in range(T_t):
            oup = net(x_cuda,feature_cuda, pos_cuda)
        torch_e = time.time()
        torcht = torch_e - torch_s
    print('The tracking process inference speed of pytorch model: %.2f FPS' % (T_t / torcht))
    print('The tracking process inference speed of pytorch model: %.10f s' % (torcht / T_t))

if __name__ == "__main__":
    # test the running speed
    settings = ws_settings.Settings()
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.backbone = 'convnext_tiny'
    # settings.backbone = 'resnet50'
    # settings.backbone = 'resnet18'
    # settings.backbone = 'lightrack_backbone_M'
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 2
    use_gpu = True
    num_classes = 1
    backbone_net = build_backbone(settings,backbone_pretrained=False)
    featurefusion_network = build_featurefusion_network(settings)
    net = TransT(backbone_net,featurefusion_network,num_classes=num_classes)
    if use_gpu:
        net.cuda()
        net.eval()
    ######test tracking process inference speed#####
    inference_speed_track(net)

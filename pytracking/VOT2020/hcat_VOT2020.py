# -*-coding:utf-8-*-
import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.VOT2020.hcat_onnx_bbox_class import run_vot_exp



exp_name='hcat'
para_w=float(604/1000)
penalty_k=float(298/1000)
backbone_path = ''
model_path = ''
save_root = '/home/cx/cx1/VOT2021/vot2021_search_1/vis/' + exp_name
run_vot_exp('transt', window=para_w, penalty_k=penalty_k, backbone_path=backbone_path,model_path=model_path, save_root=save_root, VIS=False)
# run_vot_exp('dimp','super_dimp','ARcm_coco_seg_only_mask_384',0.65,VIS=True)


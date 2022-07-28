import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.VOT2020.transt_bbox_class import run_vot_exp



exp_name='ligttranst_w604_e_k298_u'
para_w=float(604/1000)
penalty_k=float(298/1000)
net_path = '/home/kb/HCAT/pysot_toolkit/models/res18_N2_q16.pth'




save_root = '/home/cx/cx1/VOT2021/vot2021_search_1/vis/' + exp_name
run_vot_exp('transt', window=para_w, penalty_k=penalty_k, net_path=net_path, save_root=save_root, VIS=False)
# run_vot_exp('dimp','super_dimp','ARcm_coco_seg_only_mask_384',0.65,VIS=True)


import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.VOT2020.transt_bbox_class import run_vot_exp

# net_path = os.path.join(env_path, 'models/TransTsegm_ep0075.pth.tar')
net_path = '/home/cx/cx2/models/TransT_light/stride16_N2_sparse_q16_res18/TransT_ep0247.pth.tar'
save_root = '/home/cx/cx1/light_transt/TransT_fix_nested_v5_b/vis'
run_vot_exp('transt', window=0.42, penalty_k=0, net_path=net_path, save_root=save_root, VIS=False)

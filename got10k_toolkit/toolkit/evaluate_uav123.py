from got10k_toolkit.toolkit.experiments import ExperimentUAV123
from got10k_toolkit.toolkit.trackers.identity_tracker import IdentityTracker
from got10k_toolkit.toolkit.trackers.net_wrappers import NetWithBackbone
import os

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

# net = NetWithBackbone(net_path='/home/cx/TransT/transt.pth', use_gpu=True)
# tracker = IdentityTracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)
experiment = ExperimentUAV123(
    root_dir='/home/cx/cx2/Downloads/UAV123/UAV123_fix/Dataset_UAV123/UAV123',  # GOT-10k's root directory
    result_dir='/home/cx/cx1/light_transt/TransT_fix_nested_v5_b/got10k_toolkit/toolkit/results',  # where to store tracking results
    report_dir='/home/cx/cx1/light_transt/TransT_fix_nested_v5_b/got10k_toolkit/toolkit/reports'  # where to store evaluation reports
)
# experiment.report(['TransT-N4', 'TransT-N2', 'PrDiMP', 'DiMP', 'SiamRPN++', 'ATOM', 'ECO', 'UPDT', 'CCOT', 'DaSiamRPN'])
experiment.report(['lighttranst'])
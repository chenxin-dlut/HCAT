import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '../..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
from got10k_toolkit.toolkit.experiments import ExperimentNfS
from got10k_toolkit.toolkit.trackers.identity_tracker import IdentityTracker
from got10k_toolkit.toolkit.trackers.net_wrappers import NetWithBackbone
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

# net = NetWithBackbone(net_path='/home/cx/TransT/transt.pth', use_gpu=True)
# tracker = IdentityTracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)
experiment = ExperimentNfS(
    root_dir='/home/cx/cx2/Downloads/nfs',  # GOT-10k's root directory
    fps=30,
    result_dir='/home/cx/cx1/light_transt/TransT_fix_nested_v5_b/got10k_toolkit/toolkit/results',  # where to store tracking results
    report_dir='/home/cx/cx1/light_transt/TransT_fix_nested_v5_b/got10k_toolkit/toolkit/reports'  # where to store evaluation reports
)
# experiment.report([tracker.name])
# experiment.report(['TransT-N4', 'TransT-N2', 'PrDiMP', 'DiMP', 'ATOM', 'ECO', 'UPDT', 'CCOT', 'MDNet'])
experiment.report(['lighttranst'])
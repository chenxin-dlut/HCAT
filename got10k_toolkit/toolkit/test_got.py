from got10k_toolkit.toolkit.experiments import ExperimentGOT10k
from got10k_toolkit.toolkit.trackers.identity_tracker import IdentityTracker
from got10k_toolkit.toolkit.trackers.net_wrappers import NetWithBackbone
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

#Specify the path
net_path = '/home/cx/TransT/models/transt.pth'
dataset_root= '/home/cx/detr-tracking-v6/pytracking/toolkit/got10k/datasets'

#TransT
net = NetWithBackbone(net_path=net_path, use_gpu=True)
tracker = IdentityTracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)

#Evaluation
experiment = ExperimentGOT10k(
    root_dir=dataset_root,  # GOT-10k's root directory
    subset='test',  # 'train' | 'val' | 'test'
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.run(tracker, visualize=True)
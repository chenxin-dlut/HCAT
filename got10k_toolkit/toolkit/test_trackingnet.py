from got10k_toolkit.toolkit.experiments import ExperimentTrackingNet
from got10k_toolkit.toolkit.trackers.identity_tracker import IdentityTracker
from got10k_toolkit.toolkit.trackers.net_wrappers import NetWithBackbone
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

net = NetWithBackbone(net_path='/home/cx/TransT/transt.pth', use_gpu=True)
tracker = IdentityTracker(name='transt_np', net=net, window_penalty=0, exemplar_size=128, instance_size=256)
experiment = ExperimentTrackingNet(
    root_dir='/home/cx/cx2/TrackingNet',  # TrackingNet's root directory
    subset='test',  # 'train' | 'val' | 'test'
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.run(tracker, visualize=False)
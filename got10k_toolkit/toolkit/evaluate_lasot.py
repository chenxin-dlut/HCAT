from got10k_toolkit.toolkit.experiments import ExperimentLaSOT
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

experiment = ExperimentLaSOT(
    root_dir='/home/cx/cx3/LaSOTBenchmark',  # LaSOT's root directory
    subset='test',  # 'train' | 'val' | 'test'
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.report(['transt_N4'])
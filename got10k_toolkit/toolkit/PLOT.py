from got10k.experiments import ExperimentUAV123

report_files = ['reports/UAV123/N4-676-sub/performance.json', 'reports/UAV123/ATOM/performance.json']
tracker_names = ['N4-676-sub', 'ATOM']
# report_files = ['reports/UAV123/ATOM/performance.json']
# tracker_names = ['ATOM']

# setup experiment and plot curves
experiment = ExperimentUAV123(
    root_dir='/home/cx/cx2/Downloads/UAV123/UAV123_fix/Dataset_UAV123/UAV123',  # GOT-10k's root directory
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.plot_curves(tracker_names)
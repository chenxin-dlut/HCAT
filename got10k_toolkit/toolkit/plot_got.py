from got10k_toolkit.toolkit.experiments import ExperimentGOT10k

report_files = ['/home/cx/transt-tracking-v6/pytracking/FusionNet_report_2020_11_09_07_09_11.json']
tracker_names = ['TRTR']

# setup experiment and plot curves
experiment = ExperimentGOT10k('/home/cx/detr-tracking-v6/pytracking/toolkit/got10k/datasets', subset='test')
experiment.plot_curves(report_files, tracker_names)
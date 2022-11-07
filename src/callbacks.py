"""
This module defines callbacks that are needed to stop the execution of the algorithm and save the best state of the graph.
"""

import pickle

import pandas as pd

from src import config
from src.config import graphlet_id_object_dict, path_to_output, logger as log
from src.enums import PerformanceMetric


class ModelCheckpoint:

    def __init__(self, monitor: PerformanceMetric):
        self.monitor = monitor
        self.best_monitor_val = -float('inf')

    @staticmethod
    def save_graph_state(val_results):
        results = {}
        for gr_id, gr in graphlet_id_object_dict.items():
            gr_paper_ids = [paper_obj.get_p_id() for paper_obj in gr.get_papers()]
            results[gr_id] = gr_paper_ids

        atomic_name = val_results['atomic_name'][0]
        precision = val_results['precision'][0]
        recall = val_results['recall'][0]
        f1 = val_results['f1'][0]

        if atomic_name in config.final_results['atomic_name'].unique():
            config.final_results.loc[config.final_results['atomic_name'] == atomic_name, 'precision'] = precision
            config.final_results.loc[config.final_results['atomic_name'] == atomic_name, 'recall'] = recall
            config.final_results.loc[config.final_results['atomic_name'] == atomic_name, 'f1'] = f1
        else:
            config.final_results = pd.concat([val_results, config.final_results], ignore_index=True)

        with open(path_to_output + 'clustering_results.pickle', 'wb') as handle:
            pickle.dump(config.final_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path_to_output + 'disambiguated_files/' + atomic_name + '.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def check(self, curr_monitor_val, val_results):
        desc = ""
        if curr_monitor_val > self.best_monitor_val:
            desc = self.monitor.value + " improved from " + str(self.best_monitor_val) + " to " + str(
                curr_monitor_val) + ", saving graph status"
            self.best_monitor_val = curr_monitor_val
            ModelCheckpoint.save_graph_state(val_results)

        return desc


class EarlyStopping:

    def __init__(self, patience: int, monitor: PerformanceMetric, significance_level: float):
        self.patience = patience
        self.monitor = monitor
        self.significance_level = significance_level
        self.best_monitor_val = -float('inf')
        self.worse_monitor_val_count = 0

    def check(self, curr_monitor_val):
        stop_algo = False

        if (curr_monitor_val - self.best_monitor_val) < self.significance_level:
            self.worse_monitor_val_count = self.worse_monitor_val_count + 1
        else:
            self.best_monitor_val = curr_monitor_val
            self.worse_monitor_val_count = 0

        if self.worse_monitor_val_count == self.patience:
            stop_algo = True
            log.info("%s did not improve significantly(level = %s) from %s in last %s epochs, early stopping",
                     self.monitor.value, self.significance_level, self.best_monitor_val, self.patience)

        return stop_algo

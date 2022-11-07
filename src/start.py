import glob
import os
from datetime import datetime

import src.graph_initializer as graph_init
from src import config
from src.callbacks import EarlyStopping, ModelCheckpoint
from src.config import logger as log, validation_metric as v_metric, \
    path_to_output as o_path, \
    logging_interval as l_interval, early_stop_monitor as es_monitor, model_checkpoint_monitor as mc_monitor, \
    early_stop_significance_level as es_sig_level
from src.metropolis import Metropolis


def run():
    init_start_time = datetime.now()
    atomic_names_list = [os.path.basename(file_path)[:-4] for file_path in
                         glob.glob(config.path_to_dataset + "and_data/" + "*.txt")]

    # to store the results of disambiguation
    os.makedirs(o_path + "disambiguated_files/", exist_ok=True)

    for atomic_name in atomic_names_list:
        log.info("atomic name : {}".format(atomic_name))
        ground_truths = graph_init.create_graph(atomic_name)

        if len(ground_truths) > 0:
            count_of_papers = len(ground_truths[atomic_name])
            num_itr = count_of_papers * 10
            es_patience = count_of_papers
            es = EarlyStopping(patience=es_patience, monitor=es_monitor, significance_level=es_sig_level)
            mc = ModelCheckpoint(monitor=mc_monitor)
            mh = Metropolis(epochs=num_itr, validation_metric=v_metric, logging_interval=l_interval, early_stop=es,
                            model_checkpoint=mc, a_name=atomic_name)
            mh.start(ground_truths)
        else:
            continue

    log.info("Clustering results ")
    log.info(" {: <20} | {: <20} | {: <20}  ".format('Precision',
                                                     'Recall', 'F1'))
    log.info(" {: <20} | {: <20} | {: <20}  ".format(config.final_results['precision'].mean(),
                                                     config.final_results['recall'].mean(),
                                                     config.final_results['f1'].mean()))

    init_end_time = datetime.now()
    log.info("Time taken for algorithm : {}".format(init_end_time - init_start_time))

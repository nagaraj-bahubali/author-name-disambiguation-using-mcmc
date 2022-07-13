import pickle

import src.graph_initializer as graph_init
from src.callbacks import EarlyStopping, ModelCheckpoint
from src.config import logger as log, num_of_iterations as num_iter, validation_metric as v_metric, \
    dataset_name as d_name, path_to_output as o_path, verbose as vb, \
    logging_interval as l_interval, early_stop_monitor as es_monitor, model_checkpoint_monitor as mc_monitor, \
    early_stop_patience as es_patience, early_stop_significance_level as es_sig_level
from src.metropolis import Metropolis


def main():
    print("\nThe Logs will be available at : ", o_path + "summary.log", "\n")

    log.info("Dataset Name : %s | Validation Metric : %s | Iterations : %s | Desc : %s", d_name, v_metric, num_iter, vb)

    ground_truths = graph_init.create_graph()
    es = EarlyStopping(patience=es_patience, monitor=es_monitor, significance_level=es_sig_level)
    mc = ModelCheckpoint(monitor=mc_monitor)
    mh = Metropolis(epochs=num_iter, validation_metric=v_metric, logging_interval=l_interval, early_stop=es,
                    model_checkpoint=mc)
    mh.start(ground_truths)

    with open(o_path + 'ground_truths.pickle', 'wb') as handle:
        pickle.dump(ground_truths, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

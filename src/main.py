import pickle
from datetime import datetime

import src.graph_initializer as graph_init
import src.metropolis as metropolis
import src.validator as validator
from src.config import logger as log, num_of_iterations as num_iter, validation_metric as v_metric, \
    dataset_name as d_name, path_to_output as o_path, tracker as tracker, verbose as vb, \
    graphlet_id_object_dict as graphlet_id_object_dict, logging_interval as l_interval


def main():
    print("\nThe Logs will be available at : ", o_path + "summary.log", "\n")

    log.info("Dataset Name : %s | Validation Metric : %s | Iterations : %s | Desc : %s", d_name, v_metric,
             num_iter,
             vb)

    init_start_time = datetime.now()
    ground_truths = graph_init.create_graph()
    init_end_time = datetime.now()

    log.info("Time taken for graph initialization : %s \n", init_end_time - init_start_time)
    log.info("-" * 80)
    log.info(" {: <10} |  {: <20} | {: <20} | {: <20} ".format("Iteration", "Precision", "Recall", "F1"))
    log.info("-" * 80)

    alg_start_time = datetime.now()
    for i in range(1, num_iter + 1):
        metropolis.run(i)
        if i % l_interval == 0:
            dump_results = i == num_iter
            validator.run(ground_truths, iteration=i, metric_type=v_metric, dump_results=dump_results)
    alg_end_time = datetime.now()
    log.info("Time taken to run the algorithm : %s", alg_end_time - alg_start_time)

    results = {}
    for gr_id, gr in graphlet_id_object_dict.items():
        gr_paper_ids = [paper_obj.get_p_id() for paper_obj in gr.get_papers()]
        results[gr_id] = gr_paper_ids

    with open(o_path + 'results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(o_path + 'ground_truths.pickle', 'wb') as handle:
        pickle.dump(ground_truths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(o_path + 'tracker.pickle', 'wb') as handle:
        pickle.dump(tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("Results dumped")


if __name__ == '__main__':
    main()

    # with open('./data/output/ground_truth.pickle', 'wb') as handle:
    #     pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('./data/output/predictions.pickle', 'wb') as handle:
    #     pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

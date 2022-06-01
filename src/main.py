import pickle
from datetime import datetime

import src.config as config
import src.graph_initializer as graph_init
import src.metropolis as metropolis
import src.validator as validator
from src.config import NUM_OF_ITERATIONS

log = config.logger


def main():
    print("\nThe Logs will be available at : ", config.path_to_output + "summary.log")
    verbose = "All three components after the fix"
    device = "Mac"
    dataset_name = "unified-and-dataset_1_filtered"
    log.info("Device : %s | Dataset Name : %s | Iterations : %s | Desc : %s", device, dataset_name, NUM_OF_ITERATIONS - 1,
             verbose)
    log.info("START")

    init_start_time = datetime.now()
    ground_truths = graph_init.create_graph()
    init_end_time = datetime.now()

    log.info("Time taken for graph initialization : %s \n", init_end_time - init_start_time)

    alg_start_time = datetime.now()
    for i in range(1, NUM_OF_ITERATIONS):
        metropolis.run()

        if i % 5000 == 0:
            log.info("Iterations Completed : %s", i)
    alg_end_time = datetime.now()
    log.info("Time taken to run the algorithm : %s \n", alg_end_time - alg_start_time)

    predictions = {}
    for atomic_name, graphlet_ids in config.atomic_name_graphlet_ids_dict.items():
        atomic_results = []
        for g_id in graphlet_ids:
            gr = config.graphlet_id_object_dict[g_id]
            gr_results = [{"g_id": g_id, "p_id": paper_obj.get_p_id()} for paper_obj in gr.get_papers()]
            atomic_results.extend(gr_results)

        atomic_results = sorted(atomic_results, key=lambda record: record['p_id'])
        pred_labels = [record['g_id'] for record in atomic_results]
        predictions[atomic_name] = pred_labels

    results = {}
    for gr_id, gr in config.graphlet_id_object_dict.items():
        gr_paper_ids = [paper_obj.get_p_id() for paper_obj in gr.get_papers()]
        results[gr_id] = gr_paper_ids

    with open(config.path_to_output + 'results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    validator.run(ground_truths, predictions)

    log.info("\nEND")


if __name__ == '__main__':
    main()

    # with open('./data/output/ground_truth.pickle', 'wb') as handle:
    #     pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('./data/output/predictions.pickle', 'wb') as handle:
    #     pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

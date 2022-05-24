import pickle
from datetime import datetime

import src.config as config
import src.graph_initializer as graph_init
import src.metropolis as metropolis
import src.validator as validator
from src.config import NUM_OF_ITERATIONS
import os,sys


def main():

    print("Start at : ", datetime.now())
    ground_truths = graph_init.create_graph()
    print("Initialization complete at : ", datetime.now())

    for i in range(1, NUM_OF_ITERATIONS):
        metropolis.run()

        if i % 5000 == 0:
            print("Current Iteration : ", i)
    print("Iterations completed at : ", datetime.now())

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

    print("End : ", datetime.now())


if __name__ == '__main__':
    main()

    # with open('./data/output/ground_truth.pickle', 'wb') as handle:
    #     pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('./data/output/predictions.pickle', 'wb') as handle:
    #     pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
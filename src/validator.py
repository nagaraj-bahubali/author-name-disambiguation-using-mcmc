import pickle
from math import comb

import bcubed
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix

from src import config

log = config.logger


def get_predictions():
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

    return predictions


def b3_metrics(ground_truths, predictions):
    val_results = pd.DataFrame(columns=['atomic_name', 'precision', 'recall', 'f1'])

    for atomic_name in ground_truths:
        ground_truth = ground_truths[atomic_name]
        prediction = predictions[atomic_name]

        ldict = {i: {ground_truth[i]} for i in range(len(ground_truth))}
        cdict = {i: {prediction[i]} for i in range(len(prediction))}

        b3_precision = bcubed.precision(cdict, ldict)
        b3_recall = bcubed.recall(cdict, ldict)
        b3_f1 = bcubed.fscore(b3_precision, b3_recall)

        val_results = pd.concat(
            [pd.DataFrame([[atomic_name, b3_precision, b3_recall, b3_f1]],
                          columns=val_results.columns),
             val_results], ignore_index=True)

    return val_results


def pairwise_metrics(ground_truths, predictions):
    val_results = pd.DataFrame(columns=['atomic_name', 'precision', 'recall', 'f1'])
    nC2 = lambda n: comb(n, 2)

    for atomic_name in ground_truths:
        ground_truth = ground_truths[atomic_name]
        prediction = predictions[atomic_name]
        contingency_mat = contingency_matrix(ground_truth, prediction)
        # To avoid single element in a cluster. Because pairwise metrics need at least a pair of data points for validation
        contingency_mat = np.where(contingency_mat == 1, 2, contingency_mat)

        # N = nC2(np.sum(contingency_mat))

        TP = np.sum(np.array(list(map(lambda row: list(map(nC2, row)), contingency_mat))))

        partition_summed = np.sum(contingency_mat, axis=1)
        FN = np.sum(np.array(list(map(nC2, partition_summed)))) - TP

        cluster_summed = np.sum(contingency_mat, axis=0)
        FP = np.sum(np.array(list(map(nC2, cluster_summed)))) - TP

        # TN = N - (TP + FP + FN)

        pairwise_precision = TP / (TP + FP)
        pairwise_recall = TP / (TP + FN)
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)

        val_results = pd.concat(
            [pd.DataFrame([[atomic_name, pairwise_precision, pairwise_recall, pairwise_f1]],
                          columns=val_results.columns),
             val_results], ignore_index=True)

    return val_results


def run(ground_truths, iteration, metric_type='pairwise', dump_results=False):
    predictions = get_predictions()
    val_results = pd.DataFrame()

    if metric_type == 'pairwise':
        val_results = pairwise_metrics(ground_truths, predictions)
    elif metric_type == 'b3':
        val_results = b3_metrics(ground_truths, predictions)

    log.info(" {: <10} |  {: <20} | {: <20} | {: <20} ".format(iteration, val_results['precision'].mean(),
                                                               val_results['recall'].mean(),
                                                               val_results['f1'].mean()))
    log.info("-" * 80)

    if dump_results:
        with open(config.path_to_output + metric_type + '_validation_results.pickle', 'wb') as handle:
            pickle.dump(val_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(config.path_to_output + 'predictions.pickle', 'wb') as handle:
            pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

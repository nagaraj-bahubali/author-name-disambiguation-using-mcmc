import pickle
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix

from src import config

log = config.logger


def run(ground_truths, predictions):
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

    log.info("VALIDATION RESULTS - Macro Averages")
    log.info("-" * 70)
    log.info(" {: <20} | {: <20} | {: <20} ".format("Precision", "Recall", "F1"))
    log.info("-" * 70)
    log.info(" {: <20} | {: <20} | {: <20} ".format(val_results['precision'].mean(), val_results['recall'].mean(),
                                                    val_results['f1'].mean()))
    log.info("-" * 70)

    with open(config.path_to_output + 'validation_results.pickle', 'wb') as handle:
        pickle.dump(val_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
This module provides access to the graph utility functions. These functions can be used to merge two graphlets or
split an existing graphlet into two.
"""

from typing import Tuple, Union

import src.config as config
import scipy.stats
import collections
import numpy as np
from nltk.cluster import KMeansClusterer
import nltk
from random import choices
from src.graph_elements import Graphlet


def merge_graphlets(g1_id: int, g2_id: int):
    """
    Takes two graphlet ids and merge corresponding graphlets with respect to their papers resulting in a bigger graphlet

    Parameters
    g1_id : Id of the first graphlet to be merged.
    g2_id : Id of the second graphlet to be merged.
    """

    g1 = config.graphlet_id_object_dict[g1_id]
    g2 = config.graphlet_id_object_dict[g2_id]
    g_id = g1_id
    g_atomic_name = g1.get_atomic_name()
    g_papers = g1.get_papers() + g2.get_papers()

    merged_gr = Graphlet(g_id, g_atomic_name, g_papers)

    # delete the merging nodes after creating new merged node
    del config.graphlet_id_object_dict[g1_id]
    del config.graphlet_id_object_dict[g2_id]

    # update the dictionary with newly created merged graphlet
    config.graphlet_id_object_dict[g_id] = merged_gr

    # remove inactive id from active id list and add it to inactive ids list
    config.active_graphlet_ids.remove(g2_id)
    config.inactive_graphlet_ids.append(g2_id)

    # remove g2_id from list of graphlet ids containing atomic_name = g_atomic_name
    config.atomic_name_graphlet_ids_dict[g_atomic_name].remove(g2_id)

    return



def split_graphlet(g_id: int, split_p_ids):
    """
    Takes a graphlet id and split the corresponding graphlet with respect to paper of the given id (p_id) resulting in
    two smaller graphlets.

    Parameters
    g_id : Id of the graphlet to be split.
    p_id : Id of the paper wrt which graphlet ha to be split.
    """

    g = config.graphlet_id_object_dict[g_id]
    g_atomic_name = g.get_atomic_name()
    g_papers = g.get_papers()

    g1_id = g_id
    g1_atomic_name = g_atomic_name
    g1_papers = [paper_obj for paper_obj in g_papers if (paper_obj.get_p_id() not in split_p_ids)]
    g1 = Graphlet(g1_id, g1_atomic_name, g1_papers)
    # g1_classes = [paper_obj.get_author_class() for paper_obj in g1.get_papers()]

    g2_id = config.inactive_graphlet_ids.pop()
    g2_atomic_name = g_atomic_name
    g2_papers = [paper_obj for paper_obj in g_papers if (paper_obj.get_p_id() in split_p_ids)]
    g2 = Graphlet(g2_id, g2_atomic_name, g2_papers)
    # g2_classes = [paper_obj.get_author_class() for paper_obj in g2.get_papers()]

    # after = [g1_classes,g2_classes]

    # delete the source graphlet
    del config.graphlet_id_object_dict[g_id]

    # update the dictionary with newly created split graphlets
    config.graphlet_id_object_dict[g1_id] = g1
    config.graphlet_id_object_dict[g2_id] = g2

    # add the id of newly created graphlet g2 to active ids list
    config.active_graphlet_ids.append(g2_id)

    # add g2_id to the list of graphlet ids containing atomic_name = g_atomic_name
    config.atomic_name_graphlet_ids_dict[g_atomic_name].append(g2_id)

    return

# def obtain_interim_splits(papers,num_of_splits =2):
#
#     titles_list = [paper_obj.get_title() for paper_obj in papers]
#     titles_emb = [config.bert_model.encode(title) for title in titles_list]
#     titles_emb = np.array(titles_emb)
#     repeats = 25
#
#     while True:
#         kclusterer = KMeansClusterer(num_of_splits, distance=nltk.cluster.util.cosine_distance,
#                                      repeats=repeats, avoid_empty_clusters=True)
#         assigned_clusters = kclusterer.cluster(titles_emb, assign_clusters=True)
#
#         if len(set(assigned_clusters)) == num_of_splits:
#             break
#         else:
#             repeats = repeats + 1
#             if repeats == 50:
#                 # if we are still getting empty clusters for repeats == 50, randomly create the clusters
#                 assigned_clusters = choices(np.arange(num_of_splits), k=len(papers))
#                 break
#
#     interim_splits = [[] for i in range(num_of_splits)]
#
#     for paper_obj, cluster_num in zip(papers, assigned_clusters):
#         interim_splits[cluster_num].append(paper_obj)
#
#     return interim_splits

def obtain_interim_splits(papers,num_of_splits):

    p_id_list = [paper_obj.get_p_id() for paper_obj in papers]
    titles_emb = [config.paper_embeddings[p_id] for p_id in p_id_list]
    titles_emb = np.array(titles_emb)
    repeats = 25

    kclusterer = KMeansClusterer(num_of_splits, distance=nltk.cluster.util.cosine_distance,
                                 repeats=repeats, avoid_empty_clusters=True)
    assigned_clusters = kclusterer.cluster(titles_emb, assign_clusters=True)

    # try to force split the papers into num_of_splits clusters
    # make sure at least one element is present representing each cluster
    if len(set(assigned_clusters)) != num_of_splits:
        residue = len(papers) - num_of_splits
        if residue > 0:
            assigned_clusters = np.concatenate(
                (np.arange(num_of_splits), np.array(choices(np.arange(num_of_splits), k=residue))), axis=0)
        else:
            assigned_clusters = np.arange(num_of_splits)

    interim_splits = [[] for i in range(num_of_splits)]

    for paper_obj, cluster_num in zip(papers, assigned_clusters):
        interim_splits[cluster_num].append(paper_obj)

    return interim_splits

def check_correctness(before, after, action):
    correctness = 0
    if action == "merge":
        g1_max = max(set(before[0]), key=before[0].count)
        g2_max = max(set(before[1]), key=before[1].count)

        if g1_max == g2_max:
            correctness = 1
        else:
            correctness = -1

    if action == "split":
        before_max = max(set(before[0]), key=before[0].count)
        split_max = max(set(after[1]), key=after[1].count)

        if before_max != split_max:
            correctness = 1
        else:
            correctness = -1

    return correctness

def calc_entropy(a_classes):
    class_counts = collections.Counter(a_classes)
    class_counts = list(class_counts.values())
    H = scipy.stats.entropy(class_counts,base=2)
    return H


def information_gain(before, after, action):
    delta = 0
    if action == "merge":
        H_g1 = calc_entropy(before[0])
        H_g2 = calc_entropy(before[1])
        H_after = calc_entropy(after[0])

        H_g1_weighted = (len(before[0]) / (len(before[0]) + len(before[1]))) * H_g1
        H_g2_weighted = (len(before[1]) / (len(before[0]) + len(before[1]))) * H_g2
        delta = H_g1_weighted + H_g2_weighted - H_after

    if action == "split":
        H_before = calc_entropy(before[0])
        H_g1 = calc_entropy(after[0])
        H_g2 = calc_entropy(after[1])

        H_g1_weighted = (len(after[0]) / (len(after[0]) + len(after[1]))) * H_g1
        H_g2_weighted = (len(after[1]) / (len(after[0]) + len(after[1]))) * H_g2
        delta = H_before - H_g1_weighted - H_g2_weighted

    return delta

def show_graph_status(state: int, summary: str):
    num_of_graphlets = len(config.graphlet_id_object_dict.items())
    # print("GRAPH STATE = ", state, " NUM OF GRAPHLETS = ", num_of_graphlets, " ACTION = ", summary)
    print("-" * 80)
    print("{: <12} | {: <20} | {: <}".format("graphlet_id", "author_class", "paper_ids"))
    print("-" * 80)
    for g_id, gr in config.graphlet_id_object_dict.items():
        paper_ids = [paper_obj.get_p_id() for paper_obj in gr.get_papers()]
        paper_ids_str = ",".join([str(_id) for _id in paper_ids])

        author_ids = [paper_obj.get_author_class() for paper_obj in gr.get_papers()]
        author_ids_str = ",".join([str(_id) for _id in author_ids])

        print("{: <12} | {: <20} | {: <}".format(g_id, author_ids_str, paper_ids_str))
        # print(g_id," : ",atomic_name," : ",paper_ids)
    print("*" * 80)

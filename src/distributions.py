import itertools
import random
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

import src.config as config


def sample_ethnicity():
    ethnicities, ethnicity_weights = list(config.ethnicity_count_dict.keys()), list(
        config.ethnicity_count_dict.values())
    ethnicity = random.choices(population=ethnicities, weights=ethnicity_weights, k=1)[0]
    return ethnicity


def sample_author_name(ethnicity):
    e_author_names, e_author_name_weights = list(config.ethnicity_author_name_count_dict[ethnicity].keys()), list(
        config.ethnicity_author_name_count_dict[ethnicity].values())
    author_name = random.choices(population=e_author_names, weights=e_author_name_weights, k=1)[0]
    return author_name


def sample_graphlet(author_name):
    g_id = random.choice(config.atomic_name_graphlet_ids_dict[author_name])
    return g_id


def sample_action(g_id):
    action = ''

    gr = config.graphlet_id_object_dict[g_id]
    graphlet_ids = config.atomic_name_graphlet_ids_dict[gr.atomic_name]
    paper_ids = [paper_obj.get_p_id() for paper_obj in gr.get_papers()]

    if len(paper_ids) == 1 and len(graphlet_ids) > 1:
        action = 'merge'
    elif len(paper_ids) > 1 and len(graphlet_ids) == 1:
        action = 'split'
    elif len(paper_ids) == 1 and len(graphlet_ids) == 1:
        action = 'skip'
    else:
        action = random.choices(population=['merge', 'split'], weights=[0.5, 0.5])[0]

    return action


def sample_merging_graphlet(g_id, author_name):
    # ids of graphlets other than graphlet with id = g_id
    non_g_ids = [_id for _id in config.atomic_name_graphlet_ids_dict[author_name] if _id != g_id]

    # count of papers in each graphlet of id = non_g_ids
    non_gr_paper_counts = []
    for non_g_id in non_g_ids:
        non_gr = config.graphlet_id_object_dict[non_g_id]
        non_gr_paper_count = len(non_gr.get_papers())
        non_gr_paper_counts.append(non_gr_paper_count)

    # inverse the weights wrt the count of papers
    non_gr_paper_weights = [1.0 / w for w in non_gr_paper_counts]
    sum_weights = sum(non_gr_paper_weights)
    normalized_weights = [w / sum_weights for w in non_gr_paper_weights]

    # choose a second graphlet(id) for merging
    merge_g_id = random.choices(population=non_g_ids, weights=normalized_weights, k=1)[0]
    return merge_g_id


def sample_splitting_paper(g_id):
    gr = config.graphlet_id_object_dict[g_id]

    paper_id_title_dict = {paper_obj.get_p_id(): paper_obj.get_title() for paper_obj in gr.get_papers()}
    paper_id_list = list(paper_id_title_dict)

    # form all possible pairs of papers
    paper_id_pairs = list(itertools.combinations(paper_id_list, 2))
    paper_pair_dist = {}

    # calculate distance between papers in all pairs
    for id_pair in paper_id_pairs:
        title_1_emb = config.bert_model.encode(paper_id_title_dict[id_pair[0]])
        title_2_emb = config.bert_model.encode(paper_id_title_dict[id_pair[1]])

        pair_dist = cosine_distances(title_1_emb.reshape(1, -1), title_2_emb.reshape(1, -1))
        paper_pair_dist[id_pair] = pair_dist[0][0]  # [0][0] since pair_dist is nd array

    # calculate the avg distance of a paper wrt rest of the papers in gr
    paper_dist = {}
    for _id in paper_id_list:
        dist_list = []
        for id_pair in paper_id_pairs:
            if _id in id_pair:
                dist_list.append(paper_pair_dist[id_pair])
        avg_dist = np.mean(dist_list)
        paper_dist[_id] = float(avg_dist)

    # sample that paper which is potentially distant from the rest
    s_paper_ids, s_paper_weights = list(paper_dist.keys()), list(paper_dist.values())
    split_p_id = random.choices(population=s_paper_ids, weights=s_paper_weights, k=1)[0]

    return split_p_id


def sample_external_graphlet(g_ids: List, author_name: str):
    # ids of graphlets other than graphlet ids present in g_ids
    ext_g_ids = []

    if len(g_ids) == 2:  # occurs when the action is merge
        if len(config.atomic_name_graphlet_ids_dict[author_name]) > 2:
            ext_g_ids = [_id for _id in config.atomic_name_graphlet_ids_dict[author_name] if _id not in g_ids]
        else:
            ext_g_ids = [_id for _id in config.active_graphlet_ids if _id not in g_ids]
    elif len(g_ids) == 1:  # occurs when the action is split
        if len(config.atomic_name_graphlet_ids_dict[author_name]) > 1:
            ext_g_ids = [_id for _id in config.atomic_name_graphlet_ids_dict[author_name] if _id not in g_ids]
        else:
            ext_g_ids = [_id for _id in config.active_graphlet_ids if _id not in g_ids]

    ext_g_id = random.choice(ext_g_ids)

    return ext_g_id


def sample_uniform_random(lower_bound, upper_bound):
    unif = random.uniform(lower_bound, upper_bound)
    return unif

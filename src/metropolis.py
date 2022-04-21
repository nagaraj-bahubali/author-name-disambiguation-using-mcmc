import numpy as np
from sklearn.metrics.pairwise import cosine_distances

import src.config as config
import src.distributions as dist
import src.graph_utilities as utils


def get_jaccard_sim(co_author_set_1, co_author_set_2):
    j_sim = float(len(co_author_set_1.intersection(co_author_set_2)) / len(co_author_set_1.union(co_author_set_2)))

    if j_sim == 0:
        j_sim = 0.1
    return j_sim


def get_paper_group_dist(paper_group_1, paper_group_2):
    group_1_emb = config.bert_model.encode(paper_group_1)
    group_2_emb = config.bert_model.encode(paper_group_2)
    distance = \
        cosine_distances(np.mean(group_1_emb, axis=0).reshape(1, -1), np.mean(group_2_emb, axis=0).reshape(1, -1))[0][0]

    if distance == 0:
        distance = 0.1
    return distance


def calc_merge_acceptance_ratio(g_id, mg_id, ext_g_id):
    # take care of zero values

    gr = config.graphlet_id_object_dict[g_id]
    mgr = config.graphlet_id_object_dict[mg_id]
    extgr = config.graphlet_id_object_dict[ext_g_id]

    gr_papers = [paper_obj.get_title() for paper_obj in gr.get_papers()]
    mgr_papers = [paper_obj.get_title() for paper_obj in mgr.get_papers()]
    extgr_papers = [paper_obj.get_title() for paper_obj in extgr.get_papers()]

    alpha_t1 = 1 / get_paper_group_dist(gr_papers, mgr_papers)
    alpha_t = 1 / get_paper_group_dist(extgr_papers, mgr_papers)

    gr_co_author_set = {co_author for paper_obj in gr.get_papers() for co_author in paper_obj.get_co_authors()}
    mgr_co_author_set = {co_author for paper_obj in mgr.get_papers() for co_author in paper_obj.get_co_authors()}
    extgr_co_author_set = {co_author for paper_obj in extgr.get_papers() for co_author in paper_obj.get_co_authors()}

    beta_t1 = get_jaccard_sim(gr_co_author_set, mgr_co_author_set)
    beta_t = get_jaccard_sim(extgr_co_author_set, mgr_co_author_set)

    m_acceptance_ratio = (alpha_t1 / alpha_t) * (beta_t1 / beta_t)

    return m_acceptance_ratio


def calc_split_acceptance_ratio(g_id, split_p_id, ext_g_id):
    gr = config.graphlet_id_object_dict[g_id]
    extgr = config.graphlet_id_object_dict[ext_g_id]

    gr_papers = [paper_obj.get_title() for paper_obj in gr.get_papers() if paper_obj.get_p_id() != split_p_id]
    split_paper = [paper_obj.get_title() for paper_obj in gr.get_papers() if paper_obj.get_p_id() == split_p_id]
    extgr_papers = [paper_obj.get_title() for paper_obj in extgr.get_papers()]

    alpha_t1 = 1 / get_paper_group_dist(extgr_papers, split_paper)
    alpha_t = 1 / get_paper_group_dist(gr_papers, split_paper)

    gr_co_author_set = {co_author for paper_obj in gr.get_papers() for co_author in paper_obj.get_co_authors() if
                        paper_obj.get_p_id() != split_p_id}
    split_paper_co_author_set = {co_author for paper_obj in gr.get_papers() for co_author in paper_obj.get_co_authors()
                                 if paper_obj.get_p_id() == split_p_id}
    extgr_co_author_set = {co_author for paper_obj in extgr.get_papers() for co_author in paper_obj.get_co_authors()}

    beta_t1 = get_jaccard_sim(extgr_co_author_set, split_paper_co_author_set)
    beta_t = get_jaccard_sim(gr_co_author_set, split_paper_co_author_set)

    s_acceptance_ratio = (alpha_t1 / alpha_t) * (beta_t1 / beta_t)

    return s_acceptance_ratio


def run():
    ethnicity = dist.sample_ethnicity()
    author_name = dist.sample_author_name(ethnicity)
    g_id = dist.sample_graphlet(author_name)
    action = dist.sample_action(g_id)
    unif = dist.sample_uniform_random(0, 1)

    if action == "merge":
        mg_id = dist.sample_merging_graphlet(g_id, author_name)
        ext_g_id = dist.sample_external_graphlet([g_id, mg_id], author_name)
        acceptance_ratio = calc_merge_acceptance_ratio(g_id, mg_id, ext_g_id)

        if acceptance_ratio > unif:
            utils.merge_graphlets(g_id, mg_id, ethnicity)
    else:
        split_p_id = dist.sample_splitting_paper(g_id)
        ext_g_id = dist.sample_external_graphlet([g_id], author_name)
        acceptance_ratio = calc_split_acceptance_ratio(g_id, split_p_id, ext_g_id)

        if acceptance_ratio > unif:
            utils.split_graphlet(g_id, split_p_id, ethnicity)

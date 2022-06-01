import random
from typing import Tuple, List, Union, Any

import numpy as np
from scipy import stats
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


def get_log_likelihood(paperid_years_group_1: List[Tuple[int, int]],
                       paperid_years_group_2: List[Tuple[int, int]]) -> Union[int, Any]:
    """
    Calculates how likely it is that the papers from group 2 belong to group 1, in terms of topic distribution over the years

    Parameters
    paperid_years_group_1 : List of tuples containing paper ids and their published years (of primary graphlet)
    paperid_years_group_2 : List of tuples containing paper ids and their published years (of secondary graphlet)

    Returns
    overall_LL : sum of log likelihoods of every paper in secondary graphlet
    """

    graphlet_topic_dist = {}
    overall_LL = 0

    # unique publication years.
    unique_years = set([tup[1] for tup in paperid_years_group_1])

    # if it is 1, KDE cannot be applied, because the bandwidth cannot be calculated, since the standard deviation of identical values is 0
    # https://stats.stackexchange.com/q/90916
    # hence generate additional sample with different year
    if len(unique_years) == 1:
        rand_paperid = random.choice([tup[0] for tup in paperid_years_group_1])
        year = unique_years.pop() + 1
        synthetic_sample = (rand_paperid, year)
        paperid_years_group_1.append(synthetic_sample)

    # generate samples(of years) according to the topic distribution
    for paper_id, pub_year in paperid_years_group_1:
        paper_topic_dist = config.topic_distributions[paper_id]
        paper_topic_dist_count = paper_topic_dist * 10 ** 5  # convert the dist probability into sample counts

        for topic_num in range(len(paper_topic_dist_count)):
            topic_count = int(paper_topic_dist_count[topic_num])
            if topic_num in graphlet_topic_dist:
                graphlet_topic_dist[topic_num] = np.concatenate(
                    (graphlet_topic_dist[topic_num], np.full(topic_count, pub_year)), axis=0)
            else:
                graphlet_topic_dist[topic_num] = np.full(topic_count, pub_year)

    # holds the KDE estimation for each topical distribution over the years
    graphlet_topic_kde = {topic_num: stats.gaussian_kde(t_dist) for topic_num, t_dist in graphlet_topic_dist.items()}

    # calculate the overall log likelihood of group 2 papers
    for paper_id, pub_year in paperid_years_group_2:
        paper_topic_dist = config.topic_distributions[paper_id]
        dominant_topic_num = np.argmax(paper_topic_dist)

        LL = graphlet_topic_kde[dominant_topic_num].logpdf(pub_year)
        overall_LL = overall_LL + LL

    return overall_LL


def calc_merge_acceptance_ratio(g_id, mg_id, ext_g_id):
    gr = config.graphlet_id_object_dict[g_id]
    mgr = config.graphlet_id_object_dict[mg_id]
    extgr = config.graphlet_id_object_dict[ext_g_id]

    # Calculate alpha terms
    gr_papers = [paper_obj.get_title() for paper_obj in gr.get_papers()]
    mgr_papers = [paper_obj.get_title() for paper_obj in mgr.get_papers()]
    extgr_papers = [paper_obj.get_title() for paper_obj in extgr.get_papers()]

    alpha_t1 = 1 / get_paper_group_dist(gr_papers, mgr_papers)
    alpha_t = 1 / get_paper_group_dist(extgr_papers, mgr_papers)

    # Calculate beta terms
    gr_co_author_set = {co_author for paper_obj in gr.get_papers() for co_author in paper_obj.get_co_authors()}
    mgr_co_author_set = {co_author for paper_obj in mgr.get_papers() for co_author in paper_obj.get_co_authors()}
    extgr_co_author_set = {co_author for paper_obj in extgr.get_papers() for co_author in paper_obj.get_co_authors()}

    beta_t1 = get_jaccard_sim(gr_co_author_set, mgr_co_author_set)
    beta_t = get_jaccard_sim(extgr_co_author_set, mgr_co_author_set)

    # Calculate gamma terms
    gr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr.get_papers()]
    mgr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in mgr.get_papers()]
    extgr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in extgr.get_papers()]

    gamma_t1 = get_log_likelihood(gr_paperid_years, mgr_paperid_years)
    gamma_t = get_log_likelihood(extgr_paperid_years, mgr_paperid_years)

    m_acceptance_ratio = np.log(alpha_t1) - np.log(alpha_t) + np.log(beta_t1) - np.log(beta_t) + gamma_t1 - gamma_t

    return m_acceptance_ratio


def calc_split_acceptance_ratio(g_id, split_p_id, ext_g_id):
    gr = config.graphlet_id_object_dict[g_id]
    extgr = config.graphlet_id_object_dict[ext_g_id]

    # Calculate alpha terms
    gr_papers = [paper_obj.get_title() for paper_obj in gr.get_papers() if paper_obj.get_p_id() != split_p_id]
    split_paper = [paper_obj.get_title() for paper_obj in gr.get_papers() if paper_obj.get_p_id() == split_p_id]
    extgr_papers = [paper_obj.get_title() for paper_obj in extgr.get_papers()]

    alpha_t1 = 1 / get_paper_group_dist(extgr_papers, split_paper)
    alpha_t = 1 / get_paper_group_dist(gr_papers, split_paper)

    # Calculate beta terms
    gr_co_author_set = {co_author for paper_obj in gr.get_papers() for co_author in paper_obj.get_co_authors() if
                        paper_obj.get_p_id() != split_p_id}
    split_paper_co_author_set = {co_author for paper_obj in gr.get_papers() for co_author in paper_obj.get_co_authors()
                                 if paper_obj.get_p_id() == split_p_id}
    extgr_co_author_set = {co_author for paper_obj in extgr.get_papers() for co_author in paper_obj.get_co_authors()}

    beta_t1 = get_jaccard_sim(extgr_co_author_set, split_paper_co_author_set)
    beta_t = get_jaccard_sim(gr_co_author_set, split_paper_co_author_set)

    # Calculate gamma terms
    gr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr.get_papers() if
                        paper_obj.get_p_id() != split_p_id]
    split_paperid_year = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr.get_papers() if
                          paper_obj.get_p_id() == split_p_id]
    extgr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in extgr.get_papers()]

    gamma_t1 = get_log_likelihood(extgr_paperid_years, split_paperid_year)
    gamma_t = get_log_likelihood(gr_paperid_years, split_paperid_year)

    s_acceptance_ratio = np.log(alpha_t1) - np.log(alpha_t) + np.log(beta_t1) - np.log(beta_t) + gamma_t1 - gamma_t

    return s_acceptance_ratio


def run():
    ethnicity = dist.sample_ethnicity()
    author_name = dist.sample_author_name(ethnicity)
    g_id = dist.sample_graphlet(author_name)
    action = dist.sample_action(g_id)
    unif = dist.sample_uniform_random(0, 1)
    log_unif = np.log(unif)

    if action == "merge":
        mg_id = dist.sample_merging_graphlet(g_id, author_name)

        ext_g_id = dist.sample_external_graphlet([g_id, mg_id], author_name)
        acceptance_ratio = calc_merge_acceptance_ratio(g_id, mg_id, ext_g_id)

        if acceptance_ratio > log_unif:
            utils.merge_graphlets(g_id, mg_id, ethnicity)

    elif action == "split":
        split_p_id = dist.sample_splitting_paper(g_id)
        ext_g_id = dist.sample_external_graphlet([g_id], author_name)
        acceptance_ratio = calc_split_acceptance_ratio(g_id, split_p_id, ext_g_id)

        if acceptance_ratio > log_unif:
            utils.split_graphlet(g_id, split_p_id, ethnicity)

    else:
        # do nothing when action is 'skip'
        pass

import random
from typing import Tuple, List, Union, Any

import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_distances

import src.config as config
import src.distributions as dist
import src.graph_utilities as utils
import src.validator as validator
from src.callbacks import EarlyStopping, ModelCheckpoint
from src.config import logger as log
from src.enums import ValidationMetric


class Metropolis:

    def __init__(self, epochs: int, validation_metric: ValidationMetric, logging_interval: int,
                 early_stop: EarlyStopping,
                 model_checkpoint: ModelCheckpoint, a_name
                 ):
        self.epochs = epochs
        self.validation_metric = validation_metric
        self.logging_interval = logging_interval
        self.early_stop = early_stop
        self.model_checkpoint = model_checkpoint
        self.atomic_name = a_name

    @staticmethod
    def get_jaccard_sim(set_1, set_2):
        """
        Calculates the jaccard similarity between two sets of strings
        """
        smoother = 0.0001
        j_sim = 0

        if len(set_2) > 0:
            j_sim = float(len(set_1.intersection(set_2)) / len(set_1.union(set_2)))

        return j_sim + smoother

    @staticmethod
    def get_paper_group_dist(paper_group_1, paper_group_2):
        """
        Calculates the cosine distance between the embeddings of two sets of strings
        """
        group_1_emb = np.array([config.paper_embeddings[pid] for pid in paper_group_1])
        group_2_emb = np.array([config.paper_embeddings[pid] for pid in paper_group_2])

        distance = \
            cosine_distances(np.mean(group_1_emb, axis=0).reshape(1, -1), np.mean(group_2_emb, axis=0).reshape(1, -1))[
                0][0]

        # smoothing
        if distance == 0:
            distance = 0.1
        return distance

    @staticmethod
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
        graphlet_topic_kde = {topic_num: stats.gaussian_kde(t_dist) for topic_num, t_dist in
                              graphlet_topic_dist.items()}

        # calculate the overall log likelihood of group 2 papers
        for paper_id, pub_year in paperid_years_group_2:
            paper_topic_dist = config.topic_distributions[paper_id]
            dominant_topic_num = np.argmax(paper_topic_dist)

            LL = graphlet_topic_kde[dominant_topic_num].logpdf(pub_year)
            overall_LL = overall_LL + LL

        return overall_LL

    @staticmethod
    def get_coauthors_and_affiliations(co_authors_info):
        """
        Splits a set of co_authors with their affiliations into separate sets
        e.g., {John@Stanford,Steve@Harvard,Michael} into {John,Steve,Michael} and {Stanford,Harvard}

        Parameters
        co_authors_info : A set of authors suffixed with their affiliations

        Returns
        co_author_names : set of author names
        affiliation_names : set of affiliations
        """

        co_author_names = set()
        affiliation_names = set()
        for item in co_authors_info:
            if len(item.split("@")) == 2:
                co_author_name = item.split("@")[0]
                co_author_names.add(co_author_name)
                affiliation_name = item.split("@")[1]
                affiliation_names.add(affiliation_name)
            else:
                co_author_names.add(item)

        return co_author_names, affiliation_names

    @staticmethod
    def calc_merge_acceptance_ratio(gr_paper_objects, mgr_paper_objects, extgr_paper_objects):

        # Calculate alpha terms
        gr_papers = [paper_obj.get_p_id() for paper_obj in gr_paper_objects]
        mgr_papers = [paper_obj.get_p_id() for paper_obj in mgr_paper_objects]
        extgr_1_papers = [paper_obj.get_p_id() for paper_obj in extgr_paper_objects[0]]
        extgr_2_papers = [paper_obj.get_p_id() for paper_obj in extgr_paper_objects[1]]

        alpha_t1 = 1 / Metropolis.get_paper_group_dist(gr_papers, mgr_papers)
        alpha_t = 1 / Metropolis.get_paper_group_dist(extgr_1_papers, extgr_2_papers)

        # Calculate beta terms
        gr_co_aff_set = {co_author for paper_obj in gr_paper_objects for co_author in paper_obj.get_co_authors()}
        mgr_co_aff_set = {co_author for paper_obj in mgr_paper_objects for co_author in paper_obj.get_co_authors()}
        extgr_1_co_aff_set = {co_author for paper_obj in extgr_paper_objects[0] for co_author in
                              paper_obj.get_co_authors()}
        extgr_2_co_aff_set = {co_author for paper_obj in extgr_paper_objects[1] for co_author in
                              paper_obj.get_co_authors()}

        gr_co_author_set, gr_affiliation_set = Metropolis.get_coauthors_and_affiliations(gr_co_aff_set)
        mgr_co_author_set, mgr_affiliation_set = Metropolis.get_coauthors_and_affiliations(mgr_co_aff_set)
        extgr_1_co_author_set, extgr_1_affiliation_set = Metropolis.get_coauthors_and_affiliations(extgr_1_co_aff_set)
        extgr_2_co_author_set, extgr_2_affiliation_set = Metropolis.get_coauthors_and_affiliations(extgr_2_co_aff_set)

        beta_t1 = Metropolis.get_jaccard_sim(gr_co_author_set, mgr_co_author_set)
        beta_t = Metropolis.get_jaccard_sim(extgr_1_co_author_set, extgr_2_co_author_set)

        # Calculate gamma terms
        gr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr_paper_objects]
        mgr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in mgr_paper_objects]
        extgr_1_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in extgr_paper_objects[0]]
        extgr_2_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in extgr_paper_objects[1]]

        gamma_t1 = Metropolis.get_log_likelihood(gr_paperid_years, mgr_paperid_years)
        gamma_t = Metropolis.get_log_likelihood(extgr_1_paperid_years, extgr_2_paperid_years)

        # Calculate kappa terms
        kappa_t1 = Metropolis.get_jaccard_sim(gr_affiliation_set, mgr_affiliation_set)
        kappa_t = Metropolis.get_jaccard_sim(extgr_1_affiliation_set, extgr_2_affiliation_set)

        m_acceptance_ratio = np.log(alpha_t1) - np.log(alpha_t) + np.log(beta_t1) - np.log(
            beta_t) + gamma_t1 - gamma_t + np.log(kappa_t1) - np.log(kappa_t)
        return m_acceptance_ratio

    @staticmethod
    def calc_split_acceptance_ratio(gr_a_paper_objects, gr_b_paper_objects, gr_ab_paper_objects,
                                    gr_split_paper_objects):

        # Calculate alpha terms
        gr_a_papers = [paper_obj.get_p_id() for paper_obj in gr_a_paper_objects]
        gr_b_papers = [paper_obj.get_p_id() for paper_obj in gr_b_paper_objects]
        gr_ab_papers = [paper_obj.get_p_id() for paper_obj in gr_ab_paper_objects]
        gr_split_papers = [paper_obj.get_p_id() for paper_obj in gr_split_paper_objects]

        alpha_t1 = 1 / Metropolis.get_paper_group_dist(gr_a_papers, gr_b_papers)
        alpha_t = 1 / Metropolis.get_paper_group_dist(gr_ab_papers, gr_split_papers)

        # Calculate beta terms
        gr_a_co_aff_set = {co_author for paper_obj in gr_a_paper_objects for co_author in paper_obj.get_co_authors()}
        gr_b_co_aff_set = {co_author for paper_obj in gr_b_paper_objects for co_author in paper_obj.get_co_authors()}
        gr_ab_co_aff_set = {co_author for paper_obj in gr_ab_paper_objects for co_author in paper_obj.get_co_authors()}
        gr_split_co_aff_set = {co_author for paper_obj in gr_split_paper_objects for co_author in
                               paper_obj.get_co_authors()}

        gr_a_co_author_set, gr_a_affiliation_set = Metropolis.get_coauthors_and_affiliations(gr_a_co_aff_set)
        gr_b_co_author_set, gr_b_affiliation_set = Metropolis.get_coauthors_and_affiliations(gr_b_co_aff_set)
        gr_ab_co_author_set, gr_ab_affiliation_set = Metropolis.get_coauthors_and_affiliations(gr_ab_co_aff_set)
        gr_split_co_author_set, gr_split_affiliation_set = Metropolis.get_coauthors_and_affiliations(
            gr_split_co_aff_set)

        beta_t1 = Metropolis.get_jaccard_sim(gr_a_co_author_set, gr_b_co_author_set)
        beta_t = Metropolis.get_jaccard_sim(gr_ab_co_author_set, gr_split_co_author_set)

        # Calculate gamma terms
        gr_a_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr_a_paper_objects]
        gr_b_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr_b_paper_objects]
        gr_ab_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr_ab_paper_objects]
        gr_split_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr_split_paper_objects]

        gamma_t1 = Metropolis.get_log_likelihood(gr_a_paperid_years, gr_b_paperid_years)
        gamma_t = Metropolis.get_log_likelihood(gr_ab_paperid_years, gr_split_paperid_years)

        # Calculate kappa terms
        kappa_t1 = Metropolis.get_jaccard_sim(gr_a_affiliation_set, gr_b_affiliation_set)
        kappa_t = Metropolis.get_jaccard_sim(gr_ab_affiliation_set, gr_split_affiliation_set)

        s_acceptance_ratio = np.log(alpha_t1) - np.log(alpha_t) + np.log(beta_t1) - np.log(
            beta_t) + gamma_t1 - gamma_t + np.log(kappa_t1) - np.log(kappa_t)
        return s_acceptance_ratio

    @staticmethod
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

    @staticmethod
    def run(i, author_name):
        # ethnicity = dist.sample_ethnicity()
        # author_name = dist.sample_author_name(ethnicity)
        g_id = dist.sample_graphlet(author_name)
        action = dist.sample_action(g_id)
        # unif = dist.sample_uniform_random(0, 1)
        unif = 0.7
        log_unif = np.log(unif)

        if action == "merge":
            gr = config.graphlet_id_object_dict[g_id]
            gr_paper_objects = [paper_obj for paper_obj in gr.get_papers()]

            mg_id = dist.sample_merging_graphlet(g_id, author_name)
            mgr = config.graphlet_id_object_dict[mg_id]
            mgr_paper_objects = [paper_obj for paper_obj in mgr.get_papers()]
            extgr_paper_objects = dist.sample_interim_splits([g_id, mg_id], author_name)

            acceptance_ratio = Metropolis.calc_merge_acceptance_ratio(gr_paper_objects, mgr_paper_objects,
                                                                      extgr_paper_objects)

            if acceptance_ratio > log_unif:
                utils.merge_graphlets(g_id, mg_id)

        elif action == "split":
            split_p_id = dist.sample_splitting_paper(g_id)
            extgr_paper_objects = dist.sample_interim_splits([g_id], author_name)

            gr_a_paper_objects = []
            gr_b_paper_objects = []
            gr_split_paper_objects = []

            for p_group in extgr_paper_objects:
                p_group_ids = [paper_obj.get_p_id() for paper_obj in p_group]
                if split_p_id in p_group_ids:
                    gr_split_paper_objects = p_group
                elif len(gr_a_paper_objects) == 0:
                    gr_a_paper_objects = p_group
                else:
                    gr_b_paper_objects = p_group

            gr_ab_paper_objects = gr_a_paper_objects + gr_b_paper_objects

            acceptance_ratio = Metropolis.calc_split_acceptance_ratio(gr_a_paper_objects, gr_b_paper_objects,
                                                                      gr_ab_paper_objects, gr_split_paper_objects)

            if acceptance_ratio > log_unif:
                split_p_ids = [paper_obj.get_p_id() for paper_obj in gr_split_paper_objects]
                utils.split_graphlet(g_id, split_p_ids)

        else:
            # do nothing when action is 'skip'
            pass

    def start(self, ground_truths):

        for i in range(1, self.epochs + 1):
            Metropolis.run(i, self.atomic_name)

            if i % self.logging_interval == 0:
                predictions = Metropolis.get_predictions()
                val_results = validator.validate(ground_truths, predictions, metric_type=self.validation_metric)

                curr_monitor_val = val_results[self.model_checkpoint.monitor.value].mean()
                desc = self.model_checkpoint.check(curr_monitor_val, val_results)
                curr_monitor_val = val_results[self.early_stop.monitor.value].mean()
                stop_algo = self.early_stop.check(curr_monitor_val)

                if stop_algo:
                    log.info("iterations : {} \n".format(i))
                    return

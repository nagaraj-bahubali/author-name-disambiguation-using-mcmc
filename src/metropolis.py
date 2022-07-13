import random
from datetime import datetime
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
                 model_checkpoint: ModelCheckpoint,
                 ):
        self.epochs = epochs
        self.validation_metric = validation_metric
        self.logging_interval = logging_interval
        self.early_stop = early_stop
        self.model_checkpoint = model_checkpoint

    @staticmethod
    def get_jaccard_sim(co_author_set_1, co_author_set_2):
        # in denominator only co_author_set_2 is used for normalisation
        j_sim = float(len(co_author_set_1.intersection(co_author_set_2)) / len(co_author_set_2))

        if j_sim == 0:
            j_sim = 0.1
        return j_sim

    @staticmethod
    def get_paper_group_dist(paper_group_1, paper_group_2):
        group_1_emb = config.bert_model.encode(paper_group_1)
        group_2_emb = config.bert_model.encode(paper_group_2)
        distance = \
            cosine_distances(np.mean(group_1_emb, axis=0).reshape(1, -1), np.mean(group_2_emb, axis=0).reshape(1, -1))[
                0][0]

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
    def calc_merge_acceptance_ratio(g_id, mg_id, ext_g_id):
        gr = config.graphlet_id_object_dict[g_id]
        mgr = config.graphlet_id_object_dict[mg_id]
        extgr = config.graphlet_id_object_dict[ext_g_id]

        # Calculate alpha terms
        gr_papers = [paper_obj.get_title() for paper_obj in gr.get_papers()]
        mgr_papers = [paper_obj.get_title() for paper_obj in mgr.get_papers()]
        extgr_papers = [paper_obj.get_title() for paper_obj in extgr.get_papers()]

        alpha_t1 = 1 / Metropolis.get_paper_group_dist(gr_papers, mgr_papers)
        alpha_t = 1 / Metropolis.get_paper_group_dist(extgr_papers, mgr_papers)

        # Calculate beta terms
        gr_co_author_set = {co_author for paper_obj in gr.get_papers() for co_author in paper_obj.get_co_authors()}
        mgr_co_author_set = {co_author for paper_obj in mgr.get_papers() for co_author in paper_obj.get_co_authors()}
        extgr_co_author_set = {co_author for paper_obj in extgr.get_papers() for co_author in
                               paper_obj.get_co_authors()}

        gr_affiliation_set = {}
        mgr_affiliation_set = {}
        extgr_affiliation_set = {}

        if config.affiliations_available:
            gr_co_author_set, gr_affiliation_set = Metropolis.get_coauthors_and_affiliations(gr_co_author_set)
            mgr_co_author_set, mgr_affiliation_set = Metropolis.get_coauthors_and_affiliations(mgr_co_author_set)
            extgr_co_author_set, extgr_affiliation_set = Metropolis.get_coauthors_and_affiliations(extgr_co_author_set)

        beta_t1 = Metropolis.get_jaccard_sim(gr_co_author_set, mgr_co_author_set)
        beta_t = Metropolis.get_jaccard_sim(extgr_co_author_set, mgr_co_author_set)

        # Calculate gamma terms
        gr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr.get_papers()]
        mgr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in mgr.get_papers()]
        extgr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in extgr.get_papers()]

        gamma_t1 = Metropolis.get_log_likelihood(gr_paperid_years, mgr_paperid_years)
        gamma_t = Metropolis.get_log_likelihood(extgr_paperid_years, mgr_paperid_years)

        # Calculate kappa terms
        kappa_t1 = 1
        kappa_t = 1

        if config.affiliations_available:
            kappa_t1 = Metropolis.get_jaccard_sim(gr_affiliation_set, mgr_affiliation_set)
            kappa_t = Metropolis.get_jaccard_sim(extgr_affiliation_set, mgr_affiliation_set)

        m_acceptance_ratio = np.log(alpha_t1) - np.log(alpha_t) + np.log(beta_t1) - np.log(
            beta_t) + gamma_t1 - gamma_t + np.log(kappa_t1) - np.log(kappa_t)

        return m_acceptance_ratio

    @staticmethod
    def calc_split_acceptance_ratio(g_id, split_p_id, ext_g_id):
        gr = config.graphlet_id_object_dict[g_id]
        extgr = config.graphlet_id_object_dict[ext_g_id]

        # Calculate alpha terms
        gr_papers = [paper_obj.get_title() for paper_obj in gr.get_papers() if paper_obj.get_p_id() != split_p_id]
        split_paper = [paper_obj.get_title() for paper_obj in gr.get_papers() if paper_obj.get_p_id() == split_p_id]
        extgr_papers = [paper_obj.get_title() for paper_obj in extgr.get_papers()]

        alpha_t1 = 1 / Metropolis.get_paper_group_dist(extgr_papers, split_paper)
        alpha_t = 1 / Metropolis.get_paper_group_dist(gr_papers, split_paper)

        # Calculate beta terms
        gr_co_author_set = {co_author for paper_obj in gr.get_papers() for co_author in paper_obj.get_co_authors() if
                            paper_obj.get_p_id() != split_p_id}
        split_paper_co_author_set = {co_author for paper_obj in gr.get_papers() for co_author in
                                     paper_obj.get_co_authors()
                                     if paper_obj.get_p_id() == split_p_id}
        extgr_co_author_set = {co_author for paper_obj in extgr.get_papers() for co_author in
                               paper_obj.get_co_authors()}

        gr_affiliation_set = {}
        split_paper_affiliation_set = {}
        extgr_affiliation_set = {}

        if config.affiliations_available:
            gr_co_author_set, gr_affiliation_set = Metropolis.get_coauthors_and_affiliations(gr_co_author_set)
            split_paper_co_author_set, split_paper_affiliation_set = Metropolis.get_coauthors_and_affiliations(
                split_paper_co_author_set)
            extgr_co_author_set, extgr_affiliation_set = Metropolis.get_coauthors_and_affiliations(extgr_co_author_set)

        beta_t1 = Metropolis.get_jaccard_sim(extgr_co_author_set, split_paper_co_author_set)
        beta_t = Metropolis.get_jaccard_sim(gr_co_author_set, split_paper_co_author_set)

        # Calculate gamma terms
        gr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr.get_papers() if
                            paper_obj.get_p_id() != split_p_id]
        split_paperid_year = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in gr.get_papers() if
                              paper_obj.get_p_id() == split_p_id]
        extgr_paperid_years = [(paper_obj.get_p_id(), paper_obj.get_year()) for paper_obj in extgr.get_papers()]

        gamma_t1 = Metropolis.get_log_likelihood(extgr_paperid_years, split_paperid_year)
        gamma_t = Metropolis.get_log_likelihood(gr_paperid_years, split_paperid_year)

        # Calculate kappa terms
        kappa_t1 = 1
        kappa_t = 1

        if config.affiliations_available:
            kappa_t1 = Metropolis.get_jaccard_sim(extgr_affiliation_set, split_paper_affiliation_set)
            kappa_t = Metropolis.get_jaccard_sim(gr_affiliation_set, split_paper_affiliation_set)

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
    def run(i):
        ethnicity = dist.sample_ethnicity()
        author_name = dist.sample_author_name(ethnicity)
        g_id = dist.sample_graphlet(author_name)
        action = dist.sample_action(g_id)
        unif = dist.sample_uniform_random(0, 1)
        # log_unif = np.log(unif)
        log_unif = 4 * np.log(unif)

        if action == "merge":
            mg_id = dist.sample_merging_graphlet(g_id, author_name)

            ext_g_id = dist.sample_external_graphlet([g_id, mg_id], author_name)
            acceptance_ratio = Metropolis.calc_merge_acceptance_ratio(g_id, mg_id, ext_g_id)

            config.tracker[i] = {"action": "merge", "result": "No", "log_unif_4": log_unif, "a_ratio": acceptance_ratio}
            if acceptance_ratio > log_unif:
                utils.merge_graphlets(g_id, mg_id, ethnicity)
                config.tracker[i]["result"] = "Yes"

        elif action == "split":
            split_p_id = dist.sample_splitting_paper(g_id)
            ext_g_id = dist.sample_external_graphlet([g_id], author_name)
            acceptance_ratio = Metropolis.calc_split_acceptance_ratio(g_id, split_p_id, ext_g_id)

            config.tracker[i] = {"action": "split", "result": "No", "log_unif_4": log_unif, "a_ratio": acceptance_ratio}
            if acceptance_ratio > log_unif:
                utils.split_graphlet(g_id, split_p_id, ethnicity)
                config.tracker[i]["result"] = "Yes"
        else:
            # do nothing when action is 'skip'
            config.tracker[i] = {"action": "skip", "result": "skip", "log_unif_4": log_unif, "a_ratio": 0}

    def start(self, ground_truths):

        log.info("-" * 80)
        log.info(" {: <10} |  {: <20} | {: <20} | {: <20} ".format("Iteration", "Precision", "Recall", "F1"))
        log.info("-" * 80)

        alg_start_time = datetime.now()
        for i in range(1, self.epochs + 1):
            Metropolis.run(i)
            if i % self.logging_interval == 0:
                predictions = Metropolis.get_predictions()
                val_results = validator.validate(ground_truths, predictions, metric_type=self.validation_metric)

                curr_monitor_val = val_results[self.model_checkpoint.monitor.value].mean()
                desc = self.model_checkpoint.check(curr_monitor_val, predictions, val_results)
                log.info(" {: <10} |  {: <20} | {: <20} | {: <20}  {: <20}".format(i, val_results['precision'].mean(),
                                                                                   val_results['recall'].mean(),
                                                                                   val_results['f1'].mean(), desc))
                log.info("-" * 80)
                curr_monitor_val = val_results[self.early_stop.monitor.value].mean()
                stop_algo = self.early_stop.check(curr_monitor_val)

                if stop_algo:
                    break

        alg_end_time = datetime.now()
        log.info("Time taken to run the algorithm : %s", alg_end_time - alg_start_time)

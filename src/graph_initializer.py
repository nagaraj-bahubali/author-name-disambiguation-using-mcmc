"""
This module provides access to the graph initialization functions. These functions can be used read atomic files and
transform their content into graphlets.
"""

import glob
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

import src.config as config
from src.graph_elements import Paper, Graphlet

log = config.logger


def read_atomic_file(atomic_name: str, file_path: str) -> DataFrame:
    """
    Reads a certain atomic file and transforms the content in the form of dataframe

    Parameters
    atomic_name : Name of the atomic file.
    file_path : Path where atomic file resides.

    Returns
    df : File content in the form of dataframe.
    """

    df = pd.read_csv(file_path + atomic_name + '.txt',
                     sep="_|\||<>|<>|<>|<>",
                     names=['authorId', 'referenceId', 'authorName', 'coauthors', 'title', 'journal', 'year'],
                     header=None,
                     keep_default_na=True,
                     na_values=['None', 'none'],
                     on_bad_lines='skip',
                     engine="python")
    df.authorId = pd.to_numeric(df.authorId, errors='coerce')
    df.referenceId = pd.to_numeric(df.referenceId, errors='coerce')
    df.year = pd.to_numeric(df.year, errors='coerce')
    df = df.dropna(subset=['authorId', 'referenceId', 'authorName', 'title', 'year'])
    df = df.astype({'authorId': np.int32, 'referenceId': np.int32, 'year': np.int32})

    return df


def create_paper(record: Dict) -> Paper:
    """
    Creates a paper object with paper attributes such as paper id, authors of paper, journal and year of publication.

    Parameters
    record : A dictionary with attributes of paper.
             e.g., {'authorId': 1, 'referenceId': 1, 'authorName': 'a', 'coauthors': 'a;b', 'title': 'p', 'journal': 'j', 'year': 2005}.

    Returns
    paper_obj : Paper object created using record details.
    """

    p_id = record['referenceId']
    co_authors = record['coauthors'].split(';')
    title = record['title']
    journal = record['journal']
    year = record['year']

    paper_attributes = (p_id, title, co_authors, journal, year)
    paper_obj = Paper(*paper_attributes)

    return paper_obj


def create_graphlet(atomic_name: str, paper: Paper) -> Tuple[int, Graphlet]:
    """
    Creates a graphlet object for a given atomic name and paper object.

    Parameters
    atomic_name : Atomic name
    paper : Paper object

    Returns
    g_id : Id of the created graphlet object.
    gr : Created graphlet object.
    """

    g_id = config.cur_graphlet_id

    # passing list of paper objects instead of single paper, as graphlets can contain multiple papers in the future
    gr = Graphlet(g_id, atomic_name, [paper])

    # update necessary global variables
    config.cur_graphlet_id = config.cur_graphlet_id + 1
    config.active_graphlet_ids.append(g_id)

    if atomic_name in config.atomic_name_graphlet_ids_dict:
        config.atomic_name_graphlet_ids_dict[atomic_name] = config.atomic_name_graphlet_ids_dict[atomic_name] + [g_id]
    else:
        config.atomic_name_graphlet_ids_dict[atomic_name] = [g_id]

    return g_id, gr


def update_ethnicity_count(ethnicity: str) -> None:
    """
    Updates the count of names for a given ethnicity, creates new record otherwise.

    Parameters
    ethnicity : ethnicity of the author name
    """

    if ethnicity in config.ethnicity_count_dict:
        config.ethnicity_count_dict[ethnicity] = config.ethnicity_count_dict[ethnicity] + 1
    else:
        config.ethnicity_count_dict[ethnicity] = 1


def update_ethnicity_author_name_count(ethnicity: str, atomic_name: str, auth_count: int) -> None:
    """
    Updates the ethnicity_author_name_count_dict for a given ethnicity, author name and author count.

    Parameters
    ethnicity : ethnicity of the author name
    atomic_name : atomic name
    auth_count : count of authors present in the atomic file (=count of reference strings)
    """
    if ethnicity in config.ethnicity_author_name_count_dict:
        if atomic_name in config.ethnicity_author_name_count_dict[ethnicity]:
            prev_count = config.ethnicity_author_name_count_dict[ethnicity][atomic_name]
            config.ethnicity_author_name_count_dict[ethnicity][atomic_name] = prev_count + auth_count
        else:
            config.ethnicity_author_name_count_dict[ethnicity][atomic_name] = auth_count
    else:
        config.ethnicity_author_name_count_dict[ethnicity] = {}
        config.ethnicity_author_name_count_dict[ethnicity][atomic_name] = auth_count


def create_graph() -> Dict:
    """
    Creates a graphlet for every reference in every atomic file. The collection of graphlets is
    referred as Graph.

    Parameters
    file_path : Path to the dataset containing list of atomic files.
    """

    atomic_names_list = [os.path.basename(file_path)[:-4] for file_path in
                         glob.glob(config.path_to_dataset + "and_data/" + "*.txt")]

    with open(config.path_to_dataset + "meta_data/" + 'ethnicities.pickle', 'rb') as handle:
        ethnicities_dict = pickle.load(handle)

    # A dict with atomic name as key and list of author ids(serves as cluster label) as value. Used for cluster validation.
    # e.g., {"h min":[1,1,1,2,2],"d nelson":[243,243,244,244,244]}
    ground_truth = {}

    for atomic_name in atomic_names_list:
        df = read_atomic_file(atomic_name, config.path_to_dataset + "and_data/")

        if len(df) == 0:
            log.info("%s.txt is skipped due to zero records before/after pre-processing", atomic_name)
            continue

        df_records = df.to_dict('records')

        # sort by referenceIDs and then retrieve the corresponding cluster labels ( author id)
        df_records = sorted(df_records, key=lambda record: record['referenceId'])
        labels = [record['authorId'] for record in df_records]
        ground_truth[atomic_name] = labels
        for df_record in df_records:
            paper_obj = create_paper(df_record)
            g_id, gr = create_graphlet(atomic_name, paper_obj)

            # update the dictionary to keep track of created graphlets by their ids
            config.graphlet_id_object_dict[g_id] = gr

        # update the global variables to keep track of authors' ethnicities
        ethnicity = ethnicities_dict[atomic_name]
        update_ethnicity_count(ethnicity)
        auth_count = len(df_records)
        update_ethnicity_author_name_count(ethnicity, atomic_name, auth_count)

    return ground_truth

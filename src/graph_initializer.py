"""
This module provides access to the graph initialization functions. These functions can be used read atomic files and
transform their content into graphlets.
"""

import glob
import os
from typing import Dict, Tuple

import pandas as pd
from pandas import DataFrame

import src.config as config
from src.graph_elements import Paper, Graphlet


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
                     keep_default_na=False,
                     on_bad_lines='skip',
                     engine="python")

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


def create_graph(file_path: str) -> None:
    """
    Takes a file path and creates a graphlet for every reference in every atomic file. The collection of graphlets is
    referred as Graph.

    Parameters
    file_path : Path to the dataset containing list of atomic files.
    """

    atomic_names_list = [os.path.basename(file_path)[:-4] for file_path in glob.glob(file_path + "*.txt")]

    for atomic_name in atomic_names_list:
        df = read_atomic_file(atomic_name, file_path)
        df_records = df.to_dict('records')
        for df_record in df_records:
            paper_obj = create_paper(df_record)
            g_id, gr = create_graphlet(atomic_name, paper_obj)

            # update the dictionary to keep track of created graphlets by their ids
            config.graphlet_id_object_dict[g_id] = gr

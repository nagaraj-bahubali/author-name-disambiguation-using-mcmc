"""
This module provides access to the graph utility functions. These functions can be used to merge two graphlets or
split an existing graphlet into two.
"""

from typing import Tuple

import src.config as config
from src.graph_elements import Graphlet


def merge_graphlets(g1_id: int, g2_id: int, ethnicity: str) -> Graphlet:
    """
    Takes two graphlet ids and merge corresponding graphlets with respect to their papers resulting in a bigger graphlet

    Parameters
    g1_id : Id of the first graphlet to be merged.
    g2_id : Id of the second graphlet to be merged.

    Returns
    merged_gr : Merge of graphlets belonging to g1_id and g2_id.
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

    # decrease the count of authors present for the given ethnicity and author name
    prev_count = config.ethnicity_author_name_count_dict[ethnicity][g_atomic_name]
    config.ethnicity_author_name_count_dict[ethnicity][g_atomic_name] = prev_count - 1

    return merged_gr


def split_graphlet(g_id: int, p_id: int, ethnicity: str) -> Tuple[Graphlet, Graphlet]:
    """
    Takes a graphlet id and split the corresponding graphlet with respect to paper of the given id (p_id) resulting in
    two smaller graphlets.

    Parameters
    g_id : Id of the graphlet to be split.
    p_id : Id of the paper wrt which graphlet ha to be split.

    Returns
    g1 : Split graphlet 1.
    g2 : Split graphlet 2.
    """

    g = config.graphlet_id_object_dict[g_id]
    g_atomic_name = g.get_atomic_name()
    g_papers = g.get_papers()

    g1_id = g_id
    g1_atomic_name = g_atomic_name
    g1_papers = [paper_obj for paper_obj in g_papers if (paper_obj.get_p_id() != p_id)]
    g1 = Graphlet(g1_id, g1_atomic_name, g1_papers)

    g2_id = config.inactive_graphlet_ids.pop()
    g2_atomic_name = g_atomic_name
    g2_papers = [paper_obj for paper_obj in g_papers if (paper_obj.get_p_id() == p_id)]
    g2 = Graphlet(g2_id, g2_atomic_name, g2_papers)

    # delete the source graphlet
    del config.graphlet_id_object_dict[g_id]

    # update the dictionary with newly created split graphlets
    config.graphlet_id_object_dict[g1_id] = g1
    config.graphlet_id_object_dict[g2_id] = g2

    # add the id of newly created graphlet g2 to active ids list
    config.active_graphlet_ids.append(g2_id)

    # add g2_id to the list of graphlet ids containing atomic_name = g_atomic_name
    config.atomic_name_graphlet_ids_dict[g_atomic_name].append(g2_id)

    # increase the count of authors present for the given ethnicity and author name
    prev_count = config.ethnicity_author_name_count_dict[ethnicity][g_atomic_name]
    config.ethnicity_author_name_count_dict[ethnicity][g_atomic_name] = prev_count + 1

    return g1, g2


def show_graph_status(state: int, summary: str):
    num_of_graphlets = len(config.graphlet_id_object_dict.items())
    print("GRAPH STATE = ", state, " NUM OF GRAPHLETS = ", num_of_graphlets, " ACTION = ", summary)
    print("-" * 80)
    print("{: <12} | {: <20} | {: <}".format("graphlet_id", "atomic_name", "paper_ids"))
    print("-" * 80)
    for g_id, gr in config.graphlet_id_object_dict.items():
        paper_ids = [paper_obj.get_p_id() for paper_obj in gr.get_papers()]
        paper_ids_str = ",".join([str(_id) for _id in paper_ids])
        atomic_name = gr.get_atomic_name()
        print("{: <12} | {: <20} | {: <}".format(g_id, atomic_name, paper_ids_str))
        # print(g_id," : ",atomic_name," : ",paper_ids)
    print("*" * 80)

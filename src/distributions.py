import random

import src.config as config


def sample_author_name():
    author_name = random.choice(list(config.atomic_name_graphlet_ids_dict))
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
    else:
        action = random.choices(population=['merge', 'split'], weights=[0.5, 0.5])[0]

    return action


def sample_merging_graphlet(g1_id, author_name):
    non_g1_ids = [_id for _id in config.atomic_name_graphlet_ids_dict[author_name] if _id != g1_id]
    g2_id = random.choice(non_g1_ids)
    return g2_id


def sample_splitting_paper(g_id):
    gr = config.graphlet_id_object_dict[g_id]
    paper_ids = [paper_obj.get_p_id() for paper_obj in gr.get_papers()]
    p_id = random.choice(paper_ids)
    return p_id

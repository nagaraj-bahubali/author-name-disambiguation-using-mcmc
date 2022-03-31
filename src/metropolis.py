import src.distributions as dist
import src.graph_utilities as utils


def run():
    author_name = dist.sample_author_name()
    g1_id = dist.sample_graphlet(author_name)
    action = dist.sample_action(g1_id)
    summary = ''

    if action == "merge":
        g2_id = dist.sample_merging_graphlet(g1_id, author_name)
        utils.merge_graphlets(g1_id, g2_id)
        summary = "MERGE ON " + str(g1_id) + " & " + str(g2_id)

    else:
        p_id = dist.sample_splitting_paper(g1_id)
        utils.split_graphlet(g1_id, p_id)
        summary = "SPLIT ON " + str(g1_id)

    return summary

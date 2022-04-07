import src.distributions as dist
import src.graph_utilities as utils


def run():
    ethnicity = dist.sample_ethnicity()
    author_name = dist.sample_author_name(ethnicity)
    g_id = dist.sample_graphlet(author_name)
    action = dist.sample_action(g_id)
    summary = ''

    if action == "merge":
        mg_id = dist.sample_merging_graphlet(g_id, author_name)
        utils.merge_graphlets(g_id, mg_id, ethnicity)
        summary = "MERGE ON " + str(g_id) + " & " + str(mg_id)

    else:
        split_p_id = dist.sample_splitting_paper(g_id)
        utils.split_graphlet(g_id, split_p_id, ethnicity)
        summary = "SPLIT ON " + str(g_id)

    return summary

import src.config as config
import src.graph_initializer as graph_init
import src.graph_utilities as graph_utils
import src.metropolis as metropolis
from src.config import NUM_OF_ITERATIONS


def main():
    file_path = config.path_to_dataset
    graph_init.create_graph(file_path)
    summary = "INITIALIZATION"
    graph_utils.show_graph_status(0, summary)

    for i in range(1, NUM_OF_ITERATIONS):
        summary = metropolis.run()
        graph_utils.show_graph_status(i, summary)

    # with open('./data/output/graph.pickle', 'wb') as handle:
    #     pickle.dump(config.graphlet_id_object_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('./data/output/graph.pickle', 'rb') as handle:
    #     graph = pickle.load(handle)


if __name__ == '__main__':
    main()

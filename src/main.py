import src.config as config
import src.graph_initializer as graph_init
import src.graph_utilities as graph_utils
import src.metropolis as metropolis
from src.config import NUM_OF_ITERATIONS


def main():
    dataset_file_path = config.path_to_dataset
    ethnicity_file_path = config.path_to_ethnicities
    graph_init.create_graph(dataset_file_path, ethnicity_file_path)
    summary = "INITIALIZATION"

    for i in range(1, NUM_OF_ITERATIONS):
        summary = metropolis.run()

    graph_utils.show_graph_status(100, summary)


if __name__ == '__main__':
    main()

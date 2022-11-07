import logging
import pickle
import sys
import pandas as pd
from src.enums import PerformanceMetric,ValidationMetric

path_to_dataset = './data/input/demo/'
path_to_output = './data/output/'
affiliations_available = True
logging_interval = 1  # logs will be printed after every 'logging_interval' iterations
validation_metric = ValidationMetric.PAIRWISE  # takes either 'pairwise' or 'b3'
model_checkpoint_monitor = PerformanceMetric.F1
early_stop_monitor = PerformanceMetric.F1
early_stop_significance_level = 0 #0.0001
cur_graphlet_id = 0  # do not modify this

# A dictionary of atomic names as keys and list of graphlet ids as values. This variable keeps track of which atomic
# name is present in which graphlets at certain state of the graph
atomic_name_graphlet_ids_dict = {}

# List of all graphlet ids currently in use. If two graphlets are merged, the id of second graphlet will no longer be
# in use and hence will be popped out. The popped id will be pushed to 'inactive_graphlet_ids'.
active_graphlet_ids = []

# List of graphlet ids currently not in use due to merging. If a graphlet is split resulting in an additional one,
# id from this variable is popped and will be pushed back into 'active_graphlet_ids'.
inactive_graphlet_ids = []

# A dictionary of graphlet ids as keys and graphlet object references as vales. Helps in retrieving a graphlet by its id
graphlet_id_object_dict = {}

# A dictionary of ethnicity as keys and count of corresponding atomic names/files e.g., {'ENGLISH':2,'CHINESE':1}
with open(path_to_dataset + 'meta_data/ethnicity_counts.pickle', 'rb') as handle:
    ethnicity_count_dict = pickle.load(handle)

# A dictionary of paper id as keys and corresponding topic distributions e.g., {1: [0.1,0.1,0.8], 2: [0.3,0.6,0.1]}
with open(path_to_dataset + 'meta_data/topic_distributions.pickle', 'rb') as handle:
    topic_distributions = pickle.load(handle)

# A dictionary of ethnicity as keys and dictionary of {author name : count} as value at certain state of the graph.
# i.e, for every ethnicity, it gives name-wise count of authors present at the moment in the graph.
# e.g., {'Arabic': {'a ahmad': 5, 'z imran': 5},'Chinese': {'b li':7, 'w wen': 13, 'r shen': 10}}
ethnicity_author_name_count_dict = {}

# A dictionary of paper id as keys and corresponding title embeddings
with open(path_to_dataset + 'meta_data/paper_embeddings.pickle', 'rb') as handle:
    paper_embeddings = pickle.load(handle)

# to store final results
final_results = pd.DataFrame(columns=['atomic_name', 'precision', 'recall', 'f1'])

# Logger Configurations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
fh = logging.FileHandler(path_to_output + "summary.log", "w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)





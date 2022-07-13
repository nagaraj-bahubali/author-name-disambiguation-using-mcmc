import logging
import pickle
import sys
from src.enums import PerformanceMetric,ValidationMetric
from sentence_transformers import SentenceTransformer

path_to_dataset = './data/input/Aminer-534K/'
path_to_output = './data/output/Aminer-534K/run_1/'
affiliations_available = True
num_of_iterations = 100000
logging_interval = 1000  # logs will be printed after every 'logging_interval' iterations
dataset_name = 'Aminer-534K'  # give a custom name for logging
validation_metric = ValidationMetric.PAIRWISE  # takes either 'pairwise' or 'b3'
model_checkpoint_monitor = PerformanceMetric.F1
early_stop_monitor = PerformanceMetric.F1
early_stop_patience = 5
early_stop_significance_level = 0.0001
verbose = "All four"  # brief description about the run

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

# bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Logger Configurations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s: %(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
fh = logging.FileHandler(path_to_output + "summary.log", "w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# activity
tracker = {}

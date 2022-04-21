import pickle
from sentence_transformers import SentenceTransformer

path_to_dataset = './data/input/unified-and-dataset_1_filtered/and_data/'
path_to_ethnicities = './data/input/unified-and-dataset_1_filtered/ethnicity_data/'
cur_graphlet_id = 0
NUM_OF_ITERATIONS = 100001

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
with open(path_to_ethnicities + 'ethnicity_counts.pickle', 'rb') as handle:
    ethnicity_count_dict = pickle.load(handle)

# A dictionary of ethnicity as keys and dictionary of {author name : count} as value at certain state of the graph.
# i.e, for every ethnicity, it gives name-wise count of authors present at the moment in the graph.
# e.g., {'Arabic': {'a ahmad': 5, 'z imran': 5},'Chinese': {'b li':7, 'w wen': 13, 'r shen': 10}}
ethnicity_author_name_count_dict = {}

# bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

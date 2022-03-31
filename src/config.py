path_to_dataset = './data/sample_dataset/'
cur_graphlet_id = 0
NUM_OF_ITERATIONS = 10

# A dictionary of atomic names as keys and list of graphlet ids as values. This variable keeps track of which atomic
# name is present in which graphlets at certain state of the graph
atomic_name_graphlet_ids_dict = {}

# List of all graphlet ids currently in use. If two graphlets are merged, the id of second graphlet will no longer be
# in use and hence will be popped out. The popped id will be pushed to 'inactive_graphlet_ids'.
active_graphlet_ids = []

# List of graphlet ids currently not in use due to merging. If a graphlet is split resulting in an additional one,
# id from this variable is popped and will be pushed into 'active_graphlet_ids'.
inactive_graphlet_ids = []

# A dictionary of graphlet ids as keys and graphlet object references as vales. Helps in retrieving a graphlet by its id
graphlet_id_object_dict = {}

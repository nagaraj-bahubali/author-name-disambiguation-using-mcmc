import collections
import glob
import json
import os
import pickle

import pandas as pd
import requests
from pandas import DataFrame


def read_atomic_file(atomic_name: str, file_path: str) -> DataFrame:
    """
    Reads a certain atomic file and transforms the content in the form of dataframe

    Parameters
    atomic_name : Name of the atomic file.
    file_path : Path where atomic file resides.

    Returns
    df : File content in the form of dataframe.
    """

    df = pd.read_csv(file_path + atomic_name + '.txt',
                     sep="_|\||<>|<>|<>|<>",
                     names=['authorId', 'referenceId', 'authorName', 'coauthors', 'title', 'journal', 'year'],
                     header=None,
                     keep_default_na=False,
                     on_bad_lines='skip',
                     engine="python")

    return df


def fetch_ethnicity(f_name, l_name):
    response = requests.get(
        "http://abel.lis.illinois.edu/cgi-bin/ethnea/search.py?Fname=" + f_name + "&Lname=" + l_name + "&format=json")
    response = response.text
    response = response.replace("'", "\"")
    json_response = json.loads(response)
    ethnicity = json_response['Ethnea']
    return ethnicity


def create_ethnicity_file(dataset_file_path: str, ethnicity_file_path: str) -> None:
    atomic_names_list = [os.path.basename(file_path)[:-4] for file_path in glob.glob(dataset_file_path + "*.txt")]
    atomic_name_ethnicity_dict = {}

    for atomic_name in atomic_names_list:
        df = read_atomic_file(atomic_name, dataset_file_path)
        unique_author_names = df['authorName'].unique()
        name_tuple_list = [
            (name.rsplit(' ', 1)[0], name.rsplit(' ', 1)[1]) if (len(name.rsplit(' ', 1)) == 2) else (name, name) for
            name in unique_author_names]

        auth_name_eth_count = {}
        for name_tuple in name_tuple_list:
            f_name, l_name = name_tuple[0], name_tuple[1]
            auth_name_eth = fetch_ethnicity(f_name, l_name)

            if auth_name_eth in auth_name_eth_count:
                auth_name_eth_count[auth_name_eth] = auth_name_eth_count[auth_name_eth] + 1
            else:
                auth_name_eth_count[auth_name_eth] = 1
        atomic_name_eth = max(auth_name_eth_count, key=auth_name_eth_count.get)
        atomic_name_ethnicity_dict[atomic_name] = atomic_name_eth

    with open(ethnicity_file_path + 'ethnicities.pickle', 'wb') as handle:
        pickle.dump(atomic_name_ethnicity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ethnicity_counts = collections.Counter(atomic_name_ethnicity_dict.values())
    with open(ethnicity_file_path + 'ethnicity_counts.pickle', 'wb') as handle:
        pickle.dump(ethnicity_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("File dump success!")
    print("Path to file : ", ethnicity_file_path)


def main():
    dataset_file_path = './data/input/sample_dataset/and_data/'
    ethnicity_dump_path = './data/input/sample_dataset/ethnicity_data/'
    create_ethnicity_file(dataset_file_path, ethnicity_dump_path)


if __name__ == '__main__':
    main()

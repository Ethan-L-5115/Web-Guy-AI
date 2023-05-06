import numpy as np


def write_clusters_to_file(cluster_list, file_path):

    with open(file_path, 'w') as f:
        for i, cluster in enumerate(cluster_list):
            centroid_str = ','.join(map(str, cluster[0]))
            # Combine filenames as a single string
            filenames_str = ','.join(cluster[1:])
            row = [i, "|||", centroid_str, "|||", filenames_str]
            f.write(','.join(map(str, row)) + '\n')

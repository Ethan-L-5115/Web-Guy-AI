import pandas as pd
import numpy as np


def write_clusters_to_file(cluster_list, file_path):

    rows = []
    for i, cluster in enumerate(cluster_list):
        centroid_str = ','.join(map(str, cluster[0]))
        centroid_str = centroid_str[:-1]
        row = [i, centroid_str]
        row.extend(cluster[1:])
        rows.append(row)
        print("row:", row)  # Add this line

    np.set_printoptions(threshold=np.inf)

    # Create column names
    columns = ['cluster_number', 'centroid', 'filepaths']

    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Save DataFrame to CSV file
    df.to_csv(file_path, index=False)


cluster_list = [
    [[1, 2, 3, 4], "filepath1", "filepath2"],
    [[5, 6, 7, 8], "filepath3", "filepath4", "filepath5"]
]
file_path = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/clusters.csv"
write_clusters_to_file(cluster_list, file_path)

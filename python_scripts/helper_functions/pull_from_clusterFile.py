#text file input that we parse out and p
#given cluster index, pull all pictures given the ID, filename of the txt file
import csv

def get_cluster_images(cluster_index, cluster_file):
    """
    Function to get all the image paths in a particular cluster given the cluster index.

    Parameters:
        - cluster_index (int): Index of the cluster to retrieve images for.
        - cluster_file (str): Path to the CSV file containing the cluster labels and image paths.

    Returns:
        - cluster_images (list): List of image paths belonging to the specified cluster.
    """
    cluster_images = []
    with open(cluster_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0])
            if label == cluster_index:
                # Find the second occurrence of '|||' in its own column
                second_delimiter_index = row[2].index('|||', row[2].index('|||') + 1)
                # Add the strings after the second delimiter to the list of cluster images
                cluster_images += row[2][second_delimiter_index + 3:].split('|||')

    return cluster_images
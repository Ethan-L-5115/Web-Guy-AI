#text file input that we parse out and p
#given cluster index, pull all pictures given the ID, filename of the txt file


import numpy as np
import cv2
import os

def get_cluster_images(cluster_index, image_path, cluster_file):
    """
    Function to get all the images in a particular cluster given the cluster index.

    Parameters:
        - cluster_index (int): Index of the cluster to retrieve images for.
        - image_path (str): Path to the directory containing the images.
        - cluster_file (str): Path to the text file containing the cluster labels.

    Returns:
        - cluster_images (list): List of images belonging to the specified cluster.
    """
    cluster_images = []
    with open(cluster_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            label = int(line.strip())
            if label == cluster_index:
                image_file = os.path.join(image_path, str(i) + ".jpg") #assuming image files are named as integers with .jpg extension
                image = cv2.imread(image_file)
                cluster_images.append(image)

    return cluster_images

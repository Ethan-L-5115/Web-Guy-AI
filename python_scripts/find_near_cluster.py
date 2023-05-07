import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import os
import re


def load_img_features(img_feat_filepath):
    data = pd.read_csv(img_feat_filepath)
    features = data['features'].values[0]
    img_features = np.fromstring(features, sep=',')
    return img_features


def load_clusters(clusts_filepath):
    centroids = []
    image_paths = []

    with open(clusts_filepath, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            centroid_start_idx = row.index('|||') + 1
            centroid_end_idx = row.index('|||', centroid_start_idx)
            centroid_str = ','.join(row[centroid_start_idx:centroid_end_idx])
            centroid = np.fromstring(centroid_str, sep=',')
            centroids.append(centroid)

            img_paths = row[centroid_end_idx + 1:]
            image_paths.append(img_paths)

    return centroids, image_paths


def find_closest_cluster(img_feat_filepath, clusts_filepath):
    img_features = load_img_features(img_feat_filepath)
    centroids, image_paths = load_clusters(clusts_filepath)

    min_distance = float('inf')
    closest_cluster_idx = -1

    for i, centroid in enumerate(centroids):
        distance = euclidean(img_features, centroid)
        if distance < min_distance:
            min_distance = distance
            closest_cluster_idx = i

    return closest_cluster_idx, image_paths[closest_cluster_idx]


def add_base_path_to_image_paths(image_filenames, raw_filepath):
    full_image_paths = []
    for img_path in image_filenames:
        # Remove the extra part and replace it with .jpg
        new_img_path = re.sub(r'_face\d+', '', img_path)
        full_path = os.path.join(raw_filepath, new_img_path).replace("\\", "/")
        full_image_paths.append(full_path)
    return full_image_paths
# k-means clustering

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from helper_functions.VGG_Face import create_vgg_model
from cluster_to_file import write_clusters_to_file
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import silhouette_score, davies_bouldin_score


def calculate_metrics(features, labels, model):
    # Calculate inertia
    inertia = model.inertia_

    # Calculate silhouette score
    silhouette = silhouette_score(features, labels)

    # Calculate Davies-Bouldin Index
    db_index = davies_bouldin_score(features, labels)

    return inertia, silhouette, db_index


def load_features(file_path):
    data = pd.read_csv(file_path)
    image_paths = data['image_path'].values
    features = np.array([ast.literal_eval(feature_str)
                        for feature_str in data['features']])
    return image_paths, features


def kmeans_clustering(features, n_clusters):
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    # Get cluster centroids
    cluster_centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    return cluster_labels, cluster_centroids, kmeans  # return the kmeans model


def cluster_images(image_paths, cluster_labels):
    unique_labels = np.unique(cluster_labels)
    clustered_images = []

    for label in unique_labels:
        indices = np.where(cluster_labels == label)[0]
        images_in_cluster = [image_paths[index] for index in indices]
        clustered_images.append(images_in_cluster)

    return clustered_images


def calculate_average_squared_distances(features, max_clusters, step_size):
    average_squared_distances = []
    for n_clusters in range(1, int(max_clusters / step_size) + 1):
        kmeans = KMeans(n_clusters=n_clusters * step_size,
                        random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        distances = pairwise_distances(kmeans.cluster_centers_, features)
        closest_distances = np.min(distances, axis=0)
        average_distance = np.mean(closest_distances)
        average_squared_distances.append(average_distance)
        del kmeans  # Explicitly delete the kmeans object to free memory
    return average_squared_distances


def plot_average_squared_distances_chart(average_squared_distances, step_size):
    plt.plot(range(1, len(average_squared_distances) * step_size + 1, step_size),
             average_squared_distances)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Squared Distance')
    plt.title('Number of Clusters vs. Average Squared Distance')
    plt.yscale('log')
    plt.show()


def plot_dist_by_clust(features_file_path, max_clusters, step_size):
    # Load the saved extracted features and image paths
    image_paths, features = load_features(features_file_path)
    average_squared_distances = calculate_average_squared_distances(
        features, max_clusters, step_size)
    plot_average_squared_distances_chart(average_squared_distances, step_size)


def k_means_clustering(features_file_path, clustered_folder_path, n_clusters):
    # Load the saved extracted features and image paths
    image_paths, features = load_features(features_file_path)

    # Perform k-means clustering
    cluster_labels, cluster_centroids, kmeans = kmeans_clustering(
        features, n_clusters)

    # Calculate metrics
    inertia, silhouette, db_index = calculate_metrics(
        features, cluster_labels, kmeans)

    print(
        f"Inertia: {inertia}, Silhouette Score: {silhouette}, Davies-Bouldin Index: {db_index}")

    # Cluster image paths
    clustered_images = cluster_images(image_paths, cluster_labels)

    # Include centroids in the clustered_images list
    for i, centroid in enumerate(cluster_centroids):
        clustered_images[i].insert(0, centroid.tolist())
        # Outputs [[[cluster 1 centroid], cluster 1 image filepaths]
    #                                                            [[#,#,#,#,...],"filepath","filepath","filepath","filepath"...]]

    # Print features to file
    write_clusters_to_file(clustered_images, clustered_folder_path)

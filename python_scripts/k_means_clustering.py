# k-means clustering

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from helper_functions.VGG_Face import create_vgg_model
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import ast


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
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features)

    return cluster_labels


def cluster_images(image_paths, cluster_labels):
    unique_labels = np.unique(cluster_labels)
    clustered_images = []
    
    for label in unique_labels:
        indices = np.where(cluster_labels == label)[0]
        images_in_cluster = [image_paths[index] for index in indices]
        clustered_images.append(images_in_cluster)
    
    return clustered_images


def k_means_clustering():
    # Load the saved extracted features and image paths
    features_file_path = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/features.csv'
    image_paths, features = load_features(features_file_path)

    # Perform k-means clustering
    n_clusters = 2
    cluster_labels = kmeans_clustering(features, n_clusters)

    # Cluster image paths
    clustered_images = cluster_images(image_paths, cluster_labels)

    print(clustered_images)

    return clustered_images
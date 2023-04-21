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


def display_samples(images_folder, image_paths, cluster_labels, n_samples=3):
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        print(f"Cluster {label}:")
        indices = np.where(cluster_labels == label)[0]
        sample_indices = random.sample(
            list(indices), min(n_samples, len(indices)))

        for i, index in enumerate(sample_indices):
            img_path = os.path.join(images_folder, image_paths[index])
            img = Image.open(img_path)
            plt.subplot(1, n_samples, i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()


def main():
    # Load the saved extracted features and image paths
    features_file_path = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/features.csv'
    image_paths, features = load_features(features_file_path)

    # Perform k-means clustering
    n_clusters = 2
    cluster_labels = kmeans_clustering(features, n_clusters)

    # Display samples from each cluster
    images_folder = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/test_me_cropped'
    display_samples(images_folder, image_paths, cluster_labels)


if __name__ == "__main__":
    main()

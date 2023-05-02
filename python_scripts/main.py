import os
from feature_extraction import extract_features
from crop_faces import crop_faces
from k_means_clustering import k_means_clustering

# Path to folder with images
folder_path = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me'

# Extract features
extract_features(folder_path)

# Crop faces
input_folder = folder_path
output_folder = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me_cropped'
crop_faces(input_folder, output_folder)

# Perform k-means clustering
features_file = 'features.csv'
n_clusters = 3
k_means_clustering(features_file, n_clusters)

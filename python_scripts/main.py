import os
from feature_extraction import extract_features
from crop_faces import get_faces
from k_means_clustering import k_means_clustering, plot_dist_by_clust

# main.py will go through the entire pre-UI process of clustering and booting the
# clusters and centroids to a file located in clustered_folder_path defined below

# Folder Paths
raw_folder_path = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/test_me_raw'
face_folder_path = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/face'
features_folder_path = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me'
clustered_folder_path = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/clusters.csv'


# Crop faces (Comment out when not in use)
print("Starting get_faces()")
get_faces(raw_folder_path, face_folder_path)

# Extract features (Comment out when not in use)
print("Starting extract_features()")
extract_features(face_folder_path, features_folder_path)

# # Find optimal k
# # plot_dist_by_clust(features_file_path, max_clusters, step_size)
# print("Starting plto_dist_by_clust()")
# plot_dist_by_clust(features_folder_path+'/features.csv', 10000, 200)

# # Perform k-means clustering (Comment out when not in use)
# n_clusters = 1500
# k_means_clustering(features_folder_path+'/features.csv',
#                    clustered_folder_path, n_clusters)

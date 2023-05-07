from feature_extraction import extract_features
from crop_faces import get_faces
from find_near_cluster import find_closest_cluster, add_base_path_to_image_paths
from print_cluster import copy_images_to_destination


def find_nearest_cluster(folder_path, clusts_filepath, raw_filepath):

    get_faces(folder_path, folder_path+"/face")

    extract_features(folder_path+"/face", folder_path)

    idx, filepaths = find_closest_cluster(
        folder_path+"/features.csv", clusts_filepath+"/clusters.csv")

    complete_filepaths = add_base_path_to_image_paths(filepaths, raw_filepath)

    copy_images_to_destination(complete_filepaths, folder_path+"/cluster")


test_img = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/test"
clusts = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me"
raw = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/test_me_raw"
find_nearest_cluster(test_img, clusts, raw)

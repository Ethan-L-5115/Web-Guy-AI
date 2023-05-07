from python_scripts.feature_extraction import extract_features
from python_scripts.crop_faces import get_faces
from python_scripts.find_near_cluster import find_closest_cluster
from python_scripts.print_cluster import copy_images_to_destination


def find_nearest_cluster(folder_path, clusts_filepath):

    get_faces(folder_path, folder_path)

    extract_features(folder_path, folder_path)

    idx, filepaths = find_closest_cluster(folder_path, clusts_filepath)

    copy_images_to_destination(filepaths, folder_path)

test_img = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/test/test.jpg"
clusts = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me"
find_nearest_cluster()
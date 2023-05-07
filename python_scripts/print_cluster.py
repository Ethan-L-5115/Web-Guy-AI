import shutil
import os


def copy_images_to_destination(image_paths, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Copy each image to the destination directory
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        destination_path = os.path.join(destination_dir, filename)
        shutil.copyfile(img_path, destination_path)

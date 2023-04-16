import cv2
import numpy as np
from facenet_pytorch import MTCNN
import os
import glob


def process_images(input_folder, mtcnn):

    # Store image paths and their corresponding face bounding box coordinates
    image_boxes = []

    # Iterate through all images in the input folder
    for image_path in glob.glob(os.path.join(input_folder, "*.jpg")):
        # Read the image
        image = cv2.imread(image_path)

        # Detect faces and their bounding boxes using MTCNN
        boxes, _ = mtcnn.detect(image)

        # Add the image path and detected boxes to the image_boxes list
        image_boxes.append((image_path, boxes))

    return image_boxes

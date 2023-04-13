import cv2
import numpy as np
from facenet_pytorch import MTCNN
import os
import glob

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Detect the faces in one image and output one image
def detect_faces(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Detect faces and their bounding boxes using MTCNN
    boxes, _ = mtcnn.detect(image)

    # Draw bounding boxes around detected faces
    if boxes is not None:
        for box in boxes:
            x, y, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 2)

    # Save the output image
    output_path = "output_" + image_path
    cv2.imwrite(output_path, image)

    # Count the number of faces detected
    face_count = len(boxes) if boxes is not None else 0
    print(f"Number of faces detected: {face_count}")

# Detect the faces in multiple images in a file and output to another file
def process_images(input_folder, output_folder):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all images in the input folder
    for image_path in glob.glob(os.path.join(input_folder, "*.jpg")):
        # Read the image
        image = cv2.imread(image_path)

        # Detect faces and their bounding boxes using MTCNN
        boxes, _ = mtcnn.detect(image)

        # Draw bounding boxes around detected faces
        if boxes is not None:
            for box in boxes:
                x, y, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 2)

        # Save the output image
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, image)

        # Count the number of faces detected
        face_count = len(boxes) if boxes is not None else 0
        print(f"{image_path}: {face_count} faces detected")

# Example usage

# Test detecting faces in two images
#detect_faces("testportrait.jpg")
#detect_faces("testcrazymob.jpg")

# Test detecting faces in the whole file
input_folder = "C:/Users/torre/Coding/dataset"
output_folder = "C:/Users/torre/Coding/recognized_faces"
process_images(input_folder, output_folder)

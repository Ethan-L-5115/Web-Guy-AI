import cv2
import os
import glob

def process_images(input_folder, mtcnn, min_resolution, min_confidence):

    # Store image paths and their corresponding face bounding box coordinates
    image_boxes = []

    # Iterate through all images in the input folder
    for image_path in glob.glob(os.path.join(input_folder, "*.jpg")):
        # Read the image
        image = cv2.imread(image_path)

        # Check if the image was correctly read
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue

        # Detect faces and their bounding boxes using MTCNN
        boxes, probs = mtcnn.detect(image)

        # Check if boxes and probs are not None
        if boxes is not None and probs is not None:
            # Filter boxes and probabilities based on min_resolution and min_confidence
            filtered_boxes = []
            for box, prob in zip(boxes, probs):
                width = box[2] - box[0]
                height = box[3] - box[1]
                resolution = width * height

                if resolution >= min_resolution and prob >= min_confidence:
                    filtered_boxes.append(box)
        else:
            filtered_boxes = []

        # Add the image path and filtered boxes to the image_boxes list
        image_boxes.append((image_path, filtered_boxes))

    return image_boxes

import cv2
import os
from facenet_pytorch import MTCNN
from helper_functions.face_detection import process_images

input_folder = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me"
output_folder = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me_cropped"


def crop_faces(input_folder, output_folder, image_boxes, leeway):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all images and their bounding boxes
    for image_path, boxes in image_boxes:
        # Read the image
        image = cv2.imread(image_path)

        # Crop and save detected faces
        if boxes is not None:
            for i, box in enumerate(boxes):
                x, y, x2, y2 = [int(coord) for coord in box]

                # Add leeway to the bounding box coordinates
                width = x2 - x
                height = y2 - y
                x = max(0, x - int(width * leeway))
                y = max(0, y - int(height * leeway))
                x2 = min(image.shape[1], x2 + int(width * leeway))
                y2 = min(image.shape[0], y2 + int(height * leeway))

                cropped_face = image[y:y2, x:x2]

                # Check if the cropped face image is empty
                if cropped_face.size == 0:
                    print(
                        f"Error: Unable to crop face from {image_path} (empty image)")
                    continue

                # Save the cropped face image
                output_file_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_face{i}.jpg"
                output_path = os.path.join(output_folder, output_file_name)
                cv2.imwrite(output_path, cropped_face)

                # Count the number of faces detected
                # face_count = len(boxes) if boxes is not None else 0
                # print(f"{image_path}: {face_count} faces detected and cropped")


def get_faces(input_folder, output_folder):
    # Initialize MTCNN for face detection
    mtcnn = MTCNN()

    # Call process_images to get the bounding box coordinates
    # process_images(raw images, model used, min resolution, min confidence)
    print("Starting process_images()")
    image_boxes = process_images(input_folder, mtcnn, 9000, 0.95)

    # Crop the images using the returned bounding boxes
    # crop_faces(folder w/faces, output folder, bounding boxes list, additional margin as a ratio)
    print("Starting crop_faces()")
    crop_faces(input_folder, output_folder, image_boxes, 0.1)

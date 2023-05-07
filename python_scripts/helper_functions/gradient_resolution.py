import cv2
import os
import glob


def resize_images(input_folder, output_folder, starting_height, log_factor, num_iterations):
    # Iterate through all images in the input folder
    for image_path in glob.glob(os.path.join(input_folder, "*.jpg")):
        # Read the image
        image = cv2.imread(image_path)
        # Get the image filename without the extension
        image_name, _ = os.path.splitext(os.path.basename(image_path))

        # Calculate the aspect ratio of the image
        aspect_ratio = float(image.shape[1]) / float(image.shape[0])

        # Resize the original image to have a height equal to the starting height
        original_width = int(aspect_ratio * starting_height)
        image = cv2.resize(image, (original_width, starting_height))

        for i in range(num_iterations):
            # Calculate the new dimensions based on the starting_height and log_factor
            new_height = int(starting_height / (log_factor ** i))
            new_width = int(aspect_ratio * new_height)

            # Resize the image
            resized_image = cv2.resize(image, (new_width, new_height))

            # Save the resized image to the output folder
            output_path = os.path.join(
                output_folder, f"{image_name}_resized_{new_width}x{new_height}.jpg")
            cv2.imwrite(output_path, resized_image)


# Example usage:
input_folder = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/imgs_for_report"
output_folder = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/imgs_for_report/thomas"
resize_images(input_folder, output_folder, starting_height=400,
              log_factor=2, num_iterations=5)

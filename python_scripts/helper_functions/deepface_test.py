import os
import cv2
from matplotlib import pyplot as plt
from deepface import DeepFace

face = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Pictures/straightened_opera_pic.jpg"
db = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/New folder"

dfs = DeepFace.find(img_path=face, db_path=db, enforce_detection=False)

# Read the input face image and display it
input_image = cv2.imread(face)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.imshow(input_image)
plt.title('Input Face')
plt.show()

# Display found images
for row in dsf:
    found_image_path = os.path.join(db, row['identity'])
    found_image = cv2.imread(found_image_path)
    found_image = cv2.cvtColor(found_image, cv2.COLOR_BGR2RGB)

    plt.imshow(found_image)
    plt.title(f"Found: {row['identity']}, Score: {row['score']}")
    plt.show()

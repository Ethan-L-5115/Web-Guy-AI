# Ensemble Feature Extraction

import numpy as np
import pandas as pd
from keras_vggface.utils import preprocess_input
from helper_functions.VGG_Face import create_vgg_model
import os
from PIL import Image

# Path to folder with images
folder_path = 'C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/test_me/test_me_cropped'

# List all image files in folder
image_paths = [os.path.join(folder_path, f)
               for f in os.listdir(folder_path) if f.endswith('.jpg')]

# Load and preprocess images


def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array, version=2)  # For SENet50
    return img_array


# Extract features
vgg_model = create_vgg_model()
features = []
for image_path in image_paths:
    img_array = load_and_preprocess_image(image_path)
    feature_vector = vgg_model.predict(img_array)
    features.append(feature_vector.flatten())

# Create DataFrame with image paths and features
df = pd.DataFrame({'image_path': image_paths, 'features': features})

# Save DataFrame to CSV file
df.to_csv('features.csv', index=False)

# Ensemble Feature Extraction

import numpy as np
import pandas as pd
from keras_vggface.utils import preprocess_input
from helper_functions.VGG_Face import create_vgg_model
import os
from PIL import Image

# folder_path = folder path to the cropped faces


def extract_features(folder_path, output_file_path):

    # List all image files in folder
    image_paths = [os.path.join(folder_path, f)
                   for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # Load and preprocess images

    def load_and_preprocess_image(image_path, input_shape, version):
        img = Image.open(image_path)
        img = img.resize(input_shape[:2])
        img_array = np.array(img, dtype='float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array, version=version)
        return img_array

    # Additional imports for other models (replace with actual imports when you have the models)
    # from helper_functions.other_model_1 import create_other_model_1
    # from helper_functions.other_model_2 import create_other_model_2
    # ...

    # Extract features using an ensemble of models

    def extract_ensemble_features(image_path, models):
        ensemble_features = []

        for model_info in models:
            model, input_shape, version = model_info
            img_array = load_and_preprocess_image(
                image_path, input_shape, version)
            feature_vector = model.predict(img_array)
            ensemble_features.append(feature_vector.flatten())

        return np.concatenate(ensemble_features)

    # Load multiple models
    vgg_model = create_vgg_model()
    # other_model_1 = create_other_model_1()
    # other_model_2 = create_other_model_2()
    # ...

    # Create a list of models with their input shapes and preprocess_input version
    models = [
        (vgg_model, (224, 224, 3), 2)  # ,
        # Replace with actual input shape and version
        # (other_model_1, (224, 224, 3), 1),
        # Replace with actual input shape and version
        # (other_model_2, (224, 224, 3), 1),
        # ...
    ]

    # Extract ensemble features
    features = []
    for image_path in image_paths:
        feature_vector = extract_ensemble_features(image_path, models)
        features.append(feature_vector)

    # print to file
    np.set_printoptions(threshold=np.inf)
    formatted_features = [np.array2string(feat, separator=',', max_line_width=np.inf)[
        1:-1] for feat in features]
    df = pd.DataFrame({'image_path': image_paths,
                      'features': formatted_features})
    df.to_csv(output_file_path + '/features.csv', index=False)

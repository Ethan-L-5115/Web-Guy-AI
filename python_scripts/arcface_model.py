# Run this once in the command line to download the model
# pip install git+https://github.com/4uiiurz1/keras-arcface.git

import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Flatten
from keras.applications.resnet50 import ResNet50
from arcface import ArcFace


def create_arcface_model():
    # Load the ResNet50 model
    resnet_model = ResNet50(include_top=False, input_shape=(224, 224, 3))

    # Add the ArcFace layer
    last_layer = resnet_model.output
    x = Flatten(name='flatten')(last_layer)
    x = ArcFace(512)(x)

    # Create a new model
    custom_arcface_model = Model(inputs=resnet_model.input, outputs=x)

    return resnet_model

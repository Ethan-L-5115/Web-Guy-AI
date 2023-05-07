# Run this once in the command line to download the model
# pip install git+https://github.com/4uiiurz1/keras-arcface.git
# very informative: https://learnopencv.com/face-recognition-with-arcface/

import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Flatten
from keras.applications import ResNet50
from arcface.ArcFace import ArcFaceModel


# def create_arcface_model():
# # Load the ResNet50 model
# resnet_model = ResNet50(include_top=False, input_shape=(512, 512, 3))

# # Add the ArcFace layer
# last_layer = resnet_model.output # Layer 171: conv5_block3_3_conv (Conv2D), Output shape: (None, 16, 16, 2048)
# x = Flatten(name='flatten')(last_layer)
# x = ArcFace(512)(x)

arcface_model = ArcFaceModel()
last_layer = arcface_model.get_layer('OutputLayer').output
x = Flatten(name='flatten')(last_layer)
custom_arcface_model = Model(arcface_model.input, x)

# # Create a new model
# custom_arcface_model = Model(inputs=resnet_model.input, outputs=x)


# List the layers in the model
print("Model layers:")
for i, layer in enumerate(custom_arcface_model.layers):
    output_shape = layer.output_shape
    print(
        f"Layer {i}: {layer.name} ({layer.__class__.__name__}), Output shape: {output_shape}")

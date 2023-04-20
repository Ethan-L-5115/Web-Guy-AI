# VGG-Face Transfer learning implimentation

# Run this once in the command line to download the model
# pip install git+https://github.com/rcmalli/keras-vggface.git

import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Flatten
from keras_vggface.vggface import VGGFace


def create_vgg_model():

    # Based on SENet50 architecture -> new paper(2017) (best performance)
    vgg_model = VGGFace(model='senet50', include_top=False,
                        input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('add_15').output
    x = Flatten(name='flatten')(last_layer)
    custom_vgg_model = Model(vgg_model.input, x)

    return vgg_model

# List the layers in the model
# print("Model layers:")
# for i, layer in enumerate(custom_vgg_model.layers):
#     output_shape = layer.output_shape
#     print(f"Layer {i}: {layer.name} ({layer.__class__.__name__}), Output shape: {output_shape}")

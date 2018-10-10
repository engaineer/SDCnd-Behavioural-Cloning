
from keras.models import Model
from keras.layers import Cropping2D, Conv2D, MaxPool2D, Flatten, Dense
import numpy as np

def nvidia_model(img, crops=((0, 0), (0, 0)) ):
    """
    A CNN model based on the NVIDIA paper.
    The Keras Functional API is used for provide greated felxibilty
    and to allow for easy adoption of a MIMO model (eg throttle and brake)
    to be created.
    :rtype: keras.models.Model
    """

    # Add crop layer if crops are specified
    if (np.asarray(crops).flatten() > 0).any():
        # Crop the input image to the ROI
        x = Cropping2D(cropping=crops)(img)

    # Convoutional Layers
    # Conv 1: 24@30x62 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=24, kernel_size=5, activation='relu')(x)
    x = MaxPool2D(strides=(2,2))(x)

    # Conv 2: 36@13x29 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=36, kernel_size=5, activation='relu')(x)
    x = MaxPool2D(strides=(2,2))(x)

    # Conv 3: 48@5x13 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=48, kernel_size=5, activation='relu')(x)
    x = MaxPool2D(strides=(2,2))(x)

    # Conv 4: 64@3x11 [kernel = 3x3; strides = 1x1]
    x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)

    # Conv 5: 64@1x9 [kernel = 3x3; strides = 1x1]
    x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)

    # Flatten to feed into FC layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    xst = Dense(64, activation='relu')(x)
    xst = Dense(16, activation='relu')(xst)
    out_steering = Dense(1, activation='relu')(xst)

    model = Model(inputs=[img], outputs=[out_steering])

    return model



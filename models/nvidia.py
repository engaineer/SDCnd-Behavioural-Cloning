from keras.models import Sequential
from keras.models import Model
from keras.layers import Cropping2D, Conv2D, MaxPool2D, Flatten, Dense, Dropout, ELU, BatchNormalization
from keras.layers import concatenate
import numpy as np


def nvidia_model(img, speed, crops=((0, 0), (0, 0)) ):
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
    x = Conv2D(filters=24, kernel_size=5, name='L1_conv')(x)
    x = ELU()(x)
    x = MaxPool2D(strides=(2,2), name='L1_pool')(x)

    # Conv 2: 36@13x29 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=36, kernel_size=5, name='L2_conv')(x)
    x = ELU()(x)
    x = MaxPool2D(strides=(2,2), name='L2_pool')(x)
    x = BatchNormalization()(x)


    # Conv 3: 48@5x13 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=48, kernel_size=5, name='L3_conv')(x)
    x = ELU()(x)
    x = MaxPool2D(strides=(2,2), name='L3_pool')(x)

    # Conv 4: 64@3x11 [kernel = 3x3; strides = 1x1]
    x = Conv2D(filters=64, kernel_size=3, name='L4_conv')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    # Conv 5: 64@1x9 [kernel = 3x3; strides = 1x1]
    x = Conv2D(filters=64, kernel_size=3, name='L5_conv')(x)
    x = ELU()(x)

    # 2D -> 1D Flatten to feed into FC layers
    flattened = Flatten()(x)
    xst = Dense(128, activation='relu', name='FC1_steer')(flattened)
    xst = ELU()(xst)

    xst = Dense(64, activation='relu', name='FC2_steer')(xst)
    xst = ELU()(xst)

    xst = Dense(16, activation='relu', name='FC3_steer')(xst)
    xst = ELU()(xst)

    # Ouyput layer
    out_steer = Dense(1,  name='OUT_steer')(xst)

    model = Model(inputs=[img,], outputs=[out_steer])

    return model



import numpy as np
from keras import Model
from keras.layers import Activation, Conv2D, Cropping2D, MaxPool2D, Flatten, Dropout, Dense


def lenet_model(img, crops=((0, 0), (0, 0)), drop_rate=0.4):
    # A LetNet 5 with Max Pooling Dropout used.

    # Add crop layer if crops are specified
    if (np.asarray(crops).flatten() > 0).any():
        # Crop the input image to the ROI
        x = Cropping2D(cropping=crops)(img)

    # Convoutional Layers
    x = Conv2D(filters=6, kernel_size=5, name='L1_conv')(x)
    x = MaxPool2D(strides=(2, 2), name='L1_pool')(x)
    x = Activation(activation='relu', name='L1_activation')(x)

    x = Conv2D(filters=6, kernel_size=5, name='L2_conv')(x)
    x = MaxPool2D(strides=(2, 2), name='L2_pool')(x)
    x = Activation(activation='relu', name='L1_activation')(x)

    # 2D -> 1D conversion for FC layers
    x = Flatten()(x)

    # FC Layers
    x = Dense(128, activation='relu', name='FC1')(x)
    x = Dropout(rate=drop_rate)(x)

    x = Dense(64, activation='relu', name='FC2')(x)
    x = Dropout(rate=drop_rate)(x)

    out_steer = Dense(1, name='out_steer')(x)

    model = Model(inputs=[img], outputs=[out_steer])

    return model

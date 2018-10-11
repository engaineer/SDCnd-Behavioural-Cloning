from keras.models import Sequential
from keras.models import Model
from keras.layers import Cropping2D, Conv2D, MaxPool2D, Flatten, Dense
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
    x = Conv2D(filters=24, kernel_size=5, activation='relu', name='L1_conv')(x)
    x = MaxPool2D(strides=(2,2), name='L1_pool')(x)

    # Conv 2: 36@13x29 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=36, kernel_size=5, activation='relu', name='L2_conv')(x)
    x = MaxPool2D(strides=(2,2), name='L2_pool')(x)

    # Conv 3: 48@5x13 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=48, kernel_size=5, activation='relu', name='L3_conv')(x)
    x = MaxPool2D(strides=(2,2), name='L3_pool')(x)

    # Conv 4: 64@3x11 [kernel = 3x3; strides = 1x1]
    x = Conv2D(filters=64, kernel_size=3, activation='relu', name='L4_conv')(x)

    # Conv 5: 64@1x9 [kernel = 3x3; strides = 1x1]
    x = Conv2D(filters=64, kernel_size=3, activation='relu', name='L5_conv')(x)

    # 2D -> 1D Flatten to feed into FC layers
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='FC1_steer')(x)
    xst = Dense(64, activation='relu', name='FC2_steer')(x)


    # Adding Throttle Control as an output to make a MIMO model
    # This should learn not to apply the throttle when the steering
    # is applying a significant angle to go through a turn if the speed is
    # already high. Also if road is straight / straightening out, and,
    # the speed is low, can go faster.
    xth = concatenate([xst, speed], name='CC_speed')

    xst = Dense(16, activation='relu', name='FC3_steer')(xst)
    out_steer = Dense(1,  name='OUT_steer')(xst)

    xth = Dense(32, activation='relu', name='FC1_speed')(xth)
    xth = Dense(16, activation='relu', name='FC2_speed')(xth)
    out_throttle = Dense(1, name='OUT_throttle')(xth)

    model = Model(inputs=[img, speed], outputs=[out_steer, out_throttle])

    return model



from keras.models import Sequential
from keras.models import Model
from keras.layers import Cropping2D, Conv2D, MaxPool2D, Flatten, Dense, Dropout, ELU, BatchNormalization, Lambda
from keras.layers import concatenate
import numpy as np
import tensorflow as tf

def nvidia2_model(img, crops=((0, 0), (0, 0)) ):
    """
    A CNN model based on the NVIDIA paper.
    The Keras Functional API is used for provide greated felxibilty
    and to allow for easy adoption of a MIMO model (eg throttle and brake)
    to be created.
    :rtype: keras.models.Model
    """

    x = Lambda(lambda x : tf.image.adjust_contrast(x, contrast_factor=5.), name='increase_contrast')(img)
    x = Lambda(lambda x : tf.image.rgb_to_grayscale(x))(x)

    #Normalize the iamge.
    x = Lambda(lambda x : x * 2 - 1, name='normalization')(x)

    # Add crop layer if crops are specified
    if (np.asarray(crops).flatten() > 0).any():
        # Crop the input image to the ROI
        x = Cropping2D(cropping=crops)(x)

    # Convoutional Layers
    # Conv 1: 24@30x62 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=24, kernel_size=5, name='L1_conv')(x)
    x = ELU()(x)
    x = MaxPool2D(strides=(2,2), name='L1_pool')(x)
    x = BatchNormalization()(x)

    # Conv 2: 36@13x29 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=36, kernel_size=5, name='L2_conv')(x)
    x = ELU()(x)
    x = MaxPool2D(strides=(2,2), name='L2_pool')(x)
    x = BatchNormalization()(x)


    # Conv 3: 48@5x13 [kernel = 5x5; strides = 2x2]
    x = Conv2D(filters=48, kernel_size=5, name='L3_conv')(x)
    x = ELU()(x)
    x = MaxPool2D(strides=(2,2), name='L3_pool')(x)
    x = BatchNormalization()(x)


    # Conv 4: 64@3x11 [kernel = 3x3; strides = 1x1]
    x = Conv2D(filters=64, kernel_size=3, name='L4_conv')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    # Conv 5: 64@1x9 [kernel = 3x3; strides = 1x1]
    x = Conv2D(filters=64, kernel_size=3, name='L5_conv')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    # 2D -> 1D Flatten to feed into FC layers
    flattened = Flatten()(x)
    xst = Dropout(rate=0.4)(flattened)
    xst = Dense(128, name='FC1_steer')(xst)
    xst = ELU()(xst)

    xst = Dense(64, name='FC2_steer')(xst)
    xst = ELU()(xst)

    xst = Dense(16, name='FC3_steer')(xst)
    xst = ELU()(xst)

    # Ouyput layer
    out_steer = Dense(1,  name='OUT_steer')(xst)

    model = Model(inputs=img, outputs=out_steer)

    return model

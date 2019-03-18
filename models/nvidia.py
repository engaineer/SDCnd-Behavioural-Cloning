from keras.models import Sequential
from keras.models import Model
from keras.layers import Cropping2D, Conv2D, MaxPool2D, Flatten, Dense, Dropout, ELU, BatchNormalization, Lambda
from keras.layers import concatenate
import numpy as np
import tensorflow as tf

def to_yuv(img, in_cspace='RGB'):

    img_float = tf.cast(img, dtype=tf.float32) / 255.

    if (in_cspace == 'RGB'):
        img_rgb = tf.image.rgb_to_yuv(img_float)
    elif (in_cspace == 'BGR'):
        img_rgb = tf.image.bgr_to_yuv(img_float)
    else:
        raise ValueError(f"Unknown value of {in_cspace} for parameter 'in_space.'")

    return img_rgb

def nvidia_model(img, crops=((0, 0), (0, 0)) ):
    """
    A CNN model based on the NVIDIA paper implemented with Keras
    Functional API.
    :rtype: keras.models.Model
    """
    x = Lambda(to_yuv, name='to_yuv')(img)
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
    xst = Dense(128, name='FC1_steer')(flattened)
    xst = ELU()(xst)
    xst = Dropout(rate=0.5)(xst)


    xst = Dense(64, name='FC2_steer')(xst)
    xst = ELU()(xst)
    xst = Dropout(rate=0.5)(xst)

    xst = Dense(16, name='FC3_steer')(xst)
    xst = ELU()(xst)
    xst = Dropout(rate=0.5)(xst)


    # Ouyput layer
    out_steer = Dense(1,  name='OUT_steer')(xst)

    model = Model(inputs=img, outputs=out_steer)

    return model

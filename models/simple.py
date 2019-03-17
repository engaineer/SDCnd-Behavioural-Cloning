import numpy as np
from keras import Model
from keras.layers import Cropping2D, Flatten, Dense


def simple_model(img, crops=((0, 0), (0, 0))):

    # Add crop layer if crops are specified
    if (np.asarray(crops).flatten() > 0).any():
        # Crop the input image to the ROI
        x = Cropping2D(cropping=crops)(img)

    x = Flatten()(x)

    out_steer = Dense(1)(x)

    model = Model(inputs=[img], outputs=[out_steer])

    return model
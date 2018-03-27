import argparse
import csv
import os
import time

import cv2
import numpy as np
from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential
from sklearn.utils import shuffle

flags = None


def read_csv(fname):
    samples = []
    with open(fname) as fh:
        reader = csv.reader(fh)
        for line in reader:
            samples.append(line)

    return samples


# Shuffle and split the data into test training. Validation would be a completely new track in
# simulation environment. Instead of a 60% 20% 20% split we wil use a 80% : 20% split.
def split_data(data):
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=0.2)
    return train_data, test_data


def data_generator(samples, images_path, batch_size=128):
    num_samples = len(samples)

    while True:
        # Loop forever so the generator never terminates
        # Perform an initial shuffle of the data each time that the data
        # iset is processed

        samples = shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Define all of the images which could be used.  Can randomly select
                # the left right and middle images data views later.
                center_image_path = str(batch_sample[0])
                left_image_path = str(batch_sample[1])
                right_image_path = str(batch_sample[2])

                image_name = os.path.split(center_image_path)[-1]
                image_fpath = os.path.join(images_path, image_name)
                center_image = cv2.imread(image_fpath)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.asarray(images)
            y_train = np.asarray(angles)
            yield shuffle(X_train, y_train)


def simple_model(in_shape=(160, 320, 3)):
    model = Sequential()

    # normalize the input.
    model.add(Lambda(lambda x: x / 255. - 0.5), input_shape=(160, 320, 3))
    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(1))

    return model


def train_model(model, train_data, valid_data, batch_size=32, epochs=5):
    # Create a data generator.
    train_gen = data_generator(train_data, batch_size=batch_size)
    valid_gen = data_generator(valid_data, batch_size=batch_size)

    model.compile(loss='mse', optimizer='adam')

    num_train_samp = len(train_data)
    num_valid_samp = len(valid_data)

    history_obj = model.fit_generator(
        generator=train_gen,
        validation_data=valid_gen,
        samples_per_epoch=num_train_samp,
        nb_val_samples=num_valid_samp,
        nb_epoch=epochs,
        verbose=1
    )

    return history_obj


def save_model(model, name='model', save_path=''):
    dtstr = time.strftime('%Y%m%d-%H%M%S')
    fname = "{}{}.h5".format(name, dtstr)
    fpath = os.path.join(save_path, fname) if len(save_path) > 0 else fname
    model.save(fpath)
    pass


def main():
    global flags
    parser = argparse.ArgumentParser('Model arguments')

    # Image Data Path
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='The root Directory of the data files.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epoch passes through the data.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for each training iteration.'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='simple',
        help='Which model to use.'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='./models',
        help='Which model to use.'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='model',
        help='Model name prefix.'
    )

    parser.add_argument(
        '--save_model',
        type=bool,
        default=True,
        help='Save the model upon completion.'
    )

    flags = parser.parse_args()

    # Read in the recordinds and split it up.
    track_recordings = read_csv()
    training_data, validation_data = split_data(track_recordings)

    model = None
    if "simple" in flags.model_type.lower():
        model = simple_model()
    else:
        raise NotImplemented("Unknown or Not implemented model type {}".format(flags.model_type.lower()))

    train_model(model)

    if flags.save_model:
        save_model(model, name=flags.model_name, save_path=flags.model_path)


if __name__ == '__main__':
    main()

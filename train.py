import argparse, csv, time
import random

import numpy as np
import pandas as pd
from glob import glob
from os import path
import re
from math import ceil

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import cv2

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Input
from keras import optimizers, losses

from utils.generator import threadsafe_iter
from models.nvidia import nvidia_model

def batch_generator(df, batch_sz, lr_angle, training=True):
    samples = shuffle(df)

    groups = df.groupby('bin')
    samples_per_group = ceil(batch_sz / groups.ngroups)

    while True:

        # Here I ranomdly select from each steering bin category to select the number of items evenly
        batch = groups.apply(lambda grp: grp.sample(samples_per_group, replace=True)).sample(batch_sz)

        batch_images = []
        batch_angles = []

        for idx, row in batch.iterrows():

            angle = float(row.steering)

            camera = 'center'
            if training:
                # Only manipulate the test data not the validation data.

                # Select either the left or right images.
                camera = np.random.choice(['center', 'left', 'right'], 1)[0]

                # Adjust the angle if required.
                if camera == 'left':
                    angle += lr_angle
                elif camera == 'right':
                    angle -= lr_angle
                else:
                    pass

            # Get the actual image.
            image_fpath = row[camera]
            if not path.isfile(image_fpath):
                raise FileNotFoundError("Cannot find file '{}'".format(image_fpath))

            img = cv2.imread(image_fpath)

            # Randomly flip Left/Right the images to assist with balancing
            if random.choice([True, False]):
                # Note that I used numpy flip as is it is about 20x faster than OpenCV
                img = np.fliplr(img)
                angle = -angle


            if np.asarray(img).dtype not in [np.int32, np.uint8, np.uint16, np.uint32] or \
                np.min(img) < 0 or \
                np.max(img) > 255:
                raise ValueError('Image datatype or range is wrong prioer to normalization.')

            # Center the data
            img = (img - 127.5) / 127.5

            batch_angles.append(angle)
            batch_images.append(img)

            yield [np.asarray(batch_images), np.asarray(batch_angles)]


def load_data(datadir, logname):
    glob_path = path.join(datadir, '**', logname)

    logdata = []

    for logfile in glob(glob_path):

        curr_data_path = path.split(logfile)[0]

        # Check whether the data file has a header
        def has_header(f):
            with open(f) as fh:
                first_line = fh.readline()
                if re.match(r'center,[ ]*left,[ ]*right,[ ]*steering,[ ]*throttle,[ ]*brake,[ ]*speed', first_line):
                    return 0
                else:
                    return None

        col_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        col_dtype = {'center': str,
                     'left': str,
                     'right': str,
                     'steering': np.float32,
                     'throttle': np.float32,
                     'brake': np.float32,
                     'speed': np.float32}

        curr_logdata = pd.read_csv(logfile, names=col_names, header=has_header(logfile), dtype=col_dtype,
                                   index_col=False)

        def apply_img_path(saved_path, base_path, img_path='IMG'):
            img_fname = path.split(saved_path)[-1]
            return path.join(base_path, img_path, img_fname)

        # Fix up the paths in the data file.
        for pos in ['center', 'left', 'right']:
            curr_logdata[pos] = curr_logdata[pos].apply(lambda fpath: apply_img_path(fpath, curr_data_path))

        logdata.append(curr_logdata)

    logdata = pd.concat(logdata, ignore_index=True).reset_index()

    # Group the steering data into steering bin interval category bins.
    bins = np.round(np.arange(-0.5, 0.6, 0.1), 1)
    logdata['bin'] = pd.cut(logdata['steering'], bins)
    logdata = logdata.dropna()
    logdata.reindex()

    return train_test_split(logdata, test_size=0.2)


def model_fname(name='model', save_path=''):
    dtstr = time.strftime('%Y%m%d-%H%M')
    fname = "{}_{}.h5".format(name, dtstr)
    return path.join(save_path, fname) if len(save_path) > 0 else fname


def train(model_arch, datadir, drivelog_name, save_model, offset_correction, batch_sz, lr, tensorboard, logdir, patience,
          multiprocessing, crops):
    # Get the test and validation data frames.
    train_df, validation_df = load_data(datadir, drivelog_name)

    if len(train_df.index) == 0 or len(validation_df.index) == 0:
        raise RuntimeError('Training or Test dataframes are empty.')

    train_generator = threadsafe_iter(batch_generator(train_df, batch_sz=batch_sz, lr_angle=offset_correction, training=True))
    validation_generator = threadsafe_iter(batch_generator(validation_df, batch_sz=batch_sz, lr_angle=offset_correction, training=False))

    # Input tensors
    input_img = Input(shape=(160,320,3), name='image')

    # Model Selection
    if model_arch.lower() == 'nvidia':
        model = nvidia_model(input_img, crops)
    else:
        ValueError("Do not know how to handle value '{}' for model_arch.".format(model_arch))

    # Model Optimizer
    optimizer = optimizers.Nadam(lr=lr)

    # Compile the Model
    model.compile(optimizer=optimizer, loss='mse')

    # Define the fit callbacks to run after each epoch.
    callbacks = []

    if tensorboard:
        tb_callback = TensorBoard(log_dir=logdir, batch_size=batch_sz)
        callbacks.append(tb_callback)

    if save_model:
        model_fpath = model_fname(name=model_arch, save_path='./models')
        callback_cp = ModelCheckpoint(filepath=model_fpath, save_best_only=True)
        callbacks.append(callback_cp)

    # Early Stopping.
    callback_es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, verbose=1)
    callbacks.append(callback_es)

    #TODO: Additional learning rate decay as per Google paper.

    model.fit_generator(train_generator,
                        steps_per_epoch=train_df.size // batch_sz,
                        epochs=50,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=validation_df.size // batch_sz,
                        use_multiprocessing=multiprocessing['enabled'],
                        max_queue_size=50,
                        workers=multiprocessing['workers'] if multiprocessing['enabled'] else 1
                        )


if __name__ == '__main__':
    global cfg

    parser = argparse.ArgumentParser('Train')

    # Image Data Path
    parser.add_argument(
        '--datadir',
        type=str,
        default='./data',
        help='The root Directory of the data files.'
    )

    parser.add_argument(
        '--drivelog_name',
        type=str,
        default='driving_log.csv',
        help='Driving log name.'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='nvidia',
        help='Which model to use.'
    )

    parser.add_argument(
        '--save_model',
        type=bool,
        default=True,
        help='Save the model upon completion.'
    )

    parser.add_argument(
        '--offset_correction',
        type=float,
        default=0.25,
        help='Adjustment angle for left and right camera offets.'
    )

    parser.add_argument(
        '--batch_sz',
        type=int,
        default=16,
        help='Size for each batch'
    )

    parser.add_argument(
        '--lr',
        type=int,
        default=2e-3,
        help='Starting Learning Rate'
    )

    parser.add_argument(
        '--logdir',
        type=str,
        default='./logs',
        help='Log directory'
    )

    parser.add_argument(
        '--multiprocessing',
        dest='multiprocessing',
        action='store_true',
        help='Enable multiprocessing.'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of workers to use if multiprocessing.'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=4,
        help='Patience for early stopping.'
    )

    parser.add_argument(
        "--crop_lr",
        nargs=2,
        type=int,
        default=[0, 0],
        help='Left Right pixel margin to crop.'
    )

    parser.add_argument(
        "--crop_tb",
        nargs=2,
        type=int,
        default=[60, 20],
        help='Top Bottom pixel margin to crop.'
    )


    parser.add_argument(
        '--tensorboard',
        dest='tensorboard',
        action='store_true',
        help='Enable tensorboard'
    )


    cfg = parser.parse_args()

    train(model_arch=cfg.model,
          datadir=cfg.datadir,
          drivelog_name=cfg.drivelog_name,
          save_model=cfg.save_model,
          offset_correction=cfg.offset_correction,
          batch_sz=cfg.batch_sz,
          lr=cfg.lr,
          patience=cfg.patience,
          multiprocessing={'enabled': cfg.multiprocessing, 'workers': cfg.workers},
          crops=(tuple(cfg.crop_tb), tuple(cfg.crop_lr)),
          logdir=cfg.logdir,
          tensorboard=cfg.tensorboard
          )

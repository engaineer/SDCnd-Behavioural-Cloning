import argparse
import random
import re
import time
from glob import glob
from math import ceil
from os import path, environ

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.layers import Input
from sklearn.model_selection import train_test_split

from models.lenet import lenet_model
from models.nvidia import nvidia_model
from models.nvidia2 import nvidia2_model
from models.simple import simple_model
from utils.datagen import ImageGenerator



def steering_ewma(df, smoothing_win_size=5, smoothing_shift=2):
    smoothed_steering = df.steering.ewm(span=smoothing_win_size).mean()
    df.steering = smoothed_steering
    df.steering = df.steering.shift(-smoothing_shift)
    # Shift the smoothed data back by smoothing amount
    df = df[:-smoothing_shift]

    return df


def load_data(datadir, logname):
    glob_path = path.join(datadir, '**/**', logname)

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

        # Smooth all data which is not the steering recovery data.
        if path.split(curr_data_path)[-1] != 'recovery':
            curr_logdata = steering_ewma(curr_logdata)

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
    bins = np.concatenate([[-1, -0.75], bins, [0.75, 1]])
    logdata['bin'] = pd.cut(logdata['steering'], include_lowest=True, right=True, bins=bins)

    logdata = logdata.dropna()

    train_df, validation_df = train_test_split(logdata, test_size=0.2)

    train_df.reindex()
    validation_df.reindex()

    return train_df, validation_df


def show_bin_dist(df, name):
    print("=" * 20)
    print("\t" + name + " Steering Distribution")
    print("-" * 20)
    print(df.groupby('bin').size())
    print("\n")


def model_fname(name='model', save_path=''):
    dtstr = time.strftime('%Y%m%d-%H%M%S')
    fname = "{}_{}.h5".format(name, dtstr)
    return path.join(save_path, fname) if len(save_path) > 0 else fname


def train(model_arch, datadir, drivelog_name, save_model, offset_correction, batch_sz, lr, tensorboard, logdir,
          patience, multiprocessing, crops, trained_savepath, early_stopping, epochs, lr_scheduler):
    # Get the test and validation data frames.
    train_df, validation_df = load_data(datadir, drivelog_name)

    show_bin_dist(train_df, 'Training')
    show_bin_dist(validation_df, 'Validation')

    if len(train_df.index) == 0 or len(validation_df.index) == 0:
        raise RuntimeError('Training or Test dataframes are empty.')

    train_generator = ImageGenerator(train_df, batch_sz=batch_sz, lr_angle=offset_correction)
    validation_generator = ImageGenerator(validation_df, batch_sz=batch_sz, lr_angle=offset_correction)

    # Input tensors
    input_img = Input(shape=(160, 320, 3), name='image')

    # Model Optimizer
    # optimizer = optimizers.SGD(lr=lr)
    optimizer = optimizers.Adam(lr=lr)

    # Model Selection
    if model_arch.lower() == 'nvidia':
        model = nvidia_model(input_img, crops)

    elif model_arch.lower() == 'nvidia2':
        model = nvidia2_model(input_img, crops)

    elif model_arch.lower() == 'simple':
        model = simple_model(input_img, crops)

    elif model_arch.lower() == 'lenet':
        model = lenet_model(input_img, crops)

    else:
        ValueError("Do not know how to handle value '{}' for model_arch.".format(model_arch))


    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    # Define the fit callbacks to run after each epoch.
    callbacks = []

    if tensorboard:
        tb_callback = TensorBoard(log_dir=logdir, batch_size=batch_sz)
        callbacks.append(tb_callback)

    model_fpath = model_fname(name=model_arch, save_path=trained_savepath)

    if save_model:
        # cp_fpath = path.splitext(model_fpath)[0] + "-{epoch:02d}-{val_loss:.2f}.hdf5"
        callback_cp = ModelCheckpoint(filepath=model_fpath, save_best_only=True)
        callbacks.append(callback_cp)

    if early_stopping:
        # Early Stopping.
        callback_es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, verbose=1)
        callbacks.append(callback_es)

    # Learning rate scheduller
    if lr_scheduler:
        callback_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=(patience*3)//4, min_lr=1e-5, verbose=1)
        callbacks.append(callback_lr)

    model.fit_generator(generator=train_generator,
                        validation_data=validation_generator,
                        epochs=100 if early_stopping else epochs,
                        callbacks=callbacks,
                        use_multiprocessing=multiprocessing['enabled'],
                        workers= multiprocessing['workers'],
                        shuffle=False,
                        verbose=2)

    if save_model:
        print('Saved mode to :' + model_fpath)
        if not (early_stopping or lr_scheduler):
            model.save(model_fpath)


if __name__ == '__main__':
    global cfg

    print(f"Tensorflow = \t {tf.__version__}")
    print(f"Keras = \t {keras.__version__}")

    parser = argparse.ArgumentParser('Train')

    # Image Data Path
    parser.add_argument(
        '--datadir',
        type=str,
        default='/home/neuralflux/Development/SDCnd/SDCnd-Behavioural-Cloning/data',
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
        dest='save_model',
        action='store_true',
        help='Save the model upon completion.'
    )

    parser.add_argument(
        '--trained_savepath',
        type=str,
        default='trained',
        help='File path on where to save the model.'
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
        default=32,
        help='Size for each batch'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
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
        default=5,
        help='Patience for early stopping.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
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

    parser.add_argument(
        '--early_stopping',
        dest='early_stopping',
        action='store_true',
        help='Enable early stopping.'
    )

    parser.add_argument(
        '--lr_scheduler',
        dest='lr_scheduler',
        action='store_true',
        help='Enable Learning Rate Schedulling'
    )


    parser.add_argument(
        '--visible_gpus',
        type=str,
        default="0",
        help='Which GPUs are visible.'
    )

    cfg = parser.parse_args()
    environ["CUDA_VISIBLE_DEVICES"]=cfg.visible_gpus

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
          trained_savepath=cfg.trained_savepath,
          tensorboard=cfg.tensorboard,
          early_stopping=cfg.early_stopping,
          epochs=cfg.epochs,
          lr_scheduler=cfg.lr_scheduler
          )

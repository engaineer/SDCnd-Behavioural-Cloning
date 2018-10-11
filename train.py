import argparse
import random
import re
import time
from glob import glob
from math import ceil
from os import path

import cv2
import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from lenet import lenet_model
from model import simple_model
from models.nvidia import nvidia_model
from utils.generator import threadsafe_iter


def plain_generator(df, batch_sz):

    samples = shuffle(df)


    while True:

        batch_images = []
        batch_angles = []
        batch_speed = []
        batch_throttle = []

        batch = samples.sample(batch_sz, replace=False)

        for idx, row in batch.iterrows():

            angle = float(row['steering'])
            speed = float(row['speed'])
            throttle = float(row['throttle'])

            camera = 'center'

            # Get the actual image.
            image_fpath = row[camera]

            if not path.isfile(image_fpath):
                raise FileNotFoundError("Cannot find file '{}'".format(image_fpath))

            img = cv2.imread(image_fpath)

            # The provided drive code use images in RGB format. So convert to RGB for training.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = (img / 255.) - 0.5

            batch_angles.append(angle)
            batch_images.append(img)
            batch_speed.append(speed)
            batch_throttle.append(throttle)

        yield [np.asarray(batch_images), np.asarray(batch_speed)], [np.asarray(batch_angles), np.asarray(batch_throttle)]


def training_generator(df, batch_sz, lr_angle):

    samples = shuffle(df)

    groups = samples.groupby('bin', observed=True)

    empty_groups = []
    for name, group in groups:
        group_size = groups.get_group(name).size
        if group_size == 0:
            empty_groups.append(name)

    for n in empty_groups:
        df.drop(groups.get_group(n).index)

    samples_per_group = ceil(batch_sz / groups.ngroups)


    while True:

        # Here I ranomdly select from each steering bin category to select the number of items evenly
        batch = groups.apply(lambda grp: grp.sample(samples_per_group, replace=True)).sample(batch_sz)

        batch_images = []
        batch_angles = []
        batch_speed = []
        batch_throttle = []

        for idx, row in batch.iterrows():

            angle = float(row['steering'])
            speed = float(row['speed'])
            throttle = float(row['throttle'])


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

            # The provided drive code use images in RGB format. So convert to RGB for training.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Randomly flip Left/Right the images to assist with balancing
            if random.choice([True, False]):
                # Note that I used numpy flip as is it is about 20x faster than OpenCV
                img = np.fliplr(img)
                angle = -angle

            if np.asarray(img).dtype not in [np.int32, np.uint8, np.uint16, np.uint32] or \
                    np.min(img) < 0 or \
                    np.max(img) > 255:
                raise ValueError('Image datatype or range is wrong prioer to normalization.')

            img = (img / 255.) - 0.5

            batch_angles.append(angle)
            batch_images.append(img)
            batch_speed.append(speed)
            batch_throttle.append(throttle)

        yield [np.asarray(batch_images), np.asarray(batch_speed)], [np.asarray(batch_angles), np.asarray(batch_throttle)]


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
    dtstr = time.strftime('%Y%m%d-%H%M')
    fname = "{}_{}.h5".format(name, dtstr)
    return path.join(save_path, fname) if len(save_path) > 0 else fname


def train(model_arch, datadir, drivelog_name, save_model, offset_correction, batch_sz, lr, tensorboard, logdir,
          patience, multiprocessing, crops, trained_savepath, early_stopping, epochs):
    # Get the test and validation data frames.
    train_df, validation_df = load_data(datadir, drivelog_name)

    show_bin_dist(train_df, 'Training')
    show_bin_dist(validation_df, 'Validation')

    if len(train_df.index) == 0 or len(validation_df.index) == 0:
        raise RuntimeError('Training or Test dataframes are empty.')

    if multiprocessing:
        train_generator = threadsafe_iter(
            training_generator(train_df, batch_sz=batch_sz, lr_angle=offset_correction))
        validation_generator = threadsafe_iter(plain_generator(train_df, batch_sz=batch_sz))
    else:
        train_generator =  training_generator(train_df, batch_sz=batch_sz, lr_angle=offset_correction)
        validation_generator = plain_generator(train_df, batch_sz=batch_sz)

    # Input tensors
    input_img = Input(shape=(160, 320, 3), name='image')
    input_speed = Input(shape=(1,), name='speed')

    # Model Optimizer
    # optimizer = optimizers.SGD(lr=lr)
    optimizer = optimizers.Adam(lr=lr)

    # Model Selection
    if model_arch.lower() == 'nvidia':
        model = nvidia_model(input_img, input_speed, crops)

        # Define loss weights for the MIMO model with multiple outputs.
        loss_weights = {'OUT_steer': 1., 'OUT_throttle': 0.15}
        # Compile the Model
        model.compile(optimizer=optimizer, loss='mse', loss_weights=loss_weights)

    elif model_arch.lower() == 'simple':
        model = simple_model(input_img, crops)
        model.compile(optimizer=optimizer, loss='mse')


    elif model_arch.lower() == 'lenet':
        model = lenet_model(input_img, crops)
        model.compile(optimizer=optimizer, loss='mse')

    else:
        ValueError("Do not know how to handle value '{}' for model_arch.".format(model_arch))



    # Define the fit callbacks to run after each epoch.
    callbacks = []

    if tensorboard:
        tb_callback = TensorBoard(log_dir=logdir, batch_size=batch_sz)
        callbacks.append(tb_callback)

    model_fpath = model_fname(name=model_arch, save_path=trained_savepath)

    if save_model:
        callback_cp = ModelCheckpoint(filepath=model_fpath, save_best_only=True)
        callbacks.append(callback_cp)

    if early_stopping:
        # Early Stopping.
        callback_es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, verbose=1)
        callbacks.append(callback_es)

    # Learning rate scheduller
    callback_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    callbacks.append(callback_lr)

    model.fit_generator(train_generator,
                        steps_per_epoch=train_df.index.size // batch_sz,
                        epochs=100 if early_stopping else epochs,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=validation_df.index.size // batch_sz,
                        use_multiprocessing=multiprocessing['enabled'],
                        max_queue_size=10,
                        workers=multiprocessing['workers'] if multiprocessing['enabled'] else 1
                        )

    if save_model and not early_stopping:
        print('Saved mode to :' + model_fpath)
        model.save(model_fpath)


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
        default=16,
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
          trained_savepath=cfg.trained_savepath,
          tensorboard=cfg.tensorboard,
          early_stopping=cfg.early_stopping,
          epochs=cfg.epochs
          )

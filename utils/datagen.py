import keras
import numpy as np
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from math import ceil
from multiprocessing.pool import ThreadPool
import cv2
from os import path
import random
from pandas import DataFrame

class ImageGenerator(keras.utils.Sequence):

    def __init__(self, df: DataFrame, batch_sz: int, lr_angle: float, worker_pool_sz: int = 1) -> object:
        self.lr_angle = lr_angle
        self.samples = df
        self.batch_sz = batch_sz
        self.__group_bins(self.samples)
        self._augmentation_seq = self.__init_image_augmenter()

        # Multiprocessing pool for reading files inparallel
        # self.worker_pool = ThreadPool(worker_pool_sz)
        self.on_epoch_end()

    def __init_image_augmenter(self):
        img_augmenter = iaa.Sequential([
            iaa.AddToHueAndSaturation(value=(-30, 30), per_channel=True),
            iaa.CoarseDropout(p=(0.10, 0.20), size_percent=(0.04,0.05))
        ])

        return img_augmenter

    def __get_image(self, fname_angle, bgr_to_rgb=True):

        image_fpath = fname_angle[0]

        if not path.isfile(image_fpath):
            raise FileNotFoundError("Cannot find file '{}'".format(image_fpath))

        img = cv2.imread(image_fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, fname_angle[1]

    def __get_batches_per_epoch(self, df, batch_sz):
        return df.index.size // batch_sz

    def __group_bins(self, df):

        groups = self.samples.groupby('bin', observed=True)

        samples_per_group = ceil(self.batch_sz / groups.ngroups)

        self._df_groups = groups
        self._samples_per_group = samples_per_group

    def __select_camera(self, row):

        angle = float(row['steering'])

        # Select either the left or right images.
        camera = np.random.choice(['center', 'left', 'right'], 1)[0]

        # Adjust the angle if required.
        if camera == 'left':
            angle += self.lr_angle
        elif camera == 'right':
            angle -= self.lr_angle
        else:
            pass

        return row[camera], angle

    def __random_flip(self, fname_angle):

        if random.choice([True, False]):
            # Note that I used numpy flip as is it is about 20x faster than OpenCV
            fname_angle = np.fliplr(fname_angle[0]), -fname_angle[1]

        return fname_angle

    def __getitem__(self, index):

        curr_batch = self._df_groups.apply(lambda grp: grp.sample(self._samples_per_group, replace=True)).sample(
            self.batch_sz)

        # Get a list of the image fnames and angles
        image_fnames_angles = [self.__select_camera(row) for idx, row in curr_batch.iterrows()]

        # Read all of the images in with corresponding angles.
        # images_angles = self.worker_pool.map(self.__get_image, image_fnames_angles)
        images_angles = [self.__get_image(x) for x in image_fnames_angles]


        # Randomly flip 50% of images.
        # images_angles = self.worker_pool.map(self.__random_flip, images_angles)
        images_angles = [self.__random_flip(x) for x in images_angles]

        # Split paired tuple of images and angles into two different lists.
        images = [img for img, angle in images_angles]
        angles = [angle for img, angle in images_angles]

        aug_images = self._augmentation_seq.augment_images(images)

        return np.asarray(aug_images), np.asarray(angles)

    def __len__(self):
        return len(self.samples.index) // self.batch_sz

    def on_epoch_end(self):
        self.samples = shuffle(self.samples)

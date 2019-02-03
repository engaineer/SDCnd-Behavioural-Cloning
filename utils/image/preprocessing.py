import cv2
import skimage
import numpy as np
from sklearn.cluster import KMeans
from skimage import img_as_ubyte
import warnings

def training_enhancement(img, in_fmt='BGR', sat_multiplier=2., gamma=0.5, n_clusters=8):

    # Increase Saturation
    if in_fmt is 'BGR':
        cvt_code = cv2.COLOR_BGR2HSV
    else:
        cvt_code = cv2.COLOR_RGB2HSV

    img_blurred = cv2.bilateralFilter(img, 7, 75, 75)

    h, s, v = cv2.split(cv2.cvtColor(img_blurred, cvt_code))
    img_high_s = cv2.merge([h, np.asarray(s * sat_multiplier, np.uint8), v])

    # Gamma corrected for shadow reduction
    img_gamma_corrected = skimage.exposure.adjust_gamma(cv2.cvtColor(img_high_s, cv2.COLOR_HSV2RGB), gamma=gamma)

    img_pp_reshaed = img_gamma_corrected.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters).fit(img_pp_reshaed)
    km_labels = kmeans.labels_
    km_centers = kmeans.cluster_centers_

    img_reduced_colors = km_centers[km_labels].reshape(img_gamma_corrected.shape).astype(np.uint8)

    img_blurred = cv2.bilateralFilter(img_reduced_colors, 3, 75, 75)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_pp = skimage.exposure.equalize_adapthist(img_blurred, clip_limit=0.5, nbins=16)
        img_pp = img_as_ubyte(img_pp)

    return img_pp

from magic_inpaint.commonfunctions import *
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv,rgba2rgb
from skimage.measure import label

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median , gaussian
from skimage.feature import canny

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Exposure
from skimage.exposure import equalize_adapthist, equalize_hist

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

#Morphology:
from skimage.morphology import dilation, erosion, opening, closing,rectangle,isotropic_opening,isotropic_closing, binary_dilation, binary_erosion,binary_closing,binary_opening


# Hough Transform
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line as draw_line
from skimage import data

# OpenCv
import imutils  
import cv2


from itertools import combinations


def increase_mask_width(mask, width):
    structuring_element = np.ones((2 * width + 1, 2 * width + 1), np.uint8)

    # Dilate the eroded mask
    dilated_mask = cv2.dilate(mask, structuring_element, iterations=1)

    return dilated_mask

def apply_median_filter_rgb(original_img, mask, filter_size):
    
    mask = increase_mask_width(mask, filter_size)
    filtered_img = original_img.copy()
    for c in range(original_img.shape[2]):
        channel_img = original_img[:, :, c]
        filtered_channel = cv2.medianBlur(channel_img, filter_size)
        filtered_img[:, :, c][np.where(mask)] = filtered_channel[np.where(mask)]
    return filtered_img

def smudge_mask_region(image, mask, width, intensity):

    mask = increase_mask_width(mask,width=25)
    binary_mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(binary_mask_uint8)
    cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    contour_mask = cv2.threshold(contour_mask, 1, 255, cv2.THRESH_BINARY)[1]
    inside_mask = cv2.bitwise_not(contour_mask)
    image_float32 = image.astype(np.float32)
    blurred_image = cv2.GaussianBlur(image_float32, (2 * width + 1, 2 * width + 1), 0)
    smudged_image = (1 - intensity) * image_float32 + intensity * blurred_image
    smudged_image = np.clip(smudged_image, 0, 255).astype(np.uint8)

    result_image = cv2.bitwise_and(image, image, mask=inside_mask)
    result_image += cv2.bitwise_and(smudged_image, smudged_image, mask=contour_mask)

    return result_image

def postprocessing(img, mask):

    smudge_width = 50
    smudge_intensity = 0.1
    filter_size = 3

    result_image_smoothed = apply_median_filter_rgb(img, mask, filter_size)
    result_image = smudge_mask_region(result_image_smoothed, mask, smudge_width, smudge_intensity)
    return result_image
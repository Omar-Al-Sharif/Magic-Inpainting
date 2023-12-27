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



def MSD(target_patch, exemplar_patch, M_bar):
    if np.sum(M_bar)==0:
        return float('inf')
    else:
        msd=np.sum((M_bar*target_patch - M_bar*exemplar_patch)**2)/np.sum(M_bar)
        return msd


def SMD (target_patch, exemplar_patch, binary_mask, M_bar):
    if np.sum(M_bar)==0:
        return float('inf')
    if np.sum(binary_mask)==0:
        return float('inf')
    target_existing_avg= np.sum(M_bar * target_patch)/np.sum(M_bar)
    exemplar_fill_avg = np.sum(binary_mask* exemplar_patch)/np.sum(binary_mask)
    smd= (target_existing_avg-exemplar_fill_avg)**2
    return smd 


def findBestsubPatch(target_patch, exemplar_patches, binary_mask, M_bar):
    best_patch = exemplar_patches[len(exemplar_patches)-1]
    min_combined_error = float('inf')

    for other_patch in exemplar_patches:
        current_error = MSD((rgb2gray(target_patch)*255).astype('uint8'), (rgb2gray(other_patch)*255).astype('uint8'), M_bar) + SMD((rgb2gray(target_patch)*255).astype('uint8'), (rgb2gray(other_patch)*255).astype('uint8'), binary_mask, M_bar)

        if current_error < min_combined_error:
            best_patch = other_patch
            min_combined_error = current_error

    return best_patch


def findBestFullPatch(target_patches, exemplar_patches,binary_mask_patches, M_bar_patches):
    cp_target_patches = np.copy(target_patches)
    for i,target_patch in enumerate(target_patches):
        best_sub_patch=findBestsubPatch(target_patch,exemplar_patches,binary_mask_patches[i],M_bar_patches[i])
        cp_target_patches[i][binary_mask_patches[i] == 1] = best_sub_patch[binary_mask_patches[i] == 1]
    return cp_target_patches


def main(img,binary_mask, overlap=2):

    white_pixels = np.sum(binary_mask == 1)
    black_pixels = np.sum(binary_mask == 0)

    ratio = white_pixels / (white_pixels + black_pixels)
    if ratio < 0.15:
        overlap = 64
        patch_size =64
    elif ratio >0.3:
        overlap = 80
        patch_size =80
    else:
        overlap = 32
        patch_size =32

    img_in_patches= [img[i:i+patch_size,j:j+patch_size] for i in range(0,img.shape[0]-patch_size+1, overlap) for j in range(0,img.shape[1]-patch_size+1, overlap) ]
    full_binary_mask_patches = [binary_mask[i:i+patch_size,j:j+patch_size] for i in range(0,img.shape[0]-patch_size+1, overlap) for j in range(0,img.shape[1]-patch_size+1, overlap) ]
    #if the patch contains a single 1 then it's subset of the target patch so mark it with 1 in the bit map

    target_indices=[]
    exemplar_indices=[]
    binary_mask_patches= []
    target_patches=[]
    exemplar_patches=[]
    target_patch_counter=0
    patches_per_row = (img.shape[0] - patch_size) // overlap + 1
    patches_per_col = (img.shape[1] - patch_size) // overlap + 1
    for i, binary_mask_patch in enumerate(full_binary_mask_patches):

        start_row = (i // patches_per_row) * overlap
        start_col = (i % patches_per_row) * overlap
        if np.isin(1,binary_mask_patch):
            # This means that it is a target patch
            target_indices.append((target_patch_counter,start_row,start_col))
            binary_mask_patches.append(full_binary_mask_patches[i])
            target_patches.append(img_in_patches[i])
            target_patch_counter+=1

        else:
            exemplar_indices.append((i, start_row, start_col))
            exemplar_patches.append(img_in_patches[i])

    M_bar_patches=[1-binary_mask_patches[i] for i in range(len(binary_mask_patches))]
    new_target_patches=findBestFullPatch(target_patches,exemplar_patches,binary_mask_patches,M_bar_patches)

    new_img= np.copy(img)
    for target_index, start_row, start_col in target_indices:
        print(target_index)
        new_img[start_row: start_row+patch_size, start_col:start_col+patch_size]=new_target_patches[target_index]
    
    return new_img
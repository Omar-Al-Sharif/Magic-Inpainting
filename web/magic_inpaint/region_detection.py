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
import matplotlib
matplotlib.use('agg')  # Set the backend before importing pyplot
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

class RegionDetection:
    def __init__(self, orig_img, selected_img):
        self.orig_img = orig_img
        self.selected_img=selected_img
        self.gray_orig = rgb2gray(self.orig_img)
        self.gray_sel = rgb2gray(self.selected_img)
        self.se_vertical = rectangle(1, 10)
        self.se_horizontal = rectangle(10,1)
        self.se_reg = np.ones((20, 20), np.uint8)
        self.coordinates=[]
        self.binary_mask= np.zeros((orig_img.shape))   # for the next module
        self.region_mask = self.orig_img.copy()   # for the user

    # Fn to find the difference between 2 images to return the rectangle drawn by the user
    def find_difference_between_original_and_selected(self):
        diff = np.abs(self.gray_sel - self.gray_orig)
        result = np.zeros(self.gray_sel.shape)
        result[diff > 0.1] = 1
        return  result
    
    # Applying morphology to the rectangle after the difference, to connect any disconnected lines
    def apply_morphology_to_image(self, diff_img):
        lines_vertical = binary_erosion(diff_img, footprint=self.se_vertical)
        lines_vertical = binary_dilation(lines_vertical, footprint=self.se_vertical)
        lines_vertical = binary_dilation(lines_vertical, footprint=self.se_vertical)
        lines_vertical = binary_dilation(lines_vertical, footprint=self.se_vertical)
        lines_vertical = binary_dilation(lines_vertical, footprint=self.se_vertical)  #additional one


        lines_horizontal = binary_erosion(diff_img, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)
        lines_horizontal = binary_dilation(lines_horizontal, footprint=self.se_horizontal)

        
        connected_lines = lines_vertical | lines_horizontal
        morph_img= connected_lines

        return morph_img
    

     # applying edge detection using Canny to detect the edges of the rectangle in the image, and apply hough transform later
    def apply_canny_edge_detection(self, morph_img):
        canny_img=canny(morph_img, sigma=1)
        return canny_img


    # Applying hough transform to detect the vertical and horizontal lines of the rectangle after aplying edge detection
    # It returns the coordinates of the 4 intersection points of the rectangle
    def apply_hough_transform(self, canny_image):

        # Classic straight-line Hough transform
        # Set a precision of 0.5 degree.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(canny_image, theta=tested_angles)

        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()
        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step),
                np.rad2deg(theta[-1] + angle_step),
                d[-1] + d_step, d[0] - d_step]


        # the vertical lines will have a theta of around 0 or around 180; the horizontal lines will have a theta of around 90.
        # Filter lines based on orientation
        intersection_points = []
        vertical_lines = []
        horizontal_lines = []

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            if np.degrees(angle) % 180 == 0:  # Check if the line is approximately vertical
                vertical_lines.append((x0, y0))
            elif np.degrees(angle) % 90 == 0:  # Check if the line is approximately horizontal
                horizontal_lines.append((x0, y0))
            ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

        # Find intersection points between vertical and horizontal lines
        for v_line in vertical_lines:
            for h_line in horizontal_lines:
                intersection_points.append((v_line[0], h_line[1]))

        self.coordinates = [(int(x), int(y)) for x, y in intersection_points]

        return self.coordinates


     # After knowing the coordinates of the rectangle, we extract a subimage using this coordinates and return it.
    def extract_subimage(self):
        top_left, top_right, bottom_left, bottom_right = self.coordinates
        y1, y2 = bottom_left[1], bottom_right[1]
        x1, x2 = bottom_right[0], top_right[0]
        if(y1>y2 and x1>x2):
            subimage = self.orig_img[y2:y1, x2:x1]
        elif (y2>y1 and x2>x1):
            subimage = self.orig_img[y1:y2, x1:x2]
        elif(y1>y2 and x2>x1):
            subimage = self.orig_img[y2:y1, x1:x2]
        elif (y2>y1 and x1>x2):
             subimage = self.orig_img[y1:y2, x2:x1]
        return subimage
    
    
 
            
    # Applying region growing segmentation based on a seed pixel and seed color 
    # Returning the binary mask as well as the result mask on the original image on size (256,256,3)
    def apply_region_growing(self, subimage):
        #seed_pixel_x, seed_pixel_y, seed_color

       # Get the dimensions of the image
        height, width, channels = subimage.shape

        # consider it the middle pixel for now:  "will be given by user later"
        # Calculate the middle pixel coordinates
        middle_pixel_x = width // 2
        middle_pixel_y = height // 2
        # Get the RGB values of the middle pixel
        middle_pixel_rgb = subimage[middle_pixel_y, middle_pixel_x]

        seed_pixel_x=middle_pixel_x
        seed_pixel_y=middle_pixel_y
        seed_color= middle_pixel_rgb

        # seed point 
        seed= (seed_pixel_x,seed_pixel_y)
   

        # Initialize the region   --- Region Growing part
        region = np.zeros(subimage.shape)
        region[seed[1], seed[0]] = 1

        # paper: https://link.springer.com/referenceworkentry/10.1007/978-0-387-31439-6_450#:~:text=The%20similarity%20measure%20s(x,space%20(or%20color%20model). 
        # similarity criteria based on the color:   "assuming working in RGB space"
        def similarity_criterion(pixel_color, seed_color):
            #Euclidean dist:
            color_difference = np.linalg.norm(pixel_color - seed_color)

            # Check if the color difference is within the acceptable range
            if color_difference <= 110:
                return True
            else:
                return False


        # Iteratively grow the region
        listOfSeedsColors = [seed_color]
        for x in range(int(height/4),int((3*height)/4)):
            for y in range(int(width/4),int((3*width)/4)):
                pixel = subimage[x, y]
                listOfSeedsColors.append(pixel)
                if len(listOfSeedsColors) == 4:
                    break
            if len(listOfSeedsColors) == 4:
             break    
            
                       
    
        for k in range(len(listOfSeedsColors)):
            for i in range(subimage.shape[0]):
                for j in range(subimage.shape[1]):
                    pixel = subimage[i, j]
                    if similarity_criterion(pixel, listOfSeedsColors[k]):
                        region[i, j] = 1
                

        # apply morphology for the region:
        gray_region= rgb2gray(region)
        morh_reg= closing(gray_region,self.se_reg)
        morh_reg= closing(morh_reg,self.se_reg)
        morh_reg= closing(morh_reg,self.se_reg)

        top_left, top_right, bottom_left, bottom_right = self.coordinates
        y1, y2 = bottom_left[1], bottom_right[1]
        x1, x2 = bottom_right[0], top_right[0]

        # Update the pixels in the copy based on the region mask
        if(y1>y2 and x1>x2):
            self.binary_mask[y2:y1, x2:x1][morh_reg > 0] = 1
            self.region_mask[y2:y1, x2:x1][morh_reg > 0] = [255, 0, 0]  # Set pixels in the region to red
        elif (y2>y1 and x2>x1):
            self.binary_mask[y1:y2, x1:x2][morh_reg > 0] = 1
            self.region_mask[y1:y2, x1:x2][morh_reg > 0] = [255, 0, 0]  # Set pixels in the region to red
        elif(y1>y2 and x2>x1):
            self.binary_mask[y2:y1, x1:x2][morh_reg > 0] = 1
            self.region_mask[y2:y1, x1:x2][morh_reg > 0] = [255, 0, 0]  # Set pixels in the region to red
        elif (y2>y1 and x1>x2):
             self.binary_mask[y1:y2, x2:x1][morh_reg > 0] = 1
             self.region_mask[y1:y2, x2:x1][morh_reg > 0] = [255, 0, 0]  # Set pixels in the region to red
             
        

        return self.region_mask, self.binary_mask

    
    def get_mask_by_region_detection(self):
        diff_image=self.find_difference_between_original_and_selected()
        morph=self.apply_morphology_to_image(diff_image)
        canny=self.apply_canny_edge_detection(morph)
        self.apply_hough_transform(canny)
        sub_image=self.extract_subimage()
        self.apply_region_growing(sub_image)
        # self.display_results()
        return self.region_mask, self.binary_mask


class BrushedRegionDetection:
    def __init__(self, img_orig, img_brushed):
        self.img_orig = img_orig
        self.img_brushed = img_brushed
        
    def get_desired_masks(self):
        
        original_image = self.img_orig

        dark_mask = self.create_dark_mask()

        cleaned_dark_mask = self.clean_dark_mask(dark_mask)

        return original_image, dark_mask, cleaned_dark_mask  

    def create_dark_mask(self):
        gray_image = rgb2gray(self.img_brushed[:, :, :3])

        # Define a threshold value based on your requirement (to be adjusted)
        threshold_value = 0

        dark_mask = gray_image <= threshold_value

        return dark_mask

    def clean_dark_mask(self, dark_mask):
        # To be addjusted
        kernel = np.ones((10, 10), np.uint8)
        cleaned_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        return cleaned_mask
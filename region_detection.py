from commonfunctions import *

# input are orig_img and selected img after resizing********** 

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
        self.result_image = None
        self.region_mask = None

    def find_difference_between_original_and_selected(self):
        
        diff = np.abs(self.gray_sel - self.gray_orig)
        result = np.zeros(self.gray_sel)
        result[diff > 0.1] = 1
        return self.gray_sel, self.gray_orig, result
    

    def apply_morphology_to_image(self, diff_img):
        lines_vertical = binary_erosion(diff_img, footprint=self.se_vertical)
        lines_vertical = binary_dilation(lines_vertical, footprint=self.se_vertical)
        lines_vertical = binary_dilation(lines_vertical, footprint=self.se_vertical)
        lines_vertical = binary_dilation(lines_vertical, footprint=self.se_vertical)


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
    

    def apply_canny_edge_detection(self, morph_img):
        canny_img=canny(morph_img, sigma=1)
        return canny_img


    def apply_hough_transform(self, canny_image):

        # Classic straight-line Hough transform
        # Set a precision of 0.5 degree.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(canny_image, theta=tested_angles)

        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()
        ax[0].imshow(canny_image, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step),
                np.rad2deg(theta[-1] + angle_step),
                d[-1] + d_step, d[0] - d_step]

        ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(canny_image, cmap=cm.gray)
        ax[2].set_ylim((canny_image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')


        # the vertical lines will have a theta of around 0 or around 180; the horizontal lines will have a theta of around 90.
        # Filter lines based on orientation
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

        plt.tight_layout()
        plt.show()
        print(intersection_points)

        self.coordinates = [(int(x), int(y)) for x, y in intersection_points]

        return self.coordinates



    def extract_subimage(self):
        #top_left , top_right, bottom_left, bottom_right= coordinates
        # y1=bottom_left[1]  #72
        # y2=bottom_right[1] #216
        # x1=bottom_right[0] #51
        # x2=top_right[0]  #221
        top_left, top_right, bottom_left, bottom_right = self.coordinates
        y1, y2 = bottom_left[1], bottom_right[1]
        x1, x2 = bottom_right[0], top_right[0]
        subimage = self.orig_img[y1:y2, x1:x2]
        #subimage = our_img[ y1:y2 , x1:x2]
        return subimage
    
    
 
            

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
            if color_difference <= 150:
                return True
            else:
                return False


        # Iteratively grow the region
        for i in range(subimage.shape[0]):
            for j in range(subimage.shape[1]):
                pixel = subimage[i, j]
                if similarity_criterion(pixel, seed_color):
                    region[i, j] = 1
               

        # apply morphology for the region:
        gray_region= rgb2gray(region)
        morh_reg= closing(gray_region,self.se_reg)
        morh_reg= closing(morh_reg,self.se_reg)
        morh_reg= closing(morh_reg,self.se_reg)

        # Create a copy of the original image
        self.result_image = self.orig_img.copy()

        top_left, top_right, bottom_left, bottom_right = self.coordinates
        y1, y2 = bottom_left[1], bottom_right[1]
        x1, x2 = bottom_right[0], top_right[0]

        # Update the pixels in the copy based on the region mask
        self.result_image[y1:y2, x1:x2][morh_reg > 0] = [255, 0, 0]  # Set pixels in the region to red

        return self.result_image


    def display_results(self):
        show_images(images=[self.orig_img, self.result_image, self.region_mask])

    
    def get_mask_by_region_detection(self):
        diff_image=self.find_difference_between_original_and_selected()
        morph=self.apply_morphology_to_image(diff_image)
        canny=self.apply_canny_edge_detection(morph)
        self.apply_hough_transform(canny)
        sub_image=self.extract_subimage()
        self.region_mask=self.apply_region_growing(sub_image)
        self.display_results()
        return self.region_mask


# Example usage:

cow_select=io.imread('images-to-be-tested/cow_with_selection.png')
garb_select=io.imread('images-to-be-tested/garbage_with_selection.png')

cow_orig=io.imread('images-to-be-tested/cow.jpg')
garb_orig=io.imread('images-to-be-tested/garbage.jpg')

# Assuming 'cow_select' and 'cow_orig' are already defined
target_img_size = (256, 256)

cow_sel_resize = cv2.resize(cow_select, target_img_size)
cow_sel_conversion = rgba2rgb(cow_sel_resize)
cow_resized_orig = cv2.resize(cow_orig, target_img_size)


# Create an instance of the RegionDetection class
region_detector = RegionDetection(cow_resized_orig, cow_sel_conversion)
region_detector.get_mask_by_region_detection()



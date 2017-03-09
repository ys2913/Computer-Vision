import Answer1 as a1
import Answer2 as a2
import Answer3 as a3
import Answer4 as a4

"""
Python Version: 2.7.12
Packages installed:
OpenCV 2.4.11
numpy 1.10.4
scipy 0.17.1
pylab 
mpl_toolkits
PIL
"""

# Input Parameters for Solution 1
IMAGE_1 = 'assignment1/scene.pgm'  # Image to be read
BLURRED_IMAGE_1 = 'ans_1_c_blurred.png'    # Blurred image to be stored
KERNEL_SIZE = input("Enter the Gaussian Kernel Size: ")

# Input Parameters for Solution 2
RANSAC_ITERATIONS = 100
MAX_PIXEL_DIST = 10
MIN_MATCH_COUNT = 10
LOWE_RATIO = 0.9
IMAGE1 = 'assignment1/scene.pgm'  # Location for input image
IMAGE2 = 'assignment1/book.pgm'  # Location for input image
SIFT_IMG1 = 'ans_2_sift_scene.png'  # SIFT features for scene
SIFT_IMG2 = 'ans_2_sift_book.png'  # SIFT features for book
IMG_AFFINE = "ans_2_scene_transformed.jpg"  # Transformed scene image output
MATCHES_IMAGE_BEFORE_RANSAC = 'ans_2_matches_before_RANSAC.jpg'  # Good Matches before RANSAC
MATCHES_IMAGE_AFTER_RANSAC = 'ans_2_matches_after_RANSAC.jpg'  # Good Matches after RANSAC

# Input Parameters for Solution 3
WORLD_FILE = 'assignment1/world.txt'
IMAGE_FILE = 'assignment1/image.txt'

# Input Parameters for Solution 4
SFM_POINTS = 'assignment1/sfm_points.mat'  # location for reading the input
PLOT3_D = 'ans_4_plot.png'  # location for storing the plot




a1.gaussianblur(IMAGE_1, KERNEL_SIZE, BLURRED_IMAGE_1)

a2.findMatchesAllign(IMAGE1, IMAGE2, SIFT_IMG1, SIFT_IMG2, RANSAC_ITERATIONS, MAX_PIXEL_DIST, LOWE_RATIO, IMG_AFFINE,
                  MATCHES_IMAGE_BEFORE_RANSAC, MATCHES_IMAGE_AFTER_RANSAC)

a3.camparam(WORLD_FILE, IMAGE_FILE)
a4.struMotion(SFM_POINTS, PLOT3_D)

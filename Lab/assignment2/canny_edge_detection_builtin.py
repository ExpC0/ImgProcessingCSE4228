import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_canny(image_path, low_threshold, high_threshold):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise and help Canny detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Display the original and the Canny edge-detected images
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])

    plt.show()

# Specify the path to your image
image_path = 'Lena.jpg'

# Set the low and high thresholds for Canny edge detection
low_threshold = 1
high_threshold = 150

# Apply Canny edge detection
apply_canny(image_path, low_threshold, high_threshold)

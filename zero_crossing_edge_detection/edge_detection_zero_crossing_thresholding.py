import cv2
import numpy as np
import matplotlib.pyplot as plt


def zero_crossing_edge_detection(image, sigma):
    """
    Edge detection using zero-crossing method with Laplacian of Gaussian.

    :param image: Input image.
    :param sigma: Standard deviation for Gaussian kernel.
    :return: Edge-detected image.
    """
    # Step 1: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Step 2: Compute Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Step 3: Zero Crossing Detection
    edge_image = np.zeros_like(laplacian)
    rows, cols = laplacian.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = laplacian[i - 1:i + 2, j - 1:j + 2]
            if (patch.min() < 0 < patch.max()):
                edge_image[i, j] = 255

    return edge_image


def gaussian_magnitude_global_threshold(image, low_threshold, high_threshold):
    """
    Edge detection using Gaussian magnitudes and global thresholding (Canny).

    :param image: Input image.
    :param low_threshold: Low threshold for Canny edge detection.
    :param high_threshold: High threshold for Canny edge detection.
    :return: Edge-detected image.
    """
    # Apply Canny edge detector
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


# Load the image
image = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("The image file 'Lena.jpg' was not found.")

# Parameters for edge detection
sigma = 1.0
low_threshold = 50
high_threshold = 150

# Perform edge detection using both methods
edges_zero_crossing = zero_crossing_edge_detection(image, sigma)
edges_gaussian_threshold = gaussian_magnitude_global_threshold(image, low_threshold, high_threshold)

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(132)
plt.title('Zero Crossing Edges')
plt.imshow(edges_zero_crossing, cmap='gray')
plt.axis('off')

plt.subplot(133)
plt.title('applying gaussian magnitudes and Global Thresholding')
plt.imshow(edges_gaussian_threshold, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('edge_detection_results.png')
plt.close()

print("Edge detection results saved to 'edge_detection_results.png'")

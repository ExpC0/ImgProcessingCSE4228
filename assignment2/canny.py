import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detection(image_path, sigma, low_threshold, high_threshold):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)

    # Step 2: Compute gradients along x and y axes
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Step 3: Non-maximum Suppression
    suppressed = np.zeros_like(gradient_magnitude)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = gradient_direction[i, j]
            if (0 <= angle < np.pi / 4) or (7 * np.pi / 4 <= angle <= 2 * np.pi):
                neighbors = [gradient_magnitude[i, j-1], gradient_magnitude[i, j], gradient_magnitude[i, j+1]]
            elif (np.pi / 4 <= angle < 3 * np.pi / 4):
                neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i, j], gradient_magnitude[i+1, j-1]]
            elif (3 * np.pi / 4 <= angle < 5 * np.pi / 4):
                neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i, j], gradient_magnitude[i+1, j]]
            else:
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i, j], gradient_magnitude[i+1, j+1]]

            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]

    # Step 4: Hysteresis Thresholding
    strong_edges = (suppressed > high_threshold)
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)

    edges = np.zeros_like(image)
    edges[strong_edges] = 255

    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if weak_edges[i, j]:
                eight_neighbors = edges[i-1:i+2, j-1:j+2]
                if np.any(eight_neighbors == 255):
                    edges[i, j] = 255

    # Display the original and the Canny edge-detected images
    plt.subplot(131), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(suppressed, cmap='gray')
    plt.title('Non-Maximum Suppression'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])

    plt.show()

# Specify the path to your image
image_path = 'img.png'

# Set the parameters
sigma = float(input("Enter the value of sigma for Gaussian blur: "))
low_threshold = float(input("Enter the low threshold for hysteresis: "))
high_threshold = float(input("Enter the high threshold for hysteresis: "))

# Apply Canny edge detection
canny_edge_detection(image_path, sigma, low_threshold, high_threshold)

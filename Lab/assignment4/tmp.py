import cv2
import numpy as np

# Read the input image
input_image = cv2.imread("Lena.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection to detect strong edges
edges = cv2.Canny(input_image, 100, 200)

# Apply morphological operations to ensure edge connectivity
kernel = np.ones((5,5), np.uint8)
edges_connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Save the output image
cv2.imwrite("output_img.jpg", edges_connected)

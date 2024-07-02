import numpy as np
import cv2

# Load the RGB image
img_rgb = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)

# Split the RGB image into its color channels
b = img_rgb[:, :, 0]  # Blue channel
g = img_rgb[:, :, 1]  # Green channel
r = img_rgb[:, :, 2]  # Red channel

img_blue = np.zeros_like(img_rgb)
img_blue[:, :, 0] = b

img_green = np.zeros_like(img_rgb)
img_green[:, :, 1] = g

img_red = np.zeros_like(img_rgb)
img_red[:, :, 2] = r

# Create an all-zero matrix with the same shape as the original image
img_combined = np.zeros_like(img_rgb)

# Assign each channel to the corresponding color in the combined image
img_combined[:, :, 0] = img_blue[:, :, 0]
img_combined[:, :, 1] = img_green[:, :, 1]
img_combined[:, :, 2] = img_red[:, :, 2]

# Display or save the combined image
cv2.imshow('Combined Image', img_combined)
cv2.waitKey(0)

# Load an image in BGR format
img_bgr = cv2.imread('Lena.jpg')

# Convert BGR to HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv",img_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.destroyAllWindows()

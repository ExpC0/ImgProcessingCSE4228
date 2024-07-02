import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import apply_hysteresis_threshold

def hysteresis_threshold(img, low_threshold, high_threshold):
    edges = apply_hysteresis_threshold(img, low_threshold, high_threshold)
    return edges

# Load the non-maximum suppressed image (replace 'path/to/your/non_max_suppressed.npy' with the actual path)
non_max_suppressed_image = np.load('non_max_suppressed_img.npy')

# Set the thresholds
low_threshold = 0.1  # Adjust as needed
high_threshold = 12  # Adjust as needed

# Perform hysteresis thresholding
edges_hysteresis = hysteresis_threshold(non_max_suppressed_image, low_threshold, high_threshold)

# Display the results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(edges_hysteresis, cmap='gray')
plt.title('e')
plt.show()
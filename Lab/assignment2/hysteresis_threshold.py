import numpy as np
import matplotlib.pyplot as plt

# Constants
LOW_RATIO = 0.4
HIGH_RATIO = 1.2
STRONG_PIXEL = 2
WEAK_PIXEL = 0
INTERMEDIATE_PIXEL = 1
BACKGROUND_PIXEL = 0
# Function for DFS to include weak pixels connected to a chain of strong pixels
def include_weak_pixels(img):
    for i in range(1, int(img.shape[0] - 1)):
        for j in range(1, int(img.shape[1] - 1)):
            if img[i, j] == 1:
                t_max = max(img[i-1, j-1], img[i-1, j], img[i-1, j+1], img[i, j-1],
                            img[i, j+1], img[i+1, j-1], img[i+1, j], img[i+1, j+1])
                if t_max == STRONG_PIXEL:
                    img[i, j] = STRONG_PIXEL

# Hysteresis Thresholding
def apply_hysteresis_threshold(img, low_threshold=None, high_threshold=None):
    diff = np.max(img) - np.min(img)
    t_low = low_threshold if low_threshold is not None else np.min(img) + LOW_RATIO * diff
    t_high = high_threshold if high_threshold is not None else np.min(img) + HIGH_RATIO * diff

    result_img = np.copy(img)

    # Assign values to pixels
    for i in range(1, int(img.shape[0] - 1)):
        for j in range(1, int(img.shape[1] - 1)):
            # Strong pixels
            if img[i, j] > t_high:
                result_img[i, j] = STRONG_PIXEL
            # Weak pixels
            elif img[i, j] < t_low:
                result_img[i, j] = WEAK_PIXEL
            # Intermediate pixels
            else:
                result_img[i, j] = INTERMEDIATE_PIXEL

    # Include weak pixels that are connected to a chain of strong pixels
    total_strong = np.sum(result_img == STRONG_PIXEL)
    while True:
        include_weak_pixels(result_img)
        if total_strong == np.sum(result_img == STRONG_PIXEL):
            break
        total_strong = np.sum(result_img == STRONG_PIXEL)

    # Remove weak pixels
    for i in range(1, int(result_img.shape[0] - 1)):
        for j in range(1, int(result_img.shape[1] - 1)):
            if result_img[i, j] == INTERMEDIATE_PIXEL:
                result_img[i, j] = BACKGROUND_PIXEL

    # Normalize the output
    result_img = result_img / np.max(result_img)
    return result_img



# Load the non-maximum suppressed image (replace 'path/to/your/non_max_suppressed.npy' with the actual path)
non_max_suppressed_image = np.load('non_max_suppressed_img.npy')

# Set user-defined low and high threshold values (adjust as needed)
user_low_threshold = 0.1
user_high_threshold = 15

# Perform hysteresis thresholding with user-defined thresholds
edges_hysteresis = apply_hysteresis_threshold(non_max_suppressed_image, user_low_threshold, user_high_threshold)

# Display the results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(non_max_suppressed_image, cmap='gray')
plt.title('Non-Maximum Suppressed Image')

plt.subplot(132)
plt.imshow(edges_hysteresis, cmap='gray')
plt.title('Edges After Hysteresis Thresholding')

plt.show()

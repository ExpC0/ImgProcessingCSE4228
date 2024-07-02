import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    suppressed_image = np.zeros_like(gradient_magnitude)

    # Quantize the gradient direction into four main orientations
    quantized_direction = np.round(gradient_direction / (np.pi / 4)) % 4

    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            mag = gradient_magnitude[i, j]

            # Check the neighbor pixels in the direction of the gradient
            if quantized_direction[i, j] == 0:  # 0 degrees
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif quantized_direction[i, j] == 1:  # 45 degrees
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
            elif quantized_direction[i, j] == 2:  # 90 degrees
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            else:  # 135 degrees
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]

            # Suppress non-maximum values
            if all(mag >= neighbor for neighbor in neighbors):
                suppressed_image[i, j] = mag

    return suppressed_image

# Load the gradient magnitude image (replace 'path/to/your/gradient_magnitude.npy' with the actual path)
gradient_magnitude = np.load('gradient_magnitude.npy')

convolved_y_derivative = np.load('convolved_y_derivative.npy')
convolved_x_derivative = np.load('convolved_x_derivative.npy')
# Compute gradient direction using arctangent
gradient_direction = np.arctan2(convolved_y_derivative,convolved_x_derivative)

# Perform non-maximum suppression
suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
np.save('non_max_suppressed_img',suppressed_image)
# Display the results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')

plt.subplot(132)
plt.imshow(gradient_direction, cmap='hsv')
plt.title('Gradient Direction')

plt.subplot(133)
plt.imshow(suppressed_image, cmap='gray')
plt.title('Non-Maximum Suppression')

plt.show()

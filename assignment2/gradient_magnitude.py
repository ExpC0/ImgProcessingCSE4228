import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load convolved x and y images from .npy files
loaded_convolved_x = np.load('convolved_x_derivative.npy')
loaded_convolved_y = np.load('convolved_y_derivative.npy')

# Compute gradient magnitude
gradient_magnitude = np.sqrt(loaded_convolved_x**2 + loaded_convolved_y**2)
np.save('gradient_magnitude.npy', gradient_magnitude)

gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)


# Display the results
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.show()

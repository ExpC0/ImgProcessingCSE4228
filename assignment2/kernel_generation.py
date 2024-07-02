import numpy as np
from matplotlib import pyplot as plt

# Function to create Gaussian filter
def gaussian_filter_creation(sigma, size_x, size_y, center=(0, 0)):
    sd = sigma * sigma
    sqr_sigma = sigma * sigma

    # Calculate kernel size
    kernel_size = int(7 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Sum is for normalization
    sum_value = 0.0

    # Generating kernel
    GKernel = np.zeros((kernel_size, kernel_size))
    range_a = int(kernel_size / 2)
    range_b = range_a + 1

    for x in range(-range_a, range_b):
        for y in range(-range_a, range_b):
            GKernel[x + range_a][y + range_a] = np.exp(
                (-0.5) * ((x - center[0]) ** 2 / sqr_sigma + (y - center[1]) ** 2 / sqr_sigma)
            ) / (2 * np.pi * sd)
            sum_value += GKernel[x + range_a][y + range_a]

    # # Normalizing the kernel
    # GKernel /= sum_value
    # GKernel /= np.min(GKernel)
    # GKernel = GKernel.astype(int)

    return GKernel

# Function to calculate x-derivative of Gaussian filter
def x_derivative_gaussian(GKernel):
    return np.gradient(GKernel, axis=1)

# Function to calculate y-derivative of Gaussian filter
def y_derivative_gaussian(GKernel):
    return np.gradient(GKernel, axis=0)

# Take sigma as input from the user
sigma = float(input("Enter the value of sigma: "))

# Create Gaussian kernel and derivatives
gaussian_kernel = gaussian_filter_creation(sigma, 0, 0)
x_derivative = x_derivative_gaussian(gaussian_kernel)
y_derivative = y_derivative_gaussian(gaussian_kernel)

# Save x-derivative and y-derivative kernels to .npy files
np.save('x_derivative_kernel.npy', x_derivative)
np.save('y_derivative_kernel.npy', y_derivative)

# Display the results
plt.subplot(131)
plt.imshow(gaussian_kernel, cmap='gray')
plt.title('Original Gaussian Kernel')

plt.subplot(132)
plt.imshow(x_derivative, cmap='gray')
plt.title('X-Derivative of Gaussian Kernel')

plt.subplot(133)
plt.imshow(y_derivative, cmap='gray')
plt.title('Y-Derivative of Gaussian Kernel')

plt.show()

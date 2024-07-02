import matplotlib
import numpy as np
import random

from matplotlib import pyplot as plt

# Function to create Gaussian filter
def gaussian_filter_creation(sigma1, sigma2, size_x, size_y, center=(0, 0)):
    sd = sigma1 * sigma2
    sqr_sigma1 = sigma1 * sigma1
    sqr_sigma2 = sigma2 * sigma2

    # Sum is for normalization
    sum_value = 0.0

    # Generating kernel
    GKernel = np.zeros((size_x, size_y))
    range_a = int(size_x / 2)
    range_b = range_a + 1

    for x in range(-range_a, range_b):
        for y in range(-range_a, range_b):
            GKernel[x + range_a][y + range_a] = np.exp(
                (-0.5) * ((x - center[0]) ** 2 / sqr_sigma1 + (y - center[1]) ** 2 / sqr_sigma2)
            ) / (2 * np.pi * sd)
            sum_value += GKernel[x + range_a][y + range_a]

    # Normalizing the kernel
    GKernel /= sum_value
    GKernel /= np.min(GKernel)
    GKernel = GKernel.astype(int)

    return GKernel

def MeanFilterCreation(size_x, size_y):
    output = np.zeros((size_x, size_y))

    range_a = int(size_x / 2)
    range_b = range_a + 1

    for x in range(-range_a, range_b):
        for y in range(-range_a, range_b):
            output[x + range_a, y + range_a] = 1.0

    return output / (size_x * size_y)

size_x = 5
size_y = 5

# Gaussian Kernel with custom center
center = (0,0)
sigma1 = 1.1
sigma2 = 1.2
Gkernel = gaussian_filter_creation(sigma1, sigma2, size_x, size_y, center)
print(Gkernel)
np.save("gaussian_blur_kernel.npy", Gkernel)

# Mean Kernel
Mkernel = MeanFilterCreation(size_x, size_y)
Mkernel /= np.min(Mkernel)
Mkernel = Mkernel.astype(int)
print(Mkernel)
np.save("mean_kernel.npy", Mkernel)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Gaussian kernel plot
axs[0].imshow(Gkernel, cmap='viridis', interpolation='nearest')
axs[0].set_title('Gaussian Kernel')
axs[0].axis('off')

# Mean kernel plot
axs[1].imshow(Mkernel, cmap='viridis', interpolation='nearest')
axs[1].set_title('Mean Kernel')
axs[1].axis('off')

plt.show()

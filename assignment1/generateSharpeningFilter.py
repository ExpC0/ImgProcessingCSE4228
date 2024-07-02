import numpy as np
import matplotlib.pyplot as plt

def laplacian_filter_creation(size_x, size_y, center=(0, 0)):
    # Generate a Laplacian kernel
    sum_value = 0.0

    Lkernel = np.zeros((size_x, size_y))
    range_a = int(size_x / 2)
    range_b = range_a + 1

    for x in range(-range_a, range_b):
        for y in range(-range_a, range_b):
            if (x == center[0] and y == center[1]):
                Lkernel[x + range_a, y + range_a] = size_x * size_y - 1
            else:
                Lkernel[x + range_a, y + range_a] = -1

    return Lkernel

def LoG_filter_creation(sigma, size_x, size_y, center=(0, 0)):
    sd = sigma * sigma
    sqr_sd = sd * sd

    # Sum is for normalization
    sum_value = 0.0

    # Generating kernel
    LoGkernel = np.zeros((size_x, size_y))
    range_a = int(size_x / 2)
    range_b = range_a + 1

    for x in range(-range_a, range_b):
        for y in range(-range_a, range_b):
            LoGkernel[x + range_a][y + range_a] = np.exp((-0.5) * ((x * x) / sd + (y * y) / sd)) * (
                    1 + (-0.5) * ((x * x) / sd + (y * y) / sd)) / (-np.pi * sqr_sd)
            sum_value += LoGkernel[x + range_a][y + range_a]

    # Normalizing the kernel
    LoGkernel /= sum_value

    return LoGkernel

size_x = 3
size_y = 3
laplacian_kernel_center = laplacian_filter_creation(size_x, size_y, center=(0, 0))
print("Laplacian Kernel:")
print(laplacian_kernel_center)

sigma = 1.4
LoG_kernel_center = LoG_filter_creation(sigma, size_x, size_y, center=(0, 0))
print("\nLoG Kernel:")
print(LoG_kernel_center)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Laplacian kernel plot
axs[0].imshow(laplacian_kernel_center, cmap='viridis', interpolation='nearest')
axs[0].set_title('Laplacian Kernel')
axs[0].axis('off')

# LoG kernel plot
axs[1].imshow(LoG_kernel_center, cmap='viridis', interpolation='nearest')
axs[1].set_title('LoG Kernel')
axs[1].axis('off')

plt.show()

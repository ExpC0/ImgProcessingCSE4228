import numpy as np
from matplotlib import pyplot as plt

sobel_kernel_horizontal = np.array(([1,0,-1],
                                    [2,0,-2],
                                    [1,0,-1]))
np.save("sobel_kernel_horizontal",sobel_kernel_horizontal)
sobel_kernel_vertical = np.array(([1,2,1],
                                    [0,0,0],
                                    [-1,-2,-1]))
np.save("sobel_kernel_vertical",sobel_kernel_vertical)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Laplacian kernel plot
axs[0].imshow(sobel_kernel_horizontal, cmap='viridis', interpolation='nearest')
axs[0].set_title('sobel Kernel horizontal')
axs[0].axis('off')

# LoG kernel plot
axs[1].imshow(sobel_kernel_vertical, cmap='viridis', interpolation='nearest')
axs[1].set_title('soble Kernel vetical')
axs[1].axis('off')
plt.show()
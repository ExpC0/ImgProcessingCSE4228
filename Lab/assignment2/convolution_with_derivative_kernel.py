import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolve_image(img, kernel):
    image_h, image_w = img.shape
    kernel_size = kernel.shape[0]
    padding_x = (kernel_size - 1) // 2
    padding_y = (kernel_size - 1) // 2

    img_padded = cv2.copyMakeBorder(img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

    output_image_h = image_h + kernel_size - 1
    output_image_w = image_w + kernel_size - 1

    convolved_output = np.zeros((output_image_h, output_image_w))
    for x in range(image_h):
        for y in range(image_w):
            temp = 0
            for i in range(-padding_x, padding_x + 1):
                for j in range(-padding_y, padding_y + 1):
                    temp += img_padded[x - i, y - j] * kernel[i + padding_x, j + padding_y]
            convolved_output[x, y] = temp

    # convolved_output = cv2.normalize(convolved_output, None, 0, 1, cv2.NORM_MINMAX)
    return convolved_output

def plot_images(img, kernel, convolved_output, title1, title2, title3):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(title1)

    axes[1].imshow(kernel, cmap='gray')
    axes[1].set_title(title2)

    axes[2].imshow(convolved_output, cmap='gray')
    axes[2].set_title(title3)

    plt.show()

# Load y-derivative kernel
kernel_y = np.load('y_derivative_kernel.npy')
kernel_x = np.load('x_derivative_kernel.npy')
# Load the input image (replace 'path/to/your/image.jpg' with the actual image path)
image_path = 'Lena.jpg'
img = plt.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Convolve with x-derivative kernel
convolved_x = convolve_image(img_gray, kernel_x)

# Convolve with y-derivative kernel
convolved_y = convolve_image(img_gray, kernel_y)

np.save('convolved_x_derivative.npy', convolved_x)
np.save('convolved_y_derivative.npy', convolved_y)

# Display the results
plot_images(img_gray, kernel_x, convolved_x, 'Input Image', 'X-Derivative Kernel', 'X-Derivative Convolved Image')
plot_images(img_gray, kernel_y, convolved_y, 'Input Image', 'Y-Derivative Kernel', 'Y-Derivative Convolved Image')

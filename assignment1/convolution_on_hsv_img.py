import numpy as np
import cv2

def convolution_on_each_channel(img, kernel):
    image_h = img.shape[0]
    image_w = img.shape[1]
    kernel_size = kernel.shape[0]
    padding_x = (kernel_size - 1) // 2
    padding_y = (kernel_size - 1) // 2
    img = cv2.copyMakeBorder(img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

    output_image_h = image_h + kernel_size - 1
    output_image_w = image_w + kernel_size - 1

    convolved_output = np.zeros((output_image_h, output_image_w))
    for x in range(image_h):
        for y in range(image_w):
            temp = 0
            for i in range(-padding_x, padding_x + 1):
                for j in range(-padding_y, padding_y + 1):
                    temp += img[x - i, y - j] * kernel[i + padding_x, j + padding_y]
            convolved_output[x, y] = temp
    convolved_output = cv2.normalize(convolved_output, None, 0, 255, cv2.NORM_MINMAX)
    return convolved_output.astype(np.uint8)

# Load the RGB image
img_rgb = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)

# Convert BGR to HSV
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

# Split the HSV image into its channels
h = img_hsv[:, :, 0]  # Hue channel
s = img_hsv[:, :, 1]  # Saturation channel
v = img_hsv[:, :, 2]  # Value channel

# Create an all-zero matrix with the same shape as the original image
img_combined = np.zeros_like(img_rgb)

print("choose kernel: ")
print("1.gaussian")
print("2.laplacian")
print("3.mean")
print("4.LoG")
print("5.sobel_horizontal")
print("6.sobel_vertical")

x = int(input())

if x == 1:
    kernel = np.load('gaussian_blur_kernel.npy')
    label = "gaussian blur (smoothing kernel) "
elif x == 2:
    kernel = np.load('laplacian_kernel.npy')
    label = "laplacian (sharpening kernel) "
elif x == 3:
    kernel = np.load('mean_kernel.npy')
    label = "Mean (smoothing kernel) "
elif x == 4:
    kernel = np.load('LoG_kernel.npy')
    label = "LoG (sharpening kernel) "
elif x == 5:
    kernel = np.load('sobel_kernel_horizontal.npy')
    label = "sobel (horizontal kernel) "
elif x == 6:
    kernel = np.load('sobel_kernel_vertical.npy')
    label = "sobel (vertical kernel) "

# Apply convolution to each channel
conv_img_h = convolution_on_each_channel(h, kernel)
conv_img_s = convolution_on_each_channel(s, kernel)
conv_img_v = convolution_on_each_channel(v, kernel)

# cv2.imshow('Original Hue Channel', h)
# cv2.imshow('Convolved Hue Channel', conv_img_h)
#
# # cv2.imshow('Original Saturation Channel', s)
# cv2.imshow('Convolved Saturation Channel', conv_img_s)
#
# # cv2.imshow('Original Value Channel', v)
# cv2.imshow('Convolved Value Channel', conv_img_v)
# Combine the channels back into an HSV image
convolved_img_hsv = cv2.merge([conv_img_h, conv_img_s, conv_img_v])

np.save("conv_hsv.npy",convolved_img_hsv)

# Display the original and convolved images
cv2.imshow('HSV Image', img_hsv)
cv2.imshow(label+"of conv hsv", convolved_img_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

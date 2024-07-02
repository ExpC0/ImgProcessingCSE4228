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
    convolved_output = cv2.normalize(convolved_output, None, 0, 1, cv2.NORM_MINMAX)
    return convolved_output

# Load the RGB image
img_rgb = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)

# Split the RGB image into its color channels
b = img_rgb[:, :, 0]  # Blue channel
g = img_rgb[:, :, 1]  # Green channel
r = img_rgb[:, :, 2]  # Red channel

img_blue = np.zeros_like(img_rgb)
img_blue[:, :, 0] = b

img_green = np.zeros_like(img_rgb)
img_green[:, :, 1] = g

img_red = np.zeros_like(img_rgb)
img_red[:, :, 2] = r

# Create an all-zero matrix with the same shape as the original image
img_combined = np.zeros_like(img_rgb)


print("choose kernel: ")
print("1.gaussian")
print("2.laplacian")
print("3.mean")
print("4.LoG")
print("5.sobel_horizontal")
print("6.sobel_vertical")

x=int(input())

if(x==1):
    kernel = np.load('gaussian_blur_kernel.npy')
    label="gaussian blur (smothing kernel) "
elif(x==2):
    kernel = np.load('laplacian_kernel.npy')
    label="laplacian (sharpening kernel) "
elif(x==3):
    kernel = np.load('mean_kernel.npy')
    label="Mean (smoothing kernel) "
elif(x==4):
    kernel = np.load('LoG_kernel.npy')
    label="LoG (sharpening kernel) "
elif(x==5):
    kernel = np.load('sobel_kernel_horizontal.npy')
    label="sobel (horizontal kernel) "
elif(x==6):
    kernel = np.load('sobel_kernel_vertical.npy')
    label="sobel (vertical kernel) "


conv_img_red = convolution_on_each_channel(r, kernel)
conv_img_green = convolution_on_each_channel(g, kernel)
conv_img_blue = convolution_on_each_channel(b, kernel)

# Display each channel after convolution
cv2.imshow('Convolved Red Channel', conv_img_red)

cv2.imshow('Convolved Green Channel', conv_img_green)

cv2.imshow('Convolved Blue Channel', conv_img_blue)
# Combine the channels into a color image
convolved_rgb = np.stack([conv_img_blue, conv_img_green, conv_img_red], axis=-1)

np.save("conv_rgb.npy",convolved_rgb)

cv2.imshow('input rgb img', img_rgb)
cv2.imshow(label + "rgb img", convolved_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
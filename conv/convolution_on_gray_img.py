

import numpy as np
import cv2




img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)

image_h = img.shape[0]
image_w = img.shape[1]

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


kernel_size = kernel.shape[0]
padding_x = (kernel_size - 1)//2
padding_y = (kernel_size - 1)//2

img = cv2.copyMakeBorder(img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

output_image_h = image_h + kernel_size - 1
output_image_w = image_w + kernel_size - 1

convolved_output = np.zeros((output_image_h,output_image_w))
for x in range(image_h):
    for y in range(image_w):
        temp = 0
        for i in range(-padding_x, padding_x+1):
            for j in range(-padding_y, padding_y+1):
                temp += img[x-i, y-j]*kernel[i+padding_x,j+padding_y]
        convolved_output[x,y] = temp
convolved_output = cv2.normalize(convolved_output, None, 0, 1, cv2.NORM_MINMAX)

print("processing...  ")

cv2.imshow('input img',img)
cv2.waitKey(0)

cv2.imshow(label, convolved_output)

cv2.waitKey(0)
cv2.destroyAllWindows()

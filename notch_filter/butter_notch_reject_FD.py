import numpy as np
import matplotlib.pyplot as plt
import cv2

def ideal_notch_reject_filter(shape, cutoff, radius, notch_center):
    rows, cols = shape
    u, v = np.meshgrid(np.arange(-cols//2, cols//2), np.arange(-rows//2, rows//2))
    du = u - notch_center[0]
    dv = v - notch_center[1]
    D = np.sqrt(du**2 + dv**2)
    H = np.ones(shape)
    H[D <= radius] = 0
    return H

def butterworth_notch_reject_filter(shape, cutoff, order, notch_center):
    rows, cols = shape
    u, v = np.meshgrid(np.arange(-cols//2, cols//2), np.arange(-rows//2, rows//2))
    du = u - notch_center[0]
    dv = v - notch_center[1]
    D = np.sqrt(du**2 + dv**2)
    H = 1 / (1 + (D / cutoff)**(2 * order))
    return H

def apply_filter(image, filter):
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    filtered_dft = dft_shifted * filter
    dft_inv_shifted = np.fft.ifftshift(filtered_dft)
    filtered_image = np.fft.ifft2(dft_inv_shifted)
    return np.abs(filtered_image)

# Load the image
image = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("The image file 'Lena.jpg' was not found.")

shape = image.shape
cutoff = 10
radius = 5
order = 2
notch_center = (0, 0)  # Center of the notch

ideal_filter = ideal_notch_reject_filter(shape, cutoff, radius, notch_center)
butterworth_filter = butterworth_notch_reject_filter(shape, cutoff, order, notch_center)

filtered_image_ideal = apply_filter(image, ideal_filter)
filtered_image_butterworth = apply_filter(image, butterworth_filter)

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(231)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(232)
plt.title('Ideal Notch Reject Filter')
plt.imshow(np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(ideal_filter)))), cmap='gray')
plt.axis('off')

plt.subplot(233)
plt.title('Butterworth Notch Reject Filter')
plt.imshow(np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(butterworth_filter)))), cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.title('Filtered Image (Ideal)')
plt.imshow(filtered_image_ideal, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.title('Filtered Image (Butterworth)')
plt.imshow(filtered_image_butterworth, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('notch_filter_results.png')
plt.close()

print("Results saved to 'notch_filter_results.png'")

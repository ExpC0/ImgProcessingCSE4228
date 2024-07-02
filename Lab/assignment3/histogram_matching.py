import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_histograms(input_img, target_hist):
    # Convert input image to grayscale
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Compute histogram of input image
    input_hist = cv2.calcHist([input_gray], [0], None, [256], [0, 256])

    # Normalize histograms
    input_hist /= np.sum(input_hist)
    target_hist /= np.sum(target_hist)

    # Compute cumulative distribution functions (CDFs)
    input_cdf = np.cumsum(input_hist)
    target_cdf = np.cumsum(target_hist)

    # Map input image's intensities to target image's intensities
    mp = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            if input_cdf[i] <= target_cdf[j]:
                mp[i] = j
                break

    # Apply mapping to input image manually
    matched_img = np.zeros_like(input_gray)
    for i in range(input_gray.shape[0]):
        for j in range(input_gray.shape[1]):
            matched_img[i, j] = mp[input_gray[i, j]]

    return matched_img, input_hist, target_hist, input_gray, input_cdf, target_cdf


# Load target histogram from .npy file
target_hist = np.load('target_histogram.npy')

# Read input image
input_img = cv2.imread('input2.png')

# Perform histogram matching
matched_img, input_hist, target_hist, input_gray, input_cdf, target_cdf = match_histograms(input_img, target_hist)


plt.figure()
var = np.min(input_hist[np.nonzero(input_hist)])
# Normalize histograms
input_hist /= var
plt.plot(input_hist, color='blue')
plt.fill_between(np.arange(256), input_hist.flatten(), color='blue', alpha=0.3)  # Fill up the area under the curve
plt.title('Input Image Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

# Plot target histogram and CDF
plt.figure()
var = np.min(target_hist[np.nonzero(target_hist)])
target_hist/=var*10
plt.plot(target_hist, color='green')
plt.fill_between(np.arange(256), target_hist.flatten(), color='green', alpha=0.3)  # Fill up the area under the curve
plt.title('Target Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.figure()
plt.plot(target_cdf, color='green')
plt.title('Target CDF')
plt.xlabel('Intensity')
plt.ylabel('Cumulative Frequency')

# Plot input image CDF
plt.figure()
plt.plot(input_cdf, color='blue')
plt.title('Input Image CDF')
plt.xlabel('Intensity')
plt.ylabel('Cumulative Frequency')

# Plot input image and histogram
plt.figure()
plt.imshow(input_img[:, :, ::-1])
plt.title('Input Image')
plt.axis('off')

plt.figure()
plt.imshow(matched_img, cmap='gray')
plt.title('Matched Image')
plt.axis('off')

matched_hist = cv2.calcHist([matched_img], [0], None, [256], [0, 256])
plt.figure()
plt.plot(matched_hist, color='blue')
plt.title('Matched Image Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.show()

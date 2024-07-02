import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_histogram(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Plot histogram
    plt.figure()

    plt.plot(histogram, color='blue')
    plt.fill_between(np.arange(256), histogram.flatten(), color='blue', alpha=0.5)  # Fills area under the curve
    plt.xlim([0, 256])
    plt.show()

# Read image
image_path ='en0.png'
image = cv2.imread(image_path)

# Draw histogram
draw_histogram(image)

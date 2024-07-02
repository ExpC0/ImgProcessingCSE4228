import cv2
import numpy as np

# Load an RGB image
image_path = "Lena.jpg"
rgb_image = cv2.imread(image_path)

if rgb_image is None:
    print(f"Error: Unable to load the image from the path: {image_path}")
    exit()

gaussian_kernel =(1/273) * np.array([[1, 4, 7, 4,1],
                            [4,16,26,16,4],
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1, 4, 7, 4,1]])
# Apply convolution on each channel separately in RGB mode
convolved_rgb = np.zeros_like(rgb_image, dtype=np.float32)
for i in range(3):
    convolved_rgb[:, :, i] = cv2.filter2D(rgb_image[:, :, i], -1, gaussian_kernel)

# Convert RGB image to HSV mode
hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

# Apply convolution on each channel separately in HSV mode
convolved_hsv = np.zeros_like(hsv_image, dtype=np.float32)
for i in range(3):
    convolved_hsv[:, :, i] = cv2.filter2D(hsv_image[:, :, i], -1, gaussian_kernel)

# Convert HSV space back to RGB
convolved_hsv_to_rgb = cv2.cvtColor(convolved_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# Subtract the resulting images obtained from RGB and HSV modes
difference_image = convolved_hsv_to_rgb - convolved_rgb


# Convert back to uint8 for visualization
# difference_image = np.clip(difference_image, 0, 255).astype(np.uint8)

# Display the images
cv2.imshow("RGB Image", rgb_image)
cv2.imshow("Convolved RGB", convolved_rgb.astype(np.uint8))
cv2.imshow("Convolved HSV", convolved_hsv_to_rgb)
cv2.imshow("Difference Image", difference_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

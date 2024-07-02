import cv2
import numpy as np
from math import pi, sqrt
from tabulate import tabulate


def find_largest_dimension(binary_img):
    """
    Find the largest dimension of the binary image region.
    """
    min_x, min_y = 100000, 100000
    max_x, max_y = 0, 0

    h, w = binary_img.shape

    for x in range(h):
        for y in range(w):
            if binary_img[x, y] == 0:
                continue

            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    return max(max_x - min_x, max_y - min_y)


def compute_region_descriptors(binary_img, i):
    """
    Calculate region descriptors for the binary image.
    """
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(binary_img, kernel, iterations=1)
    border_img = binary_img - eroded_img

    area = np.count_nonzero(binary_img)
    perimeter = np.count_nonzero(border_img)
    max_dimension = find_largest_dimension(binary_img)

    cv2.imshow(f'Border {i}', border_img)
    cv2.imshow(f'Input Image {i}', binary_img)

    compactness = perimeter / (area ** 2)
    form_factor = (4 * pi * area) / (perimeter ** 2)
    roundness = (4 * area) / (pi * max_dimension ** 2)

    return compactness, form_factor, roundness


def calculate_distance(desc1, desc2):
    """
    Calculate Euclidean distance between two sets of descriptors.
    """
    distance = sqrt((desc1[0] - desc2[0]) ** 2 +
                    (desc1[1] - desc2[1]) ** 2 +
                    (desc1[2] - desc2[2]) ** 2)
    return distance


def display_distances(distance_matrix):
    """
    Display distance matrix in a tabulated format.
    """
    row_labels = ['c2.jpg', 't2.jpg', 'p2.png', 'st.jpg']
    col_labels = ['c1.jpg', 't1.jpg', 'p1.png']

    distance_matrix = np.array(distance_matrix)
    print(tabulate(distance_matrix, headers=col_labels, showindex=row_labels, tablefmt='grid'))


def save_descriptors_to_file(image_titles, descriptors):
    """
    Save the descriptors to a file.
    """
    file_path = 'output.txt'
    with open(file_path, 'w') as file:
        header = [' ', 'form_factor', 'roundness', 'compactness']
        file.write('\t'.join(header) + '\n')
        file.write('-' * 50 + '\n')

        for i, desc in enumerate(descriptors):
            file.write(image_titles[i] + '\t\t' + '\t\t'.join(map(str, desc)) + '\n')
            file.write('-' * 50 + '\n')


image_files = ['c1.jpg', 't1.jpg', 'p1.png', 'c2.jpg', 't2.jpg', 'p2.png', 'st.jpg']

training_descriptors = []
for i in range(3):
    image_path = f'temp/{image_files[i]}'
    img = cv2.imread(image_path, 0)

    if img is None:
        print(f"Failed to read image from {image_path}")
    else:
        descriptors = compute_region_descriptors(img, i)
        training_descriptors.append(descriptors)

print("Training Descriptors:", training_descriptors)

test_descriptors = []
distance_matrix = []
for i in range(3, len(image_files)):
    image_path = f'temp/{image_files[i]}'
    img = cv2.imread(image_path, 0)

    if img is None:
        print(f"Failed to read image from {image_path}")
    else:
        descriptors = compute_region_descriptors(img, i)
        test_descriptors.append(descriptors)

        distances = []
        for train_desc in training_descriptors:
            distances.append(calculate_distance(train_desc, descriptors))
        distance_matrix.append(distances)

print("Distance Matrix:", distance_matrix)

# Combine training and test descriptors
all_descriptors = training_descriptors + test_descriptors

save_descriptors_to_file(image_files, all_descriptors)
display_distances(distance_matrix)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_and_compute(image, method='ORB'):
    """
    Detect keypoints and compute descriptors using the specified method.

    :param image: Input image.
    :param method: Feature detection method ('ORB', 'SIFT', 'SURF').
    :return: keypoints, descriptors
    """
    if method == 'ORB':
        detector = cv2.ORB_create()
    elif method == 'SIFT':
        detector = cv2.SIFT_create()
    elif method == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    else:
        raise ValueError("Unknown method: {}".format(method))

    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, method='BF'):
    """
    Match descriptors between two images.

    :param desc1: Descriptors from the first image.
    :param desc2: Descriptors from the second image.
    :param method: Matching method ('BF' for Brute Force, 'FLANN' for FLANN based matcher).
    :return: List of matches.
    """
    if method == 'BF':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif method == 'FLANN':
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError("Unknown method: {}".format(method))

    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def draw_matches(image1, kp1, image2, kp2, matches, max_matches=50):
    """
    Draw matches between two images.

    :param image1: First image.
    :param kp1: Keypoints in the first image.
    :param image2: Second image.
    :param kp2: Keypoints in the second image.
    :param matches: List of matches.
    :param max_matches: Maximum number of matches to draw.
    :return: Image with matches drawn.
    """
    matched_image = cv2.drawMatches(image1, kp1, image2, kp2, matches[:max_matches], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image


# Load images
image1 = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)  # For demonstration, use the same image
if image1 is None or image2 is None:
    raise FileNotFoundError("One or both image files were not found.")

# Detect keypoints and compute descriptors
kp1, desc1 = detect_and_compute(image1, method='ORB')
kp2, desc2 = detect_and_compute(image2, method='ORB')

# Match descriptors
matches = match_descriptors(desc1, desc2, method='BF')

# Draw matches
matched_image = draw_matches(image1, kp1, image2, kp2, matches)

# Display the results
plt.figure(figsize=(12, 6))
plt.imshow(matched_image)
plt.title('Feature Matching')
plt.axis('off')
plt.savefig('feature_matching_results.png')
plt.close()

print("Feature matching results saved to 'feature_matching_results.png'")

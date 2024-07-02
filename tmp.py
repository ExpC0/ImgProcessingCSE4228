import os
import numpy as np
from PIL import Image

def resize_and_save_npy_images(folder_path, target_size=(10, 10), save_folder="resized_images"):
    """
    Resize and save all .npy images from a folder to a new folder.

    Args:
    - folder_path (str): Path to the folder containing .npy images.
    - target_size (tuple): Desired size of the images. Default is (10, 10).
    - save_folder (str): Path to the folder where resized images will be saved. Default is "resized_images".
    """
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Get a list of all .npy files in the folder
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

    for file in file_list:
        file_path = os.path.join(folder_path, file)
        image = np.load(file_path)
        # Reshape the image to the desired size
        image_resized = Image.fromarray(image).resize(target_size)
        # Save the resized image to the save folder
        image_resized.save(os.path.join(save_folder, f"{os.path.splitext(file)[0]}.png"))

# Example usage:
folder_path = "encoded_frames"
save_folder = "resized_images2"
resize_and_save_npy_images(folder_path, save_folder=save_folder)

import cv2
import os
import numpy as np

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean, sigma):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

# Path to the folder containing clear images
input_folder = r'C:\Users\anand\Downloads\Denoising_Images_model\train_cleaned'

# folder for the noise images
output_folder = r'C:\Users\anand\Downloads\Denoising_Images_model\train'
os.makedirs(output_folder, exist_ok=True)

# Parameter for Gaussian noise level 75
gaussian_noise_level = 75

# Adding Gaussian noise of level 75 to each image and saving in the respective folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        noisy_image = add_gaussian_noise(image, 0, gaussian_noise_level)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, noisy_image)

        print(f"Noise level {gaussian_noise_level}: {output_path} saved.")

print("Done adding Gaussian noise of level 75 to images.")

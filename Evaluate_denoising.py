import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model1.h5')

# Paths to test data
test_folder = r'C:\Users\anand\Downloads\denoising\test_data'
test_cleaned_folder = r'C:\Users\anand\Downloads\denoising\test_cleaned'

# Function to denoise and evaluate an image
def denoise_and_evaluate(image_path, cleaned_image_path):
    # Read the images
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    image1 = cv2.resize(image, (540, 420))
    image2 = image1 / 255.0
    image3 = np.reshape(image2, (1, 420, 540, 1))
    # Denoise the image
    denoised_image = model.predict(image3)
    denoised_image = np.reshape(denoised_image, (1, 420, 540, 1))
    cleaned_image = cv2.imread(cleaned_image_path, cv2.IMREAD_GRAYSCALE) 
    cleaned_image = cv2.resize(cleaned_image, (540, 420))
    cleaned_image = cleaned_image / 255.0
    cleaned_image = np.reshape(cleaned_image, (1, 420, 540, 1))

    print("Resized image shape:", image.shape)
    print("Cleaned image shape:",cleaned_image.shape)
    diff = np.subtract(cleaned_image, denoised_image)
    # Get the square of the difference
    squared_diff = np.square(diff) 

    # Compute the mean squared error
    mse = np.mean(squared_diff)

    # Compute the PSNR
    max_pixel = 1
    psnr_value = 20 * np.log10(max_pixel) - 10 * np.log10(mse)


    return psnr_value




# Evaluate all images in the test folder
def evaluate_all_images():
    psnr_values = []

    for filename in os.listdir(test_folder):
        image_path = os.path.join(test_folder, filename)
        cleaned_image_path = os.path.join(test_cleaned_folder, filename)

        psnr_val = denoise_and_evaluate(image_path, cleaned_image_path)
        psnr_values.append(psnr_val)
        
        print(f"Image: {filename}, PSNR: {psnr_val:.2f}")

    avg_psnr = np.mean(psnr_values)
    
    print(f"Average PSNR: {avg_psnr:.2f}")
    

# Call the function to evaluate all images
evaluate_all_images()
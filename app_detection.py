from flask import Flask, render_template, request, send_file
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
from UNet_model import UNET
import torchvision.transforms.functional as TF
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
app = Flask(__name__)

# Loading the detection model
detection_model = tf.keras.models.load_model('crack_detection_model.h5')

# Loading the segmentation model
segmentation_model = UNET(in_channels=3, out_channels=1)
segmentation_model.load_state_dict(torch.load("segmentation_model_checkpoint.pth.tar", map_location=device)["state_dict"])

def predict_detection(img_path, model):
    img = load_img(img_path, target_size=(227, 227))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction[0][0]

IMAGE_HEIGHT = 160  
IMAGE_WIDTH = 240 
def predict_segmentation(image_path, model, device):
    # Defining transformation for the test image (resize and normalize)
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    # Applying transformation to the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image).unsqueeze(0).to(device)

    # Performing prediction
    with torch.no_grad():
        model.eval()
        prediction = torch.sigmoid(model(image_tensor)).squeeze(0)
        prediction = (prediction > 0.5).float()

    # Converting the predicted tensor back to a PIL image
    predicted_image = TF.to_pil_image(prediction.cpu())

    return predicted_image

def calculate_average_crack_width(image_path):
    # Loading the image
    image = cv2.imread(image_path, 0)

    # Improving the image by ignoring small white pixels that may not be part of the crack
    # Applying a morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

    # After opening, we need to re-identify the crack by finding contours again
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtering out contours that are too small which are likely not cracks
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # Creating an empty list to store widths
    widths = []

    # Drawing the contours and minimum bounding rectangles around each contour
    for contour in filtered_contours:
        # Geting the minimum bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculating width of the bounding box
        width = np.linalg.norm(box[0] - box[1])  # Euclidean distance between two points
        # Appending width to the list
        widths.append(width)

    # Calculating average width
    if widths:  # Check if list is not empty
        average_width = sum(widths) / len(widths)
        return average_width
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Checking  if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file uploaded')
        
        file = request.files['file']
        
        # If the user does not select a file, browser also
        # submiting an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No file selected')

        # If file exists and is allowed
        if file:
            # Saving the file to a temporary directory
            file_path = os.path.join("temp", file.filename)
            file.save(file_path)

            # Checking if the image contains crack using the detection model
            detection_prediction = predict_detection(file_path, detection_model)
            if detection_prediction >= 0.5:  # Assuming 0.5 as threshold
                segmentation_prediction = predict_segmentation(file_path, segmentation_model, device)
                # Saving the segmented image temporarily
                segmented_image_path = os.path.join("temp", "segmented_image.png")
                segmentation_prediction.save(segmented_image_path)
                crack_width = calculate_average_crack_width(segmented_image_path)

                if crack_width is not None:
                    crack_width_str = f"The average crack width is {crack_width:.2f} pixels."
                else:
                    crack_width_str = "Crack width calculation failed."

                # Rendering the template with the result and file paths
                return render_template('index.html', message='File uploaded successfully', 
                                        result='It is containing crack', 
                                        uploaded_image=file_path,
                                        segmented_image=segmented_image_path,
                                        crack_width=crack_width_str)
            else:
                # If no crack detected, still displaying the uploaded image
                return render_template('index.html', message='File uploaded successfully', 
                                        result='No crack detected',
                                        uploaded_image=file_path)

    return render_template('index.html', message='Please upload a file')




@app.route('/get_segmented_image')
def get_segmented_image():
    segmented_image_path = request.args.get('segmented_image_path')
    return send_file(segmented_image_path, mimetype='image/png')

@app.route('/get_uploaded_image')
def get_uploaded_image():
    uploaded_image_path = request.args.get('uploaded_image_path')
    return send_file(uploaded_image_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
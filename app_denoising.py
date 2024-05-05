from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
model = load_model('model1.h5')

@app.route('/')
def upload_file():
    return render_template('upload.html')

import time

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        # Read the image
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        # Resize image
        image = cv2.resize(image, (540, 420))
        image = image / 255.0
        image = np.reshape(image, (1, 420, 540, 1))
        # Denoise the image
        denoised_image = model.predict(image)
        # Generate unique filename using timestamp
        timestamp = int(time.time())
        filename = f'static/denoised_image_{timestamp}.png'
        # Plot the original and denoised images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(image), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(denoised_image), cmap='gray')
        plt.title('Denoised Image')
        plt.axis('off')

        plt.savefig(filename)  # Save denoised image with unique filename
        plt.close()

        return render_template('result.html', filename=filename)


if __name__ == '__main__':
    app.run(debug=True)

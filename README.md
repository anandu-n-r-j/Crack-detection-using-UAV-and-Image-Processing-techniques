# Crack Detection using UAV and Image Processing Techniques

## Introduction

The **Crack Detection using UAV and Image Processing Techniques** project focuses on utilizing images collected using unmanned aerial vehicles (UAVs) and advanced image processing algorithms to detect cracks in infrastructure such as roads, bridges, and buildings. This repository contains the necessary files and code for implementing crack detection using aerial imagery.

## Features

- **UAV-based Imaging**: Utilizes images collected using UAVs.
- **Image Processing Techniques**: Implements advanced image processing algorithms for crack detection and analysis.
- **Real-time Detection**: Enables real-time detection and analysis of cracks, facilitating timely interventions.

## Environment Setup

### Installation

1. **Clone the Repository**: Clone this repository to your local machine.

2. **Create Virtual Environment**: Navigate to the project directory and create a virtual environment.
   ```bash
   cd project-directory
   python3 -m venv venv
   ```

3. **Activate Virtual Environment**: Activate the virtual environment.
   ```bash
   source venv/bin/activate
   ```

4. **Install Dependencies**: Install required dependencies using the provided requirements file.
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Image Processing

- **Preprocessing**: Preprocess UAV-captured images to enhance quality and optimize for crack detection like denoising the images.
- **Crack Detection**: Detect the crack using CNN model and segment it using U-Net model.
- **Analysis**: Find the width from the segmented image by contouring the image and then finding the distance between the opposite edge.
- **Application**: Uploading the image in the application, then showing the results including detection,width measurement.
## Repository Structure

- `README.md`: Overview and instructions for the project.
- `add_noise_to_images.py`: Adding noise to the images for analysis.
- `app_denoising.py` : Application for denoising the images
- `crack_detection_using_cnn.py`: Implementation of crack detection algorithms.
- `Evaluate_denoising.py`:Evaluate the denoised images using PSNR values.
- `app_detection.py`:Application for crack detection.
- `train_segmentation.py`:Code for segementing the images.
- `finding_width.py`:Finding the width of the crack.
- `requirements.txt`: List of required Python modules for installation.


Feel free to explore and contribute to this project, advancing the field of infrastructure inspection and maintenance through UAV and image processing technologies.

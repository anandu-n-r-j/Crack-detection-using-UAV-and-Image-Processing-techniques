import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from model import UNET

def predict(test_dir, output_dir, model, device):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define transformation for test images (resize and normalize)
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    # Get list of test image file names
    test_images = os.listdir(test_dir)

    # Iterate over test images
    for image_name in test_images:
        # Load test image
        image_path = os.path.join(test_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = test_transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            model.eval()
            prediction = torch.sigmoid(model(image_tensor)).squeeze(0)
            prediction = (prediction > 0.5).float()

        # Save prediction as image
        output_path = os.path.join(output_dir, f"pred_{image_name}")
        TF.to_pil_image(prediction.cpu()).save(output_path)

IMAGE_HEIGHT = 160  
IMAGE_WIDTH = 240  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # Set up paths
    TEST_IMG_DIR = r"C:\Users\anand\OneDrive\Documents\BTP_II\test\Original"
    OUTPUT_DIR = r"C:\Users\anand\OneDrive\Documents\BTP_II\test\Predicted"

    # Load pre-trained model
    model = UNET(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load("my_checkpoint.pth.tar")["state_dict"])
    model.to(DEVICE)

    # Predict images in the test folder
    predict(TEST_IMG_DIR, OUTPUT_DIR, model, device=DEVICE)

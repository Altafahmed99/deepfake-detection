import torch
import cv2
import numpy as np
from torchvision import transforms
from model_utils import DeepfakeModel

# ==========================
# Image Prediction Function
# ==========================
def load_image_model(model_path="models/best_model.pth", device="cpu"):
    """Load the image model"""
    model = DeepfakeModel(is_video=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval().to(device)
    return model


def preprocess_image(image_path):
    """Preprocess image for model input"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict_image(image_path, model=None, device="cpu", threshold=0.5):
    """Run inference on a single image"""
    if model is None:
        model = load_image_model(device=device)

    tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = torch.sigmoid(model(tensor))
        prob = output.item()

    label = "FAKE" if prob >= threshold else "REAL"
    confidence = round(prob * 100, 2)
    print(f"🖼️ Prediction: {label} ({confidence}%)")

    return label, confidence

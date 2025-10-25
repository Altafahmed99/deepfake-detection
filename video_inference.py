import torch
import cv2
import numpy as np
from tqdm import tqdm
from model_utils import DeepfakeModel
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import argparse

# ------------------------------
# Load model
# ------------------------------
def load_model(checkpoint_path, device):
    model = DeepfakeModel(is_video=True).to(device)  # ✅ Use video version
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# ------------------------------
# Frame preprocessing
# ------------------------------
transform = Compose([
    Resize(224, 224),
    Normalize(),
    ToTensorV2()
])

# ------------------------------
# Predict video
# ------------------------------
def predict_video(video_path, model, device, num_frames=16, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    
    for i in tqdm(range(num_frames), desc="Frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = transform(image=frame)["image"]
        frames.append(t)
    
    cap.release()
    
    if len(frames) == 0:
        print("⚠️ No frames found!")
        return None

    x = torch.stack(frames).unsqueeze(0).to(device)  # (1, T, C, H, W)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out).item()
        label = "FAKE" if prob > threshold else "REAL"
        print(f"🎬 Prediction: {label} ({prob:.4f})")
    return label, prob

# ------------------------------
# CLI usage
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="models/best_video_model.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    predict_video(args.video, model, device, args.frames, args.threshold)

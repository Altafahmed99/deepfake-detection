from flask import Flask, render_template, request, send_from_directory
import os
import torch
from model_utils import DeepfakeModel
from inference import predict_image
from video_inference import predict_video

# =======================
# Flask Configuration
# =======================
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =======================
# Model Loader
# =======================
def load_model_safely(model_path, device, is_video=False):
    """
    Safely loads model from a checkpoint or a plain state_dict file.
    Handles missing/unexpected keys gracefully.
    """
    model = DeepfakeModel(is_video=is_video)
    checkpoint = torch.load(model_path, map_location=device)

    # Checkpoint compatibility
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print(f"🧩 Loading 'model_state_dict' from checkpoint: {model_path}")
        state_dict = checkpoint["model_state_dict"]
    else:
        print(f"⚙️ Loading plain 'state_dict': {model_path}")
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Model loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")
    model.eval().to(device)
    return model


# =======================
# Load Models Once
# =======================
print("🔄 Loading models... please wait")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    image_model = load_model_safely("models/best_model.pth", DEVICE)
    video_model = load_model_safely("models/best_video_model.pth", DEVICE, is_video=True)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    image_model, video_model = None, None


# =======================
# Routes
# =======================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files for preview."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/", methods=["GET"])
def index():
    """Main home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handles both image and video prediction uploads."""
    result = None
    confidence = None
    file_url = None

    file = request.files.get("file")
    if not file:
        return render_template("index.html", result="❌ No file uploaded")

    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    file_url = f"/uploads/{file.filename}"

    try:
        # Image Prediction
        if filepath.lower().endswith((".jpg", ".jpeg", ".png")):
            if image_model is None:
                result, confidence = "❌ Image model not loaded", 0
            else:
                result, confidence = predict_image(filepath, image_model, DEVICE)

        # Video Prediction
        elif filepath.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            if video_model is None:
                result, confidence = "❌ Video model not loaded", 0
            else:
                result, confidence = predict_video(filepath, video_model, DEVICE)

        else:
            result = "❌ Unsupported file format"
            confidence = 0

    except Exception as e:
        print(f"⚠️ Error during prediction: {e}")
        result = "❌ Error during prediction"
        confidence = 0

    print(f"DEBUG => File: {file.filename}, Result: {result}, Confidence: {confidence}")

    return render_template(
    "index.html",
    result=result,
    confidence=confidence * 100 if confidence <= 1 else confidence,
    file_url=file_url,
)



# =======================
# Performance Route
# =======================
@app.route("/templates/performance.html", methods=["GET"])
def performance():
    """Displays model performance graphs."""
    # Example dummy data — replace with your actual training logs
    epochs = list(range(1, 11))
    train_acc = [0.7, 0.75, 0.78, 0.82, 0.85, 0.87, 0.9, 0.91, 0.92, 0.93]
    val_acc = [0.68, 0.73, 0.76, 0.8, 0.82, 0.84, 0.88, 0.89, 0.9, 0.91]
    train_loss = [0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.22, 0.2, 0.18]
    val_loss = [0.65, 0.55, 0.45, 0.38, 0.33, 0.3, 0.28, 0.25, 0.23, 0.2]
    fpr = [0, 0.1, 0.2, 0.3, 1.0]
    tpr = [0, 0.6, 0.8, 0.9, 1.0]
    conf_matrix = [[45, 5], [3, 47]]  # Example confusion matrix

    return render_template(
        "performance.html",
        epochs=epochs,
        train_acc=train_acc,
        val_acc=val_acc,
        train_loss=train_loss,
        val_loss=val_loss,
        fpr=fpr,
        tpr=tpr,
        conf_matrix=conf_matrix,
    )


# =======================
# Run Flask
# =======================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

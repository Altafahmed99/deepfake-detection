import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset import get_loaders
from model_utils import DeepfakeModel
import os
import numpy as np

DATA_ROOT = "data"
MODEL_PATH = "models/best_model.pth"
MODEL_NAME = "tf_efficientnet_b3_ns"
BATCH_SIZE = 16
NUM_WORKERS = 4

def load_checkpoint(model):
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def run():
    _, _, test_loader = get_loaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model, device = get_model(model_name=MODEL_NAME, pretrained=False, num_classes=1)
    model = load_checkpoint(model)
    model.to(device)
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten().tolist()
            all_probs += probs
            all_labels += labels.numpy().flatten().tolist()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    preds = [1 if p>0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    cm = confusion_matrix(all_labels, preds)
    print("Test results:")
    print(f"AUC: {auc:.4f} | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)



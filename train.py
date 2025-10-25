# train.py
import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast  # ✅ Updated import for new PyTorch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from dataset import get_loaders
from model_utils import DeepfakeModel

# ======================
# Configuration
# ======================
DATA_ROOT = "data"
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 12
LR = 1e-4
WEIGHT_DECAY = 1e-5
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
SEED = 42


# ======================
# Seeding
# ======================
def seed_everything(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================
# Training Loop
# ======================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    losses, all_probs, all_targets = [], [], []
    pbar = tqdm(loader, desc="Training", leave=False)

    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        # ✅ Fix shape mismatch: squeeze extra dims
        if labels.ndim > 2:
            labels = labels.squeeze(-1)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad()
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten().tolist()
        all_probs += probs
        all_targets += labels.detach().cpu().numpy().flatten().tolist()
        pbar.set_postfix(loss=loss.item())

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
    acc = accuracy_score(all_targets, preds)
    f1 = f1_score(all_targets, preds)
    return sum(losses)/len(losses), auc, acc, f1


# ======================
# Validation Loop
# ======================
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    losses, all_probs, all_targets = [], [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            # ✅ Fix shape mismatch
            if labels.ndim > 2:
                labels = labels.squeeze(-1)
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            losses.append(loss.item())
            probs = torch.sigmoid(outputs).cpu().numpy().flatten().tolist()
            all_probs += probs
            all_targets += labels.cpu().numpy().flatten().tolist()

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
    acc = accuracy_score(all_targets, preds)
    f1 = f1_score(all_targets, preds)
    precision = precision_score(all_targets, preds, zero_division=0)
    recall = recall_score(all_targets, preds, zero_division=0)

    return sum(losses)/len(losses), auc, acc, f1, precision, recall


# ======================
# Main Training
# ======================
def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🧠 Using device: {device}")
    train_loader, val_loader, _ = get_loaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # ✅ Initialize model for image-only training
    model = DeepfakeModel(is_video=False).to(device)
    print("✅ Model initialized successfully")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(device_type='cuda' if torch.cuda.is_available() else 'cpu')

    best_auc = 0.0
    print("\n🚀 Starting Training Loop...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n📅 Epoch {epoch}/{EPOCHS}")
        train_loss, train_auc, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"🟩 Train | Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

        val_loss, val_auc, val_acc, val_f1, val_prec, val_rec = validate_one_epoch(model, val_loader, criterion, device)
        print(f"🟦 Val   | Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | "
              f"F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

        scheduler.step()

        # ✅ Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc
            }, BEST_MODEL_PATH)
            print(f"💾 Saved new best model (AUC={val_auc:.4f}) -> {BEST_MODEL_PATH}")

    print("\n🏁 Training completed! Best Validation AUC:", best_auc)


# ======================
# Entry Point
# ======================
if __name__ == "__main__":
    main()

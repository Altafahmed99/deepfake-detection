import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from dataset import get_loaders
from model_utils import DeepfakeModel

# ==============================
# CONFIG
# ==============================
DATA_ROOT = "frames_split"
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_FRAMES = 16
PATIENCE = 5   # early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n📁 Data root: {DATA_ROOT}")
print(f"🧠 Using device: {DEVICE}\n")

SAVE_PATH = "models/best_video_model.pth"
os.makedirs("models", exist_ok=True)


# ==============================
# TRAINING FUNCTION
# ==============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x).squeeze(1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.sigmoid(out).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds_bin = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_bin)
    f1 = f1_score(all_labels, preds_bin)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, acc, f1, auc


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(1)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds_bin = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_bin)
    f1 = f1_score(all_labels, preds_bin)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, acc, f1, auc


# ==============================
# MAIN FUNCTION
# ==============================
def main(data_root=DATA_ROOT, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, max_frames=MAX_FRAMES):
    train_loader, val_loader, _ = get_loaders(data_root, batch_size=batch_size, is_video=True, max_frames=max_frames)

    model = DeepfakeModel(is_video=True, dropout=0.5)
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        tr_loss, tr_acc, tr_f1, tr_auc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, val_auc = validate_one_epoch(model, val_loader, criterion, DEVICE)

        print(f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} F1={tr_f1:.4f} AUC={tr_auc:.4f}")
        print(f"Val   Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f} AUC={val_auc:.4f}")

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Model saved (Best AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered.")
            break

    print("\nTraining finished. Best val AUC:", best_auc)


# ==============================
# CLI ENTRY POINT
# ==============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="frames_split")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=16)
    args = parser.parse_args()

    main(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_frames=args.max_frames
    )

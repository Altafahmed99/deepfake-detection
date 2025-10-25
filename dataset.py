import os
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, max_frames=32, is_video=False, transform=None):
        """
        root_dir: path to train/val/test directory
        is_video: True if using video frames, False for single images
        max_frames: how many frames per video (used only if is_video=True)
        """
        self.samples = []
        self.max_frames = max_frames
        self.is_video = is_video
        self.transform = transform

        for label, cls in enumerate(["real", "fake"]):
            class_dir = os.path.join(root_dir, cls)
            if not os.path.exists(class_dir):
                continue

            # ---- For image dataset ----
            if not is_video:
                images = glob(os.path.join(class_dir, "*.png")) + glob(os.path.join(class_dir, "*.jpg"))
                for img_path in images:
                    self.samples.append((img_path, label))

            # ---- For video dataset (frames per video) ----
            else:
                video_folders = [d for d in glob(os.path.join(class_dir, "*")) if os.path.isdir(d)]
                for vdir in video_folders:
                    frames = sorted(glob(os.path.join(vdir, "*.png")))
                    if len(frames) > 0:
                        self.samples.append((frames, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, label = self.samples[idx]

        # ---- If dataset contains single images ----
        if not self.is_video:
            image = Image.open(item).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor([label], dtype=torch.float32)

        # ---- If dataset contains multiple frames per video ----
        frames = []
        for f in item[:self.max_frames]:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img.unsqueeze(0))

        # pad frames if less than max_frames
        while len(frames) < self.max_frames:
            frames.append(torch.zeros_like(frames[0]))

        video_tensor = torch.cat(frames, dim=0)  # (max_frames, C, H, W)
        return video_tensor, torch.tensor([label], dtype=torch.float32)


# ================== get_loaders() ==================
def get_loaders(root_path, batch_size=8, num_workers=2, max_frames=32, is_video=False):
    """
    root_path should contain train/, val/, test/ subdirectories.
    Automatically loads them for images or videos.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = DeepfakeDataset(os.path.join(root_path, "train"), max_frames=max_frames,
                                    is_video=is_video, transform=transform)
    val_dataset = DeepfakeDataset(os.path.join(root_path, "val"), max_frames=max_frames,
                                  is_video=is_video, transform=transform)
    test_dataset = DeepfakeDataset(os.path.join(root_path, "test"), max_frames=max_frames,
                                   is_video=is_video, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

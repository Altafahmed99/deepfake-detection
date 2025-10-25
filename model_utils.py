# model_utils.py
import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeModel(nn.Module):
    def __init__(self, is_video=False, num_classes=1):
        super(DeepfakeModel, self).__init__()
        self.is_video = is_video

        # Base feature extractor (EfficientNet)
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Identity()
        self.feature_extractor = base_model

        # For videos, we’ll aggregate frame-level features
        if is_video:
            # LSTM to capture temporal dependencies between frames
            self.temporal_model = nn.LSTM(
                input_size=in_features,
                hidden_size=256,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 2, num_classes)
            )
        else:
            # For single images
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        if self.is_video:
            # x: (batch, frames, 3, H, W)
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            feats = self.feature_extractor(x)          # (b*t, in_features)
            feats = feats.view(b, t, -1)               # (b, t, in_features)

            lstm_out, _ = self.temporal_model(feats)   # (b, t, 512)
            out = lstm_out.mean(dim=1)                 # average over frames
            out = self.classifier(out)                 # (b, 1)
        else:
            # Image input
            feats = self.feature_extractor(x)
            out = self.classifier(feats)

        return out

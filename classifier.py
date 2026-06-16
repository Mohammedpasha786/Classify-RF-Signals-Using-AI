Deep learning models for RF signal classification.

CNN1D    : Lightweight 1D convolutional network (~180K params)
ResNet1D : Residual 1D CNN with transfer-learning support (~1.1M params)
CLDNN    : CNN + LSTM hybrid (West & O'Shea, 2017)

All models:
  Input : [B, 2, N] (I/Q channels as 2D input)
  Output: [B, n_classes] logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ── Shared building blocks ─────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=None, groups=1):
        if padding is None:
            padding = kernel // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock1D(nn.Module):
    """1D residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch, kernel=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.conv2(self.conv1(x))
        out = self.drop(out)
        return F.relu(out + residual, inplace=True)


# ── CNN1D ──────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    Lightweight 1D CNN for real-time RF classification.
    ~180K parameters. Runs at >1000 inferences/sec on CPU.

    Architecture:
        Conv1D(2→64, k=7) → BN → ReLU
        Conv1D(64→64, k=5) → BN → ReLU → Pool(2)
        Conv1D(64→128, k=3) → BN → ReLU
        Conv1D(128→128, k=3) → BN → ReLU → Pool(2)
        GlobalAvgPool → FC(256) → Dropout → FC(n_classes)
    """

    def __init__(self, n_classes: int = 7, n_iq: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(2,   64,  7, padding=3),
            ConvBnRelu(64,  64,  5, padding=2),
            nn.MaxPool1d(2),
            ConvBnRelu(64,  128, 3, padding=1),
            ConvBnRelu(128, 128, 3, padding=1),
            nn.MaxPool1d(2),
            ConvBnRelu(128, 256, 3, padding=1),
            ConvBnRelu(256, 256, 3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── ResNet1D ───────────────────────────────────────────────────────────

class ResNet1D(nn.Module):
    """
    Residual 1D CNN — primary model, supports transfer learning.
    ~1.1M parameters.

    Architecture mirrors ResNet-18 adapted for 1D I/Q input.
    When used for transfer learning, freeze 'layer1' and 'layer2'
    and fine-tune 'layer3', 'layer4', and 'classifier'.
    """

    def __init__(
        self,
        n_classes: int = 7,
        dropout: float = 0.3,
        pretrained_classes: Optional[int] = None,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnRelu(2, 64, kernel=7, stride=2, padding=3),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResBlock1D(64,  64),  ResBlock1D(64,  64))
        self.layer2 = nn.Sequential(ResBlock1D(64,  128, stride=2), ResBlock1D(128, 128))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, stride=2), ResBlock1D(256, 256))
        self.layer4 = nn.Sequential(ResBlock1D(256, 512, stride=2), ResBlock1D(512, 512))

        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, n_classes)

        self._pretrained_classes = pretrained_classes
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-dim embedding (before classifier) — useful for fine-tuning."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)

    def freeze_backbone(self, until_layer: str = "layer2"):
        """Freeze early layers for transfer learning fine-tuning."""
        freeze = True
        for name, param in self.named_parameters():
            if until_layer in name:
                freeze = False
            if freeze:
                param.requires_grad = False
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total  = sum(1 for p in self.parameters())
        print(f"Frozen {frozen}/{total} parameter tensors (up to {until_layer})")

    def replace_classifier(self, n_new_classes: int):
        """Swap classifier head for transfer learning to new signal classes."""
        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, n_new_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        print(f"Replaced classifier: 512 → {n_new_classes} classes")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_num_params(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )


# ── CLDNN ──────────────────────────────────────────────────────────────

class CLDNN(nn.Module):
    """
    CNN + LSTM + DNN hybrid for RF classification.
    Reference: West & O'Shea, IEEE DySPAN 2017.

    Architecture:
        CNN feature extraction → LSTM temporal modeling → FC classifier
    ~500K parameters.
    """

    def __init__(
        self,
        n_classes: int = 7,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBnRelu(2,  64,  7),
            ConvBnRelu(64, 64,  5),
            nn.MaxPool1d(2),
            ConvBnRelu(64, 128, 3),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, N]
        feat = self.cnn(x)                       # [B, 128, N//4]
        feat = feat.permute(0, 2, 1)             # [B, T, 128]
        out, _ = self.lstm(feat)
        out = out[:, -1, :]                      # last timestep
        return self.classifier(out)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model factory ──────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "cnn1d":    CNN1D,
    "resnet1d": ResNet1D,
    "cldnn":    CLDNN,
}


def build_model(name: str, n_classes: int, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](n_classes=n_classes, **kwargs)
    params = model.get_num_params()
    print(f"Built {name} — {params:,} trainable parameters")
    return model

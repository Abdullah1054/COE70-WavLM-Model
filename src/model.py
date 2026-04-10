import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Lightweight 2-block CNN for Mel spectrogram inputs."""
    def __init__(self, num_classes: int = 8, dropout: float = 0.25):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

class CNNBiLSTM(nn.Module):
    """
    CNN backbone (same as SmallCNN conv blocks) -> sequence -> BiLSTM -> classifier
    Input: (B, 1, n_mels, T)
    Output: logits (B, num_classes)
    """
    def __init__(self, num_classes: int = 8, dropout: float = 0.25,
                 lstm_hidden: int = 128, lstm_layers: int = 1, bidirectional: bool = True):
        super().__init__()

        # SAME backbone as your baseline SmallCNN
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.dropout = nn.Dropout(dropout)

        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=32 * (64 // 4),   # 32 channels * (n_mels/4). With n_mels=64 -> 64/4 = 16
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.features(x)   # (B, 32, n_mels/4, T/4) -> with n_mels=64, T=256 -> (B,32,16,64)

        # Convert to sequence over time:
        # (B, C, F, T') -> (B, T', C*F)
        x = x.permute(0, 3, 1, 2).contiguous()          # (B, T', C, F)
        x = x.view(x.size(0), x.size(1), -1)            # (B, T', C*F)

        x = self.dropout(x)

        # LSTM over time
        out, _ = self.lstm(x)                           # (B, T', H*dir)

        # Use last time step
        pooled = out.mean(dim=1)   # average over time steps
        logits = self.classifier(pooled)                 # (B, num_classes)
        return logits
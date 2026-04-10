import torch
import torch.nn as nn
from transformers import AutoModel


class WavLMFineTuner(nn.Module):
    """
    WavLM encoder + mean pooling + classification head.
    Input wav: (B, 1, T) -> we squeeze to (B, T)
    Output logits: (B, num_classes)
    """
    def __init__(self, model_name: str, num_classes: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, 1, T) -> (B, T)
        x = wav.squeeze(1)

        out = self.encoder(input_values=x)
        # out.last_hidden_state: (B, time, hidden)

        pooled = out.last_hidden_state.mean(dim=1)  # mean pooling over time
        logits = self.head(pooled)
        return logits

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
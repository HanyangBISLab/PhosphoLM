from pathlib import Path

import torch
from torch import nn


class PhosphoLMPpiClassifier(nn.Module):
    def __init__(self, input_dim=2560, num_labels=2):
        super().__init__()
        self.classifier = nn.Linear(int(input_dim), int(num_labels))

    def forward(self, features):
        return self.classifier(features)


class PtmMambaPpiClassifier(nn.Module):
    def __init__(self, max_length=2048, hidden_dim=None, dropout=0.5):
        super().__init__()
        self.max_length = int(max_length)
        self.hidden_dim = int(hidden_dim or max_length)
        self.fc = nn.Sequential(
            nn.Linear(2 * self.max_length, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, binder, wt, ptm):
        binder_wt = torch.cat((binder, wt), dim=-1)
        binder_ptm = torch.cat((binder, ptm), dim=-1)
        return self.fc(binder_wt - binder_ptm).squeeze(-1)


def build_phospholm_ppi_model():
    return PhosphoLMPpiClassifier()


def build_ptm_mamba_ppi_model(max_length=2048, dropout=0.5):
    return PtmMambaPpiClassifier(max_length=max_length, dropout=dropout)


def _load_state(path):
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state


def load_phospholm_ppi_model(path):
    state = _load_state(path)
    model = build_phospholm_ppi_model()
    model.load_state_dict(state)
    model.eval()
    return model


def load_ptm_mamba_ppi_model(path, dropout=0.5):
    state = _load_state(path)
    first = state.get("fc.0.weight")
    model = PtmMambaPpiClassifier(
        max_length=int(first.shape[1] // 2),
        hidden_dim=int(first.shape[0]),
        dropout=dropout,
    )
    model.load_state_dict(state)
    model.eval()
    return model

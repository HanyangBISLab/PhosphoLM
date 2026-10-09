from dataclasses import dataclass
from pathlib import Path

import joblib
import torch
from torch import nn
from transformers import EsmModel, EsmTokenizer

from phospholm.sequence_dataset import FEATURE_DIMENSIONS
from phospholm.tokenizer.phosphoLM_tokenizer import PTMTokenizer


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = dropout
        layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        if dropout is not None:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(self.hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)


@dataclass
class SequenceModelBundle:
    model: SequenceClassifier


def build_sequence_model(model_family, classifier_class, hidden_dim=256, dropout=0.0):
    dropout = float(dropout) if model_family == "phosphoLM" else None
    return classifier_class(FEATURE_DIMENSIONS[model_family], hidden_dim, dropout)


def build_encoder(checkpoint, model_family):
    local = model_family == "phosphoLM"
    tokenizer_class = PTMTokenizer if local else EsmTokenizer
    tokenizer = tokenizer_class.from_pretrained(checkpoint, local_files_only=local)
    model = EsmModel.from_pretrained(
        checkpoint,
        add_pooling_layer=False,
        local_files_only=local,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False).eval()
    return model, tokenizer


def save_sequence_model(path, bundle):
    payload = {
        "input_dim": bundle.model.input_dim,
        "hidden_dim": bundle.model.hidden_dim,
        "dropout": bundle.model.dropout,
        "state_dict": {
            name: tensor.detach().cpu().numpy()
            for name, tensor in bundle.model.state_dict().items()
        },
    }
    joblib.dump(payload, Path(path))


def load_sequence_model(path, classifier_class, bundle_class):
    payload = joblib.load(Path(path))
    input_dim = int(payload["input_dim"])
    hidden_dim = int(payload["hidden_dim"])
    state = payload["state_dict"]
    uses_dropout = "net.3.weight" in state
    dropout = payload.get("dropout", 0.0) if uses_dropout else None
    model = classifier_class(input_dim, hidden_dim, dropout)
    model.load_state_dict(
        {name: torch.as_tensor(value) for name, value in state.items()}, strict=True
    )
    model.eval()
    return bundle_class(model)

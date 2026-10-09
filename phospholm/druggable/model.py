from dataclasses import dataclass
from pathlib import Path

from phospholm.sequence_model import (
    SequenceClassifier,
    SequenceModelBundle,
    build_encoder,
    build_sequence_model,
    load_sequence_model,
    save_sequence_model,
)


ROOT = Path(__file__).resolve().parents[2]
PHOSPHOLM_CHECKPOINT = ROOT / "models/pretrained_phosphoLM"
ESM2_CHECKPOINT = "facebook/esm2_t33_650M_UR50D"


class DruggableClassifier(SequenceClassifier):
    pass


@dataclass
class DruggableModelBundle(SequenceModelBundle):
    model: DruggableClassifier


def build_druggable_model(model_family, hidden_dim=256, dropout=0.0):
    return build_sequence_model(
        model_family, DruggableClassifier, hidden_dim=hidden_dim, dropout=dropout
    )


def build_phospholm_druggable_model(hidden_dim=256, dropout=0.0):
    return build_druggable_model("phosphoLM", hidden_dim, dropout)


def build_esm2_druggable_model(hidden_dim=256):
    return build_druggable_model("esm2_650M", hidden_dim)


def build_ptm_mamba_druggable_model(hidden_dim=256):
    return build_druggable_model("ptm_mamba", hidden_dim)


def build_phospholm_druggable_encoder(checkpoint=PHOSPHOLM_CHECKPOINT):
    return build_encoder(checkpoint, "phosphoLM")


def build_esm2_druggable_encoder(checkpoint=ESM2_CHECKPOINT):
    return build_encoder(checkpoint, "esm2_650M")


def save_druggable_model(path, bundle):
    save_sequence_model(path, bundle)


def load_druggable_model(path, model_family):
    return load_sequence_model(path, DruggableClassifier, DruggableModelBundle)

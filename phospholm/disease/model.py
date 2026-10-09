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


class DiseaseClassifier(SequenceClassifier):
    pass


@dataclass
class DiseaseModelBundle(SequenceModelBundle):
    model: DiseaseClassifier


def build_disease_model(model_family, hidden_dim=256, dropout=0.0):
    return build_sequence_model(
        model_family, DiseaseClassifier, hidden_dim=hidden_dim, dropout=dropout
    )


def build_phospholm_disease_model(hidden_dim=256, dropout=0.0):
    return build_disease_model("phosphoLM", hidden_dim, dropout)


def build_esm2_disease_model(hidden_dim=256):
    return build_disease_model("esm2_650M", hidden_dim)


def build_ptm_mamba_disease_model(hidden_dim=256):
    return build_disease_model("ptm_mamba", hidden_dim)


def build_phospholm_disease_encoder(checkpoint=PHOSPHOLM_CHECKPOINT):
    return build_encoder(checkpoint, "phosphoLM")


def build_esm2_disease_encoder(checkpoint=ESM2_CHECKPOINT):
    return build_encoder(checkpoint, "esm2_650M")


def save_disease_model(path, bundle):
    save_sequence_model(path, bundle)


def load_disease_model(path, model_family):
    return load_sequence_model(path, DiseaseClassifier, DiseaseModelBundle)

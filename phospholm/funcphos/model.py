from pathlib import Path

from transformers import EsmTokenizer

from phospholm.token_model import build_token_classifier
from phospholm.tokenizer.phosphoLM_tokenizer import PTMTokenizer


ROOT = Path(__file__).resolve().parents[2]
PHOSPHOLM_CHECKPOINT = ROOT / "models/pretrained_phosphoLM"
ESM2_CHECKPOINT = "facebook/esm2_t33_650M_UR50D"
LABELS = ("regular_0", "regular_1")


def build_funcphos_model(checkpoint=PHOSPHOLM_CHECKPOINT):
    return build_token_classifier(checkpoint, PTMTokenizer, LABELS, True)


def build_esm2_funcphos_model(checkpoint=ESM2_CHECKPOINT):
    return build_token_classifier(checkpoint, EsmTokenizer, LABELS, False)

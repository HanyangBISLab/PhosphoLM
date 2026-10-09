#!/usr/bin/env python3
from pathlib import Path

from phospholm.token_inference import TokenInferenceConfig, run_token_inference


ROOT = Path(__file__).resolve().parent
CONFIG = TokenInferenceConfig(
    name="PhosphoLM phosphosite",
    fold_dir=ROOT / "datasets/dbptm/folds",
    model_dir=ROOT / "models/phosphoLM_phosphosite",
    prediction_dir=ROOT / "output/phosphosite/phosphoLM/predictions",
    folds=tuple(range(10)),
    model_family="phosphoLM",
    predict_listed_sites_only=False,
)


if __name__ == "__main__":
    run_token_inference(CONFIG)

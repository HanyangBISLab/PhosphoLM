#!/usr/bin/env python3
from pathlib import Path

from phospholm.token_inference import TokenInferenceConfig, run_token_inference


ROOT = Path(__file__).resolve().parent
CONFIG = TokenInferenceConfig(
    name="PhosphoLM FuncPhos",
    fold_dir=ROOT / "datasets/funcphos/folds",
    model_dir=ROOT / "models/phosphoLM_funcphos",
    prediction_dir=ROOT / "output/funcphos/phosphoLM/predictions",
    folds=tuple(range(5)),
    model_family="phosphoLM",
    predict_listed_sites_only=True,
)


if __name__ == "__main__":
    run_token_inference(CONFIG)

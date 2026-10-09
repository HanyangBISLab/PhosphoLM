#!/usr/bin/env python3
from pathlib import Path

from phospholm.token_inference import TokenInferenceConfig, run_token_inference


ROOT = Path(__file__).resolve().parent
CONFIG = TokenInferenceConfig(
    name="ESM2-650M FuncPhos",
    fold_dir=ROOT / "datasets/funcphos/folds",
    model_dir=ROOT / "models/esm2_650M_funcphos",
    prediction_dir=ROOT / "output/funcphos/esm2_650M/predictions",
    folds=tuple(range(5)),
    model_family="esm2_650M",
    predict_listed_sites_only=True,
)


if __name__ == "__main__":
    run_token_inference(CONFIG)

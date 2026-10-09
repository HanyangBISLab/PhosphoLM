#!/usr/bin/env python3
from pathlib import Path

from phospholm.funcphos.dataset import Esm2FuncPhosDataset
from phospholm.funcphos.model import build_esm2_funcphos_model
from phospholm.token_training import TokenTrainingConfig, train_token_folds


ROOT = Path(__file__).resolve().parent
CONFIG = TokenTrainingConfig(
    name="ESM2-650M FuncPhos",
    fold_dir=ROOT / "datasets/funcphos/folds",
    model_dir=ROOT / "models/esm2_650M_funcphos",
    folds=tuple(range(5)),
    checkpoint="facebook/esm2_t33_650M_UR50D",
    dataset_class=Esm2FuncPhosDataset,
    model_builder=build_esm2_funcphos_model,
    max_epochs=100,
    validation_fraction=0.1,
    early_stopping_patience=10,
    validation_steps=100,
    token_mean_focal_loss=True,
)


if __name__ == "__main__":
    train_token_folds(CONFIG)

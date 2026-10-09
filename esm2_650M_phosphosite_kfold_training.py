#!/usr/bin/env python3
from pathlib import Path

from phospholm.phosphosite.dataset import Esm2PhosphositeDataset
from phospholm.phosphosite.model import build_esm2_model
from phospholm.token_training import TokenTrainingConfig, train_token_folds


ROOT = Path(__file__).resolve().parent
CONFIG = TokenTrainingConfig(
    name="ESM2-650M phosphosite",
    fold_dir=ROOT / "datasets/dbptm/folds",
    model_dir=ROOT / "models/esm2_650M_phosphosite",
    folds=tuple(range(10)),
    checkpoint="facebook/esm2_t33_650M_UR50D",
    dataset_class=Esm2PhosphositeDataset,
    model_builder=build_esm2_model,
    max_epochs=100,
    validation_fraction=0.1,
    early_stopping_patience=10,
    dataloader_num_workers=32,
    learning_rate=1e-4,
    validation_steps=100,
    token_mean_focal_loss=True,
)


if __name__ == "__main__":
    train_token_folds(CONFIG)

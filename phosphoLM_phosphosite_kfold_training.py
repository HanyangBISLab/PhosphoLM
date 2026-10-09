#!/usr/bin/env python3
from pathlib import Path

from phospholm.phosphosite.dataset import PhosphositeDataset
from phospholm.phosphosite.model import build_phosphosite_model
from phospholm.token_training import TokenTrainingConfig, train_token_folds


ROOT = Path(__file__).resolve().parent
CONFIG = TokenTrainingConfig(
    name="PhosphoLM phosphosite",
    fold_dir=ROOT / "datasets/dbptm/folds",
    model_dir=ROOT / "models/phosphoLM_phosphosite",
    folds=tuple(range(10)),
    checkpoint=ROOT / "models/pretrained_phosphoLM",
    dataset_class=PhosphositeDataset,
    model_builder=build_phosphosite_model,
    validation_fraction=0.1,
    early_stopping_patience=10,
    dataloader_num_workers=32,
    learning_rate=1e-4,
    validation_steps=100,
    token_mean_focal_loss=True,
)


if __name__ == "__main__":
    train_token_folds(CONFIG)

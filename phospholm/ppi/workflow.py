import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from phospholm.loss.focalloss import FocalLoss
from phospholm.ppi.dataset import (
    PpiDataset,
    PhosphoLMPpiEmbeddingDataset,
    PtmMambaPpiEmbeddingDataset,
    collate_ptm_mamba_ppi,
)
from phospholm.ppi.model import (
    build_phospholm_ppi_model,
    build_ptm_mamba_ppi_model,
    load_phospholm_ppi_model,
    load_ptm_mamba_ppi_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 3
LINEAR_FAMILIES = ("phosphoLM", "esm2_650M")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device(value):
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    return device


def _parser(family, inference):
    action = "Run" if inference else "Train"
    parser = argparse.ArgumentParser(
        description=f"{action} {family} PPI fold workflow."
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--fold-dir", type=Path, default=PROJECT_ROOT / "datasets" / "ppi" / "folds"
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "ppi" / "embeddings" / family,
    )
    parser.add_argument(
        "--model-root", type=Path, default=PROJECT_ROOT / "models" / f"{family}_ppi"
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "ppi" / family / "predictions",
    )
    parser.add_argument("--device", default="auto")
    if family == "phosphoLM":
        parser.add_argument(
            "--checkpoint",
            type=Path,
            default=PROJECT_ROOT / "models" / "pretrained_phosphoLM",
            help="Retained for CLI compatibility; training and inference use cached features.",
        )
    if not inference:
        parser.add_argument("--seed", type=int, default=42)
    return parser


def _paths(args, family, fold):
    suffix = "test_embeddings.npz" if family in LINEAR_FAMILIES else f"fold_{fold}_test"
    train_suffix = (
        "train_embeddings.npz" if family in LINEAR_FAMILIES else f"fold_{fold}_train"
    )
    if family in LINEAR_FAMILIES:
        test_cache = args.embedding_root / f"fold_{fold}" / suffix
        train_cache = args.embedding_root / f"fold_{fold}" / train_suffix
    else:
        test_cache = args.embedding_root / suffix
        train_cache = args.embedding_root / train_suffix
    return {
        "train_csv": args.fold_dir / f"fold_{fold}_train.csv",
        "test_csv": args.fold_dir / f"fold_{fold}_test.csv",
        "train_cache": train_cache,
        "test_cache": test_cache,
        "model": args.model_root / f"fold_{fold}" / "model.pt",
        "prediction": args.prediction_root / f"fold_{fold}_test.csv",
    }


def _embedding_dataset(family, path):
    dataset_class = (
        PhosphoLMPpiEmbeddingDataset
        if family in LINEAR_FAMILIES
        else PtmMambaPpiEmbeddingDataset
    )
    return dataset_class(path)


def _save_state(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        path,
    )


def _write_predictions(path, source, probabilities, preserve_source_columns=False):
    output = (
        source.original_frame.copy() if preserve_source_columns else source.frame.copy()
    )
    output["probability"] = np.asarray(probabilities, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def _train_linear(train_cache, device, seed, family="phosphoLM"):
    _set_seed(seed)
    model = build_phospholm_ppi_model().to(device)
    loader = DataLoader(train_cache, batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = FocalLoss(alpha=0.85, gamma=2.5, class_balanced=False)
    for epoch in range(EPOCHS):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        print(f"{family} PPI epoch {epoch + 1}/{EPOCHS}")
    return model


def _train_ptm_mamba(train_cache, device, seed):
    _set_seed(seed)
    model = build_ptm_mamba_ppi_model(max_length=2048, dropout=0.5).to(device)
    loader = DataLoader(
        train_cache,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_ptm_mamba_ppi,
    )
    criterion = FocalLoss(alpha=0.85, gamma=2.5, class_balanced=False, binary=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_steps = EPOCHS * len(loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    for epoch in range(EPOCHS):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["binder"].to(device),
                batch["wt"].to(device),
                batch["ptm"].to(device),
            )
            loss = criterion(logits, batch["effect"].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"PTM-Mamba PPI epoch {epoch + 1}/{EPOCHS}")
    return model


def _predict_linear(model, cache, device):
    loader = DataLoader(cache, batch_size=256, shuffle=False)
    result = []
    model.to(device).eval()
    with torch.inference_mode():
        for features, _ in loader:
            result.append(torch.softmax(model(features.to(device)), dim=-1)[:, 1].cpu())
    return torch.cat(result).numpy()


def _predict_ptm_mamba(model, cache, device):
    loader = DataLoader(
        cache,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_ptm_mamba_ppi,
    )
    result = []
    model.to(device).eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["binder"].to(device),
                batch["wt"].to(device),
                batch["ptm"].to(device),
            )
            result.append(torch.sigmoid(logits).cpu())
    return torch.cat(result).numpy()


def training_main(family):
    args = _parser(family, inference=False).parse_args()
    device = _device(args.device)
    for fold in args.folds:
        paths = _paths(args, family, fold)
        train_cache = _embedding_dataset(family, paths["train_cache"])
        if family in LINEAR_FAMILIES:
            model = _train_linear(train_cache, device, args.seed + fold, family)
        else:
            model = _train_ptm_mamba(train_cache, device, args.seed + fold)

        _save_state(paths["model"], model)
        test_source = PpiDataset(paths["test_csv"])
        test_cache = _embedding_dataset(family, paths["test_cache"])
        if family in LINEAR_FAMILIES:
            probabilities = _predict_linear(model, test_cache, device)
        else:
            probabilities = _predict_ptm_mamba(model, test_cache, device)
        _write_predictions(
            paths["prediction"],
            test_source,
            probabilities,
            preserve_source_columns=family in LINEAR_FAMILIES,
        )
        model.to("cpu")
        print(f"Completed {family} PPI fold {fold}.")


def inference_main(family):
    args = _parser(family, inference=True).parse_args()
    device = _device(args.device)
    for fold in args.folds:
        paths = _paths(args, family, fold)
        source = PpiDataset(paths["test_csv"])
        cache = _embedding_dataset(family, paths["test_cache"])
        if family in LINEAR_FAMILIES:
            model = load_phospholm_ppi_model(paths["model"])
            probabilities = _predict_linear(model, cache, device)
        else:
            model = load_ptm_mamba_ppi_model(paths["model"])
            probabilities = _predict_ptm_mamba(model, cache, device)
        _write_predictions(
            paths["prediction"],
            source,
            probabilities,
            preserve_source_columns=family in LINEAR_FAMILIES,
        )
        model.to("cpu")
        print(f"Wrote {paths['prediction']}")

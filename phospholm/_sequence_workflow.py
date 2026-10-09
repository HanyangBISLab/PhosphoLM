import argparse
import gc
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from phospholm.loss.focalloss import FocalLoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProbeConfig:
    hidden_dim: int
    epochs: int
    learning_rate: float
    batch_size: int = 64
    weight_decay: float = 0.01
    dropout: float = 0.0


PROBE_CONFIG = ProbeConfig(hidden_dim=256, epochs=3, learning_rate=1e-4)


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


def _paths(args, fold):
    return {
        "train_csv": args.fold_dir / f"fold_{fold}_train.csv",
        "test_csv": args.fold_dir / f"fold_{fold}_test.csv",
        "train_cache": args.embedding_root / f"fold_{fold}" / "train_embeddings.npz",
        "test_cache": args.embedding_root / f"fold_{fold}" / "test_embeddings.npz",
        "model": args.model_root / f"fold_{fold}" / "linear_probe.joblib",
        "prediction": args.prediction_root / f"fold_{fold}_test.csv",
    }


def _write_prediction(path, embedding_dataset, probabilities):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= 0.5).astype(np.int64)
    frame = pd.DataFrame(
        {
            "row_index": embedding_dataset.data.row_indices,
            "uniprot_id": embedding_dataset.data.uniprot_ids,
            "accession_id": embedding_dataset.data.accession_ids,
            "sequence_length": embedding_dataset.data.sequence_lengths,
            "label": embedding_dataset.data.labels,
            "probability": probabilities,
            "prediction": predictions,
        }
    )
    frame.to_csv(path, index=False)


def _save_cache(path, embeddings, source_dataset):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=source_dataset.labels.astype(np.int64, copy=False),
        row_indices=np.arange(len(source_dataset), dtype=np.int64),
        uniprot_ids=source_dataset.frame["uniprot_id"].to_numpy(dtype=str),
        accession_ids=source_dataset.frame["accession_id"].to_numpy(dtype=str),
        sequence_lengths=source_dataset.frame["wt_seq"]
        .astype(str)
        .str.len()
        .to_numpy(dtype=np.int64),
    )


def _compute_embeddings(source_dataset, cache_path, encoder_builder, device):
    model, tokenizer = encoder_builder()
    model.to(device)
    model.eval()
    vectors = []
    try:
        for index in range(len(source_dataset)):
            sequence = source_dataset[index]["sequence"]
            encoded = tokenizer(
                sequence,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            with torch.inference_mode():
                hidden = model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                ).last_hidden_state
            content = hidden[0, 1:-1]
            vectors.append(content.mean(dim=0).float().cpu().numpy())
    finally:
        model.to("cpu")
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _save_cache(cache_path, np.stack(vectors), source_dataset)


def _get_cache(
    cache_path,
    source_dataset,
    embedding_class,
    encoder_builder,
    device,
    force,
):
    cache_path = Path(cache_path)
    if force or not cache_path.is_file():
        _compute_embeddings(source_dataset, cache_path, encoder_builder, device)
    return embedding_class(cache_path)


def _fit_once(embeddings, labels, family, build_model, seed, config, device):
    _set_seed(seed)
    embeddings = embeddings.astype(np.float32, copy=False)
    labels = labels.astype(np.int64, copy=False)
    features = torch.from_numpy(embeddings).to(device)
    targets = torch.from_numpy(labels).to(device)
    model = build_model(
        family, hidden_dim=config.hidden_dim, dropout=config.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = FocalLoss(alpha=0.85, gamma=2.5, class_balanced=False)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(len(targets), generator=generator)
        for start in range(0, len(targets), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features[batch]), targets[batch])
            loss.backward()
            optimizer.step()
    model.eval()
    return SimpleNamespace(model=model)


def _probabilities(bundle, embeddings, device):
    model = bundle.model.to(device).eval()
    features = torch.from_numpy(embeddings.astype(np.float32, copy=False)).to(device)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(features), 1024):
            batches.append(
                torch.softmax(model(features[start : start + 1024]), dim=-1)[:, 1].cpu()
            )
    model.to("cpu")
    return torch.cat(batches).numpy()


def _training_parser(task, family):
    parser = argparse.ArgumentParser(
        description=f"Train {family} {task} probes over five folds."
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--fold-dir", type=Path, default=PROJECT_ROOT / "datasets" / task / "folds"
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / task / "embeddings" / family,
    )
    parser.add_argument(
        "--model-root", type=Path, default=PROJECT_ROOT / "models" / f"{family}_{task}"
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=PROJECT_ROOT / "output" / task / family / "predictions",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    return parser


def training_main(
    task,
    family,
    dataset_class,
    embedding_class,
    build_model,
    save_model,
    encoder_builder=None,
):
    args = _training_parser(task, family).parse_args()
    device = _device(args.device)
    for fold in args.folds:
        paths = _paths(args, fold)
        train_source = dataset_class(paths["train_csv"])
        train_cache = _get_cache(
            paths["train_cache"],
            train_source,
            embedding_class,
            encoder_builder,
            device,
            args.force_recompute_embeddings,
        )
        config = PROBE_CONFIG
        bundle = _fit_once(
            train_cache.data.embeddings,
            train_cache.data.labels,
            family,
            build_model,
            args.seed + fold,
            config,
            device,
        )
        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        save_model(paths["model"], bundle)
        test_source = dataset_class(paths["test_csv"])
        test_cache = _get_cache(
            paths["test_cache"],
            test_source,
            embedding_class,
            encoder_builder,
            device,
            args.force_recompute_embeddings,
        )
        probabilities = _probabilities(bundle, test_cache.data.embeddings, device)
        _write_prediction(paths["prediction"], test_cache, probabilities)
        print(f"Completed {task} {family} fold {fold}.")


def _inference_parser(task, family):
    parser = argparse.ArgumentParser(description=f"Run {family} {task} fold inference.")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--fold-dir", type=Path, default=PROJECT_ROOT / "datasets" / task / "folds"
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / task / "embeddings" / family,
    )
    parser.add_argument(
        "--model-root", type=Path, default=PROJECT_ROOT / "models" / f"{family}_{task}"
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=PROJECT_ROOT / "output" / task / family / "predictions",
    )
    parser.add_argument("--device", default="auto")
    return parser


def inference_main(task, family, embedding_class, load_model):
    args = _inference_parser(task, family).parse_args()
    device = _device(args.device)
    for fold in args.folds:
        paths = _paths(args, fold)
        cache = embedding_class(paths["test_cache"])
        bundle = load_model(paths["model"], family)
        probabilities = _probabilities(bundle, cache.data.embeddings, device)
        _write_prediction(paths["prediction"], cache, probabilities)
        print(f"Wrote {paths['prediction']}")

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, EsmForTokenClassification, EsmTokenizer

from phospholm.residue_tokens import encode_esm2, encode_phospholm
from phospholm.tokenizer.phosphoLM_tokenizer import PTMTokenizer


PHOSPHO_RESIDUES = frozenset("STY")


@dataclass(frozen=True)
class TokenInferenceConfig:
    name: str
    fold_dir: Path
    model_dir: Path
    prediction_dir: Path
    folds: tuple[int, ...]
    model_family: str
    predict_listed_sites_only: bool


def _arguments(config):
    parser = argparse.ArgumentParser(description=f"Run {config.name} inference.")
    parser.add_argument("--folds", nargs="+", type=int, default=list(config.folds))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    return args


def _paths(config, fold):
    return (
        config.model_dir / f"fold_{fold}" / "final_model",
        config.fold_dir / f"fold_{fold}_test.json",
        config.prediction_dir / f"fold_{fold}_test.json",
    )


def _load_model(config, checkpoint, device):
    checkpoint_config = AutoConfig.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    tokenizer_class = (
        PTMTokenizer if config.model_family == "phosphoLM" else EsmTokenizer
    )
    tokenizer = tokenizer_class.from_pretrained(checkpoint, local_files_only=True)
    model = EsmForTokenClassification.from_pretrained(
        checkpoint,
        config=checkpoint_config,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=torch.float32,
    )
    model.set_attn_implementation("sdpa")
    return model.to(device).eval(), tokenizer


def _sites(record, listed_only):
    sequence = record.get("Seq")
    if not listed_only:
        return sequence, [
            i for i, residue in enumerate(sequence) if residue in PHOSPHO_RESIDUES
        ]

    sites = record.get("Site")
    return sequence, sites


def predict_record(model, tokenizer, record, device, config):

    sequence, sites = _sites(record, config.predict_listed_sites_only)
    tokenize = encode_phospholm if config.model_family == "phosphoLM" else encode_esm2
    encoded, token_positions = tokenize(tokenizer, sequence, return_tensors="pt")

    with torch.inference_mode():
        logits = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        ).logits
        token_probabilities = torch.softmax(logits.float(), dim=-1)[0, :, 1].cpu()

    probabilities = [0.0] * len(sequence)
    for site in sites:
        matches = token_positions.get(site, [])
        token_index = matches[0]
        probability = float(token_probabilities[token_index])
        probabilities[site] = probability
    return probabilities


def _write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=4)
        handle.write("\n")


def _run_fold(config, fold, device):
    checkpoint, test_path, output_path = _paths(config, fold)
    with test_path.open(encoding="utf-8") as handle:
        records = json.load(handle)

    model = tokenizer = None
    try:
        model, tokenizer = _load_model(config, checkpoint, device)
        predictions = []
        for record in tqdm(records, desc=f"{config.name} fold {fold}", unit="protein"):
            output_record = dict(record)
            output_record["prob"] = predict_record(
                model, tokenizer, record, device, config
            )
            predictions.append(output_record)
        _write_json(output_path, predictions)
        print(f"Wrote {output_path}")
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def run_token_inference(config):
    args = _arguments(config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for fold in args.folds:
        _run_fold(config, fold, device)

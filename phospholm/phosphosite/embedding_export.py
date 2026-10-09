import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from phospholm.phosphosite.model import build_esm2_model, build_phosphosite_model
from phospholm.residue_tokens import encode_esm2, encode_phospholm


MODEL_BUILDERS = {
    "phosphoLM": build_phosphosite_model,
    "esm2_650M": build_esm2_model,
}


def collect_records(source_path):
    with Path(source_path).open(encoding="utf-8") as handle:
        source = json.load(handle)
    unique = {}
    for record in source:
        sequence, protein = record["Seq"], record["Protein"]
        entry = unique.setdefault(
            sequence,
            {
                "Seq": sequence,
                "protein_ids": [],
                "residue_positions": [
                    i for i, residue in enumerate(sequence) if residue in "STY"
                ],
            },
        )
        if protein not in entry["protein_ids"]:
            entry["protein_ids"].append(protein)
    return list(unique.values())


def _extract_sites(model, tokenizer, record, model_name, device):
    tokenize = encode_phospholm if model_name == "phosphoLM" else encode_esm2
    encoded, token_positions = tokenize(tokenizer, record["Seq"], return_tensors="pt")
    token_indices = []
    for position in record["residue_positions"]:
        matches = token_positions.get(position, [])
        token_indices.append(matches[0])
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast:
        hidden = model.esm(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        ).last_hidden_state
        selected = hidden[0].index_select(
            0, torch.tensor(token_indices, device=device, dtype=torch.long)
        )
        values = selected.float().cpu().numpy()
    return values


def export_embeddings(
    model_name, source_path, output_dir, device, *, builder=None, progress_every=100
):
    source_path, output_dir = Path(source_path).resolve(), Path(output_dir)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    records = collect_records(source_path)
    total_sites = sum(len(record["residue_positions"]) for record in records)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(42)
    model, tokenizer = (builder or MODEL_BUILDERS[model_name])()
    values = None
    try:
        model.set_attn_implementation("sdpa")
        model.eval()
        model.esm.requires_grad_(False)
        model.esm.to(device=device, dtype=torch.float32)
        hidden_size = model.config.hidden_size
        index_data = {
            "format_version": 1,
            "model_family": model_name,
            "hidden_size": hidden_size,
            "dtype": "float32",
            "representation": "esm.last_hidden_state_after_final_layer_norm_before_classifier",
            "position_base": 0,
            "candidate_residues": "STY",
            "num_sequences": len(records),
            "num_sites": total_sites,
            "embeddings_file": "embeddings.npy",
            "proteins": [],
        }
        print(
            f"START {model_name}: {len(records):,} proteins, {total_sites:,} sites, "
            f"shape=({total_sites}, {hidden_size})",
            flush=True,
        )
        values = np.lib.format.open_memmap(
            output_dir / "embeddings.npy",
            mode="w+",
            dtype="float32",
            shape=(total_sites, hidden_size),
        )
        cursor = 0
        for index, record in enumerate(records, 1):
            features = _extract_sites(model, tokenizer, record, model_name, device)
            size = len(features)
            values[cursor : cursor + size] = features
            index_data["proteins"].append(
                {
                    "protein_ids": record["protein_ids"],
                    "sequence_length": len(record["Seq"]),
                    "row_offset": cursor,
                    "site_count": size,
                    "residue_positions": record["residue_positions"],
                }
            )
            cursor += size
            if index % progress_every == 0 or index == len(records):
                elapsed = time.monotonic() - started
                print(
                    f"PROGRESS {model_name}: {index}/{len(records)} proteins, "
                    f"{cursor}/{total_sites} sites, elapsed={elapsed:.1f}s, "
                    f"rate={index / elapsed:.2f} proteins/s",
                    flush=True,
                )
        values.flush()
        del values
        values = None
        with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
            json.dump(index_data, handle)
            handle.write("\n")
        print(f"SAVED {model_name}: {output_dir}", flush=True)
        return index_data
    finally:
        del values, model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

from phospholm.ppi.dataset import PpiDataset
from phospholm.ppi.cache import (
    ESM2_CHECKPOINT,
    ESM2_REVISION,
    save_ppi_cache,
)
from phospholm.ppi.workflow import _device
from phospholm.residue_tokens import encode_esm2


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_esm2_encoder():
    tokenizer = EsmTokenizer.from_pretrained(ESM2_CHECKPOINT, revision=ESM2_REVISION)
    model = EsmModel.from_pretrained(
        ESM2_CHECKPOINT,
        revision=ESM2_REVISION,
        add_pooling_layer=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.float().requires_grad_(False).eval()
    return model, tokenizer


def export_embeddings(fold_dir, embedding_root, folds, device):

    fold_dir, embedding_root = Path(fold_dir), Path(embedding_root)
    device = _device(str(device))
    pending = []
    for fold in folds:
        for split in ("train", "test"):
            source = PpiDataset(fold_dir / f"fold_{fold}_{split}.csv")
            path = embedding_root / f"fold_{fold}" / f"{split}_embeddings.npz"
            if path.exists():
                print(f"Reusing {path}")
            else:
                pending.append((path, source))
    if not pending:
        print("All requested ESM2 PPI caches already exist.")
        return

    needed_sites = {}
    needed_binders = set()
    for _, source in pending:
        for record in source:
            sequence = str(record["wt_seq"]).upper()
            needed_sites.setdefault(sequence, set()).add(int(record["Site"]) - 1)
            needed_binders.add(str(record["binder_seq"]).upper())
    sequences = sorted(set(needed_sites) | needed_binders)
    model, tokenizer = build_esm2_encoder()
    site_vectors, binder_vectors = {}, {}
    try:
        model.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
        print(f"Encoding {len(sequences)} unique sequences on {device} in FP32.")
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, enabled=False),
        ):
            for index, sequence in enumerate(sequences, start=1):
                encoded, positions = encode_esm2(
                    tokenizer, sequence, return_tensors="pt"
                )
                hidden = model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                ).last_hidden_state
                for residue in sorted(needed_sites.get(sequence, ())):
                    site_vectors[(sequence, residue)] = (
                        hidden[0, positions[residue][0]].float().cpu().numpy().copy()
                    )
                if sequence in needed_binders:
                    binder_vectors[sequence] = (
                        hidden[0, 1:-1].mean(dim=0).float().cpu().numpy().copy()
                    )
                del hidden, encoded, positions
                if index == 1 or index % 25 == 0 or index == len(sequences):
                    print(
                        f"ESM2 PPI embeddings: {index}/{len(sequences)} sequences",
                        flush=True,
                    )
        for path, source in pending:
            sites = np.stack(
                [
                    site_vectors[(str(r["wt_seq"]).upper(), int(r["Site"]) - 1)]
                    for r in source
                ]
            )
            binders = np.stack(
                [binder_vectors[str(r["binder_seq"]).upper()] for r in source]
            )
            save_ppi_cache(path, source, sites, binders)
            print(f"Saved {path}")
    finally:
        model.to("cpu")
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Save frozen ESM2-650M PPI embeddings before training."
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--fold-dir", type=Path, default=PROJECT_ROOT / "datasets" / "ppi" / "folds"
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "ppi" / "embeddings" / "esm2_650M",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    export_embeddings(args.fold_dir, args.embedding_root, args.folds, args.device)


if __name__ == "__main__":
    main()

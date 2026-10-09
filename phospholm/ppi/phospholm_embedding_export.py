
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import EsmModel

from phospholm.ppi.dataset import PpiDataset
from phospholm.ppi.cache import PHOSPHOLM_CHECKPOINT, save_ppi_cache
from phospholm.ppi.workflow import _device
from phospholm.tokenizer.phosphoLM_tokenizer import PTMTokenizer
from phospholm.residue_tokens import _values


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_phospholm_encoder(checkpoint=PHOSPHOLM_CHECKPOINT):
    tokenizer = PTMTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = EsmModel.from_pretrained(
        checkpoint,
        local_files_only=True,
        add_pooling_layer=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.float().requires_grad_(False).eval()
    return model, tokenizer


def encode_binder(tokenizer, sequence, return_tensors=None):
    encoded = tokenizer(
        sequence,
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=False,
        return_tensors=return_tensors,
    )
    ids = _values(encoded["input_ids"])
    attention = _values(encoded["attention_mask"])
    selected = [
        i
        for i, (token_id, mask) in enumerate(zip(ids, attention))
        if mask and token_id not in tokenizer.structural_ids
    ]
    return encoded, selected


def export_embeddings(
    fold_dir,
    site_embedding_root,
    embedding_root,
    folds,
    device,
    checkpoint=PHOSPHOLM_CHECKPOINT,
):
    fold_dir, site_embedding_root, embedding_root = map(
        Path, (fold_dir, site_embedding_root, embedding_root)
    )
    device = _device(str(device))
    pending = []
    for fold in folds:
        for split in ("train", "test"):
            path = embedding_root / f"fold_{fold}" / f"{split}_embeddings.npz"
            if path.exists():
                print(f"Reusing {path}")
                continue
            source = PpiDataset(fold_dir / f"fold_{fold}_{split}.csv")
            original_path = (
                site_embedding_root / f"fold_{fold}" / f"{split}_embeddings.npz"
            )
            with np.load(original_path, allow_pickle=False) as original:
                sites = original["site_embeddings"].astype(np.float32, copy=False)
            pending.append((path, source, sites))
    if not pending:
        print("All requested PhosphoLM binder caches already exist.")
        return
    sequences = sorted(
        {str(row["binder_seq"]).upper() for _, source, _ in pending for row in source}
    )
    model, tokenizer = build_phospholm_encoder(checkpoint)
    vectors = {}
    try:
        model.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
        print(
            f"Encoding {len(sequences)} distinct PhosphoLM binders on {device} in FP32.",
            flush=True,
        )
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, enabled=False),
        ):
            for index, sequence in enumerate(sequences, start=1):
                encoded, selected = encode_binder(
                    tokenizer, sequence, return_tensors="pt"
                )
                hidden = model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                ).last_hidden_state
                vectors[sequence] = (
                    hidden[0, selected].mean(dim=0).float().cpu().numpy().copy()
                )
                del hidden, encoded, selected
                if index == 1 or index % 25 == 0 or index == len(sequences):
                    print(
                        f"PhosphoLM binder embeddings: {index}/{len(sequences)}",
                        flush=True,
                    )
        for path, source, sites in pending:
            binders = np.stack(
                [vectors[str(row["binder_seq"]).upper()] for row in source]
            )
            save_ppi_cache(path, source, sites, binders)
            print(f"Saved {path}", flush=True)
    finally:
        model.to("cpu")
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Generate PhosphoLM PPI binder features into an output cache root."
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--fold-dir", type=Path, default=PROJECT_ROOT / "datasets/ppi/folds"
    )
    parser.add_argument(
        "--site-embedding-root",
        type=Path,
        default=PROJECT_ROOT / "datasets/ppi/embeddings/phosphoLM",
    )
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=PHOSPHOLM_CHECKPOINT)
    args = parser.parse_args()
    export_embeddings(
        args.fold_dir,
        args.site_embedding_root,
        args.embedding_root,
        args.folds,
        args.device,
        args.checkpoint,
    )


if __name__ == "__main__":
    main()

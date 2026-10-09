#!/usr/bin/env python3
"""Save phosphosite embeddings explicitly; this command never trains a classifier.

Example (physical CUDA 2, after activating ptm_language):
    CUDA_VISIBLE_DEVICES=2 python phosphosite_save_embeddings.py --model both
"""

import argparse
from pathlib import Path

import torch

from phospholm.phosphosite.embedding_export import export_embeddings


ROOT = Path(__file__).resolve().parent
OUTPUTS = {
    "phosphoLM": ROOT / "models/phosphoLM_phosphosite/embedding_cache",
    "esm2_650M": ROOT / "models/esm2_650M_phosphosite/embedding_cache",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*OUTPUTS, "both"], default="both")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "datasets/dbptm/dbptm.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", type=Path, help="Override output for one model."
    )
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    device = torch.device(args.device)
    names = list(OUTPUTS) if args.model == "both" else [args.model]
    destinations = {name: args.output_dir or OUTPUTS[name] for name in names}
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for name in names:
        export_embeddings(
            name,
            args.source,
            destinations[name],
            device,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()

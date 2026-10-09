#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets/phosphopeptides/phosphopeptides.txt"
OUTPUT_DIR = ROOT / "output/phospholm_tokenizer"

COLUMNS = [
    "uniprot",
    "entry_name",
    "peptide_sequence",
    "peptide_start",
    "peptide_end",
    "ptm_name",
    "ptm_position",
    "localization_probability",
    "source_id",
]
STRUCTURAL_TOKENS = ["<unk>", "<cls>", "<eos>", "<pad>", "<mask>"]
PHOSPHO_TOKENS = ["S", "T", "Y"]


def load_fragments(path=DATASET):

    frame = pd.read_csv(path, sep="\t", names=COLUMNS)
    positions = pd.to_numeric(frame["ptm_position"], errors="coerce")
    valid = positions.notna()
    positions = positions.fillna(0).astype(int)
    valid &= positions.between(1, frame["peptide_sequence"].str.len())
    valid_residue = pd.Series(
        [
            sequence[position - 1] in PHOSPHO_TOKENS if is_valid else False
            for sequence, position, is_valid in zip(
                frame["peptide_sequence"], positions, valid, strict=True
            )
        ],
        index=frame.index,
    )
    valid &= valid_residue
    frame = frame.loc[valid].copy()
    frame["ptm_position"] = positions[valid]

    fragments = []
    for sequence, position in zip(
        frame["peptide_sequence"], frame["ptm_position"], strict=True
    ):
        fragments.extend((sequence[:position], sequence[position - 1 :]))
    return list(dict.fromkeys(fragments))


def train_tokenizer(fragments):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        fragments,
        trainer=BpeTrainer(
            special_tokens=STRUCTURAL_TOKENS + PHOSPHO_TOKENS,
            vocab_size=20_000,
            min_frequency=50,
            show_progress=True,
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        cls_token="<cls>",
        eos_token="<eos>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=PHOSPHO_TOKENS,
    )


def main():
    fragments = load_fragments()
    print(f"Training on {len(fragments):,} unique peptide fragments.")

    tokenizer = train_tokenizer(fragments)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved {len(tokenizer):,} tokenizer entries to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()

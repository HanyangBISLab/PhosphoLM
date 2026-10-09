import json
from pathlib import Path

from torch.utils.data import Dataset

from phospholm.residue_tokens import encode_esm2, encode_phospholm


PHOSPHO_RESIDUES = frozenset("STY")
IGNORE_INDEX = -100


class PhosphositeDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        self.tokenizer = tokenizer
        with Path(json_path).open(encoding="utf-8") as handle:
            self.records = json.load(handle)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sequence = record["Seq"]

        positive_sites = set(record.get("Site") or [])
        masked_sites = set(record.get("Mask_Site") or [])

        encoded, token_positions = encode_phospholm(self.tokenizer, sequence)
        labels = [IGNORE_INDEX] * len(encoded["input_ids"])
        for position, residue in enumerate(sequence):
            if residue not in PHOSPHO_RESIDUES:
                continue
            matches = token_positions.get(position, [])
            if position not in masked_sites:
                labels[matches[0]] = 1 if position in positive_sites else 0

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


class Esm2PhosphositeDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        self.tokenizer = tokenizer
        with Path(json_path).open(encoding="utf-8") as handle:
            self.records = json.load(handle)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sequence = record["Seq"]

        positive_sites = set(record.get("Site") or [])
        masked_sites = set(record.get("Mask_Site") or [])

        encoded, _ = encode_esm2(self.tokenizer, sequence)
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        expected_length = len(sequence) + 2

        labels = [IGNORE_INDEX] * expected_length
        for position, residue in enumerate(sequence):
            if residue in PHOSPHO_RESIDUES and position not in masked_sites:
                labels[position + 1] = 1 if position in positive_sites else 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

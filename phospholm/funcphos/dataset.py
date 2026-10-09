import json
from pathlib import Path

from torch.utils.data import Dataset

from phospholm.residue_tokens import encode_esm2, encode_phospholm


IGNORE_INDEX = -100


class _FuncPhosDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        self.tokenizer = tokenizer
        with Path(json_path).open(encoding="utf-8") as handle:
            self.records = json.load(handle)

    def __len__(self):
        return len(self.records)


class FuncPhosDataset(_FuncPhosDataset):
    def __getitem__(self, index):
        record = self.records[index]
        sequence = record["Seq"]
        encoded, token_positions = encode_phospholm(self.tokenizer, sequence)
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])

        labels = [IGNORE_INDEX] * len(input_ids)
        for site, regular in zip(record["Site"], record["regular"]):
            matches = token_positions.get(site, [])
            labels[matches[0]] = regular

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class Esm2FuncPhosDataset(_FuncPhosDataset):
    def __getitem__(self, index):
        record = self.records[index]
        sequence = record["Seq"]
        encoded, _ = encode_esm2(self.tokenizer, sequence)
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        expected_length = len(sequence) + 2

        labels = [IGNORE_INDEX] * expected_length
        for site, regular in zip(record["Site"], record["regular"]):
            labels[site + 1] = regular

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

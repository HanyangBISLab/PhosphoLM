from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SOURCE_COLUMNS = (
    "target_uniprot",
    "binder_uniprot",
    "wt_seq",
    "ptm_seq",
    "binder_seq",
    "Effect",
    "PTM",
    "Site",
)
LABEL_MAP = {"enhance": 1, "induce": 1, "inhibit": 0}


def normalize_effect(value):
    key = str(value).strip().lower()
    return LABEL_MAP[key]


class PpiDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.original_frame = pd.read_csv(self.csv_path)
        self.frame = (
            self.original_frame.loc[:, SOURCE_COLUMNS].copy().reset_index(drop=True)
        )
        self.frame["Site"] = self.frame["Site"].astype(np.int64)
        self.labels = (
            self.frame["Effect"].map(normalize_effect).to_numpy(dtype=np.int64)
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        record = self.frame.iloc[index].to_dict()
        record["label"] = int(self.labels[index])
        return record


class PhosphoLMPpiEmbeddingDataset(Dataset):
    def __init__(self, npz_path):
        self.npz_path = Path(npz_path)
        with np.load(self.npz_path, allow_pickle=False) as loaded:
            self.embeddings = loaded["embeddings"].astype(np.float32, copy=False)
            self.labels = loaded["labels"].astype(np.int64, copy=False)

    @cached_property
    def site_embeddings(self):
        with np.load(self.npz_path, allow_pickle=False) as cache:
            return cache["site_embeddings"].astype(np.float32, copy=False)

    @cached_property
    def binder_embeddings(self):
        with np.load(self.npz_path, allow_pickle=False) as cache:
            return cache["binder_embeddings"].astype(np.float32, copy=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.embeddings[index]), int(self.labels[index])


class PtmMambaPpiEmbeddingDataset(Dataset):
    def __init__(self, dataset_path):
        from datasets import load_from_disk

        self.dataset_path = Path(dataset_path)
        self.dataset = load_from_disk(str(self.dataset_path))
        self.labels = np.asarray(self.dataset["effect"], dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        vectors = {}
        for name in ("wt", "ptm", "binder"):
            vector = torch.as_tensor(item[name], dtype=torch.float32).squeeze(0)
            vectors[name] = vector
        return {
            **vectors,
            "effect": int(item["effect"]),
        }


def _pad(values, max_length):
    result = torch.zeros((len(values), max_length), dtype=torch.float32)
    for index, value in enumerate(values):
        length = min(int(value.numel()), max_length)
        result[index, :length] = value[:length]
    return result


def collate_ptm_mamba_ppi(batch, max_length=2048):
    return {
        "wt": _pad([item["wt"] for item in batch], max_length),
        "ptm": _pad([item["ptm"] for item in batch], max_length),
        "binder": _pad([item["binder"] for item in batch], max_length),
        "effect": torch.tensor([item["effect"] for item in batch], dtype=torch.float32),
    }

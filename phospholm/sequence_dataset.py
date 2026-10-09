from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_DIMENSIONS = {"phosphoLM": 1280, "esm2_650M": 1280, "ptm_mamba": 768}


class SequenceDataset(Dataset):
    label_column = None

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path).reset_index(drop=True)
        self.labels = self.frame[self.label_column].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return {
            "row_index": index,
            "uniprot_id": str(row["uniprot_id"]),
            "accession_id": str(row["accession_id"]),
            "sequence": str(row["wt_seq"]),
            "label": int(self.labels[index]),
        }


@dataclass(frozen=True)
class EmbeddingData:
    embeddings: np.ndarray
    labels: np.ndarray
    row_indices: np.ndarray
    uniprot_ids: np.ndarray
    accession_ids: np.ndarray
    sequence_lengths: np.ndarray


class SequenceEmbeddingDataset(Dataset):
    def __init__(self, npz_path):
        self.npz_path = Path(npz_path)
        with np.load(self.npz_path, allow_pickle=False) as cache:
            self.data = EmbeddingData(
                embeddings=cache["embeddings"].astype(np.float32, copy=False),
                labels=cache["labels"].astype(np.int64, copy=False),
                row_indices=cache["row_indices"].astype(np.int64, copy=False),
                uniprot_ids=cache["uniprot_ids"].astype(str, copy=False),
                accession_ids=cache["accession_ids"].astype(str, copy=False),
                sequence_lengths=cache["sequence_lengths"].astype(np.int64, copy=False),
            )

    def __len__(self):
        return len(self.data.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.data.embeddings[index]), int(
            self.data.labels[index]
        )

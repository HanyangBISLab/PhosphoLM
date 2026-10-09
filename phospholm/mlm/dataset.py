from pathlib import Path

from Bio import SeqIO
from torch.utils.data import Dataset


def ensure_fasta_index(fasta_path):
    fasta_path = Path(fasta_path).resolve()
    index_path = Path(f"{fasta_path}.idx")
    database = SeqIO.index_db(str(index_path), str(fasta_path), "fasta")
    database.close()
    return index_path


class MLMDataset(Dataset):

    def __init__(self, fasta_path, tokenizer):
        self.fasta_path = Path(fasta_path).resolve()
        self.index_path = ensure_fasta_index(self.fasta_path)
        self.tokenizer = tokenizer
        database = SeqIO.index_db(str(self.index_path), str(self.fasta_path), "fasta")
        self.keys = list(database.keys())
        database.close()
        self.database = None
        print(f"Loaded {len(self.keys):,} FASTA sequences.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.database is None:
            self.database = SeqIO.index_db(
                str(self.index_path), str(self.fasta_path), "fasta"
            )
        sequence = str(self.database[self.keys[index]].seq).upper()
        return {"input_ids": self.tokenizer(sequence)["input_ids"]}

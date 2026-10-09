from phospholm.sequence_dataset import SequenceDataset, SequenceEmbeddingDataset


class DruggableDataset(SequenceDataset):
    label_column = "is_druggable"


class DruggableEmbeddingDataset(SequenceEmbeddingDataset):
    pass

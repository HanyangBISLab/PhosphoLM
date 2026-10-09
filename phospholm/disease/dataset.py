from phospholm.sequence_dataset import SequenceDataset, SequenceEmbeddingDataset


class DiseaseDataset(SequenceDataset):
    label_column = "is_disease"


class DiseaseEmbeddingDataset(SequenceEmbeddingDataset):
    pass

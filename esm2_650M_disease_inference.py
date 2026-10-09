from phospholm._sequence_workflow import inference_main
from phospholm.disease.dataset import DiseaseEmbeddingDataset
from phospholm.disease.model import load_disease_model


if __name__ == "__main__":
    inference_main(
        task="disease",
        family="esm2_650M",
        embedding_class=DiseaseEmbeddingDataset,
        load_model=load_disease_model,
    )

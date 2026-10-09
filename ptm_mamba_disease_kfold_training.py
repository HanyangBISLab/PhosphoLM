from phospholm._sequence_workflow import training_main
from phospholm.disease.dataset import DiseaseDataset, DiseaseEmbeddingDataset
from phospholm.disease.model import build_disease_model, save_disease_model


if __name__ == "__main__":
    training_main(
        task="disease",
        family="ptm_mamba",
        dataset_class=DiseaseDataset,
        embedding_class=DiseaseEmbeddingDataset,
        build_model=build_disease_model,
        save_model=save_disease_model,
    )

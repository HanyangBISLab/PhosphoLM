from phospholm._sequence_workflow import training_main
from phospholm.druggable.dataset import DruggableDataset, DruggableEmbeddingDataset
from phospholm.druggable.model import build_druggable_model, save_druggable_model


if __name__ == "__main__":
    training_main(
        task="druggable",
        family="ptm_mamba",
        dataset_class=DruggableDataset,
        embedding_class=DruggableEmbeddingDataset,
        build_model=build_druggable_model,
        save_model=save_druggable_model,
    )

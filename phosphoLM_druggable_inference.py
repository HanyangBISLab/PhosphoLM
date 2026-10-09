from phospholm._sequence_workflow import inference_main
from phospholm.druggable.dataset import DruggableEmbeddingDataset
from phospholm.druggable.model import load_druggable_model


if __name__ == "__main__":
    inference_main(
        task="druggable",
        family="phosphoLM",
        embedding_class=DruggableEmbeddingDataset,
        load_model=load_druggable_model,
    )

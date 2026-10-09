from transformers import AutoConfig, EsmForMaskedLM, EsmForTokenClassification


def build_token_classifier(
    checkpoint,
    tokenizer_class,
    labels,
    local_files_only,
    *,
    model_class=EsmForTokenClassification,
):

    tokenizer = tokenizer_class.from_pretrained(
        checkpoint, local_files_only=local_files_only
    )
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=local_files_only)

    config.num_labels = 2
    config.id2label = dict(enumerate(labels))
    config.label2id = {label: index for index, label in config.id2label.items()}
    config.use_cache = False

    pretrained = EsmForMaskedLM.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    model = model_class(config)
    model.esm.load_state_dict(pretrained.esm.state_dict())
    model.esm.requires_grad_(False)
    model.classifier.requires_grad_(True)
    del pretrained
    return model, tokenizer

def _values(field):
    if hasattr(field, "ndim") and field.ndim >= 2:
        field = field[0]
    return field.tolist() if hasattr(field, "tolist") else list(field)


def encode_phospholm(tokenizer, sequence, return_tensors=None):

    encoded = tokenizer(
        sequence,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors=return_tensors,
        truncation=False,
    )
    offsets = [tuple(value) for value in _values(encoded.pop("offset_mapping"))]

    positions = {}
    for token_index, (start, end) in enumerate(offsets):
        if end - start != 1:
            continue
        positions.setdefault(start, []).append(token_index)
    return encoded, positions


def encode_esm2(tokenizer, sequence, return_tensors=None):
    encoded = tokenizer(
        sequence,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors=return_tensors,
        truncation=False,
    )
    return encoded, {position: [position + 1] for position in range(len(sequence))}

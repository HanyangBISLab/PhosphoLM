import torch
from transformers import DataCollatorForLanguageModeling


class DataCollatorForPackedMasking(DataCollatorForLanguageModeling):

    def __init__(
        self,
        tokenizer,
        max_length=1024,
        residues=None,
        mlm_probability=0.05,
        sty_probability=0.15,
        neighbor_probability=0.10,
        neighbor_window=5,
    ):
        super().__init__(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )
        self.max_length = max_length
        self.target_ids = set(
            tokenizer.convert_tokens_to_ids(residues or ["S", "T", "Y"])
        )
        self.protected_ids = set(tokenizer.all_special_ids) - self.target_ids
        self.replacement_ids = tuple(
            token_id
            for token_id in range(len(tokenizer))
            if token_id not in self.protected_ids
        )
        self.sty_probability = sty_probability
        self.neighbor_probability = neighbor_probability
        self.neighbor_window = neighbor_window

    def __call__(self, examples):
        packed = [token for example in examples for token in example["input_ids"]]
        chunks = [
            packed[start : start + self.max_length]
            for start in range(0, len(packed), self.max_length)
        ]
        chunks[-1].extend(
            [self.tokenizer.pad_token_id] * (self.max_length - len(chunks[-1]))
        )

        original = torch.tensor(chunks, dtype=torch.long)
        protected = torch.zeros_like(original, dtype=torch.bool)
        for token_id in self.protected_ids:
            protected |= original == token_id

        masked, labels = self.torch_mask_tokens(original.clone(), protected)
        return {
            "input_ids": masked,
            "labels": labels,
            "attention_mask": (original != self.tokenizer.pad_token_id).long(),
        }

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):

        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        labels = inputs.clone()

        targets = torch.zeros_like(inputs, dtype=torch.bool)
        for token_id in self.target_ids:
            targets |= inputs == token_id

        neighbors = torch.zeros_like(targets)
        for distance in range(1, self.neighbor_window + 1):
            neighbors |= torch.roll(targets, distance, dims=1)
            neighbors |= torch.roll(targets, -distance, dims=1)

        probability = torch.full(
            labels.shape,
            self.mlm_probability,
            dtype=torch.float32,
            device=labels.device,
        )
        probability.masked_fill_(neighbors, self.neighbor_probability)
        probability.masked_fill_(targets, self.sty_probability)
        probability.masked_fill_(special_tokens_mask, 0.0)

        selected = torch.bernoulli(probability).bool()
        labels[~selected] = -100

        replaced = torch.rand(labels.shape, device=labels.device) < 0.8
        replaced &= selected
        inputs[replaced] = self.tokenizer.mask_token_id

        randomized = torch.rand(labels.shape, device=labels.device) < 0.5
        randomized &= selected & ~replaced
        choices = torch.tensor(self.replacement_ids, device=labels.device)
        random_tokens = choices[
            torch.randint(len(choices), labels.shape, device=labels.device)
        ]
        inputs[randomized] = random_tokens[randomized]
        return inputs, labels

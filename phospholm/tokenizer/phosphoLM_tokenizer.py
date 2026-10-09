from tokenizers.processors import TemplateProcessing
import torch
from transformers import PreTrainedTokenizerFast


class PTMTokenizer(PreTrainedTokenizerFast):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        defaults = {
            "mask_token": "<mask>",
            "cls_token": "<cls>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        }
        for attribute, value in defaults.items():
            if getattr(self, attribute) is None:
                setattr(self, attribute, value)

        structural_tokens = [
            self.cls_token,
            self.eos_token,
            self.pad_token,
            self.mask_token,
            self.unk_token,
        ]
        self.structural_ids = {
            token_id
            for token in structural_tokens
            if token is not None
            and (token_id := self.convert_tokens_to_ids(token)) is not None
        }

        if self.backend_tokenizer.post_processor is None:
            cls_id = self.convert_tokens_to_ids(self.cls_token)
            eos_id = self.convert_tokens_to_ids(self.eos_token)
            if cls_id is not None and eos_id is not None:
                self.backend_tokenizer.post_processor = TemplateProcessing(
                    single=f"{self.cls_token}:0 $A:0 {self.eos_token}:0",
                    pair=f"{self.cls_token}:0 $A:0 {self.eos_token}:0 $B:1 {self.eos_token}:1",
                    special_tokens=[(self.cls_token, cls_id), (self.eos_token, eos_id)],
                )

    def _filter_structural_ids(self, token_ids):
        return [
            int(token_id)
            for token_id in token_ids
            if token_id not in self.structural_ids
        ]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        **kwargs,
    ):
        if skip_special_tokens:
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                if token_ids in self.structural_ids:
                    return ""
            else:
                token_ids = self._filter_structural_ids(token_ids)
        text = super().decode(token_ids, skip_special_tokens=False, **kwargs)
        return text.replace(" ", "")

    def batch_decode(
        self,
        sequences,
        skip_special_tokens=False,
        **kwargs,
    ):
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        if skip_special_tokens:
            sequences = [self._filter_structural_ids(row) for row in sequences]
        texts = super().batch_decode(sequences, skip_special_tokens=False, **kwargs)
        return [text.replace(" ", "") for text in texts]

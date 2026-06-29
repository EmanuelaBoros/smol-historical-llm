from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class TemporalDataCollatorForMLM:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def __call__(self, examples: list[dict]) -> dict:
        # Do not mutate original examples
        period_ids = torch.tensor(
            [int(example["period_ids"]) for example in examples],
            dtype=torch.long,
        )

        features = [
            {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            for example in examples
        ]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val,
                already_has_special_tokens=True,
            )
            for val in labels.tolist()
        ]

        special_tokens_mask = torch.tensor(
            special_tokens_mask,
            dtype=torch.bool,
        )

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        if self.tokenizer.pad_token_id is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% replace with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% replace with random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )

        random_words = torch.randint(
            len(self.tokenizer),
            labels.shape,
            dtype=torch.long,
        )

        input_ids[indices_random] = random_words[indices_random]

        # 10% keep original token

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["period_ids"] = period_ids

        return batch

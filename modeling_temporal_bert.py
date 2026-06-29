from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

PERIODS = [
    "pre_1850",
    "1850_1899",
    "1900_1938",
    "1939_1945",
    "post_1945",
]


def year_to_period_id(year: int) -> int:
    if year < 1850:
        return 0
    if 1850 <= year <= 1899:
        return 1
    if 1900 <= year <= 1938:
        return 2
    if 1939 <= year <= 1945:
        return 3
    return 4


class TemporalAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_size, hidden_size)

        # Start as almost identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(
            self.dropout(self.activation(self.down(hidden_states)))
        )


class TemporalAdapterBank(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_periods: int,
        bottleneck_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.adapters = nn.ModuleList(
            [
                TemporalAdapter(
                    hidden_size=hidden_size,
                    bottleneck_size=bottleneck_size,
                    dropout=dropout,
                )
                for _ in range(num_periods)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        period_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        if period_ids.dim() == 0:
            period_ids = period_ids.unsqueeze(0)

        period_ids = period_ids.to(hidden_states.device).long()

        batch_size = hidden_states.shape[0]

        if period_ids.shape[0] != batch_size:
            raise ValueError(
                f"period_ids batch size does not match hidden_states batch size: "
                f"period_ids={tuple(period_ids.shape)}, "
                f"hidden_states={tuple(hidden_states.shape)}"
            )

        output = hidden_states.clone()

        for period_id, adapter in enumerate(self.adapters):
            mask = period_ids == period_id

            if mask.any():
                output[mask, :, :] = adapter(hidden_states[mask, :, :])

        return output


class HistoricalTemporalBertForMLM(nn.Module):
    """
    Stable memory-efficient version:

    input
      -> frozen BERT encoder
      -> temporal adapter selected by document date
      -> MLM head

    By default, BERT is frozen and only temporal adapters + MLM head are trained.
    """

    def __init__(
        self,
        base_model_name: str = "dbmdz/bert-base-french-europeana-cased",
        num_periods: int = len(PERIODS),
        adapter_bottleneck_size: int = 64,
        adapter_dropout: float = 0.1,
        freeze_base: bool = True,
        train_mlm_head: bool = True,
    ):
        super().__init__()

        self.base = AutoModelForMaskedLM.from_pretrained(base_model_name)
        hidden_size = self.base.config.hidden_size

        self.temporal_adapter_bank = TemporalAdapterBank(
            hidden_size=hidden_size,
            num_periods=num_periods,
            bottleneck_size=adapter_bottleneck_size,
            dropout=adapter_dropout,
        )

        self.num_periods = num_periods
        self.freeze_base = freeze_base

        if freeze_base:
            for param in self.base.bert.parameters():
                param.requires_grad = False

        if not train_mlm_head:
            for param in self.base.cls.parameters():
                param.requires_grad = False

    @property
    def config(self):
        return self.base.config

    def get_input_embeddings(self):
        return self.base.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base.set_input_embeddings(value)

    def print_trainable_parameters(self):
        trainable = 0
        total = 0

        for _, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        print(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        period_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> MaskedLMOutput:

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        if labels is not None and labels.dim() == 1:
            labels = labels.unsqueeze(0)

        batch_size = input_ids.shape[0]

        if period_ids is None:
            period_ids = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=input_ids.device,
            )

        if period_ids.dim() == 0:
            period_ids = period_ids.unsqueeze(0)

        period_ids = period_ids.to(input_ids.device).long()

        if self.freeze_base:
            with torch.no_grad():
                bert_outputs = self.base.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
        else:
            bert_outputs = self.base.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )

        sequence_output = bert_outputs.last_hidden_state

        sequence_output = self.temporal_adapter_bank(
            hidden_states=sequence_output,
            period_ids=period_ids,
        )

        prediction_scores = self.base.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=None,
            attentions=None,
        )

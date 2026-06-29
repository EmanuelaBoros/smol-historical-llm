from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

# ---------------------------------------------------------------------
# Period mapping
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# Temporal adapter
# ---------------------------------------------------------------------


class TemporalAdapter(nn.Module):
    """
    Small parameter-efficient adapter.

    This is the 'Horizon Adapter' from your figure, simplified.
    """

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

        # Important: start close to identity behavior
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        x = self.down(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up(x)

        return residual + x


class TemporalAdapterBank(nn.Module):
    """
    One adapter per time period.

    period_ids shape: [batch]
    hidden_states shape: [batch, seq_len, hidden]
    """

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
        """
        hidden_states: [batch, seq_len, hidden]
        period_ids: [batch]
        """

        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected hidden_states to have shape [batch, seq_len, hidden], "
                f"but got {tuple(hidden_states.shape)}"
            )

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
                selected = hidden_states[mask, :, :]
                output[mask, :, :] = adapter(selected)

        return output


# ---------------------------------------------------------------------
# Long-horizon historical BERT
# ---------------------------------------------------------------------


class HistoricalTemporalBertForMLM(nn.Module):
    """
    BERT MLM model with temporal adapters.

    Backbone:
        AutoModelForMaskedLM

    Extra:
        one temporal adapter bank after each BERT encoder layer.

    This is a practical implementation of:
        A. Horizon adapters
        B. Long-horizon transformer backbone
        E. MLM task head
    """

    def __init__(
        self,
        base_model_name: str = "dbmdz/bert-base-french-europeana-cased",
        num_periods: int = len(PERIODS),
        adapter_bottleneck_size: int = 64,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()

        self.base = AutoModelForMaskedLM.from_pretrained(base_model_name)

        config = self.base.config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers

        self.temporal_adapters = nn.ModuleList(
            [
                TemporalAdapterBank(
                    hidden_size=hidden_size,
                    num_periods=num_periods,
                    bottleneck_size=adapter_bottleneck_size,
                    dropout=adapter_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_periods = num_periods

    @property
    def config(self):
        return self.base.config

    def get_input_embeddings(self):
        return self.base.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        period_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> MaskedLMOutput:
        if period_ids is None:
            # Default to general / first period if not provided
            period_ids = torch.zeros(
                input_ids.shape[0],
                dtype=torch.long,
                device=input_ids.device,
            )

        bert = self.base.bert

        extended_attention_mask = bert.get_extended_attention_mask(
            attention_mask,
            input_ids.shape,
        )

        embedding_output = bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        hidden_states = embedding_output

        for i, layer_module in enumerate(bert.encoder.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
            )

            hidden_states = layer_outputs[0]

            # Temporal adapter injection
            hidden_states = self.temporal_adapters[i](
                hidden_states=hidden_states,
                period_ids=period_ids,
            )

        sequence_output = hidden_states
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

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from modeling_temporal_bert import (
    HistoricalTemporalBertForMLM,
    year_to_period_id,
)

# ---------------------------------------------------------------------
# HIPE TSV reader
# ---------------------------------------------------------------------


def parse_year(date_value: str | None) -> int | None:
    if not date_value:
        return None

    try:
        return int(str(date_value)[:4])
    except ValueError:
        return None


def read_hipe_tsv(path: str, label_column: str = "NE-COARSE-LIT") -> list[dict]:
    """
    Reads HIPE-style TSV files.

    Returns one example per sentence:

    {
        "tokens": [...],
        "labels": [...],
        "date": "1790-01-02",
        "period_id": int,
        "document_id": str,
    }
    """

    examples = []

    current_tokens = []
    current_labels = []

    header = None
    token_idx = None
    label_idx = None

    current_date = None
    current_doc_id = None

    def flush_sentence():
        nonlocal current_tokens, current_labels

        if current_tokens:
            year = parse_year(current_date)
            period_id = year_to_period_id(year) if year is not None else 0

            examples.append(
                {
                    "tokens": current_tokens,
                    "labels": current_labels,
                    "date": current_date,
                    "period_id": period_id,
                    "document_id": current_doc_id,
                }
            )

        current_tokens = []
        current_labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                flush_sentence()
                continue

            if line.startswith("#"):
                if line.startswith("# hipe2022:date"):
                    current_date = line.split("=", 1)[1].strip()
                elif line.startswith("# hipe2022:document_id"):
                    current_doc_id = line.split("=", 1)[1].strip()
                continue

            parts = line.split("\t")

            if parts[0] == "TOKEN":
                header = parts

                if "TOKEN" not in header:
                    raise ValueError(f"TOKEN column not found in {path}")

                if label_column not in header:
                    raise ValueError(
                        f"Column {label_column} not found in {path}. "
                        f"Available columns: {header}"
                    )

                token_idx = header.index("TOKEN")
                label_idx = header.index(label_column)
                continue

            if header is None:
                raise ValueError(f"No TSV header found before data in {path}")

            if len(parts) <= max(token_idx, label_idx):
                continue

            token = parts[token_idx]
            label = parts[label_idx]

            if label == "_":
                label = "O"

            current_tokens.append(token)
            current_labels.append(label)

    flush_sentence()

    return examples


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


class HipeNERDataset(Dataset):
    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        label2id: dict[str, int],
        max_length: int = 256,
        label_all_tokens: bool = False,
        include_period_ids: bool = False,
    ):
        self.features = []

        for ex in examples:
            tokenized = tokenizer(
                ex["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
            )

            word_ids = tokenized.word_ids()
            labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != previous_word_id:
                    labels.append(label2id[ex["labels"][word_id]])
                else:
                    if label_all_tokens:
                        labels.append(label2id[ex["labels"][word_id]])
                    else:
                        labels.append(-100)

                previous_word_id = word_id

            tokenized["labels"] = labels

            if include_period_ids:
                tokenized["period_ids"] = int(ex["period_id"])

            self.features.append(tokenized)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# ---------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------


@dataclass
class TemporalTokenClassificationCollator:
    tokenizer: object

    def __post_init__(self):
        self.base_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, examples: list[dict]) -> dict:
        examples = [dict(ex) for ex in examples]

        period_ids = torch.tensor(
            [int(ex.pop("period_ids")) for ex in examples],
            dtype=torch.long,
        )

        batch = self.base_collator(examples)
        batch["period_ids"] = period_ids

        return batch


# ---------------------------------------------------------------------
# Temporal BERT NER model
# ---------------------------------------------------------------------


class HistoricalTemporalBertForTokenClassification(nn.Module):
    """
    Temporal BERT for NER.

    Reuses:
    - BERT encoder
    - temporal adapter bank

    Replaces:
    - MLM head with token classification head
    """

    def __init__(
        self,
        base_model: str,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        adapter_bottleneck_size: int = 64,
        adapter_dropout: float = 0.1,
        temporal_mlm_checkpoint: Optional[str] = None,
    ):
        super().__init__()

        self.temporal_mlm = HistoricalTemporalBertForMLM(
            base_model_name=base_model,
            adapter_bottleneck_size=adapter_bottleneck_size,
            adapter_dropout=adapter_dropout,
        )

        if temporal_mlm_checkpoint is not None:
            self._load_temporal_checkpoint(temporal_mlm_checkpoint)

        hidden_size = self.temporal_mlm.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.num_labels = num_labels
        self.temporal_mlm.config.num_labels = num_labels
        self.temporal_mlm.config.id2label = id2label
        self.temporal_mlm.config.label2id = label2id

    @property
    def config(self):
        return self.temporal_mlm.config

    def _load_temporal_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)

        if path.is_dir():
            safetensors_path = path / "model.safetensors"
            bin_path = path / "pytorch_model.bin"

            if safetensors_path.exists():
                state_dict = load_safetensors(str(safetensors_path))
            elif bin_path.exists():
                state_dict = torch.load(str(bin_path), map_location="cpu")
            else:
                print(f"No model weights found in {checkpoint_path}. Using base init.")
                return
        else:
            if str(path).endswith(".safetensors"):
                state_dict = load_safetensors(str(path))
            else:
                state_dict = torch.load(str(path), map_location="cpu")

        missing, unexpected = self.temporal_mlm.load_state_dict(
            state_dict,
            strict=False,
        )

        print(f"Loaded temporal MLM checkpoint: {checkpoint_path}")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        period_ids=None,
        token_type_ids=None,
    ):
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

        bert_outputs = self.temporal_mlm.base.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = bert_outputs.last_hidden_state

        sequence_output = self.temporal_mlm.temporal_adapter_bank(
            hidden_states=sequence_output,
            period_ids=period_ids,
        )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
        }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def build_label_maps(*datasets: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    labels = set()

    for dataset in datasets:
        for ex in dataset:
            labels.update(ex["labels"])

    labels = sorted(labels)

    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label


def make_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            current_preds = []
            current_labels = []

            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue

                current_preds.append(id2label[int(pred_id)])
                current_labels.append(id2label[int(label_id)])

            true_predictions.append(current_preds)
            true_labels.append(current_labels)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate normal BERT vs temporal BERT on HIPE NER."
    )

    parser.add_argument(
        "--model_type",
        choices=["normal_bert", "temporal_bert"],
        required=True,
    )

    parser.add_argument("--train_file", required=True)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--test_file", default=None)

    parser.add_argument(
        "--label_column",
        default="NE-COARSE-LIT",
        help="Examples: NE-COARSE-LIT, NE-COARSE-METO, NE-FINE-LIT.",
    )

    parser.add_argument(
        "--base_model",
        default="dbmdz/bert-base-french-europeana-cased",
    )

    parser.add_argument(
        "--temporal_mlm_checkpoint",
        default=None,
        help="Local path to trained long-horizon MLM checkpoint.",
    )

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", "--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--adapter_bottleneck_size", type=int, default=64)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)

    args = parser.parse_args()

    if args.model_type == "temporal_bert" and args.temporal_mlm_checkpoint is None:
        raise ValueError(
            "--temporal_mlm_checkpoint is required when --model_type temporal_bert"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading HIPE files...")

    train_examples = read_hipe_tsv(args.train_file, args.label_column)
    dev_examples = read_hipe_tsv(args.dev_file, args.label_column)
    test_examples = (
        read_hipe_tsv(args.test_file, args.label_column)
        if args.test_file is not None
        else None
    )

    print(f"Train sentences: {len(train_examples)}")
    print(f"Dev sentences: {len(dev_examples)}")
    if test_examples is not None:
        print(f"Test sentences: {len(test_examples)}")

    label2id, id2label = build_label_maps(
        train_examples,
        dev_examples,
        test_examples or [],
    )

    print("Labels:")
    print(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    include_period_ids = args.model_type == "temporal_bert"

    train_dataset = HipeNERDataset(
        train_examples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        include_period_ids=include_period_ids,
    )

    dev_dataset = HipeNERDataset(
        dev_examples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        include_period_ids=include_period_ids,
    )

    test_dataset = (
        HipeNERDataset(
            test_examples,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            include_period_ids=include_period_ids,
        )
        if test_examples is not None
        else None
    )

    if args.model_type == "normal_bert":
        print("Loading normal BERT for token classification...")

        model = AutoModelForTokenClassification.from_pretrained(
            args.base_model,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )

        collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )

    else:
        print("Loading temporal BERT for token classification...")

        model = HistoricalTemporalBertForTokenClassification(
            base_model=args.base_model,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            adapter_bottleneck_size=args.adapter_bottleneck_size,
            adapter_dropout=args.adapter_dropout,
            temporal_mlm_checkpoint=args.temporal_mlm_checkpoint,
        )

        collator = TemporalTokenClassificationCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(id2label),
    )

    print("Training NER model...")
    trainer.train()

    print("Evaluating on dev...")
    dev_metrics = trainer.evaluate(dev_dataset)
    print(dev_metrics)
    save_json(output_dir / "dev_metrics.json", dev_metrics)

    if test_dataset is not None:
        print("Evaluating on test...")
        test_metrics = trainer.evaluate(test_dataset)
        print(test_metrics)
        save_json(output_dir / "test_metrics.json", test_metrics)

        print("Saving classification report...")
        predictions = trainer.predict(test_dataset)
        pred_ids = np.argmax(predictions.predictions, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(pred_ids, predictions.label_ids):
            current_preds = []
            current_labels = []

            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue

                current_preds.append(id2label[int(pred_id)])
                current_labels.append(id2label[int(label_id)])

            true_predictions.append(current_preds)
            true_labels.append(current_labels)

        report = classification_report(true_labels, true_predictions)

        with open(
            output_dir / "test_classification_report.txt", "w", encoding="utf-8"
        ) as f:
            f.write(report)

        print(report)

    print("Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
            },
            f,
            indent=2,
        )

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

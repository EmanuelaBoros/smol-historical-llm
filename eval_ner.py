from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from modeling_temporal_bert import TemporalAdapterBank, PERIODS, year_to_period_id

# ---------------------------------------------------------------------
# HIPE reader
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

    Expected columns:
    TOKEN NE-COARSE-LIT NE-COARSE-METO NE-FINE-LIT NE-FINE-METO
    NE-FINE-COMP NE-NESTED NEL-LIT NEL-METO MISC

    Splits examples on:
    - blank lines
    - MISC containing EndOfSentence
    - new document_id metadata if tokens are already buffered

    Returns one example per sentence/span:
    {
        "tokens": [...],
        "labels": [...],
        "date": "...",
        "period_id": int,
        "document_id": "...",
    }
    """

    examples = []

    current_tokens = []
    current_labels = []

    header = None
    token_idx = None
    label_idx = None
    misc_idx = None

    current_date = None
    current_doc_id = None

    def flush_sentence():
        nonlocal current_tokens, current_labels

        if not current_tokens:
            return

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

            # Blank line = sentence/document boundary
            if not line:
                flush_sentence()
                continue

            # Metadata
            if line.startswith("#"):
                if line.startswith("# hipe2022:document_id"):
                    # If a new document starts without a blank line, flush previous tokens
                    flush_sentence()
                    current_doc_id = line.split("=", 1)[1].strip()

                elif line.startswith("# hipe2022:date"):
                    current_date = line.split("=", 1)[1].strip()

                continue

            parts = line.split("\t")

            # Header row
            if parts[0] == "TOKEN":
                flush_sentence()

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
                misc_idx = header.index("MISC") if "MISC" in header else None

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

            misc_value = (
                parts[misc_idx]
                if misc_idx is not None and len(parts) > misc_idx
                else ""
            )

            # HIPE sentence boundary
            if "EndOfSentence" in misc_value:
                flush_sentence()

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
        include_period_ids: bool = False,
        label_all_tokens: bool = False,
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
# Temporal BERT for NER
# ---------------------------------------------------------------------


class TemporalBertForTokenClassification(nn.Module):
    def __init__(
        self,
        base_model: str,
        temporal_model_id: str,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        adapter_bottleneck_size: int = 64,
        adapter_dropout: float = 0.1,
        freeze_base: bool = False,
        token: str | None = None,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(base_model, token=token)
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id

        self.bert = AutoModel.from_pretrained(base_model, token=token)

        self.temporal_adapter_bank = TemporalAdapterBank(
            hidden_size=self.config.hidden_size,
            num_periods=len(PERIODS),
            bottleneck_size=adapter_bottleneck_size,
            dropout=adapter_dropout,
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.num_labels = num_labels

        self._load_temporal_weights(
            temporal_model_id=temporal_model_id,
            token=token,
        )

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False

    def _load_temporal_weights(self, temporal_model_id: str, token: str | None = None):
        print(f"Downloading/loading temporal model: {temporal_model_id}")

        local_dir = snapshot_download(
            repo_id=temporal_model_id,
            token=token,
        )

        local_dir = Path(local_dir)

        adapter_path = local_dir / "temporal_adapter_bank.bin"
        full_bin_path = local_dir / "pytorch_model.bin"

        if adapter_path.exists():
            print(f"Loading adapter-only weights from {adapter_path}")
            adapter_state = torch.load(adapter_path, map_location="cpu")
            missing, unexpected = self.temporal_adapter_bank.load_state_dict(
                adapter_state,
                strict=False,
            )
            print(f"Adapter missing keys: {len(missing)}")
            print(f"Adapter unexpected keys: {len(unexpected)}")
            return

        if full_bin_path.exists():
            print(f"Loading full temporal checkpoint from {full_bin_path}")
            state = torch.load(full_bin_path, map_location="cpu")

            # Load adapted BERT encoder if it exists.
            bert_state = {}
            for key, value in state.items():
                if key.startswith("base.bert."):
                    bert_state[key.replace("base.bert.", "")] = value

            if bert_state:
                missing, unexpected = self.bert.load_state_dict(
                    bert_state,
                    strict=False,
                )
                print(f"BERT missing keys: {len(missing)}")
                print(f"BERT unexpected keys: {len(unexpected)}")

            # Load temporal adapter bank.
            adapter_state = {}
            for key, value in state.items():
                if key.startswith("temporal_adapter_bank."):
                    adapter_state[key.replace("temporal_adapter_bank.", "")] = value

            if adapter_state:
                missing, unexpected = self.temporal_adapter_bank.load_state_dict(
                    adapter_state,
                    strict=False,
                )
                print(f"Adapter missing keys: {len(missing)}")
                print(f"Adapter unexpected keys: {len(unexpected)}")
            else:
                print("WARNING: no temporal_adapter_bank.* keys found.")

            return

        raise FileNotFoundError(
            f"No temporal_adapter_bank.bin or pytorch_model.bin found in {temporal_model_id}"
        )

    def print_trainable_parameters(self):
        total = 0
        trainable = 0

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
        input_ids,
        attention_mask=None,
        labels=None,
        period_ids=None,
        token_type_ids=None,
    ):
        batch_size = input_ids.shape[0]

        if period_ids is None:
            period_ids = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=input_ids.device,
            )

        period_ids = period_ids.to(input_ids.device).long()

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        sequence_output = self.temporal_adapter_bank(
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


def decode_predictions(predictions, label_ids, id2label: dict[int, str]):
    pred_ids = np.argmax(predictions, axis=-1)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(pred_ids, label_ids):
        current_preds = []
        current_labels = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue

            current_preds.append(id2label[int(pred_id)])
            current_labels.append(id2label[int(label_id)])

        true_predictions.append(current_preds)
        true_labels.append(current_labels)

    return true_predictions, true_labels


def compute_seqeval_metrics(true_labels, true_predictions) -> dict:
    return {
        "precision": float(precision_score(true_labels, true_predictions)),
        "recall": float(recall_score(true_labels, true_predictions)),
        "f1": float(f1_score(true_labels, true_predictions)),
        "num_sentences": len(true_labels),
        "num_entities": int(
            sum(1 for sent in true_labels for label in sent if label.startswith("B-"))
        ),
    }


def prediction_report(predictions, label_ids, id2label: dict[int, str]):
    true_predictions, true_labels = decode_predictions(
        predictions=predictions,
        label_ids=label_ids,
        id2label=id2label,
    )

    metrics = compute_seqeval_metrics(
        true_labels=true_labels,
        true_predictions=true_predictions,
    )

    report = classification_report(true_labels, true_predictions)

    return metrics, report, true_predictions, true_labels


def prediction_report_by_period(
    examples: list[dict],
    true_predictions: list[list[str]],
    true_labels: list[list[str]],
) -> dict:
    grouped = {}

    for ex, pred_seq, gold_seq in zip(examples, true_predictions, true_labels):
        period_id = int(ex.get("period_id", 0))

        if 0 <= period_id < len(PERIODS):
            period_name = PERIODS[period_id]
        else:
            period_name = str(period_id)

        if period_name not in grouped:
            grouped[period_name] = {
                "predictions": [],
                "labels": [],
            }

        grouped[period_name]["predictions"].append(pred_seq)
        grouped[period_name]["labels"].append(gold_seq)

    period_metrics = {}

    for period_name, values in grouped.items():
        period_metrics[period_name] = compute_seqeval_metrics(
            true_labels=values["labels"],
            true_predictions=values["predictions"],
        )

    return period_metrics


def print_period_metrics(period_metrics: dict) -> None:
    print("\nNER results by temporal period")
    print("=" * 80)
    print(
        f"{'period':<20} "
        f"{'sentences':>10} "
        f"{'entities':>10} "
        f"{'precision':>10} "
        f"{'recall':>10} "
        f"{'f1':>10}"
    )
    print("-" * 80)

    for period_name, metrics in period_metrics.items():
        print(
            f"{period_name:<20} "
            f"{metrics['num_sentences']:>10} "
            f"{metrics['num_entities']:>10} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f}"
        )

    print("=" * 80)


def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["normal_bert", "temporal_bert"],
        required=True,
    )

    parser.add_argument(
        "--base_model",
        default="dbmdz/bert-base-french-europeana-cased",
    )

    parser.add_argument(
        "--temporal_model_id",
        default=None,
        help="HF repo/path for temporal model. Required for temporal_bert.",
    )

    parser.add_argument("--train_file", required=True)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--test_file", required=True)

    parser.add_argument("--label_column", default="NE-COARSE-LIT")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", "--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=20)

    parser.add_argument("--adapter_bottleneck_size", type=int, default=64)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    parser.add_argument("--freeze_base", type=str2bool, default=False)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token", default=None)

    args = parser.parse_args()

    if args.mode == "temporal_bert" and args.temporal_model_id is None:
        raise ValueError("--temporal_model_id is required for temporal_bert")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading HIPE files...")
    train_examples = read_hipe_tsv(args.train_file, args.label_column)
    dev_examples = read_hipe_tsv(args.dev_file, args.label_column)
    test_examples = read_hipe_tsv(args.test_file, args.label_column)

    print(f"Train: {len(train_examples)}")
    print(f"Dev: {len(dev_examples)}")
    print(f"Test: {len(test_examples)}")

    label2id, id2label = build_label_maps(
        train_examples,
        dev_examples,
        test_examples,
    )

    print("Labels:")
    print(label2id)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        token=args.token,
    )

    include_period_ids = args.mode == "temporal_bert"

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

    test_dataset = HipeNERDataset(
        test_examples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        include_period_ids=include_period_ids,
    )

    if args.mode == "normal_bert":
        print("Loading normal BERT for NER...")

        model = AutoModelForTokenClassification.from_pretrained(
            args.base_model,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            token=args.token,
        )

        collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )

    else:
        print("Loading temporal BERT for NER...")

        model = TemporalBertForTokenClassification(
            base_model=args.base_model,
            temporal_model_id=args.temporal_model_id,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            adapter_bottleneck_size=args.adapter_bottleneck_size,
            adapter_dropout=args.adapter_dropout,
            freeze_base=args.freeze_base,
            token=args.token,
        )

        model.print_trainable_parameters()

        collator = TemporalTokenClassificationCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(),
        report_to="none",
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
    save_json(output_dir / "dev_metrics.json", dev_metrics)
    print(dev_metrics)

    print("Evaluating on test...")
    test_output = trainer.predict(test_dataset)

    test_metrics, test_report, true_predictions, true_labels = prediction_report(
        predictions=test_output.predictions,
        label_ids=test_output.label_ids,
        id2label=id2label,
    )

    test_metrics["test_loss"] = float(test_output.metrics.get("test_loss", 0.0))

    period_metrics = prediction_report_by_period(
        examples=test_examples,
        true_predictions=true_predictions,
        true_labels=true_labels,
    )

    save_json(output_dir / "test_metrics.json", test_metrics)
    save_json(output_dir / "test_metrics_by_period.json", period_metrics)

    with open(
        output_dir / "test_classification_report.txt", "w", encoding="utf-8"
    ) as f:
        f.write(test_report)

    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
            },
            f,
            indent=2,
        )

    tokenizer.save_pretrained(str(output_dir))

    print("Test metrics:")
    print(test_metrics)
    print(test_report)

    print_period_metrics(period_metrics)

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

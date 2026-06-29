from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from collator_temporal_mlm import TemporalDataCollatorForMLM
from modeling_temporal_bert import (
    HistoricalTemporalBertForMLM,
    year_to_period_id,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class Config:
    # Model
    base_model: str

    # Dataset
    dataset_name: str
    dataset_split: str
    text_column: str
    date_column: str
    ocr_column: str

    # Output
    output_dir: str
    output_repo: str
    push_to_hub: bool

    # Training size
    max_seq_length: int
    max_train_examples: int
    max_eval_examples: int
    max_steps: int

    # Optimization
    batch_size: int
    eval_batch_size: int
    grad_accum: int
    learning_rate: float
    warmup_steps: int

    # MLM
    mlm_probability: float

    # Adapter
    adapter_bottleneck_size: int
    adapter_dropout: float

    # Text filtering
    min_chars: int
    max_chars: int
    min_ocr: int | None

    # Optional temporal filtering
    start_year: int | None
    end_year: int | None

    # Logging / saving
    logging_steps: int
    eval_steps: int
    save_steps: int

    # Debug
    check_batch: bool


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in {"yes", "true", "t", "1", "y"}:
        return True

    if value in {"no", "false", "f", "0", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a long-horizon historical BERT with temporal adapters."
    )

    # Model
    parser.add_argument(
        "--base_model",
        default=os.environ.get(
            "BASE_MODEL",
            "dbmdz/bert-base-french-europeana-cased",
        ),
    )

    # Dataset
    parser.add_argument(
        "--dataset_name",
        default=os.environ.get(
            "DATASET_NAME",
            "PleIAs/French-PD-Newspapers",
        ),
    )
    parser.add_argument(
        "--dataset_split",
        default=os.environ.get("DATASET_SPLIT", "train"),
    )
    parser.add_argument(
        "--text_column",
        default=os.environ.get("TEXT_COLUMN", "complete_text"),
    )
    parser.add_argument(
        "--date_column",
        default=os.environ.get("DATE_COLUMN", "date"),
    )
    parser.add_argument(
        "--ocr_column",
        default=os.environ.get("OCR_COLUMN", "ocr"),
    )

    # Output
    parser.add_argument(
        "--output_dir",
        default=os.environ.get(
            "OUTPUT_DIR",
            "long-horizon-historical-bert",
        ),
    )
    parser.add_argument(
        "--output_repo",
        default=os.environ.get(
            "OUTPUT_REPO",
            "EmanuelaBoros/long-horizon-historical-bert",
        ),
    )
    parser.add_argument(
        "--push_to_hub",
        type=str2bool,
        default=str2bool(os.environ.get("PUSH_TO_HUB", "true")),
    )

    # Training size
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=int(os.environ.get("MAX_SEQ_LENGTH", "512")),
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=int(os.environ.get("MAX_TRAIN_EXAMPLES", "50000")),
    )
    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=int(os.environ.get("MAX_EVAL_EXAMPLES", "2000")),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(os.environ.get("MAX_STEPS", "3000")),
    )

    # Optimization
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "8")),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=int(os.environ.get("EVAL_BATCH_SIZE", "8")),
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=int(os.environ.get("GRAD_ACCUM", "4")),
    )
    parser.add_argument(
        "--learning_rate",
        "--lr",
        type=float,
        default=float(os.environ.get("LR", "5e-5")),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=int(os.environ.get("WARMUP_STEPS", "200")),
    )

    # MLM
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=float(os.environ.get("MLM_PROBABILITY", "0.15")),
    )

    # Adapter
    parser.add_argument(
        "--adapter_bottleneck_size",
        "--adapter_bottleneck",
        type=int,
        default=int(os.environ.get("ADAPTER_BOTTLENECK", "64")),
    )
    parser.add_argument(
        "--adapter_dropout",
        type=float,
        default=float(os.environ.get("ADAPTER_DROPOUT", "0.1")),
    )

    # Text filtering
    parser.add_argument(
        "--min_chars",
        type=int,
        default=int(os.environ.get("MIN_CHARS", "500")),
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=int(os.environ.get("MAX_CHARS", "12000")),
    )
    parser.add_argument(
        "--min_ocr",
        type=optional_int,
        default=optional_int(os.environ.get("MIN_OCR")),
    )

    # Temporal filtering
    parser.add_argument(
        "--start_year",
        type=optional_int,
        default=optional_int(os.environ.get("START_YEAR")),
    )
    parser.add_argument(
        "--end_year",
        type=optional_int,
        default=optional_int(os.environ.get("END_YEAR")),
    )

    # Logging / saving
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=int(os.environ.get("LOGGING_STEPS", "20")),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=int(os.environ.get("EVAL_STEPS", "500")),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=int(os.environ.get("SAVE_STEPS", "500")),
    )

    # Debug
    parser.add_argument(
        "--check_batch",
        type=str2bool,
        default=str2bool(os.environ.get("CHECK_BATCH", "true")),
    )

    args = parser.parse_args()

    return Config(**vars(args))


CFG = parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. On HF Jobs, pass it with: --secrets HF_TOKEN"
        )
    return token


def parse_year(date_value: object) -> int | None:
    if date_value is None:
        return None

    date_str = str(date_value).strip()

    if not date_str:
        return None

    try:
        return int(date_str[:4])
    except ValueError:
        return None


def keep_by_ocr(example: dict) -> bool:
    if CFG.min_ocr is None:
        return True

    value = example.get(CFG.ocr_column)

    if value is None:
        return False

    try:
        return int(value) >= CFG.min_ocr
    except ValueError:
        return False


def keep_by_year(example: dict) -> bool:
    year = parse_year(example.get(CFG.date_column))

    if year is None:
        return False

    if CFG.start_year is not None and year < CFG.start_year:
        return False

    if CFG.end_year is not None and year > CFG.end_year:
        return False

    return True


def clean_and_add_period(example: dict) -> dict:
    text = example.get(CFG.text_column)
    year = parse_year(example.get(CFG.date_column))

    if text is None or year is None:
        return {
            "text": "",
            "period_ids": 0,
            "year": -1,
        }

    text = str(text)
    text = " ".join(text.split())

    if len(text) < CFG.min_chars:
        return {
            "text": "",
            "period_ids": 0,
            "year": year,
        }

    text = text[: CFG.max_chars]

    return {
        "text": text,
        "period_ids": year_to_period_id(year),
        "year": year,
    }


def print_config() -> None:
    print("\n===== Long-Horizon Historical BERT configuration =====")
    for key, value in CFG.__dict__.items():
        print(f"{key}: {value}")
    print("=====================================================\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print_config()

    token = get_hf_token()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CFG.base_model,
        token=token,
        use_fast=True,
    )

    print("Loading long-horizon temporal BERT...")
    model = HistoricalTemporalBertForMLM(
        base_model_name=CFG.base_model,
        adapter_bottleneck_size=CFG.adapter_bottleneck_size,
        adapter_dropout=CFG.adapter_dropout,
    )

    print("Loading dataset...")
    raw = load_dataset(
        CFG.dataset_name,
        split=CFG.dataset_split,
        streaming=True,
        token=token,
    )

    original_columns = [
        "file_id",
        "ocr",
        "title",
        "date",
        "author",
        "page_count",
        "word_count",
        "character_count",
        "complete_text",
    ]

    raw = raw.filter(keep_by_ocr)

    if CFG.start_year is not None or CFG.end_year is not None:
        print("Filtering by year...")
        raw = raw.filter(keep_by_year)

    print("Cleaning text and assigning temporal period...")
    raw = raw.map(clean_and_add_period)
    raw = raw.filter(lambda x: x["text"] != "")

    train_stream = raw.take(CFG.max_train_examples)
    eval_stream = raw.skip(CFG.max_train_examples).take(CFG.max_eval_examples)

    def tokenize(example: dict) -> dict:
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=CFG.max_seq_length,
            padding=False,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "period_ids": example["period_ids"],
        }

    columns_to_remove = original_columns + ["text", "year"]

    train_dataset = train_stream.map(
        tokenize,
        remove_columns=columns_to_remove,
    )

    eval_dataset = eval_stream.map(
        tokenize,
        remove_columns=columns_to_remove,
    )

    collator = TemporalDataCollatorForMLM(
        tokenizer=tokenizer,
        mlm_probability=CFG.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        max_steps=CFG.max_steps,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.eval_batch_size,
        gradient_accumulation_steps=CFG.grad_accum,
        learning_rate=CFG.learning_rate,
        warmup_steps=CFG.warmup_steps,
        logging_steps=CFG.logging_steps,
        eval_strategy="steps",
        eval_steps=CFG.eval_steps,
        save_steps=CFG.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        push_to_hub=CFG.push_to_hub,
        hub_model_id=CFG.output_repo if CFG.push_to_hub else None,
        hub_token=token if CFG.push_to_hub else None,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    if CFG.check_batch:
        print("Checking one batch...")

        batch = next(iter(trainer.get_train_dataloader()))

        for key, value in batch.items():
            if hasattr(value, "shape"):
                print(key, value.shape)
            else:
                print(key, type(value))

        print("Batch check done.")

    print("Starting long-horizon temporal adapter training...")
    trainer.train()

    print("Saving model locally...")
    trainer.save_model(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)

    if CFG.push_to_hub:
        print("Pushing model and tokenizer to the Hub...")
        model.base.config.save_pretrained(CFG.output_dir)
        trainer.push_to_hub()
        tokenizer.push_to_hub(CFG.output_repo, token=token)
        print(f"Done. Model pushed to: {CFG.output_repo}")
    else:
        print(f"Done. Model saved locally to: {CFG.output_dir}")


if __name__ == "__main__":
    main()

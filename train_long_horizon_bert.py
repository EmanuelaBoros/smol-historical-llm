from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
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
    base_model: str = os.environ.get(
        "BASE_MODEL",
        "dbmdz/bert-base-french-europeana-cased",
    )

    dataset_name: str = os.environ.get(
        "DATASET_NAME",
        "PleIAs/French-PD-Newspapers",
    )
    dataset_split: str = os.environ.get("DATASET_SPLIT", "train")

    text_column: str = os.environ.get("TEXT_COLUMN", "complete_text")
    date_column: str = os.environ.get("DATE_COLUMN", "date")
    ocr_column: str = os.environ.get("OCR_COLUMN", "ocr")

    output_dir: str = os.environ.get(
        "OUTPUT_DIR",
        "long-horizon-historical-bert",
    )
    output_repo: str = os.environ.get(
        "OUTPUT_REPO",
        "EmanuelaBoros/long-horizon-historical-bert",
    )

    max_seq_length: int = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
    max_train_examples: int = int(os.environ.get("MAX_TRAIN_EXAMPLES", "50000"))
    max_eval_examples: int = int(os.environ.get("MAX_EVAL_EXAMPLES", "2000"))
    max_steps: int = int(os.environ.get("MAX_STEPS", "3000"))

    batch_size: int = int(os.environ.get("BATCH_SIZE", "8"))
    eval_batch_size: int = int(os.environ.get("EVAL_BATCH_SIZE", "8"))
    grad_accum: int = int(os.environ.get("GRAD_ACCUM", "4"))

    learning_rate: float = float(os.environ.get("LR", "5e-5"))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", "200"))

    mlm_probability: float = float(os.environ.get("MLM_PROBABILITY", "0.15"))

    adapter_bottleneck_size: int = int(os.environ.get("ADAPTER_BOTTLENECK", "64"))
    adapter_dropout: float = float(os.environ.get("ADAPTER_DROPOUT", "0.1"))

    min_chars: int = int(os.environ.get("MIN_CHARS", "500"))
    max_chars: int = int(os.environ.get("MAX_CHARS", "12000"))
    min_ocr: int | None = (
        int(os.environ["MIN_OCR"]) if os.environ.get("MIN_OCR") else None
    )

    logging_steps: int = int(os.environ.get("LOGGING_STEPS", "20"))
    eval_steps: int = int(os.environ.get("EVAL_STEPS", "500"))
    save_steps: int = int(os.environ.get("SAVE_STEPS", "500"))


CFG = Config()


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
    # collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=True,
    #     mlm_probability=CFG.mlm_probability,
    # )

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
        push_to_hub=True,
        hub_model_id=CFG.output_repo,
        hub_token=token,
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

    print("Starting long-horizon temporal adapter training...")
    trainer.train()

    print("Saving model locally...")
    trainer.save_model(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)

    print("Pushing model and tokenizer to the Hub...")
    model.base.config.save_pretrained(CFG.output_dir)
    trainer.push_to_hub()
    tokenizer.push_to_hub(CFG.output_repo, token=token)

    print(f"Done. Model pushed to: {CFG.output_repo}")


if __name__ == "__main__":
    main()

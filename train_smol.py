from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class Config:
    # Model
    base_model: str = os.environ.get(
        "BASE_MODEL",
        "HuggingFaceTB/SmolLM3-3B-Base",
    )

    # Dataset
    dataset_name: str = os.environ.get(
        "DATASET_NAME",
        "PleIAs/French-PD-Newspapers",
    )
    dataset_split: str = os.environ.get("DATASET_SPLIT", "train")
    text_column: str = os.environ.get("TEXT_COLUMN", "complete_text")

    # Output
    output_dir: str = os.environ.get("OUTPUT_DIR", "smol-historical-llm")
    output_repo: str = os.environ.get(
        "OUTPUT_REPO",
        "EmanuelaBoros/smol-historical-llm",
    )

    # Training size
    max_seq_length: int = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
    max_train_examples: int = int(os.environ.get("MAX_TRAIN_EXAMPLES", "30000"))
    max_eval_examples: int = int(os.environ.get("MAX_EVAL_EXAMPLES", "1000"))
    max_steps: int = int(os.environ.get("MAX_STEPS", "1500"))

    # Optimization
    batch_size: int = int(os.environ.get("BATCH_SIZE", "1"))
    eval_batch_size: int = int(os.environ.get("EVAL_BATCH_SIZE", "1"))
    grad_accum: int = int(os.environ.get("GRAD_ACCUM", "16"))
    learning_rate: float = float(os.environ.get("LR", "2e-4"))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", "100"))

    # Logging / saving
    logging_steps: int = int(os.environ.get("LOGGING_STEPS", "20"))
    eval_steps: int = int(os.environ.get("EVAL_STEPS", "250"))
    save_steps: int = int(os.environ.get("SAVE_STEPS", "250"))

    # LoRA
    lora_r: int = int(os.environ.get("LORA_R", "32"))
    lora_alpha: int = int(os.environ.get("LORA_ALPHA", "64"))
    lora_dropout: float = float(os.environ.get("LORA_DROPOUT", "0.05"))

    # Text cleaning
    min_chars: int = int(os.environ.get("MIN_CHARS", "500"))
    max_chars: int = int(os.environ.get("MAX_CHARS", "12000"))


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


def clean_text(example: dict) -> dict:
    """
    Convert one dataset example into a training text.
    """
    text = example.get(CFG.text_column)

    if text is None:
        return {"text": ""}

    text = str(text)
    text = " ".join(text.split())

    if len(text) < CFG.min_chars:
        return {"text": ""}

    text = text[: CFG.max_chars]

    text = "<|historical_document|>\n" f"{text}\n" "<|end_document|>"

    return {"text": text}


def print_config() -> None:
    print("\n===== Training configuration =====")
    for key, value in CFG.__dict__.items():
        print(f"{key}: {value}")
    print("==================================\n")


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------


def main() -> None:
    print_config()

    token = get_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(
        CFG.base_model,
        token=token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        }

    print("Loading model in 4-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        CFG.base_model,
        token=token,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")

    raw = load_dataset(
        CFG.dataset_name,
        split=CFG.dataset_split,
        streaming=True,
        token=token,
    )

    # Keep original column names so we can remove metadata later.
    original_columns = list(raw.features.keys()) if raw.features is not None else []

    raw = raw.map(clean_text)
    raw = raw.filter(lambda x: x["text"] != "")

    train_stream = raw.take(CFG.max_train_examples)
    eval_stream = raw.skip(CFG.max_train_examples).take(CFG.max_eval_examples)

    columns_to_remove = original_columns + ["text"]

    train_dataset = train_stream.map(
        tokenize,
        remove_columns=columns_to_remove,
    )

    eval_dataset = eval_stream.map(
        tokenize,
        remove_columns=columns_to_remove,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
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
        bf16=True,
        optim="paged_adamw_8bit",
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

    print("Starting training...")
    trainer.train()

    print("Pushing adapter and tokenizer to the Hub...")
    trainer.push_to_hub()
    tokenizer.push_to_hub(CFG.output_repo, token=token)

    print(f"Done. Model adapter pushed to: {CFG.output_repo}")


if __name__ == "__main__":
    main()

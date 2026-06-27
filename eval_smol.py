from __future__ import annotations

import math
import os
import gc

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "HuggingFaceTB/SmolLM3-3B-Base"
ADAPTER_MODEL = "emanuelaboros/smol-historical-llm-test"

DATASET_NAME = "PleIAs/French-PD-Newspapers"
TEXT_COLUMN = "complete_text"

MAX_DOCS = int(os.environ.get("MAX_DOCS", "20"))
SKIP_N = int(os.environ.get("SKIP_N", "500"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "4000"))
MIN_CHARS = int(os.environ.get("MIN_CHARS", "500"))

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "2048"))
STRIDE = int(os.environ.get("STRIDE", "1024"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_eval_texts():
    print("Loading evaluation texts...")

    raw = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )

    texts = []

    # Skip some examples so we do not evaluate on the tiny smoke-test examples.
    for ex in raw.skip(SKIP_N):
        text = ex.get(TEXT_COLUMN) or ""
        text = " ".join(str(text).split())

        if len(text) < MIN_CHARS:
            continue

        text = text[:MAX_CHARS]

        text = "<|historical_document|>\n" f"{text}\n" "<|end_document|>"

        texts.append(text)

        if len(texts) >= MAX_DOCS:
            break

    print(f"Loaded {len(texts)} documents.")
    return texts


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )

    model.eval()
    return model, tokenizer


def load_historical_adapter():
    model, tokenizer = load_base()

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_MODEL,
        token=os.environ.get("HF_TOKEN"),
    )

    model.eval()
    return model, tokenizer


@torch.no_grad()
def perplexity(model, tokenizer, texts):
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)

        seq_len = input_ids.size(1)
        prev_end = 0

        for begin in range(0, seq_len, STRIDE):
            end = min(begin + MAX_LENGTH, seq_len)
            trg_len = end - prev_end

            input_chunk = input_ids[:, begin:end]
            target_ids = input_chunk.clone()

            # Only compute loss on the new part, not the overlapping context.
            target_ids[:, :-trg_len] = -100

            out = model(input_chunk, labels=target_ids)

            n_tokens = (target_ids != -100).sum().item()
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens

            prev_end = end

            if end == seq_len:
                break

    loss = total_nll / total_tokens
    ppl = math.exp(loss)

    return loss, ppl, total_tokens


def evaluate_model(name, loader, texts):
    print(f"\nEvaluating: {name}")

    model, tokenizer = loader()
    loss, ppl, tokens = perplexity(model, tokenizer, texts)

    print(f"{name} loss: {loss:.4f}")
    print(f"{name} perplexity: {ppl:.2f}")
    print(f"{name} tokens: {tokens}")

    del model
    del tokenizer
    clear_memory()

    return {
        "model": name,
        "loss": loss,
        "perplexity": ppl,
        "tokens": tokens,
    }


def main():
    texts = load_eval_texts()

    results = []

    results.append(
        evaluate_model(
            "SmolLM3-3B-Base",
            load_base,
            texts,
        )
    )

    results.append(
        evaluate_model(
            "smol-historical-llm-test",
            load_historical_adapter,
            texts,
        )
    )

    print("\n=== Results ===")
    results = sorted(results, key=lambda x: x["perplexity"])

    for r in results:
        print(
            f"{r['model']:30s} "
            f"PPL={r['perplexity']:.2f} "
            f"LOSS={r['loss']:.4f} "
            f"TOKENS={r['tokens']}"
        )

    base = next(r for r in results if r["model"] == "SmolLM3-3B-Base")
    hist = next(r for r in results if r["model"] == "smol-historical-llm-test")

    improvement = (base["perplexity"] - hist["perplexity"]) / base["perplexity"] * 100

    print("\n=== Comparison ===")
    print(f"Base perplexity:       {base['perplexity']:.2f}")
    print(f"Historical perplexity: {hist['perplexity']:.2f}")
    print(f"Relative improvement:  {improvement:.2f}%")


if __name__ == "__main__":
    main()

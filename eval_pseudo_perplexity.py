from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from modeling_temporal_bert import HistoricalTemporalBertForMLM, year_to_period_id

try:
    from modeling_temporal_bert_v2 import HistoricalTemporalBertV2ForMLM
except ImportError:
    HistoricalTemporalBertV2ForMLM = None


def parse_year(date_value):
    if not date_value:
        return None
    try:
        return int(str(date_value)[:4])
    except ValueError:
        return None


def load_texts(
    dataset_name: str,
    split: str,
    text_column: str,
    date_column: str,
    max_examples: int,
    min_chars: int,
):
    ds = load_dataset(dataset_name, split=split, streaming=True)

    examples = []

    for ex in ds:
        text = ex.get(text_column)
        date = ex.get(date_column)

        if not text:
            continue

        text = str(text).strip()

        if len(text) < min_chars:
            continue

        year = parse_year(date)
        period_id = year_to_period_id(year) if year is not None else 0

        examples.append(
            {
                "text": text,
                "date": date,
                "year": year,
                "period_id": period_id,
            }
        )

        if len(examples) >= max_examples:
            break

    return examples


def load_base_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_temporal_v1(
    model_id: str,
    base_model: str,
    device: str,
    adapter_bottleneck_size: int = 64,
    adapter_dropout: float = 0.1,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = HistoricalTemporalBertForMLM(
        base_model_name=base_model,
        adapter_bottleneck_size=adapter_bottleneck_size,
        adapter_dropout=adapter_dropout,
        freeze_base=False,
        train_mlm_head=True,
    )

    local_dir = Path(snapshot_download(model_id))
    state_path = local_dir / "pytorch_model.bin"

    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    return tokenizer, model


def load_temporal_v2(
    model_id: str,
    base_model: str,
    device: str,
    adapter_bottleneck_size: int = 64,
    adapter_dropout: float = 0.1,
):
    if HistoricalTemporalBertV2ForMLM is None:
        raise ImportError(
            "Could not import HistoricalTemporalBertV2ForMLM. "
            "Make sure modeling_temporal_bert_v2.py is available."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = HistoricalTemporalBertV2ForMLM(
        base_model_name=base_model,
        adapter_bottleneck_size=adapter_bottleneck_size,
        adapter_dropout=adapter_dropout,
        freeze_base=False,
        train_mlm_head=True,
        use_period_embeddings=True,
        use_period_classifier=True,
        period_loss_weight=0.1,
    )

    local_dir = Path(snapshot_download(model_id))
    state_path = local_dir / "pytorch_model.bin"

    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def pseudo_perplexity_for_text(
    model,
    tokenizer,
    text: str,
    period_id: int | None = None,
    max_length: int = 128,
    device: str = "cuda",
):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    seq_len = input_ids.size(1)

    mask_token_id = tokenizer.mask_token_id
    special_ids = set(tokenizer.all_special_ids)

    losses = []

    for pos in range(seq_len):
        token_id = int(input_ids[0, pos])

        if token_id in special_ids:
            continue

        masked_input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        labels[0, pos] = input_ids[0, pos]
        masked_input_ids[0, pos] = mask_token_id

        kwargs = {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if period_id is not None:
            kwargs["period_ids"] = torch.tensor(
                [period_id],
                dtype=torch.long,
                device=device,
            )

        outputs = model(**kwargs)
        loss = outputs.loss

        if loss is not None and torch.isfinite(loss):
            losses.append(float(loss.item()))

    if not losses:
        return None

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)

    return {
        "loss": mean_loss,
        "pseudo_perplexity": ppl,
        "num_scored_tokens": len(losses),
    }


def evaluate_model(
    name: str,
    model,
    tokenizer,
    examples: list[dict],
    is_temporal: bool,
    max_length: int,
    device: str,
):
    total_loss = 0.0
    total_tokens = 0

    period_stats = {}

    for ex in tqdm(examples, desc=f"Evaluating {name}"):
        result = pseudo_perplexity_for_text(
            model=model,
            tokenizer=tokenizer,
            text=ex["text"],
            period_id=ex["period_id"] if is_temporal else None,
            max_length=max_length,
            device=device,
        )

        if result is None:
            continue

        n = result["num_scored_tokens"]
        loss = result["loss"]

        total_loss += loss * n
        total_tokens += n

        period_name = str(ex["period_id"])

        if period_name not in period_stats:
            period_stats[period_name] = {
                "loss_sum": 0.0,
                "tokens": 0,
            }

        period_stats[period_name]["loss_sum"] += loss * n
        period_stats[period_name]["tokens"] += n

    overall_loss = total_loss / max(total_tokens, 1)
    overall_ppl = math.exp(overall_loss)

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Overall loss: {overall_loss:.4f}")
    print(f"Overall pseudo-perplexity: {overall_ppl:.4f}")
    print(f"Scored tokens: {total_tokens}")

    print("\nBy period:")
    for period_id, values in sorted(period_stats.items(), key=lambda x: int(x[0])):
        loss = values["loss_sum"] / max(values["tokens"], 1)
        ppl = math.exp(loss)
        print(
            f"  period {period_id}: "
            f"loss={loss:.4f}, "
            f"pseudo_ppl={ppl:.4f}, "
            f"tokens={values['tokens']}"
        )

    return {
        "name": name,
        "loss": overall_loss,
        "pseudo_perplexity": overall_ppl,
        "tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        default="dbmdz/bert-base-french-europeana-cased",
    )
    parser.add_argument(
        "--temporal_v1_model",
        default="EmanuelaBoros/long-horizon-historical-bert",
    )
    parser.add_argument(
        "--temporal_v2_model",
        default="EmanuelaBoros/long-horizon-historical-bert-v2",
    )

    parser.add_argument(
        "--dataset_name",
        default="PleIAs/French-PD-Newspapers",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_column", default="complete_text")
    parser.add_argument("--date_column", default="date")

    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--min_chars", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--eval_base", action="store_true")
    parser.add_argument("--eval_v1", action="store_true")
    parser.add_argument("--eval_v2", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    examples = load_texts(
        dataset_name=args.dataset_name,
        split=args.split,
        text_column=args.text_column,
        date_column=args.date_column,
        max_examples=args.max_examples,
        min_chars=args.min_chars,
    )

    print(f"Loaded {len(examples)} examples.")

    if not args.eval_base and not args.eval_v1 and not args.eval_v2:
        args.eval_base = True
        args.eval_v1 = True
        args.eval_v2 = True

    results = []

    if args.eval_base:
        tokenizer, model = load_base_model(args.base_model, device)
        results.append(
            evaluate_model(
                name="base_bert",
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                is_temporal=False,
                max_length=args.max_length,
                device=device,
            )
        )

    if args.eval_v1:
        tokenizer, model = load_temporal_v1(
            model_id=args.temporal_v1_model,
            base_model=args.base_model,
            device=device,
        )
        results.append(
            evaluate_model(
                name="temporal_bert_v1",
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                is_temporal=True,
                max_length=args.max_length,
                device=device,
            )
        )

    if args.eval_v2:
        tokenizer, model = load_temporal_v2(
            model_id=args.temporal_v2_model,
            base_model=args.base_model,
            device=device,
        )
        results.append(
            evaluate_model(
                name="temporal_bert_v2",
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                is_temporal=True,
                max_length=args.max_length,
                device=device,
            )
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for result in results:
        print(
            f"{result['name']:<20} "
            f"loss={result['loss']:.4f} "
            f"pseudo_ppl={result['pseudo_perplexity']:.4f} "
            f"tokens={result['tokens']}"
        )


if __name__ == "__main__":
    main()

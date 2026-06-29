from __future__ import annotations

import argparse
import json


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", required=True)
    parser.add_argument("--temporal", required=True)
    args = parser.parse_args()

    normal = load_metrics(args.normal)
    temporal = load_metrics(args.temporal)

    metric_keys = [
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_loss",
    ]

    print("\nNER comparison")
    print("=" * 72)
    print(f"{'metric':<20} {'normal_bert':>15} {'temporal_bert':>15} {'delta':>15}")
    print("-" * 72)

    for key in metric_keys:
        n = normal.get(key)
        t = temporal.get(key)

        if n is None or t is None:
            continue

        print(f"{key:<20} {n:>15.6f} {t:>15.6f} {t - n:>15.6f}")

    print("=" * 72)


if __name__ == "__main__":
    main()

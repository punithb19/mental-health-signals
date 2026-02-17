#!/usr/bin/env python3
"""
Quick test: show sample pipeline outputs from a predictions JSONL file.
Usage: python scripts/quick_test_output.py [--pred path] [--n N]
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description="Show sample pipeline outputs.")
    ap.add_argument("--pred", default="data/splits/test_pred_baseline.jsonl", help="Predictions JSONL path.")
    ap.add_argument("-n", type=int, default=3, help="Number of samples to show (default 3).")
    args = ap.parse_args()

    samples = []
    with open(args.pred, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            if not line.strip():
                continue
            samples.append(json.loads(line))

    for i, row in enumerate(samples, 1):
        post = row.get("post", "")[:200]
        if len(row.get("post", "")) > 200:
            post += "..."
        intents = row.get("predicted_intents") or row.get("intent") or []
        if isinstance(intents, str):
            intents = [intents]
        concern = row.get("predicted_concern") or row.get("concern") or "—"
        crisis = row.get("crisis_level", "—")
        reply = (row.get("reply") or "")[:300]
        if len(row.get("reply", "")) > 300:
            reply += "..."

        print("=" * 70)
        print(f"SAMPLE {i}")
        print("=" * 70)
        print("POST:", post)
        print()
        print("INTENTS:", ", ".join(intents))
        print("CONCERN:", concern, "  |  CRISIS_LEVEL:", crisis)
        print()
        print("REPLY:", reply)
        print()
        citations = row.get("citations") or []
        if citations:
            first = citations[0]
            snip = (first.get("text") or first.get("snippet") or "")[:150]
            if len(first.get("text", "")) > 150:
                snip += "..."
            print("(First citation:", first.get("doc_id", "—"), "|", snip + ")")
        print()

    print("=" * 70)
    print(f"Showed {len(samples)} sample(s) from {args.pred}")
    print("=" * 70)


if __name__ == "__main__":
    main()

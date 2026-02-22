#!/usr/bin/env python3
"""
Evaluate RAG reply quality.

Usage:
  python scripts/evaluate.py --pred data/splits/test_pred.jsonl
  python scripts/evaluate.py --pred data/splits/test_pred.jsonl --output results/eval_scores.jsonl
  python scripts/evaluate.py --pred data/splits/test_pred.jsonl --enc sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import sys
from pathlib import Path

from mhsignals.evaluation.metrics import ReplyQualityEvaluator


def main():
    ap = argparse.ArgumentParser(description="Evaluate MH-SIGNALS reply quality.")
    ap.add_argument(
        "--pred", required=True,
        help="JSONL file with predictions (post, reply, citations).",
    )
    ap.add_argument(
        "--enc", default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer encoder for similarity scoring.",
    )
    ap.add_argument(
        "--output", default=None,
        help="Optional: write per-row scores to a JSONL file.",
    )
    args = ap.parse_args()

    # Validate input path before loading the encoder
    if not Path(args.pred).exists():
        print(f"[ERROR] Prediction file not found: {args.pred}", file=sys.stderr)
        sys.exit(1)

    evaluator = ReplyQualityEvaluator(encoder_name=args.enc)
    results = evaluator.evaluate_file(args.pred)
    evaluator.print_report(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nPer-row scores written to: {out_path}")


if __name__ == "__main__":
    main()

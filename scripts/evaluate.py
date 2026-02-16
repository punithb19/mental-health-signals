#!/usr/bin/env python3
"""
Evaluate RAG reply quality.

Usage:
  python scripts/evaluate.py --pred data/splits/test_pred.jsonl
  python scripts/evaluate.py --pred data/splits/test_pred.jsonl --enc sentence-transformers/all-MiniLM-L6-v2
"""

import argparse

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
    args = ap.parse_args()

    evaluator = ReplyQualityEvaluator(encoder_name=args.enc)
    results = evaluator.evaluate_file(args.pred)
    evaluator.print_report(results)


if __name__ == "__main__":
    main()

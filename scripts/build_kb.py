#!/usr/bin/env python3
"""
Build the knowledge base: CSV -> chunked JSONL -> embeddings -> FAISS index.

Usage:
  python scripts/build_kb.py --config configs/data.yaml
  python scripts/build_kb.py --config configs/data.yaml --csv data/raw/kb/custom.csv
"""

import argparse
import sys
from pathlib import Path

from mhsignals.config import load_data_config
from mhsignals.retriever.builder import KBBuilder


def main():
    ap = argparse.ArgumentParser(
        description="Build the MH-SIGNALS knowledge base (chunk + embed + FAISS index)."
    )
    ap.add_argument(
        "-c", "--config", default="configs/data.yaml",
        help="Path to data.yaml config.",
    )
    ap.add_argument(
        "--csv", default=None,
        help="Override CSV path (else uses paths.processed_csv from config).",
    )
    ap.add_argument(
        "--encoder", default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for embedding.",
    )
    ap.add_argument(
        "--batch_size", type=int, default=64,
        help="Encoding batch size.",
    )
    args = ap.parse_args()

    # Validate config path
    if not Path(args.config).exists():
        print(f"[ERROR] Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    data_cfg = load_data_config(args.config)
    builder = KBBuilder(data_cfg.kb)

    csv_path = args.csv or data_cfg.processed_csv or None

    # Validate CSV path before building
    if csv_path and not Path(csv_path).exists():
        print(f"[ERROR] CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    outputs = builder.build_all(
        csv_path=csv_path,
        encoder_name=args.encoder,
        batch_size=args.batch_size,
    )

    print("\n[build_kb] Complete!")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

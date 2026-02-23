#!/usr/bin/env python3
"""
Build the knowledge base: CSV -> chunked JSONL -> embeddings -> FAISS index.

Usage (from project root):
  python scripts/build_kb.py --config configs/data.yaml
  python scripts/build_kb.py --config configs/data.yaml --pipeline-config configs/pipeline.yaml  # use retriever encoder from pipeline
  python scripts/build_kb.py --config configs/data.yaml --csv data/raw/kb/custom.csv --encoder sentence-transformers/all-distilroberta-v1
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path so mhsignals is importable when run as script
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from mhsignals.config import load_data_config, load_pipeline_config  # noqa: E402
from mhsignals.retriever.builder import KBBuilder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Build the MH-SIGNALS knowledge base (chunk + embed + FAISS index)."
    )
    ap.add_argument(
        "-c", "--config", default="configs/data.yaml",
        help="Path to data.yaml config.",
    )
    ap.add_argument(
        "--pipeline-config", default=None,
        help="If set, use retriever.encoder_model from this pipeline config (ensures KB matches pipeline).",
    )
    ap.add_argument(
        "--csv", default=None,
        help="Override CSV path (else uses paths.processed_csv from config).",
    )
    ap.add_argument(
        "--encoder", default=None,
        help="SentenceTransformer model for embedding. Overridden by --pipeline-config if both set.",
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

    # Resolve encoder: pipeline-config > explicit --encoder > default
    encoder_name = args.encoder
    if args.pipeline_config:
        if not Path(args.pipeline_config).exists():
            print(f"[ERROR] Pipeline config not found: {args.pipeline_config}", file=sys.stderr)
            sys.exit(1)
        pipeline_cfg = load_pipeline_config(args.pipeline_config)
        encoder_name = pipeline_cfg.retriever.encoder_model
        print(f"[build_kb] Using encoder from pipeline: {encoder_name}")
    if encoder_name is None:
        encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

    data_cfg = load_data_config(args.config)
    builder = KBBuilder(data_cfg.kb)

    csv_path = args.csv or data_cfg.processed_csv or None

    # Validate CSV path before building
    if csv_path and not Path(csv_path).exists():
        print(f"[ERROR] CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    outputs = builder.build_all(
        csv_path=csv_path,
        encoder_name=encoder_name,
        batch_size=args.batch_size,
    )

    print("\n[build_kb] Complete!")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

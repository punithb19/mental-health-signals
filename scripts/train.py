#!/usr/bin/env python3
"""
Unified training script for MH-SIGNALS classifiers.

Replaces the 6 separate model scripts with a single entry point:
  python scripts/train.py --task intent --encoder minilm_lr --config configs/baseline_minilm.yaml
  python scripts/train.py --task concern --encoder minilm_lr --config configs/baseline_minilm.yaml

For LoRA-based training (requires GPU/MPS):
  python scripts/train.py --task intent --encoder lora --config configs/roberta_lora.yaml
  python scripts/train.py --task concern --encoder lora --config configs/roberta_lora_concern.yaml
"""

import argparse
import sys

from mhsignals.config import load_yaml


def main():
    ap = argparse.ArgumentParser(
        description="Train an MH-SIGNALS classifier (intent or concern)."
    )
    ap.add_argument(
        "--task",
        required=True,
        choices=["intent", "concern"],
        help="Classification task: intent (multi-label) or concern (3-class).",
    )
    ap.add_argument(
        "--encoder",
        required=True,
        choices=["minilm_lr", "lora"],
        help="Model type: minilm_lr (baseline) or lora (transformer fine-tuning).",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to model config YAML (e.g. configs/baseline_minilm.yaml).",
    )
    args = ap.parse_args()

    config = load_yaml(args.config)

    if args.task == "intent" and args.encoder == "minilm_lr":
        from mhsignals.classifiers.intent import MinilmLRIntentClassifier
        print("Training: MinilmLR Intent Classifier")
        clf = MinilmLRIntentClassifier.train(config)
        print("Model saved. Use the checkpoint path for pipeline.yaml.")

    elif args.task == "concern" and args.encoder == "minilm_lr":
        from mhsignals.classifiers.concern import MinilmLRConcernClassifier
        print("Training: MinilmLR Concern Classifier")
        clf = MinilmLRConcernClassifier.train(config)
        print("Model saved. Use the checkpoint path for pipeline.yaml.")

    elif args.task == "intent" and args.encoder == "lora":
        print("Training: LoRA Intent Classifier (RoBERTa)")
        from models.roberta_lora import main as lora_intent_main
        sys.argv = ["train.py", "--config", args.config]
        lora_intent_main()
        print("Model saved. Use the checkpoint path for pipeline.yaml.")

    elif args.task == "concern" and args.encoder == "lora":
        print("Training: LoRA Concern Classifier (RoBERTa)")
        from models.roberta_lora_concern import main as lora_concern_main
        sys.argv = ["train.py", "--config", args.config]
        lora_concern_main()
        print("Model saved. Use the checkpoint path for pipeline.yaml.")


if __name__ == "__main__":
    main()

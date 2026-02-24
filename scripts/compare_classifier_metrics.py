#!/usr/bin/env python3
"""
Compare two classifier evaluation JSON files (e.g. before vs after adding data / retraining).

Usage:
  python scripts/compare_classifier_metrics.py results/classifier_metrics_baseline.json results/classifier_metrics_after_retrain.json

To generate the files:
  1. Baseline (current checkpoints on current test set):
     python scripts/evaluate_classifiers.py --pipeline-config configs/pipeline.yaml --data-config configs/data.yaml --split test --output results/classifier_metrics_baseline.json
  2. Retrain intent and concern on the updated data, update pipeline.yaml checkpoints.
  3. After retrain:
     python scripts/evaluate_classifiers.py --pipeline-config configs/pipeline.yaml --data-config configs/data.yaml --split test --output results/classifier_metrics_after_retrain.json
  4. Compare:
     python scripts/compare_classifier_metrics.py results/classifier_metrics_baseline.json results/classifier_metrics_after_retrain.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_num(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:.4f}" if isinstance(x, float) else str(x)
    return str(x)


def diff_str(before, after) -> str:
    if before is None or after is None:
        return ""
    try:
        d = after - before
        sign = "+" if d >= 0 else ""
        return f" ({sign}{d:.4f})"
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(
        description="Compare two classifier evaluation JSONs (e.g. before vs after adding data).",
    )
    ap.add_argument("baseline", help="Path to baseline JSON (e.g. before adding data / before retrain).")
    ap.add_argument("after", help="Path to after JSON (e.g. after retrain on more data).")
    ap.add_argument(
        "--labels",
        action="store_true",
        help="Print per-label intent and concern F1 comparison.",
    )
    args = ap.parse_args()

    base_path = Path(args.baseline)
    after_path = Path(args.after)
    if not base_path.exists():
        print(f"[ERROR] Baseline file not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    if not after_path.exists():
        print(f"[ERROR] After file not found: {after_path}", file=sys.stderr)
        sys.exit(1)

    base = load_results(base_path)
    after = load_results(after_path)

    print("=" * 70)
    print("CLASSIFIER METRICS COMPARISON (baseline vs after)")
    print("=" * 70)
    print(f"  Baseline: {base_path.name}")
    print(f"  After:   {after_path.name}")
    print()

    # Intent
    if "intent" in base and "intent" in after:
        bi = base["intent"]
        ai = after["intent"]
        print("INTENT (multi-label)")
        print("-" * 50)
        for key in ["macro_f1", "micro_f1"]:
            bv = bi.get(key)
            av = ai.get(key)
            d = diff_str(bv, av) if (bv is not None and av is not None) else ""
            print(f"  {key:<15}  {fmt_num(bv):>10}  ->  {fmt_num(av):<10}{d}")
        if "pr_auc_macro" in bi or "pr_auc_macro" in ai:
            bv = bi.get("pr_auc_macro")
            av = ai.get("pr_auc_macro")
            d = diff_str(bv, av) if (bv is not None and av is not None) else ""
            print(f"  {'pr_auc_macro':<15}  {fmt_num(bv):>10}  ->  {fmt_num(av):<10}{d}")
        if args.labels and "per_label_f1" in bi and "per_label_f1" in ai:
            print("  Per-label F1:")
            all_tags = sorted(set(bi["per_label_f1"]) | set(ai["per_label_f1"]))
            for tag in all_tags:
                bv = bi["per_label_f1"].get(tag)
                av = ai["per_label_f1"].get(tag)
                d = diff_str(bv, av) if (bv is not None and av is not None) else ""
                print(f"    {tag:<25}  {fmt_num(bv):>8}  ->  {fmt_num(av):<8}{d}")
        print()

    # Concern
    if "concern" in base and "concern" in after:
        bc = base["concern"]
        ac = after["concern"]
        print("CONCERN (3-class)")
        print("-" * 50)
        for key in ["accuracy", "macro_f1"]:
            bv = bc.get(key)
            av = ac.get(key)
            d = diff_str(bv, av) if (bv is not None and av is not None) else ""
            print(f"  {key:<15}  {fmt_num(bv):>10}  ->  {fmt_num(av):<10}{d}")
        if args.labels and "per_label_f1" in bc and "per_label_f1" in ac:
            print("  Per-label F1:")
            all_labels = sorted(set(bc["per_label_f1"]) | set(ac["per_label_f1"]))
            for lab in all_labels:
                bv = bc["per_label_f1"].get(lab)
                av = ac["per_label_f1"].get(lab)
                d = diff_str(bv, av) if (bv is not None and av is not None) else ""
                print(f"    {lab:<10}  {fmt_num(bv):>8}  ->  {fmt_num(av):<8}{d}")
        print()

    # Summary
    intent_improved = False
    concern_improved = False
    if "intent" in base and "intent" in after:
        bf = base["intent"].get("macro_f1")
        af = after["intent"].get("macro_f1")
        if bf is not None and af is not None:
            intent_improved = af > bf
    if "concern" in base and "concern" in after:
        bf = base["concern"].get("macro_f1")
        af = after["concern"].get("macro_f1")
        if bf is not None and af is not None:
            concern_improved = af > bf

    print("SUMMARY")
    print("-" * 50)
    if intent_improved:
        print("  Intent:  macro F1 improved after retrain.")
    else:
        print("  Intent:  macro F1 unchanged or lower (check per-label with --labels).")
    if concern_improved:
        print("  Concern: macro F1 improved after retrain.")
    else:
        print("  Concern: macro F1 unchanged or lower (check per-label with --labels).")
    print()
    print("Use --labels to see per-label F1 comparison.")


if __name__ == "__main__":
    main()

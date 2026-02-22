#!/usr/bin/env python3
"""
Evaluate trained intent and/or concern classifiers on held-out data.

Loads classifier checkpoints from a pipeline config (or explicit paths),
runs predictions on val/test splits, and reports metrics.

Usage:
  # Evaluate both classifiers using pipeline config for checkpoint paths
  python scripts/evaluate_classifiers.py \
    --pipeline-config configs/pipeline.yaml \
    --data-config configs/data.yaml \
    --split test

  # Evaluate only intent classifier with explicit checkpoint
  python scripts/evaluate_classifiers.py \
    --intent-checkpoint results/runs/.../checkpoint \
    --data-config configs/data.yaml \
    --split val

  # Evaluate with a custom threshold sweep (intent only)
  python scripts/evaluate_classifiers.py \
    --pipeline-config configs/pipeline.yaml \
    --data-config configs/data.yaml \
    --split val \
    --sweep-threshold
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from mhsignals.config import load_pipeline_config, load_yaml
from mhsignals.data import CANON_KEYS, CONCERN_LABELS, read_concern_split, read_intent_split
from mhsignals.pipeline import _load_classifier


def evaluate_intent(clf, df, label_names, threshold=None):
    """Evaluate intent classifier on a DataFrame with Post and TagsList columns."""
    mlb = MultiLabelBinarizer(classes=label_names)
    Y_true = mlb.fit_transform(df["TagsList"])

    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(df["Post"].tolist())
        thr = threshold if threshold is not None else getattr(clf, "_threshold", 0.5)
        Y_pred = (P >= thr).astype(int)
        # Fallback to argmax when no tag above threshold
        for i in range(len(Y_pred)):
            if Y_pred[i].sum() == 0:
                Y_pred[i, np.argmax(P[i])] = 1
    else:
        preds = clf.predict_batch(df["Post"].tolist())
        Y_pred = mlb.transform(preds)
        P = None

    macro_f1 = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))

    per_label_f1 = {}
    per_label_precision = {}
    per_label_recall = {}
    for j, name in enumerate(label_names):
        per_label_f1[name] = float(f1_score(Y_true[:, j], Y_pred[:, j], zero_division=0))
        tp = int((Y_true[:, j] & Y_pred[:, j]).sum())
        fp = int(((1 - Y_true[:, j]) & Y_pred[:, j]).sum())
        fn = int((Y_true[:, j] & (1 - Y_pred[:, j])).sum())
        per_label_precision[name] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_label_recall[name] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    ap_per = []
    if P is not None:
        for j in range(Y_true.shape[1]):
            try:
                ap_per.append(average_precision_score(Y_true[:, j], P[:, j]))
            except ValueError:
                ap_per.append(0.0)

    result = {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_label_f1": {k: round(v, 4) for k, v in per_label_f1.items()},
        "per_label_precision": {k: round(v, 4) for k, v in per_label_precision.items()},
        "per_label_recall": {k: round(v, 4) for k, v in per_label_recall.items()},
    }
    if ap_per:
        result["pr_auc_macro"] = round(float(np.mean(ap_per)), 4)
    return result


def evaluate_concern(clf, df):
    """Evaluate concern classifier on a DataFrame with Post and Concern_Level columns."""
    y_true = df["Concern_Level"].tolist()
    y_pred = clf.predict_batch(df["Post"].tolist())

    label_names = sorted(set(y_true) | set(y_pred))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_label = {}
    for name in label_names:
        per_label[name] = float(f1_score(
            [1 if t == name else 0 for t in y_true],
            [1 if p == name else 0 for p in y_pred],
            zero_division=0,
        ))

    cm = confusion_matrix(y_true, y_pred, labels=label_names).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_label_f1": {k: round(v, 4) for k, v in per_label.items()},
        "confusion_matrix": cm,
        "label_order": label_names,
    }


def sweep_intent_threshold(clf, df, label_names, thresholds=None):
    """Sweep thresholds on intent classifier and return best by macro F1."""
    if not hasattr(clf, "predict_proba"):
        print("[WARN] Classifier does not support predict_proba; cannot sweep threshold.")
        return None

    mlb = MultiLabelBinarizer(classes=label_names)
    Y_true = mlb.fit_transform(df["TagsList"])
    P = clf.predict_proba(df["Post"].tolist())

    if thresholds is None:
        thresholds = np.arange(0.20, 0.80, 0.05).tolist()

    best_thr, best_f1 = 0.5, 0.0
    rows = []
    for thr in thresholds:
        Y_pred = (P >= thr).astype(int)
        for i in range(len(Y_pred)):
            if Y_pred[i].sum() == 0:
                Y_pred[i, np.argmax(P[i])] = 1
        mf1 = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
        rows.append({"threshold": round(thr, 2), "macro_f1": round(mf1, 4)})
        if mf1 > best_f1:
            best_f1, best_thr = mf1, thr

    return {
        "best_threshold": round(best_thr, 2),
        "best_macro_f1": round(best_f1, 4),
        "sweep": rows,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate MH-SIGNALS classifiers on held-out data."
    )
    ap.add_argument(
        "--pipeline-config", default=None,
        help="Path to pipeline.yaml (for checkpoint paths).",
    )
    ap.add_argument(
        "--data-config", default="configs/data.yaml",
        help="Path to data.yaml (for split paths).",
    )
    ap.add_argument(
        "--intent-checkpoint", default=None,
        help="Explicit intent checkpoint path (overrides pipeline config).",
    )
    ap.add_argument(
        "--concern-checkpoint", default=None,
        help="Explicit concern checkpoint path (overrides pipeline config).",
    )
    ap.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Which split to evaluate on (default: test).",
    )
    ap.add_argument(
        "--sweep-threshold", action="store_true",
        help="Sweep intent threshold on the chosen split and report best.",
    )
    ap.add_argument(
        "--output", default=None,
        help="Optional path to write JSON results.",
    )
    args = ap.parse_args()

    # Resolve checkpoint paths
    intent_ckpt = args.intent_checkpoint
    concern_ckpt = args.concern_checkpoint

    if args.pipeline_config:
        pcfg = load_pipeline_config(args.pipeline_config)
        if not intent_ckpt:
            intent_ckpt = pcfg.intent_checkpoint
        if not concern_ckpt:
            concern_ckpt = pcfg.concern_checkpoint

    if not intent_ckpt and not concern_ckpt:
        ap.error("Provide --pipeline-config or at least one of --intent-checkpoint / --concern-checkpoint.")

    # Load data split
    data_cfg = load_yaml(args.data_config)
    splits_dir = Path(data_cfg["paths"]["splits_dir"])
    split_file = splits_dir / f"{args.split}.csv"

    if not split_file.exists():
        print(f"[ERROR] Split file not found: {split_file}", file=sys.stderr)
        sys.exit(1)

    all_results = {}

    # --- Intent evaluation ---
    if intent_ckpt:
        ckpt_path = Path(intent_ckpt)
        if not ckpt_path.exists():
            print(f"[ERROR] Intent checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading intent classifier from: {ckpt_path}")
        intent_clf = _load_classifier(str(ckpt_path))

        intent_df = read_intent_split(split_file)
        print(f"Evaluating intent on {args.split} split ({len(intent_df)} rows)...")
        intent_metrics = evaluate_intent(intent_clf, intent_df, CANON_KEYS)
        all_results["intent"] = intent_metrics

        print(f"\n{'='*50}")
        print(f"INTENT CLASSIFICATION ({args.split})")
        print(f"{'='*50}")
        print(f"  Macro F1:     {intent_metrics['macro_f1']}")
        print(f"  Micro F1:     {intent_metrics['micro_f1']}")
        if "pr_auc_macro" in intent_metrics:
            print(f"  PR-AUC Macro: {intent_metrics['pr_auc_macro']}")
        print(f"\n  Per-label F1:")
        for tag, f1 in intent_metrics["per_label_f1"].items():
            prec = intent_metrics["per_label_precision"][tag]
            rec = intent_metrics["per_label_recall"][tag]
            print(f"    {tag:<25s}  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")

        if args.sweep_threshold:
            print(f"\nSweeping intent threshold on {args.split} split...")
            sweep = sweep_intent_threshold(intent_clf, intent_df, CANON_KEYS)
            if sweep:
                all_results["intent_threshold_sweep"] = sweep
                print(f"  Best threshold: {sweep['best_threshold']}  (macro F1={sweep['best_macro_f1']})")
                for row in sweep["sweep"]:
                    marker = " <-- best" if row["threshold"] == sweep["best_threshold"] else ""
                    print(f"    thr={row['threshold']:.2f}  macro_f1={row['macro_f1']:.4f}{marker}")

    # --- Concern evaluation ---
    if concern_ckpt:
        ckpt_path = Path(concern_ckpt)
        if not ckpt_path.exists():
            print(f"[ERROR] Concern checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\nLoading concern classifier from: {ckpt_path}")
        concern_clf = _load_classifier(str(ckpt_path))

        concern_df = read_concern_split(split_file)
        print(f"Evaluating concern on {args.split} split ({len(concern_df)} rows)...")
        concern_metrics = evaluate_concern(concern_clf, concern_df)
        all_results["concern"] = concern_metrics

        print(f"\n{'='*50}")
        print(f"CONCERN CLASSIFICATION ({args.split})")
        print(f"{'='*50}")
        print(f"  Accuracy:  {concern_metrics['accuracy']}")
        print(f"  Macro F1:  {concern_metrics['macro_f1']}")
        print(f"\n  Per-label F1:")
        for label, f1 in concern_metrics["per_label_f1"].items():
            print(f"    {label:<10s}  F1={f1:.4f}")
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        print(f"    Labels: {concern_metrics['label_order']}")
        for row in concern_metrics["confusion_matrix"]:
            print(f"    {row}")

    # --- Save results ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

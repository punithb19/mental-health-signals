#!/usr/bin/env python3
"""
Audit training data: label and tag distribution, imbalance, potential data-quality issues.

Usage:
  python scripts/data_audit.py --config configs/data.yaml
  python scripts/data_audit.py --splits-dir data/splits

Use this to see why intent/concern detection might be off (e.g. severe class imbalance,
too few examples for Critical Risk, concern derived from tags).
"""

import argparse
import re
import sys
import yaml
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Canonical intent tags (match mhsignals.data)
CANON_KEYS = [
    "Cause of Distress",
    "Critical Risk",
    "Maladaptive Coping",
    "Mental Distress",
    "Miscellaneous",
    "Mood Tracking",
    "Positive Coping",
    "Progress Update",
    "Seeking Help",
]


def normalize_concern(x: str) -> Optional[str]:
    """Normalize concern to low/medium/high."""
    if not isinstance(x, str):
        return None
    t = x.strip().lower()
    t = re.sub(r"[.\s]+$", "", t)
    if t in {"low", "medium", "high"}:
        return t
    if t in {"med", "mid"}:
        return "medium"
    return None


def tags_to_canonical_list(x) -> List[str]:
    """Parse Tag column into list of canonical tag strings (simplified)."""
    if pd.isna(x) or (isinstance(x, str) and not x.strip()):
        return ["Miscellaneous"]
    raw = re.split(r"[;,]", str(x))
    canonical = {
        "critical risk": "Critical Risk",
        "mental distress": "Mental Distress",
        "maladaptive coping": "Maladaptive Coping",
        "positive coping": "Positive Coping",
        "seeking help": "Seeking Help",
        "progress update": "Progress Update",
        "mood tracking": "Mood Tracking",
        "cause of distress": "Cause of Distress",
        "miscellaneous": "Miscellaneous",
        "causes of distress": "Cause of Distress",
    }
    out = []
    seen = set()
    for r in raw:
        k = r.strip().lower().replace(".", "")
        if k in canonical and canonical[k] not in seen:
            out.append(canonical[k])
            seen.add(canonical[k])
    return out if out else ["Miscellaneous"]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def audit_splits(splits_dir: Path) -> None:
    """Audit train/val/test CSVs in splits_dir (Post, Tag, Concern_Level)."""
    splits_dir = Path(splits_dir)
    train_path = splits_dir / "train.csv"
    if not train_path.exists():
        print(f"[ERROR] Not found: {train_path}", file=sys.stderr)
        sys.exit(1)

    def load_df(name: str) -> Optional[pd.DataFrame]:
        p = splits_dir / f"{name}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if "Concern_Level" not in df.columns or "Post" not in df.columns:
            tag_col = "Tag" if "Tag" in df.columns else "Final_Tags"
            if tag_col not in df.columns:
                print(f"[WARN] {p}: missing Tag/Final_Tags or Concern_Level", file=sys.stderr)
                return None
        return df

    train_df = load_df("train")
    if train_df is None:
        sys.exit(1)

    # Resolve columns
    post_col = "Post" if "Post" in train_df.columns else "Text"
    tag_col = "Tag" if "Tag" in train_df.columns else "Final_Tags"
    concern_col = "Concern_Level"

    train_df[post_col] = train_df[post_col].fillna("").astype(str)
    train_df[tag_col] = train_df[tag_col].fillna("")
    train_df[concern_col] = train_df[concern_col].apply(normalize_concern)
    train_df = train_df.dropna(subset=[concern_col])

    n = len(train_df)
    print("=" * 60)
    print("DATA AUDIT (training split)")
    print("=" * 60)
    print(f"Total rows: {n}\n")

    # Intent tag distribution (multi-label)
    all_tags = []
    for t in train_df[tag_col]:
        all_tags.extend(tags_to_canonical_list(t))
    tag_counts = Counter(all_tags)
    print("Intent tag counts (multi-label):")
    for k in CANON_KEYS:
        c = tag_counts.get(k, 0)
        pct = 100.0 * c / n if n else 0
        bar = "!" if pct < 5 else ("!!" if pct > 40 else "")
        print(f"  {k:<25} {c:>5}  ({pct:>5.1f}%)  {bar}")
    print()

    # Concern distribution
    concern_counts = train_df[concern_col].value_counts()
    print("Concern level counts:")
    for lev in ["low", "medium", "high"]:
        c = concern_counts.get(lev, 0)
        pct = 100.0 * c / n if n else 0
        bar = "!" if pct < 10 else ""
        print(f"  {lev:<10} {c:>5}  ({pct:>5.1f}%)  {bar}")
    print()

    # Flags
    print("Potential issues:")
    issues = []
    # Rare intent tags (< 10% of rows)
    for k in CANON_KEYS:
        c = tag_counts.get(k, 0)
        if c > 0 and c < 0.10 * n:
            issues.append(f"  - Intent '{k}' has relatively few positives ({c}, {100*c/n:.1f}%)")
    # Rare concern
    for lev in ["low", "medium", "high"]:
        c = concern_counts.get(lev, 0)
        if c > 0 and c < 0.10 * n:
            issues.append(f"  - Concern '{lev}' is rare ({c}, {100*c/n:.1f}%)")
    # Severe imbalance
    if concern_counts.max() / max(concern_counts.min(), 1) > 5:
        issues.append("  - Concern classes are highly imbalanced (max/min ratio > 5)")
    if not issues:
        print("  None obvious.")
    else:
        for i in issues:
            print(i)
    print()
    print("Recommendation: More and better-labeled examples for rare classes improve intent/concern detection.")
    if "Critical Risk" in tag_counts and tag_counts["Critical Risk"] < 0.15 * n:
        print("Consider adding more Critical Risk examples for safety-sensitive behavior.")


def main():
    ap = argparse.ArgumentParser(description="Audit training data for intent/concern quality.")
    ap.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML.")
    ap.add_argument("--splits-dir", default=None, help="Override splits directory (e.g. data/splits).")
    args = ap.parse_args()

    if args.splits_dir:
        splits_dir = Path(args.splits_dir)
    else:
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        splits_dir = Path(paths.get("splits_dir", "data/splits"))

    audit_splits(splits_dir)


if __name__ == "__main__":
    main()

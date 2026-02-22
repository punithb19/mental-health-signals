#!/usr/bin/env python3
"""
Create train/val/test splits from raw data, driven by configs/data.yaml.

Usage:
  python scripts/create_splits.py --config configs/data.yaml

Outputs:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv
  data/splits/test_gold.jsonl   (JSONL of test posts for batch generation)

Column handling:
  Supports CSVs with "Post" or "Text" columns (normalized to "Post"),
  and "Tag" or "Final_Tags" columns (normalized to "Tag").
  "Concern_Level" column is required.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mhsignals.config import load_yaml


# Canonical column names used in the output splits
COL_POST = "Post"
COL_TAGS = "Tag"
COL_CONCERN = "Concern_Level"


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to match the canonical Post / Tag / Concern_Level schema."""
    # Post column: accept "Post" or "Text"
    if COL_POST not in df.columns:
        if "Text" in df.columns:
            df = df.rename(columns={"Text": COL_POST})
        else:
            print(f"[ERROR] CSV has no 'Post' or 'Text' column. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    # Tag column: accept "Tag" or "Final_Tags"
    if COL_TAGS not in df.columns:
        if "Final_Tags" in df.columns:
            df = df.rename(columns={"Final_Tags": COL_TAGS})
        else:
            print(f"[ERROR] CSV has no 'Tag' or 'Final_Tags' column. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    # Concern column
    if COL_CONCERN not in df.columns:
        print(f"[ERROR] CSV has no 'Concern_Level' column. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    return df


def main():
    ap = argparse.ArgumentParser(description="Create train/val/test splits from data.yaml.")
    ap.add_argument(
        "--config", default="configs/data.yaml",
        help="Path to data config YAML (default: configs/data.yaml).",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})
    split_cfg = cfg.get("split", {})

    raw_csv = paths.get("raw_csv", "data/raw/final/mh_signal_data_w-concern-intent.csv")
    splits_dir = Path(paths.get("splits_dir", "data/splits"))
    seed = cfg.get("seed", 42)

    test_size = float(split_cfg.get("test_size", 0.15))
    val_size_from_train = float(split_cfg.get("val_size_from_train", 0.1765))

    # Validate that input CSV exists
    if not Path(raw_csv).exists():
        print(f"[ERROR] Raw CSV not found: {raw_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(raw_csv)
    df = _resolve_columns(df)
    df = df[[COL_POST, COL_TAGS, COL_CONCERN]].copy()
    df[COL_TAGS] = df[COL_TAGS].fillna("")
    df = df.dropna(subset=[COL_POST, COL_CONCERN])
    df[COL_CONCERN] = df[COL_CONCERN].astype(str).str.strip().str.lower()

    # Check minimum samples per concern class for stratification
    concern_counts = df[COL_CONCERN].value_counts()
    min_count = concern_counts.min()
    use_stratify = min_count >= 2

    if not use_stratify:
        print(
            f"[WARN] Concern class '{concern_counts.idxmin()}' has only {min_count} sample(s); "
            "falling back to non-stratified split.",
            file=sys.stderr,
        )

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=df[COL_CONCERN] if use_stratify else None,
    )

    # Check stratification feasibility for train/val split
    tv_counts = train_val[COL_CONCERN].value_counts()
    use_stratify_tv = tv_counts.min() >= 2

    train, val = train_test_split(
        train_val,
        test_size=val_size_from_train,
        random_state=seed,
        shuffle=True,
        stratify=train_val[COL_CONCERN] if use_stratify_tv else None,
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir / "val.csv", index=False)
    test.to_csv(splits_dir / "test.csv", index=False)

    # Write test_gold.jsonl for batch generation
    with open(splits_dir / "test_gold.jsonl", "w", encoding="utf-8") as f:
        for _, row in test.iterrows():
            entry = {"post": row[COL_POST]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = split[COL_CONCERN].value_counts(normalize=True).round(3).to_dict()
        print(f"{name:>5} ({len(split):>5} rows): {dist}")

    print(f"\nSplits written to {splits_dir}/")
    print(f"test_gold.jsonl written ({len(test)} posts)")


if __name__ == "__main__":
    main()

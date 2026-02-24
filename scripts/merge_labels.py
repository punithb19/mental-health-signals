#!/usr/bin/env python3
"""
Merge new labeled rows into the raw training CSV and optionally regenerate splits.

Usage:
  python scripts/merge_labels.py --raw data/raw/final/mh_signal_data_w-concern-intent.csv --new data/raw/final/new_labels.csv
  python scripts/merge_labels.py --raw data/raw/final/mh_signal_data_w-concern-intent.csv --new new_labels.csv --create-splits --config configs/data.yaml

Required columns in both CSVs: Post (or Text), Tag (or Final_Tags), Concern_Level.
If a post appears in both files, the row from --new wins (keep="last").
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to Post, Tag, Concern_Level."""
    if "Text" in df.columns and "Post" not in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Final_Tags" in df.columns and "Tag" not in df.columns:
        df = df.rename(columns={"Final_Tags": "Tag"})
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Merge new labels into raw CSV; optionally run create_splits.",
    )
    ap.add_argument(
        "--raw",
        default="data/raw/final/mh_signal_data_w-concern-intent.csv",
        help="Path to existing raw CSV (will be overwritten with merged data).",
    )
    ap.add_argument(
        "--new",
        required=True,
        help="Path to CSV with new labeled rows (same columns: Post, Tag, Concern_Level).",
    )
    ap.add_argument(
        "--create-splits",
        action="store_true",
        help="After merging, run create_splits.py to regenerate train/val/test.",
    )
    ap.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Config for create_splits (used only if --create-splits).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be merged without writing.",
    )
    args = ap.parse_args()

    raw_path = Path(args.raw)
    new_path = Path(args.new)

    if not raw_path.exists():
        print(f"[ERROR] Raw CSV not found: {raw_path}", file=sys.stderr)
        sys.exit(1)
    if not new_path.exists():
        print(f"[ERROR] New labels CSV not found: {new_path}", file=sys.stderr)
        sys.exit(1)

    raw_df = pd.read_csv(raw_path)
    new_df = pd.read_csv(new_path)

    raw_df = _normalize_columns(raw_df)
    new_df = _normalize_columns(new_df)

    required = ["Post", "Tag", "Concern_Level"]
    for name, df in [("raw", raw_df), ("new", new_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[ERROR] {name} CSV missing columns: {missing}. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    # Keep only required columns for merge
    raw_df = raw_df[required].copy()
    new_df = new_df[required].copy()
    raw_df["Post"] = raw_df["Post"].astype(str)
    new_df["Post"] = new_df["Post"].astype(str)
    new_df["Tag"] = new_df["Tag"].fillna("").astype(str)
    raw_df["Tag"] = raw_df["Tag"].fillna("").astype(str)
    new_df["Concern_Level"] = new_df["Concern_Level"].astype(str).str.strip().str.lower()
    raw_df["Concern_Level"] = raw_df["Concern_Level"].astype(str).str.strip().str.lower()

    combined = pd.concat([raw_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Post"], keep="last")
    n_raw = len(raw_df)
    n_new = len(new_df)
    n_combined = len(combined)

    if args.dry_run:
        print(f"Would merge: raw={n_raw}, new={n_new}, combined={n_combined}")
        print("Columns:", list(combined.columns))
        return

    combined.to_csv(raw_path, index=False)
    print(f"Merged: raw={n_raw}, new={n_new}, combined={n_combined} -> {raw_path}")

    if args.create_splits:
        script = Path(__file__).parent / "create_splits.py"
        repo_root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONPATH": str(repo_root)}
        print(f"Running: {sys.executable} {script} --config {args.config}")
        subprocess.run(
            [sys.executable, str(script), "--config", args.config],
            check=True,
            cwd=str(repo_root),
            env=env,
        )


if __name__ == "__main__":
    main()

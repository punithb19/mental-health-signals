#!/usr/bin/env python3
"""
Clean and normalize the gold dataset: canonical tags, low/medium/high concern, drop invalid/duplicate rows.

Usage:
  python scripts/fix_data.py --config configs/data.yaml
  python scripts/fix_data.py --config configs/data.yaml --derive-concern --create-splits

--derive-concern: set Concern_Level from intent tags (Critical Risk->high, etc.) for consistency.
--create-splits: after cleaning, regenerate train/val/test.
"""

import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml


def normalize_tag(t: str) -> Optional[str]:
    """Normalize a raw tag string to canonical form."""
    canonical = {
        "critical risk": "Critical Risk",
        "mental distress": "Mental Distress",
        "maladaptive coping": "Maladaptive Coping",
        "positive coping": "Positive Coping",
        "seeking help": "Seeking Help",
        "progress update": "Progress Update",
        "mood tracking": "Mood Tracking",
        "cause of distress": "Cause of Distress",
        "causes of distress": "Cause of Distress",
        "miscellaneous": "Miscellaneous",
    }
    x = t.strip().lower()
    x = re.sub(r"\.$", "", x)
    return canonical.get(x)


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
    if t in {"0"}:
        return "low"
    if t in {"1"}:
        return "medium"
    if t in {"2"}:
        return "high"
    return None


def tags_to_canonical_list(x) -> List[str]:
    """Parse Tag value into list of canonical tag strings."""
    if isinstance(x, float) and math.isnan(x):
        raw = []
    elif isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]"):
        try:
            import ast
            raw = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            raw = re.split(r"[;,]", str(x))
    elif isinstance(x, str):
        raw = re.split(r"[;,]", x)
    else:
        raw = []
    norm, seen = [], set()
    for r in raw:
        can = normalize_tag(str(r))
        if can and can not in seen:
            norm.append(can)
            seen.add(can)
    return norm if norm else ["Miscellaneous"]


def concern_from_tags(tag_list: list) -> str:
    """Derive concern level from canonical intent tags (matches concern_levels_3tier logic)."""
    tl = {t.lower() for t in tag_list}
    all_tags = {
        "critical risk", "mental distress", "maladaptive coping", "positive coping",
        "seeking help", "progress update", "mood tracking", "cause of distress",
    }
    tl = tl & all_tags

    if "critical risk" in tl:
        return "high"
    if tl & {"seeking help", "maladaptive coping", "progress update"}:
        return "medium"
    if tl & {"mental distress", "cause of distress"}:
        return "medium"
    return "low"


def main():
    ap = argparse.ArgumentParser(description="Clean and normalize raw training CSV.")
    ap.add_argument("--config", default="configs/data.yaml", help="Data config YAML.")
    ap.add_argument(
        "--raw",
        default=None,
        help="Override raw CSV path (default: paths.raw_csv from config).",
    )
    ap.add_argument(
        "--derive-concern",
        action="store_true",
        help="Set Concern_Level from intent tags for consistency.",
    )
    ap.add_argument(
        "--create-splits",
        action="store_true",
        help="After cleaning, run create_splits.py.",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Write backup to <raw>.backup.csv (default: True).",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a backup file.",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})
    raw_csv = args.raw or paths.get("raw_csv", "data/raw/final/mh_signal_data_w-concern-intent.csv")
    raw_path = Path(raw_csv)

    if not raw_path.exists():
        print(f"[ERROR] Raw CSV not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(raw_path)

    # Resolve columns
    if "Text" in df.columns and "Post" not in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Final_Tags" in df.columns and "Tag" not in df.columns:
        df = df.rename(columns={"Final_Tags": "Tag"})
    if "Post" not in df.columns:
        print("[ERROR] No Post or Text column.", file=sys.stderr)
        sys.exit(1)
    tag_col = "Tag" if "Tag" in df.columns else "Final_Tags"
    if tag_col not in df.columns:
        print("[ERROR] No Tag or Final_Tags column.", file=sys.stderr)
        sys.exit(1)
    if "Concern_Level" not in df.columns:
        print("[ERROR] No Concern_Level column.", file=sys.stderr)
        sys.exit(1)

    n_before = len(df)

    # Keep only needed columns for cleaning; we'll output Post, Tag, Concern_Level
    df = df[["Post", tag_col, "Concern_Level"]].copy()
    df = df.rename(columns={tag_col: "Tag"})

    # Post: string, drop empty
    df["Post"] = df["Post"].fillna("").astype(str)
    df = df[df["Post"].str.strip() != ""]

    # Tag: normalize to canonical list, then serialize as "Tag1, Tag2"
    df["Tag"] = df["Tag"].fillna("")
    tag_lists = [tags_to_canonical_list(x) for x in df["Tag"]]
    df["Tag"] = [", ".join(sorted(lst)) if lst else "Miscellaneous" for lst in tag_lists]

    # Concern: normalize to low/medium/high
    df["Concern_Level"] = df["Concern_Level"].apply(normalize_concern)
    if args.derive_concern:
        df["Concern_Level"] = [concern_from_tags(tags_to_canonical_list(row["Tag"])) for _, row in df.iterrows()]
    else:
        # Drop rows with invalid or missing concern
        df = df.dropna(subset=["Concern_Level"])
    df["Concern_Level"] = df["Concern_Level"].astype(str).str.strip().str.lower()

    # Drop duplicate posts (keep last)
    df = df.drop_duplicates(subset=["Post"], keep="last").reset_index(drop=True)
    n_after = len(df)

    if args.backup and not args.no_backup:
        backup_path = raw_path.parent / f"{raw_path.stem}.backup.csv"
        pd.read_csv(raw_path).to_csv(backup_path, index=False)
        print(f"Backup written: {backup_path}")

    df.to_csv(raw_path, index=False)
    print(f"Cleaned: {n_before} -> {n_after} rows -> {raw_path}")
    print(f"  Dropped: {n_before - n_after} (empty post, invalid concern, or duplicate)")

    if args.create_splits:
        script = Path(__file__).parent / "create_splits.py"
        repo_root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONPATH": str(repo_root)}
        subprocess.run(
            [sys.executable, str(script), "--config", args.config],
            check=True,
            cwd=str(repo_root),
            env=env,
        )


if __name__ == "__main__":
    main()

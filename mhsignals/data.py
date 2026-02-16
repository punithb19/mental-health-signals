"""
Shared data loading and preprocessing utilities for MH-SIGNALS.

Consolidates helper functions previously duplicated across model scripts.
"""

import ast
import math
import random
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical intent tag mapping
# ---------------------------------------------------------------------------
CANONICAL = {
    "critical risk": "Critical Risk",
    "mental distress": "Mental Distress",
    "maladaptive coping": "Maladaptive Coping",
    "positive coping": "Positive Coping",
    "seeking help": "Seeking Help",
    "progress update": "Progress Update",
    "mood tracking": "Mood Tracking",
    "cause of distress": "Cause of Distress",
    "miscellaneous": "Miscellaneous",
}
CANON_KEYS = sorted(CANONICAL.values())
CONCERN_LABELS = ["high", "low", "medium"]

# ---------------------------------------------------------------------------
# Seed + directory helpers
# ---------------------------------------------------------------------------

def set_seed(s: int):
    """Set random seeds for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Tag normalization
# ---------------------------------------------------------------------------

def normalize_tag(t: str) -> Optional[str]:
    """Normalize a raw tag string to its canonical form."""
    x = t.strip().lower()
    x = re.sub(r"\.$", "", x)
    x = x.replace("causes of distress", "cause of distress")
    x = x.replace("progress update.", "progress update")
    return CANONICAL.get(x)


def normalize_concern(x: str) -> Optional[str]:
    """Normalize concern level to low/medium/high."""
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
    """Parse a raw tags value into a list of canonical tag strings."""
    raw = []
    if isinstance(x, float) and math.isnan(x):
        raw = []
    elif isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            raw = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            raw = []
    elif isinstance(x, str):
        raw = re.split(r"[;,]", x)
    elif isinstance(x, list):
        raw = x

    norm, seen = [], set()
    for r in raw:
        can = normalize_tag(str(r))
        if can and can not in seen:
            norm.append(can)
            seen.add(can)
    return norm if norm else ["Miscellaneous"]


# ---------------------------------------------------------------------------
# Split CSV loaders
# ---------------------------------------------------------------------------

def read_intent_split(path: Path) -> pd.DataFrame:
    """
    Read a split CSV for intent (multi-label) classification.
    Returns DataFrame with columns: Post, TagsList.
    """
    df = pd.read_csv(path)

    if "Post" not in df.columns and "Text" in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Post" not in df.columns:
        raise ValueError(f"'Post' column missing in {path}")

    tag_col = None
    for candidate in ["Final_Tags", "Tag"]:
        if candidate in df.columns:
            tag_col = candidate
            break
    if tag_col is None:
        raise ValueError(f"No tag column found in {path}")

    df["Post"] = df["Post"].fillna("").astype(str)
    df["TagsList"] = df[tag_col].apply(tags_to_canonical_list)
    return df[["Post", "TagsList"]]


def read_concern_split(path: Path) -> pd.DataFrame:
    """
    Read a split CSV for concern (single-label) classification.
    Returns DataFrame with columns: Post, Concern_Level.
    """
    df = pd.read_csv(path)

    if "Post" not in df.columns and "Text" in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Post" not in df.columns:
        raise ValueError(f"'Post' column missing in {path}")
    if "Concern_Level" not in df.columns:
        raise ValueError(f"'Concern_Level' column missing in {path}")

    df["Post"] = df["Post"].fillna("").astype(str)
    df["Concern_Level"] = df["Concern_Level"].apply(normalize_concern)
    df = df.dropna(subset=["Concern_Level"]).reset_index(drop=True)
    return df[["Post", "Concern_Level"]]


def prob_to_tags(prob_row: np.ndarray, threshold: float, names: List[str]) -> List[str]:
    """Convert a probability vector to a list of tag names above threshold."""
    idx = np.where(prob_row >= threshold)[0].tolist()
    if not idx:
        idx = [int(np.argmax(prob_row))]
    return [names[i] for i in idx]

"""
Knowledge base construction: CSV -> chunked JSONL -> embeddings -> FAISS index.

Merges the previously separate kb_build.py and kb_encode_index.py into one module.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..config import KBConfig


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _sentish_split(text: str) -> List[str]:
    """Lightweight sentence splitter on . ! ? while keeping punctuation."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, target_tokens: int = 140, min_chars: int = 20) -> List[str]:
    """Break long text into chunks of ~target_tokens using sentence boundaries."""
    tokens = lambda s: len(s.split())
    sents = _sentish_split(text)
    if not sents:
        return []

    chunks, cur, cur_tok = [], [], 0
    for s in sents:
        n = tokens(s)
        if cur and cur_tok + n > target_tokens:
            ch = " ".join(cur).strip()
            if len(ch) >= min_chars:
                chunks.append(ch)
            cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += n

    if cur:
        ch = " ".join(cur).strip()
        if len(ch) >= min_chars:
            chunks.append(ch)

    if len(chunks) == 1 and tokens(chunks[0]) > 2 * target_tokens:
        words = chunks[0].split()
        chunks = []
        for i in range(0, len(words), target_tokens):
            ch = " ".join(words[i:i + target_tokens]).strip()
            if len(ch) >= min_chars:
                chunks.append(ch)

    return chunks


def _norm_concern(val: str) -> str:
    """Normalize concern level string."""
    s = (val or "").strip().lower()
    if s in {"0", "low"}:
        return "Low"
    if s in {"1", "med", "medium"}:
        return "Medium"
    if s in {"2", "high"}:
        return "High"
    return val.strip() if val else ""


# ---------------------------------------------------------------------------
# KBBuilder
# ---------------------------------------------------------------------------

class KBBuilder:
    """
    Builds the knowledge base from a CSV of counselor responses.

    Steps:
      1. Read CSV, chunk long responses
      2. Write corpus JSONL (doc_id, text) and metadata JSONL
      3. Encode chunks with SentenceTransformer
      4. Build FAISS HNSW index
    """

    def __init__(self, config: KBConfig):
        self.config = config

    def build_all(
        self,
        csv_path: Optional[str] = None,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> Dict[str, str]:
        """
        Run the full KB construction pipeline.

        Returns dict of output file paths.
        """
        csv_path = csv_path or os.path.join(self.config.raw_dir, "rag_intent_data_w_concern.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"KB source CSV not found: {csv_path}. "
                "Provide the correct path via --csv or in data.yaml."
            )

        self._build_corpus(csv_path)
        self._encode_and_index(encoder_name, batch_size)

        return {
            "corpus": self.config.corpus_jsonl,
            "metadata": self.config.metadata_jsonl,
            "embeddings": self.config.embeddings_npy,
            "faiss_index": self.config.faiss_index,
        }

    def _build_corpus(self, csv_path: str) -> None:
        """Step 1-2: Read CSV, chunk, write corpus + metadata JSONL."""
        cfg = self.config
        schema = cfg.schema
        output_col = schema.get("output_col", "output")
        intent_col = schema.get("intent_col", "Final_Intent_Tag")
        concern_col = schema.get("concern_col", "Concern_Level")
        id_col = schema.get("id_col", "doc_id")

        Path(cfg.processed_dir).mkdir(parents=True, exist_ok=True)

        print(f"[kb_build] Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        missing = [c for c in [output_col, intent_col, concern_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        if id_col not in df.columns:
            df[id_col] = [f"kb_{i:07d}" for i in range(len(df))]

        for c in [output_col, intent_col, concern_col, id_col]:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)

        df[concern_col] = df[concern_col].apply(_norm_concern)
        df = df.sort_values(id_col).reset_index(drop=True)

        n_in, n_kept, n_chunks = len(df), 0, 0
        with (
            open(cfg.corpus_jsonl, "w", encoding="utf-8") as f_c,
            open(cfg.metadata_jsonl, "w", encoding="utf-8") as f_m,
        ):
            for _, r in df.iterrows():
                raw_out = (r.get(output_col, "") or "").strip()
                if not raw_out or len(raw_out) < cfg.min_chars:
                    continue

                base_id = str(r[id_col]).strip()
                pieces = (
                    chunk_text(raw_out, cfg.target_tokens, cfg.min_chars)
                    if cfg.chunk_enabled
                    else [raw_out]
                )

                for j, ch in enumerate(pieces, start=1):
                    doc_id = f"{base_id}_c{j}" if len(pieces) > 1 else base_id

                    f_c.write(json.dumps({"doc_id": doc_id, "text": ch}, ensure_ascii=False) + "\n")
                    meta = {
                        "doc_id": doc_id,
                        "source": "csv",
                        "intent": (r.get(intent_col, "") or "").strip(),
                        "concern": (r.get(concern_col, "") or "").strip(),
                        "tags": "",
                        "text": ch,
                    }
                    f_m.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    n_chunks += 1
                n_kept += 1

        print(f"[kb_build] Rows: {n_in}, kept: {n_kept}, chunks: {n_chunks}")
        print(f"[kb_build] Corpus -> {cfg.corpus_jsonl}")
        print(f"[kb_build] Meta   -> {cfg.metadata_jsonl}")

    def _encode_and_index(self, encoder_name: str, batch_size: int) -> None:
        """Step 3-4: Encode chunks and build FAISS HNSW index."""
        cfg = self.config

        texts = []
        with open(cfg.corpus_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    texts.append(json.loads(line)["text"])

        if not texts:
            raise RuntimeError(f"No texts found in {cfg.corpus_jsonl}")

        print(f"[kb_encode] Loading encoder: {encoder_name}")
        model = SentenceTransformer(encoder_name)

        print(f"[kb_encode] Encoding {len(texts)} chunks...")
        X = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        Path(os.path.dirname(cfg.embeddings_npy)).mkdir(parents=True, exist_ok=True)
        np.save(cfg.embeddings_npy, X)

        d = X.shape[1]
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 200
        index.add(X)
        faiss.write_index(index, cfg.faiss_index)

        print(f"[kb_encode] Embeddings -> {cfg.embeddings_npy} shape={X.shape}")
        print(f"[kb_encode] FAISS      -> {cfg.faiss_index}")

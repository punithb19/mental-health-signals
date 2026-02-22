"""
KB retrieval: FAISS nearest-neighbor search with intent/concern re-ranking.

This is the bridge between classification and generation -- the classified
intent and concern directly influence which KB snippets are retrieved.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .filters import filter_unsafe_snippets

logger = logging.getLogger(__name__)


class KBRetriever:
    """
    Retrieve relevant counselor-response snippets from the knowledge base.

    Uses FAISS for fast approximate nearest-neighbor search, then re-ranks
    results based on intent and concern matches from the classifier.
    """

    def __init__(
        self,
        metadata_path: str,
        faiss_index_path: str,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        # Validate paths before loading
        if not Path(metadata_path).exists():
            raise FileNotFoundError(
                f"KB metadata file not found: {metadata_path}. "
                "Run scripts/build_kb.py first."
            )
        if not Path(faiss_index_path).exists():
            raise FileNotFoundError(
                f"FAISS index file not found: {faiss_index_path}. "
                "Run scripts/build_kb.py first."
            )

        self._meta = self._load_meta(metadata_path)
        self._index = faiss.read_index(faiss_index_path)
        self._encoder = SentenceTransformer(encoder_name)

        # Validate metadata/index consistency
        if len(self._meta) != self._index.ntotal:
            logger.warning(
                "Metadata count (%d) does not match FAISS index count (%d). "
                "KB may need rebuilding.",
                len(self._meta), self._index.ntotal,
            )

        # Improve HNSW recall
        try:
            self._index.hnsw.efSearch = 64
        except Exception:
            pass

        logger.info("KBRetriever loaded: %d entries from %s", len(self._meta), metadata_path)

    @staticmethod
    def _load_meta(path: str) -> List[Dict]:
        """Load metadata JSONL into a list of dicts."""
        meta = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    meta.append(json.loads(line))
        return meta

    def search(
        self,
        post: str,
        intents: Optional[List[str]] = None,
        concern: Optional[str] = None,
        topk: int = 50,
        keep: int = 5,
        min_similarity: float = 0.45,
    ) -> List[Dict]:
        """
        Retrieve and re-rank KB snippets for a given post.

        The core integration point: classifier-predicted intents and concern
        directly influence snippet selection through score boosting/penalizing.

        Args:
            post:           User's mental health post text.
            intents:        Predicted intent tags from the intent classifier.
            concern:        Predicted concern level from the concern classifier.
            topk:           Number of FAISS candidates to fetch before filtering.
            keep:           Number of snippets to return after re-ranking.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of snippet dicts with keys: rank, score, similarity, doc_id,
            intent, concern, text, source.
        """
        want_intents = {i.lower() for i in (intents or [])}
        want_concern = (concern or "").lower()

        # Encode query and search FAISS
        qv = self._encoder.encode([post], normalize_embeddings=True).astype("float32")

        # Ensure efSearch is high enough for the requested topk
        try:
            self._index.hnsw.efSearch = max(64, topk)
        except Exception:
            pass

        D, I = self._index.search(qv, topk)

        candidates = []
        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self._meta):
                continue

            similarity = float(D[0][rank])
            if similarity < min_similarity:
                continue

            m = self._meta[idx].copy()
            score = similarity

            # --- Intent re-ranking ---
            # Snippets matching the predicted intent get boosted;
            # non-matching ones are penalized.
            if want_intents:
                snippet_intent = m.get("intent", "").lower()
                if any(w in snippet_intent for w in want_intents):
                    score *= 1.2
                else:
                    score *= 0.5

            # --- Concern re-ranking ---
            # High-concern posts strongly prefer high-concern KB entries.
            if want_concern:
                snippet_concern = m.get("concern", "").lower()
                if snippet_concern == want_concern:
                    score *= 1.2
                elif want_concern == "high" and snippet_concern != "high":
                    score *= 0.5
                elif want_concern != "high" and snippet_concern == "high":
                    score *= 0.8

            candidates.append({
                "rank": rank + 1,
                "score": score,
                "similarity": similarity,
                **m,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_snippets = candidates[:keep]

        safe_snippets = filter_unsafe_snippets(top_snippets)

        if len(safe_snippets) == 0 and len(top_snippets) > 0:
            logger.warning(
                "All %d retrieved snippets were filtered as unsafe; "
                "generator will receive no context and use fallback response.",
                len(top_snippets),
            )

        # Renumber ranks after filtering (consistent with RAG script)
        for i, snippet in enumerate(safe_snippets):
            snippet["rank"] = i + 1

        logger.info(
            "Retrieved %d snippets (from %d candidates) for post: %.60s...",
            len(safe_snippets),
            len(candidates),
            post,
        )

        return safe_snippets

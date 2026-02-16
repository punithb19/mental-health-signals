"""
Content safety filters for retrieved KB snippets.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# Patterns that indicate unsafe self-harm methods or pro-suicide content
UNSAFE_PATTERNS = [
    r"\b(hang|hanging|hanged)\s+(myself|yourself|themselves)",
    r"\b(jump|jumping|jumped)\s+(off|from)",
    r"\bmethod[s]?\s+to\s+(kill|die|suicide)",
    r"\bhow\s+to\s+(kill|die|suicide)",
    r"\b(pills?|medication)\s+(to|and)\s+(die|kill|overdose)",
    r"\bpro-?suicide\b",
    r"\bbetter\s+dead\b.*\bhow\b",
]


def filter_unsafe_snippets(snippets: List[Dict]) -> List[Dict]:
    """
    Remove snippets containing explicit self-harm methods or pro-suicide content.
    Returns only safe snippets with renumbered ranks.
    """
    filtered = []
    for snippet in snippets:
        text_lower = snippet.get("text", "").lower()
        if not any(re.search(pat, text_lower) for pat in UNSAFE_PATTERNS):
            filtered.append(snippet)
        else:
            logger.warning(
                "Filtered unsafe snippet: %s", snippet.get("doc_id", "unknown")
            )

    for i, snippet in enumerate(filtered):
        snippet["rank"] = i + 1

    return filtered

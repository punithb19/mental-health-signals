"""
Safety module: crisis detection, response validation, and content filtering.

Handles the safety-critical aspects of the RAG pipeline:
  - Multi-tier crisis detection (immediate / high / medium / none)
  - Response validation (instruction leakage, persona hallucinations, toxicity)
  - Crisis resource footers
  - Interaction logging
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crisis levels and keywords
# ---------------------------------------------------------------------------

@dataclass
class CrisisResult:
    """Result of crisis detection."""
    is_crisis: bool
    level: str  # "immediate", "high", "medium", "none"


IMMEDIATE_KEYWORDS = [
    "kill myself", "end my life", "suicide",
    "take my life", "overdose", "gonna jump",
    "going to jump", "planning to die", "planning to kill myself",
    "wrote a note", "saying goodbye",
]

HIGH_RISK_KEYWORDS = [
    "want to die", "cut myself", "cutting myself", "self-harm", "harm myself",
    "cut my", "cutting my",
    "better off dead", "everyone would be better", "no reason to live",
    "nothing to live for", "can't do this anymore", "give up", "hanging", "pills",
    "can't go on",
]

MEDIUM_RISK_KEYWORDS = [
    "hopeless", "worthless", "burden", "pointless", "no point",
    "can't take it", "breaking down", "falling apart",
]


# ---------------------------------------------------------------------------
# CrisisDetector
# ---------------------------------------------------------------------------

class CrisisDetector:
    """Detect crisis severity from post text and predicted concern level."""

    def detect(self, post: str, concern: Optional[str] = None) -> CrisisResult:
        """
        Multi-tier crisis detection.

        Args:
            post:    The user's post text.
            concern: Predicted concern level (from classifier), e.g. "high".

        Returns:
            CrisisResult with is_crisis and level.
        """
        post_lower = post.lower()

        # Tier 1: Immediate danger
        if any(kw in post_lower for kw in IMMEDIATE_KEYWORDS):
            return CrisisResult(is_crisis=True, level="immediate")

        # Concern-level override
        if concern and concern.lower() == "high":
            if any(kw in post_lower for kw in HIGH_RISK_KEYWORDS + IMMEDIATE_KEYWORDS):
                return CrisisResult(is_crisis=True, level="immediate")
            return CrisisResult(is_crisis=True, level="high")

        # Tier 2: High risk
        if any(kw in post_lower for kw in HIGH_RISK_KEYWORDS):
            return CrisisResult(is_crisis=True, level="high")

        # Tier 3: Medium risk
        if any(kw in post_lower for kw in MEDIUM_RISK_KEYWORDS):
            return CrisisResult(is_crisis=False, level="medium")

        return CrisisResult(is_crisis=False, level="none")

    @staticmethod
    def get_crisis_response(level: str) -> str:
        """Return appropriate crisis intervention message for the given level."""
        if level == "immediate":
            return (
                "\n\n--- IMMEDIATE SAFETY CONCERN DETECTED ---\n"
                "If you're in immediate danger, please:\n"
                "- Call emergency services (911 in US, 999 in UK, 112 in EU)\n"
                "- Contact a crisis helpline:\n"
                "  US: 988 Suicide & Crisis Lifeline\n"
                "  US: Text HOME to 741741 (Crisis Text Line)\n"
                "  International: https://findahelpline.com\n"
                "- Go to your nearest emergency room\n"
                "- Reach out to a trusted person immediately\n\n"
                "You deserve support from trained professionals who can help keep you safe."
            )
        elif level == "high":
            return (
                "\n\n--- SAFETY RESOURCES ---\n"
                "What you're experiencing sounds very difficult. Please consider:\n"
                "- Contacting a crisis helpline (988 in US, text HOME to 741741)\n"
                "- Reaching out to a mental health professional\n"
                "- Talking to a trusted friend or family member\n"
                "- If thoughts worsen, seek immediate help\n"
                "Resources: https://findahelpline.com"
            )
        return ""


# ---------------------------------------------------------------------------
# ResponseValidator
# ---------------------------------------------------------------------------

class ResponseValidator:
    """Validate generated responses for safety, quality, and grounding."""

    REJECT_MARKERS = [
        "guidelines:", "rules:", "you must", "do not copy",
        "using only the evidence", "your grounded response",
        "critical rules", "end of evidence", "end of post",
        "task:", "evidence snippets:", "ref 1", "ref 2", "snippet 1",
    ]

    PERSONA_MARKERS = [
        "i can relate", "i have been there", "i know how it feels",
        "i went through", "i have a job", "i am unemployed",
        "i have dealt with", "i struggle with", "my cat", "my husband",
        "my wife", "my boyfriend", "my girlfriend", "my children",
        "my kids", "my son", "my daughter", "my family", "my parents",
        "i have children", "i have kids", "i'm married", "i am married",
        "i think", "i believe", "i would", "i will", "i'd like", "i'd be happy",
        "i want", "what can i say",
    ]

    TOXIC_MARKERS = [
        "too good for you", "you are selfish", "you were selfish",
        "you should be ashamed", "it is your fault", "it's your fault",
        "you are right to feel ashamed", "you are pathetic",
        "you deserve to feel", "you are a burden",
        "[name]", "[insert", "counseling session", "our session",
        "in our meeting", "previous session",
        "therapist", "counselor", "my goal as a",
        "justified", "hero", "you will not be a hero",
    ]

    def is_valid(self, reply: str, post: str, snippets: List[Dict]) -> bool:
        """
        Validate a generated reply for safety and quality.

        Checks:
          - Minimum length
          - Starts with capital letter
          - No instruction leakage
          - No persona hallucinations
          - No toxic content
          - No excessive formatting (lists, bullets)
          - Minimum sentence count
        """
        if not reply or len(reply) < 40:
            logger.warning("Response too short (%d chars)", len(reply) if reply else 0)
            return False

        if not re.match(r"^[A-Z]", reply):
            logger.warning("Response doesn't start with capital letter")
            return False

        rl = reply.lower()

        if any(m in rl for m in self.REJECT_MARKERS):
            logger.warning("Instruction leakage detected")
            return False

        if any(m in rl for m in self.PERSONA_MARKERS):
            logger.warning("Persona/first-person hallucination detected")
            return False

        if any(m in rl for m in self.TOXIC_MARKERS):
            logger.warning("Toxic content detected")
            return False

        if "**" in reply or reply.count("-") > 6 or reply.count("\u2022") > 4:
            logger.warning("List formatting detected (likely copied)")
            return False

        sents = re.split(r"[.!?]+", reply)
        valid_sents = [s for s in sents if len(s.strip()) > 10]
        if len(valid_sents) < 2 and len(reply) < 60:
            logger.warning("Too few sentences: %d", len(valid_sents))
            return False

        return True

    def check_grounding(self, reply: str, snippets: List[Dict],
                        min_overlap: int = 3, min_ratio: float = 0.08) -> bool:
        """
        Check if the reply is sufficiently grounded in retrieved snippets.
        Returns True if grounded, False otherwise.
        """
        snippet_words = set()
        for s in snippets:
            text = s.get("text", "").lower()
            snippet_words |= set(re.findall(r"\b\w{4,}\b", text))

        reply_words = set(re.findall(r"\b\w{4,}\b", reply.lower()))
        overlap = len(snippet_words & reply_words)
        ratio = overlap / len(reply_words) if reply_words else 0

        if overlap >= min_overlap and ratio >= min_ratio:
            return True

        logger.warning("Low grounding: overlap=%d, ratio=%.2f", overlap, ratio)
        return False


# ---------------------------------------------------------------------------
# Interaction logging
# ---------------------------------------------------------------------------

def log_interaction(
    post: str,
    concern: Optional[str],
    crisis_level: str,
    reply: str,
    snippets: List[Dict],
    log_dir: str = "logs/interactions",
) -> None:
    """
    Log interactions for quality monitoring and safety review.
    High-risk interactions are written to a separate file.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "crisis_level": crisis_level,
        "predicted_concern": concern,
        "post_length": len(post),
        "post_hash": hash(post),
        "num_snippets": len(snippets),
        "reply_length": len(reply),
        "reply_hash": hash(reply),
        "requires_review": crisis_level in ["immediate", "high"],
    }

    if crisis_level in ["immediate", "high"]:
        log_file = os.path.join(
            log_dir, f"high_risk_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
    else:
        log_file = os.path.join(
            log_dir, f"general_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if crisis_level in ["immediate", "high"]:
        logger.warning("HIGH-RISK interaction logged: %s", crisis_level)

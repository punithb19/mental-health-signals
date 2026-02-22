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
# Crisis levels and detection
# ---------------------------------------------------------------------------

@dataclass
class CrisisResult:
    """Result of crisis detection."""
    is_crisis: bool
    level: str  # "immediate", "high", "medium", "none"


# ---------------------------------------------------------------------------
# Crisis keywords (safety net) + ML signals
# ---------------------------------------------------------------------------
#
# Why keywords exist alongside ML classifiers:
#   The intent classifier may not predict "Critical Risk" reliably --
#   it depends on training data quality and model capacity. Keywords
#   provide a deterministic safety net so that unambiguous crisis
#   language is NEVER missed. The ML classifiers (intent + concern)
#   provide ADDITIONAL detection for paraphrased or subtle language
#   that keywords can't cover.
#
# Hybrid strategy:
#   1. Keywords catch explicit crisis phrases → immediate/high
#   2. ML signals (Critical Risk intent, high concern) catch
#      what keywords miss → immediate/high/medium
#   3. Both layers run; whichever is more severe wins.

IMMEDIATE_KEYWORDS = [
    "kill myself", "end my life", "end it all", "suicide",
    "take my life", "overdose", "gonna jump",
    "going to jump", "planning to die", "planning to kill myself",
    "wrote a note", "saying goodbye", "want to end it",
    "don't want to be alive", "don't want to live",
    "don't want to exist", "ready to die",
]

HIGH_RISK_KEYWORDS = [
    "want to die", "cut myself", "cutting myself", "self-harm", "harm myself",
    "cut my", "cutting my",
    "better off dead", "better off without me", "better without me",
    "everyone would be better", "world would be better",
    "no reason to live", "nothing to live for",
    "can't do this anymore", "hanging", "pills",
    "can't go on", "no one cares", "nobody cares",
    "wish i was dead", "wish i were dead", "rather be dead",
    "don't see myself being here", "no future",
]

MEDIUM_RISK_KEYWORDS = [
    "hopeless", "worthless", "burden", "pointless", "no point",
    "can't take it", "breaking down", "falling apart",
]


# ---------------------------------------------------------------------------
# CrisisDetector
# ---------------------------------------------------------------------------

class CrisisDetector:
    """
    Hybrid crisis detection: keyword safety net + ML classifier signals.

    Keywords provide deterministic coverage for known crisis phrases.
    ML classifiers (intent tags + concern level) provide generalization
    to paraphrased or subtle crisis language.
    """

    def detect(
        self,
        post: str,
        intents: Optional[List[str]] = None,
        concern: Optional[str] = None,
    ) -> CrisisResult:
        """
        Multi-tier crisis detection combining keywords + ML signals.

        Can be called in two modes:
          - Pre-classification (intents/concern=None): keywords only
          - Post-classification (with intents/concern): full hybrid

        Args:
            post:     The user's post text.
            intents:  Predicted intent tags (may include "Critical Risk").
            concern:  Predicted concern level ("low", "medium", "high").

        Returns:
            CrisisResult with is_crisis and level.
        """
        post_lower = post.lower()

        # --- Layer 1: Keyword safety net ---
        keyword_level = self._keyword_check(post_lower, concern)

        # --- Layer 2: ML-driven signals ---
        ml_level = self._ml_check(intents, concern)

        # Take the more severe of the two layers
        level = self._max_severity(keyword_level, ml_level)

        if level == "immediate":
            return CrisisResult(is_crisis=True, level="immediate")
        elif level == "high":
            return CrisisResult(is_crisis=True, level="high")
        elif level == "medium":
            return CrisisResult(is_crisis=False, level="medium")
        return CrisisResult(is_crisis=False, level="none")

    @staticmethod
    def _keyword_check(post_lower: str, concern: Optional[str] = None) -> str:
        """Determine crisis level from keyword matching."""
        concern_high = concern is not None and concern.lower() == "high"

        if any(kw in post_lower for kw in IMMEDIATE_KEYWORDS):
            return "immediate"

        if any(kw in post_lower for kw in HIGH_RISK_KEYWORDS):
            return "immediate" if concern_high else "high"

        if any(kw in post_lower for kw in MEDIUM_RISK_KEYWORDS):
            return "high" if concern_high else "medium"

        if concern_high:
            return "medium"

        return "none"

    @staticmethod
    def _ml_check(
        intents: Optional[List[str]],
        concern: Optional[str],
    ) -> str:
        """Determine crisis level from ML classifier outputs."""
        if intents is None:
            return "none"

        has_critical_risk = "Critical Risk" in intents
        has_distress = "Mental Distress" in intents
        concern_high = concern is not None and concern.lower() == "high"
        concern_medium = concern is not None and concern.lower() == "medium"

        if has_critical_risk and concern_high:
            return "immediate"
        if has_critical_risk:
            return "high"
        if concern_high and has_distress:
            return "high"
        if concern_medium and has_distress:
            return "medium"

        return "none"

    @staticmethod
    def _max_severity(a: str, b: str) -> str:
        """Return the more severe of two crisis levels."""
        order = {"none": 0, "medium": 1, "high": 2, "immediate": 3}
        return a if order.get(a, 0) >= order.get(b, 0) else b

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
        "what can i say",
    ]

    # Regex patterns for first-person speech (assistant should not use "I")
    FIRST_PERSON_PATTERNS = [
        r"\bi'm\b", r"\bi am\b", r"\bi've\b", r"\bi have\b",
        r"\bi was\b", r"\bi will\b", r"\bi would\b", r"\bi can\b",
        r"\bi know\b", r"\bi think\b", r"\bi believe\b",
        r"\bi feel\b", r"\bi want\b", r"\bi need\b",
        r"\bi'd\b", r"\bi'll\b",
    ]

    TOXIC_MARKERS = [
        "too good for you", "you are selfish", "you were selfish",
        "you should be ashamed", "it is your fault", "it's your fault",
        "you are right to feel ashamed", "you are pathetic",
        "you deserve to feel", "you are a burden",
        "[name]", "[insert", "counseling session", "our session",
        "in our meeting", "previous session",
        "my goal as a", "as your therapist", "as your counselor",
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

        # Check for first-person speech patterns (assistant should not say "I")
        first_person_count = sum(
            1 for pat in self.FIRST_PERSON_PATTERNS if re.search(pat, rl)
        )
        if first_person_count >= 2:
            logger.warning("First-person speech detected (%d patterns)", first_person_count)
            return False

        if any(m in rl for m in self.TOXIC_MARKERS):
            logger.warning("Toxic content detected")
            return False

        # Reject hallucinated contact info (phone numbers, emails, URLs)
        if re.search(r"\(\d{3}\)\s*\d{3}[-.]?\d{4}", reply):
            logger.warning("Hallucinated phone number detected")
            return False
        if re.search(r"[\w.-]+@[\w.-]+\.\w{2,}", reply):
            logger.warning("Hallucinated email address detected")
            return False
        if re.search(r"http[s]?://(?!findahelpline)", rl):
            logger.warning("Hallucinated URL detected")
            return False

        if "**" in reply or reply.count("\u2022") > 4:
            logger.warning("List formatting detected (likely copied)")
            return False

        # Reject numbered lists like "1. ... 2. ... 3. ..."
        numbered_items = re.findall(r"(?:^|\n)\s*\d+\.\s", reply)
        if len(numbered_items) > 3:
            logger.warning("Numbered list detected (%d items)", len(numbered_items))
            return False

        sents = re.split(r"[.!?]+", reply)
        valid_sents = [s for s in sents if len(s.strip()) > 10]
        if len(valid_sents) < 2 and len(reply) < 80:
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

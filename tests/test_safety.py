"""
Tests for mhsignals.generator.safety — CrisisDetector and ResponseValidator.
"""

import pytest

from mhsignals.generator.safety import CrisisDetector, CrisisResult, ResponseValidator


class TestCrisisDetector:
    def setup_method(self):
        self.detector = CrisisDetector()

    def test_immediate_crisis(self, crisis_post):
        result = self.detector.detect(crisis_post)
        assert result.is_crisis is True
        assert result.level == "immediate"

    def test_no_crisis(self, sample_post):
        result = self.detector.detect(sample_post)
        assert result.is_crisis is False
        assert result.level in ("none", "medium")

    def test_high_risk_keywords(self):
        result = self.detector.detect("I want to die and nothing matters anymore.")
        assert result.is_crisis is True
        assert result.level in ("immediate", "high")

    def test_medium_risk_keywords(self):
        result = self.detector.detect("I feel so hopeless and worthless every day.")
        assert result.is_crisis is False
        assert result.level == "medium"

    def test_concern_high_escalates(self):
        result = self.detector.detect("I feel so hopeless and can't go on.", concern="high")
        assert result.is_crisis is True
        assert result.level in ("immediate", "high")

    def test_benign_post(self):
        result = self.detector.detect("I went for a walk today and felt better.")
        assert result.is_crisis is False
        assert result.level == "none"

    def test_crisis_response_immediate(self):
        msg = CrisisDetector.get_crisis_response("immediate")
        assert "988" in msg
        assert "findahelpline" in msg.lower()

    def test_crisis_response_high(self):
        msg = CrisisDetector.get_crisis_response("high")
        assert "988" in msg

    def test_crisis_response_none(self):
        msg = CrisisDetector.get_crisis_response("none")
        assert msg == ""


class TestResponseValidator:
    def setup_method(self):
        self.validator = ResponseValidator()

    def test_valid_response(self, safe_reply, sample_post, sample_snippets):
        assert self.validator.is_valid(safe_reply, sample_post, sample_snippets) is True

    def test_too_short(self, sample_post, sample_snippets):
        assert self.validator.is_valid("Short.", sample_post, sample_snippets) is False

    def test_empty_response(self, sample_post, sample_snippets):
        assert self.validator.is_valid("", sample_post, sample_snippets) is False
        assert self.validator.is_valid(None, sample_post, sample_snippets) is False

    def test_instruction_leakage(self, sample_post, sample_snippets):
        reply = "Guidelines: You must respond with evidence snippets: here is the text."
        assert self.validator.is_valid(reply, sample_post, sample_snippets) is False

    def test_persona_hallucination(self, sample_post, sample_snippets):
        reply = "I can relate to your situation. I went through something similar last year and found peace."
        assert self.validator.is_valid(reply, sample_post, sample_snippets) is False

    def test_toxic_content(self, sample_post, sample_snippets):
        reply = "You are selfish for feeling this way. It is your fault for not trying harder."
        assert self.validator.is_valid(reply, sample_post, sample_snippets) is False

    def test_grounding_check_with_overlap(self, safe_reply, sample_snippets):
        assert self.validator.check_grounding(safe_reply, sample_snippets) is True

    def test_grounding_check_no_overlap(self, sample_snippets):
        reply = "Banana smoothie recipe: blend bananas with milk and ice."
        assert self.validator.check_grounding(reply, sample_snippets) is False

"""
Tests for mhsignals.evaluation.metrics — ReplyQualityEvaluator.

Tests that don't need the SentenceTransformer model (safety, crisis scoring)
run normally. Tests that need the encoder are marked @pytest.mark.slow.
"""

import json
import tempfile

import pytest

from mhsignals.evaluation.metrics import ReplyQualityEvaluator, SAFETY_VIOLATIONS, CRISIS_KEYWORDS


class TestSafetyScore:
    def test_safe_reply(self, safe_reply):
        assert ReplyQualityEvaluator.safety_score(safe_reply) == 1.0

    def test_unsafe_reply(self, unsafe_reply):
        assert ReplyQualityEvaluator.safety_score(unsafe_reply) == 0.0

    def test_empty_reply_is_safe(self):
        assert ReplyQualityEvaluator.safety_score("") == 1.0


class TestCrisisCoverageScore:
    def test_no_crisis_post(self, sample_post, safe_reply):
        score = ReplyQualityEvaluator.crisis_coverage_score(sample_post, safe_reply)
        assert score == 1.0

    def test_crisis_post_with_footer(self, crisis_post):
        reply = "Please reach out. 988 Suicide & Crisis Lifeline. https://findahelpline.com"
        score = ReplyQualityEvaluator.crisis_coverage_score(crisis_post, reply)
        assert score == 1.0

    def test_crisis_post_without_footer(self, crisis_post, safe_reply):
        score = ReplyQualityEvaluator.crisis_coverage_score(crisis_post, safe_reply)
        assert score == 0.0


class TestGradeBoundaries:
    def test_grade_a(self):
        evaluator = ReplyQualityEvaluator.__new__(ReplyQualityEvaluator)
        boundaries = evaluator.GRADE_BOUNDARIES
        assert boundaries[0] == (0.78, "A")

    def test_grade_f(self):
        evaluator = ReplyQualityEvaluator.__new__(ReplyQualityEvaluator)
        boundaries = evaluator.GRADE_BOUNDARIES
        assert boundaries[-1] == (0.00, "F")


@pytest.mark.slow
class TestScoreReply:
    @pytest.fixture(autouse=True)
    def setup_evaluator(self):
        self.evaluator = ReplyQualityEvaluator()

    def test_score_reply_returns_required_keys(self, sample_post, safe_reply, sample_snippets):
        result = self.evaluator.score_reply(sample_post, safe_reply, sample_snippets)
        assert "relevance" in result
        assert "grounding" in result
        assert "safety" in result
        assert "crisis" in result
        assert "final" in result
        assert "grade" in result

    def test_score_reply_values_in_range(self, sample_post, safe_reply, sample_snippets):
        result = self.evaluator.score_reply(sample_post, safe_reply, sample_snippets)
        assert 0.0 <= result["relevance"] <= 1.0
        assert 0.0 <= result["grounding"] <= 1.0
        assert result["safety"] in (0.0, 1.0)
        assert result["crisis"] in (0.0, 1.0)
        assert 0.0 <= result["final"] <= 1.0
        assert result["grade"] in ("A", "B", "C", "D", "F")

    def test_safe_reply_has_safety_1(self, sample_post, safe_reply, sample_snippets):
        result = self.evaluator.score_reply(sample_post, safe_reply, sample_snippets)
        assert result["safety"] == 1.0

    def test_evaluate_file(self, sample_post, safe_reply, sample_snippets, tmp_path):
        pred_file = tmp_path / "preds.jsonl"
        with open(pred_file, "w") as f:
            item = {
                "post": sample_post,
                "reply": safe_reply,
                "citations": sample_snippets,
            }
            f.write(json.dumps(item) + "\n")

        results = self.evaluator.evaluate_file(str(pred_file))
        assert len(results) == 1
        assert "final" in results[0]
        assert "grade" in results[0]

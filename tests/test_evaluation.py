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


class TestEvaluateFileEdgeCases:
    """Test evaluate_file robustness: missing files, empty files, malformed lines."""

    def _make_evaluator(self):
        """Create evaluator without loading the real encoder (mock it)."""
        from unittest.mock import MagicMock, patch
        import numpy as np

        with patch("mhsignals.evaluation.metrics.SentenceTransformer") as MockST:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = np.random.randn(1, 384).astype(np.float32)
            MockST.return_value = mock_encoder
            return ReplyQualityEvaluator()

    def test_missing_file_raises(self):
        evaluator = self._make_evaluator()
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate_file("/nonexistent/path/preds.jsonl")

    def test_empty_file_returns_empty(self, tmp_path):
        pred_file = tmp_path / "empty.jsonl"
        pred_file.write_text("")
        evaluator = self._make_evaluator()
        results = evaluator.evaluate_file(str(pred_file))
        assert results == []

    def test_malformed_json_skipped(self, tmp_path):
        pred_file = tmp_path / "bad.jsonl"
        pred_file.write_text(
            '{"post": "valid post", "reply": "A valid reply here.", "citations": []}\n'
            'not valid json\n'
            '{"post": "another valid", "reply": "Another good reply.", "citations": []}\n'
        )
        evaluator = self._make_evaluator()
        results = evaluator.evaluate_file(str(pred_file))
        assert len(results) == 2

    def test_missing_post_field_skipped(self, tmp_path):
        pred_file = tmp_path / "no_post.jsonl"
        pred_file.write_text(
            '{"reply": "A reply without post.", "citations": []}\n'
            '{"post": "has post", "reply": "A valid reply here.", "citations": []}\n'
        )
        evaluator = self._make_evaluator()
        results = evaluator.evaluate_file(str(pred_file))
        assert len(results) == 1
        assert results[0]["post"].startswith("has post")

    def test_empty_post_field_skipped(self, tmp_path):
        pred_file = tmp_path / "empty_post.jsonl"
        pred_file.write_text(
            '{"post": "", "reply": "Reply to nothing.", "citations": []}\n'
            '{"post": "real post", "reply": "Real reply here.", "citations": []}\n'
        )
        evaluator = self._make_evaluator()
        results = evaluator.evaluate_file(str(pred_file))
        assert len(results) == 1

    def test_mixed_valid_invalid(self, tmp_path):
        pred_file = tmp_path / "mixed.jsonl"
        lines = [
            '{"post": "ok1", "reply": "Reply one is valid.", "citations": []}',
            'broken json',
            '{"no_post_key": true}',
            '',
            '{"post": "ok2", "reply": "Reply two is valid.", "citations": []}',
        ]
        pred_file.write_text("\n".join(lines) + "\n")
        evaluator = self._make_evaluator()
        results = evaluator.evaluate_file(str(pred_file))
        assert len(results) == 2


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

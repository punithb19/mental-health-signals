"""
Tests for mhsignals.pipeline — MHSignalsPipeline with mocked components.

These tests verify pipeline orchestration, crisis detection paths,
and input validation without loading real ML models.
"""

from unittest.mock import MagicMock, patch

import pytest

from mhsignals.generator.safety import CrisisDetector, CrisisResult, ResponseValidator
from mhsignals.pipeline import MHSignalsPipeline, Response


class TestPipelineProcess:
    """Test pipeline process() with mocked classifiers, retriever, and generator."""

    def _make_pipeline(
        self,
        intents=None,
        concern="medium",
        snippets=None,
        reply="This is a supportive response.",
    ):
        """Create a pipeline with mocked components."""
        intent_clf = MagicMock()
        intent_clf.predict.return_value = intents or ["Mental Distress"]

        concern_clf = MagicMock()
        concern_clf.predict.return_value = concern

        retriever = MagicMock()
        retriever.search.return_value = snippets or [
            {"doc_id": "kb_001", "intent": "Mental Distress", "concern": "Medium",
             "similarity": 0.8, "text": "Some helpful advice here."},
        ]

        generator = MagicMock()
        generator.generate.return_value = reply

        return MHSignalsPipeline(
            intent_classifier=intent_clf,
            concern_classifier=concern_clf,
            retriever=retriever,
            generator=generator,
            enable_logging=False,
        )

    def test_basic_response_structure(self, sample_post):
        pipeline = self._make_pipeline()
        response = pipeline(sample_post)

        assert isinstance(response, Response)
        assert response.post == sample_post
        assert isinstance(response.intents, list)
        assert isinstance(response.concern, str)
        assert isinstance(response.reply, str)
        assert isinstance(response.snippets, list)
        assert response.crisis_level in ("none", "medium", "high", "immediate")
        assert isinstance(response.crisis_detected, bool)
        assert response.disclaimer

    def test_intents_and_concern_propagated(self, sample_post):
        pipeline = self._make_pipeline(
            intents=["Seeking Help", "Mood Tracking"],
            concern="low",
        )
        response = pipeline(sample_post)

        assert response.intents == ["Seeking Help", "Mood Tracking"]
        assert response.concern == "low"

    def test_crisis_post_immediate_short_circuits(self, crisis_post):
        pipeline = self._make_pipeline()
        response = pipeline(crisis_post)

        assert response.crisis_detected is True
        assert response.crisis_level == "immediate"
        assert "988" in response.reply
        # Generator should NOT have been called for immediate crisis
        pipeline.generator.generate.assert_not_called()

    def test_non_crisis_calls_generator(self, sample_post):
        pipeline = self._make_pipeline()
        response = pipeline(sample_post)

        pipeline.generator.generate.assert_called_once()
        assert response.reply == "This is a supportive response."

    def test_critical_risk_triggers_crisis(self):
        """When ML classifier predicts Critical Risk + high concern,
        pipeline should trigger immediate crisis response."""
        post = "I don't see myself being here tomorrow."
        pipeline = self._make_pipeline(
            intents=["Critical Risk", "Mental Distress"],
            concern="high",
        )
        response = pipeline(post)

        assert response.crisis_detected is True
        assert response.crisis_level == "immediate"
        assert "988" in response.reply
        pipeline.generator.generate.assert_not_called()

    def test_medium_concern_with_distress_adds_footer(self):
        """Medium concern + Mental Distress = medium crisis level,
        which adds a safety footer but still generates a response."""
        post = "I feel so hopeless and worthless every day."
        pipeline = self._make_pipeline(
            intents=["Mental Distress", "Mood Tracking"],
            concern="medium",
        )
        response = pipeline(post)

        # Generator should have been called
        pipeline.generator.generate.assert_called_once()
        # Crisis level should be medium (adds footer)
        assert response.crisis_level == "medium"

    def test_empty_post_raises(self):
        pipeline = self._make_pipeline()
        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline("")

    def test_whitespace_post_raises(self):
        pipeline = self._make_pipeline()
        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline("   \n\t  ")

    def test_response_to_dict(self, sample_post):
        pipeline = self._make_pipeline()
        response = pipeline(sample_post)
        d = response.to_dict()

        assert isinstance(d, dict)
        assert "post" in d
        assert "intents" in d
        assert "concern" in d
        assert "reply" in d
        assert "snippets" in d
        assert "disclaimer" in d

    def test_response_to_json(self, sample_post):
        import json
        pipeline = self._make_pipeline()
        response = pipeline(sample_post)
        j = response.to_json()

        parsed = json.loads(j)
        assert parsed["post"] == sample_post

    def test_process_batch(self, sample_post):
        pipeline = self._make_pipeline()
        posts = [sample_post, "I went for a walk and felt better."]
        responses = pipeline.process_batch(posts)

        assert len(responses) == 2
        for r in responses:
            assert isinstance(r, Response)


class TestPipelineFromConfig:
    """Test from_config validation."""

    def test_missing_intent_checkpoint_raises(self, tmp_path):
        import yaml
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text(yaml.dump({
            "intent_checkpoint": "",
            "concern_checkpoint": "some/path",
        }))
        with pytest.raises(ValueError, match="intent_checkpoint"):
            MHSignalsPipeline.from_config(str(cfg_path))

    def test_missing_concern_checkpoint_raises(self, tmp_path):
        import yaml
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text(yaml.dump({
            "intent_checkpoint": "some/path",
            "concern_checkpoint": "",
        }))
        with pytest.raises(ValueError, match="concern_checkpoint"):
            MHSignalsPipeline.from_config(str(cfg_path))

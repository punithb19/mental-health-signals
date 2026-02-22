"""
Tests for mhsignals.generator.prompt — PromptBuilder.
"""

from mhsignals.generator.prompt import PromptBuilder


class TestPromptBuilder:
    def setup_method(self):
        self.builder = PromptBuilder()

    def test_build_returns_string(self, sample_post, sample_snippets):
        prompt = self.builder.build(sample_post, sample_snippets)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_instruction(self, sample_post, sample_snippets):
        prompt = self.builder.build(sample_post, sample_snippets)
        assert "supportive" in prompt.lower()
        assert "advice" in prompt.lower()

    def test_includes_post(self, sample_post, sample_snippets):
        prompt = self.builder.build(sample_post, sample_snippets)
        assert sample_post in prompt

    def test_includes_snippet_text(self, sample_post, sample_snippets):
        prompt = self.builder.build(sample_post, sample_snippets)
        for snippet in sample_snippets:
            # First 40 chars of each snippet should appear in prompt
            assert snippet["text"][:40] in prompt

    def test_respects_max_chars(self, sample_post, sample_snippets):
        prompt = self.builder.build(sample_post, sample_snippets, max_chars=200)
        assert len(prompt) <= 200

    def test_concern_affects_tone(self, sample_post, sample_snippets):
        prompt_high = self.builder.build(
            sample_post, sample_snippets, concern="high",
        )
        prompt_medium = self.builder.build(
            sample_post, sample_snippets, concern="medium",
        )
        assert "professional" in prompt_high.lower()
        assert "practical" in prompt_medium.lower() or "warm" in prompt_medium.lower()

    def test_empty_snippets(self, sample_post):
        prompt = self.builder.build(sample_post, [])
        assert isinstance(prompt, str)
        assert sample_post in prompt

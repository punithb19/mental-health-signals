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
        assert "Instruction:" in prompt

    def test_includes_post(self, sample_post, sample_snippets):
        prompt = self.builder.build(sample_post, sample_snippets)
        assert sample_post in prompt

    def test_includes_snippet_text(self, sample_post, sample_snippets):
        prompt = self.builder.build(sample_post, sample_snippets)
        for snippet in sample_snippets:
            assert snippet["text"][:50] in prompt

    def test_respects_max_chars(self, sample_post, sample_snippets):
        prompt = self.builder.build(sample_post, sample_snippets, max_chars=200)
        assert len(prompt) <= 200

    def test_empty_snippets(self, sample_post):
        prompt = self.builder.build(sample_post, [])
        assert isinstance(prompt, str)
        assert sample_post in prompt

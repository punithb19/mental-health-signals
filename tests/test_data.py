"""
Tests for mhsignals.data — tag normalization, concern normalization, canonical maps.
"""

from mhsignals.data import (
    CANON_KEYS,
    CONCERN_LABELS,
    normalize_concern,
    normalize_tag,
)


class TestNormalizeTag:
    def test_canonical_tags(self):
        assert normalize_tag("critical risk") == "Critical Risk"
        assert normalize_tag("Mental Distress") == "Mental Distress"
        assert normalize_tag("seeking help") == "Seeking Help"

    def test_strips_whitespace_and_period(self):
        assert normalize_tag("  mood tracking.  ") == "Mood Tracking"
        assert normalize_tag("progress update.") == "Progress Update"

    def test_causes_of_distress_alias(self):
        assert normalize_tag("causes of distress") == "Cause of Distress"

    def test_unknown_tag_returns_none(self):
        assert normalize_tag("nonexistent tag") is None
        assert normalize_tag("") is None


class TestNormalizeConcern:
    def test_standard_levels(self):
        assert normalize_concern("low") == "low"
        assert normalize_concern("medium") == "medium"
        assert normalize_concern("high") == "high"

    def test_case_insensitive(self):
        assert normalize_concern("LOW") == "low"
        assert normalize_concern("High") == "high"
        assert normalize_concern("MEDIUM") == "medium"

    def test_aliases(self):
        assert normalize_concern("med") == "medium"
        assert normalize_concern("mid") == "medium"

    def test_strips_whitespace(self):
        assert normalize_concern("  high  ") == "high"
        assert normalize_concern("low.") == "low"

    def test_invalid_returns_none(self):
        assert normalize_concern("") is None
        assert normalize_concern("extreme") is None
        assert normalize_concern(123) is None


class TestCanonicalConstants:
    def test_nine_intent_tags(self):
        assert len(CANON_KEYS) == 9

    def test_three_concern_labels(self):
        assert set(CONCERN_LABELS) == {"high", "low", "medium"}

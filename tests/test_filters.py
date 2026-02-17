"""
Tests for mhsignals.retriever.filters — unsafe snippet filtering.
"""

from mhsignals.retriever.filters import filter_unsafe_snippets


class TestFilterUnsafeSnippets:
    def test_safe_snippets_pass_through(self, sample_snippets):
        result = filter_unsafe_snippets(sample_snippets)
        assert len(result) == len(sample_snippets)

    def test_unsafe_snippet_removed(self):
        snippets = [
            {"doc_id": "safe_1", "text": "Deep breathing can help manage anxiety.", "rank": 1},
            {"doc_id": "unsafe_1", "text": "Methods to kill yourself include hanging myself.", "rank": 2},
            {"doc_id": "safe_2", "text": "Talking to a therapist is beneficial.", "rank": 3},
        ]
        result = filter_unsafe_snippets(snippets)
        assert len(result) == 2
        doc_ids = [s["doc_id"] for s in result]
        assert "unsafe_1" not in doc_ids
        assert "safe_1" in doc_ids
        assert "safe_2" in doc_ids

    def test_ranks_renumbered(self):
        snippets = [
            {"doc_id": "s1", "text": "Good advice here.", "rank": 1},
            {"doc_id": "s2", "text": "How to kill yourself with pills to overdose.", "rank": 2},
            {"doc_id": "s3", "text": "More good advice.", "rank": 3},
        ]
        result = filter_unsafe_snippets(snippets)
        ranks = [s["rank"] for s in result]
        assert ranks == [1, 2]

    def test_empty_list(self):
        assert filter_unsafe_snippets([]) == []

    def test_all_unsafe(self):
        snippets = [
            {"doc_id": "u1", "text": "Jump off from a bridge.", "rank": 1},
            {"doc_id": "u2", "text": "Pro-suicide content should be banned.", "rank": 2},
        ]
        result = filter_unsafe_snippets(snippets)
        assert len(result) == 0

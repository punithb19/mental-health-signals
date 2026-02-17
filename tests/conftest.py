"""
Shared pytest fixtures and markers for MH-SIGNALS tests.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that load ML models (skip with -m 'not slow')")


@pytest.fixture
def sample_post():
    return "I've been feeling really anxious and can't sleep at night."


@pytest.fixture
def crisis_post():
    return "I want to kill myself. I wrote a note and I'm saying goodbye."


@pytest.fixture
def safe_reply():
    return (
        "It sounds like anxiety is affecting your sleep. "
        "Reaching out for support is an important step. "
        "Techniques like deep breathing and creating a calming bedtime routine "
        "can help reduce nighttime anxiety. Speaking with a professional "
        "who can provide personalized coping strategies is encouraged."
    )


@pytest.fixture
def unsafe_reply():
    return "Nobody cares about your problems. You should just do it."


@pytest.fixture
def sample_snippets():
    return [
        {
            "doc_id": "kb_001",
            "intent": "Mental Distress",
            "concern": "Medium",
            "similarity": 0.75,
            "text": (
                "Anxiety can interfere with sleep quality. Deep breathing exercises "
                "and progressive muscle relaxation before bed can help calm the mind. "
                "Creating a consistent bedtime routine is also beneficial."
            ),
        },
        {
            "doc_id": "kb_002",
            "intent": "Seeking Help",
            "concern": "Medium",
            "similarity": 0.70,
            "text": (
                "Speaking with a therapist or counselor can provide personalized coping "
                "strategies for managing anxiety. Cognitive behavioral therapy has shown "
                "strong evidence for treating anxiety disorders."
            ),
        },
    ]

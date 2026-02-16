"""
Abstract base class for all MH-SIGNALS classifiers.

Every classifier (intent or concern, any encoder) implements this interface
so the Pipeline can call predict() without knowing the underlying model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class BaseClassifier(ABC):
    """
    Common interface for intent and concern classifiers.

    Subclasses must implement:
      - predict(text)       -> labels for a single post
      - predict_batch(texts) -> labels for multiple posts
      - save(path)          -> persist model artifacts to disk
      - load(path)          -> class method to restore a saved classifier
      - train(config)       -> class method to train and return a new classifier
    """

    @abstractmethod
    def predict(self, text: str) -> Union[List[str], str]:
        """
        Classify a single post.

        Returns:
          - For intent: List[str] of predicted tag names
          - For concern: str ("low", "medium", or "high")
        """
        ...

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Any]:
        """Classify a batch of posts. Returns a list of predictions."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save all model artifacts (weights, config, label mapping) to `path`."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseClassifier":
        """Load a previously saved classifier from `path`."""
        ...

    @classmethod
    @abstractmethod
    def train(cls, config: Dict) -> "BaseClassifier":
        """Train a new classifier from a config dict and return it."""
        ...

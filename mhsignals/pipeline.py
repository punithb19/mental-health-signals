"""
MHSignalsPipeline: end-to-end mental health signal detection and response generation.

This is the core architectural component that connects:
  1. Crisis detection    (fast keyword triage)
  2. Intent classification (multi-label: what the user is expressing)
  3. Concern classification (3-class: how severe/urgent)
  4. KB retrieval (FAISS search re-ranked by intent + concern)
  5. Response generation (Flan-T5, grounded in retrieved snippets)
  6. Safety validation + crisis footer

Problem statement:
  Given a mental health support-group post, automatically classify the user's
  intent signals and concern severity, retrieve relevant counselor-authored
  guidance informed by those classifications, and generate a safe, grounded,
  empathetic response -- all as a single end-to-end pipeline where classification
  directly drives retrieval and generation quality.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .classifiers.base import BaseClassifier
from .config import GeneratorConfig, PipelineConfig, RetrieverConfig, load_pipeline_config
from .generator.generate import ResponseGenerator
from .generator.safety import CrisisDetector, CrisisResult, ResponseValidator, log_interaction
from .retriever.search import KBRetriever

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is an automated support resource, NOT professional mental health care. "
    "For personalized help, please consult a licensed mental health professional."
)


@dataclass
class Response:
    """Structured output from the pipeline."""
    post: str
    intents: List[str]
    concern: str
    crisis_level: str
    crisis_detected: bool
    snippets: List[Dict]
    reply: str
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


class MHSignalsPipeline:
    """
    End-to-end pipeline: post -> classify -> retrieve -> generate -> validate.

    This pipeline ensures that classification results (intent tags and concern
    level) directly inform which KB snippets are retrieved for generation,
    closing the gap between signal detection and response quality.

    Usage:
        from mhsignals import MHSignalsPipeline

        pipeline = MHSignalsPipeline.from_config("configs/pipeline.yaml")
        response = pipeline("I can't focus before my exams and I'm panicking.")
        print(response.reply)
    """

    def __init__(
        self,
        intent_classifier: BaseClassifier,
        concern_classifier: BaseClassifier,
        retriever: KBRetriever,
        generator: ResponseGenerator,
        crisis_detector: Optional[CrisisDetector] = None,
        response_validator: Optional[ResponseValidator] = None,
        retriever_config: Optional[RetrieverConfig] = None,
        log_dir: str = "logs/interactions",
        enable_logging: bool = True,
    ):
        self.intent_clf = intent_classifier
        self.concern_clf = concern_classifier
        self.retriever = retriever
        self.generator = generator
        self.crisis_detector = crisis_detector or CrisisDetector()
        self.response_validator = response_validator or ResponseValidator()
        self._retriever_config = retriever_config
        self._log_dir = log_dir
        self._enable_logging = enable_logging

    @classmethod
    def from_config(cls, config_path: str) -> "MHSignalsPipeline":
        """
        Build pipeline from a pipeline.yaml config file.

        The config specifies paths to:
          - Saved intent classifier checkpoint
          - Saved concern classifier checkpoint
          - KB metadata + FAISS index
          - Generator model name + settings
        """
        cfg = load_pipeline_config(config_path)

        # Load classifiers from saved checkpoints
        intent_clf = _load_classifier(cfg.intent_checkpoint)
        concern_clf = _load_classifier(cfg.concern_checkpoint)

        # Load retriever
        retriever = KBRetriever(
            metadata_path=cfg.kb.metadata_jsonl,
            faiss_index_path=cfg.kb.faiss_index,
            encoder_name=cfg.retriever.encoder_model,
        )

        # Load generator
        generator = ResponseGenerator(cfg.generator)

        return cls(
            intent_classifier=intent_clf,
            concern_classifier=concern_clf,
            retriever=retriever,
            generator=generator,
            retriever_config=cfg.retriever,
            log_dir=cfg.log_dir,
        )

    def __call__(self, post: str) -> Response:
        """
        Process a single post through the full pipeline.

        Steps:
          1. Crisis triage (keyword-based, fast)
          2. Intent classification (model inference)
          3. Concern classification (model inference)
          4. KB retrieval (FAISS search, re-ranked by intent + concern)
          5. Response generation (Flan-T5 with grounding)
          6. Safety validation + crisis footer
        """
        return self.process(post)

    def process(self, post: str) -> Response:
        """Process a single post through the full pipeline."""

        # ---- Step 1: Crisis triage (fast, keyword-based) ----
        # Run crisis detection first as a fast safety gate.
        # Concern isn't known yet, so we detect without it initially.
        crisis_preliminary = self.crisis_detector.detect(post, concern=None)

        if crisis_preliminary.level == "immediate":
            crisis_reply = CrisisDetector.get_crisis_response("immediate")
            response = Response(
                post=post,
                intents=[],
                concern="high",
                crisis_level="immediate",
                crisis_detected=True,
                snippets=[],
                reply=crisis_reply,
            )
            self._log(response)
            return response

        # ---- Step 2: Intent classification ----
        intents = self.intent_clf.predict(post)
        logger.info("Predicted intents: %s", intents)

        # ---- Step 3: Concern classification ----
        concern = self.concern_clf.predict(post)
        logger.info("Predicted concern: %s", concern)

        # ---- Re-check crisis with concern level ----
        crisis = self.crisis_detector.detect(post, concern=concern)

        if crisis.is_crisis and crisis.level == "immediate":
            crisis_reply = CrisisDetector.get_crisis_response("immediate")
            response = Response(
                post=post,
                intents=intents,
                concern=concern,
                crisis_level="immediate",
                crisis_detected=True,
                snippets=[],
                reply=crisis_reply,
            )
            self._log(response)
            return response

        # ---- Step 4: KB retrieval (classification-informed) ----
        ret_cfg = self._retriever_config
        snippets = self.retriever.search(
            post=post,
            intents=intents,
            concern=concern,
            topk=ret_cfg.topk if ret_cfg else 50,
            keep=ret_cfg.keep if ret_cfg else 5,
            min_similarity=ret_cfg.min_similarity if ret_cfg else 0.45,
        )

        # ---- Step 5: Response generation ----
        reply = self.generator.generate(
            post=post,
            snippets=snippets,
            intents=intents,
            concern=concern,
        )

        # ---- Step 6: Crisis footer if needed ----
        if crisis.is_crisis:
            footer = CrisisDetector.get_crisis_response(crisis.level)
            reply += footer

        response = Response(
            post=post,
            intents=intents,
            concern=concern,
            crisis_level=crisis.level,
            crisis_detected=crisis.is_crisis,
            snippets=[
                {
                    "doc_id": s.get("doc_id", ""),
                    "intent": s.get("intent", ""),
                    "concern": s.get("concern", ""),
                    "similarity": s.get("similarity", 0.0),
                    "text": s.get("text", ""),
                }
                for s in snippets
            ],
            reply=reply,
        )

        self._log(response)
        return response

    def process_batch(self, posts: List[str]) -> List[Response]:
        """Process multiple posts. Each post goes through the full pipeline."""
        return [self.process(post) for post in posts]

    def _log(self, response: Response) -> None:
        """Log interaction for safety monitoring."""
        if not self._enable_logging:
            return
        try:
            log_interaction(
                post=response.post,
                concern=response.concern,
                crisis_level=response.crisis_level,
                reply=response.reply,
                snippets=response.snippets,
                log_dir=self._log_dir,
            )
        except Exception as e:
            logger.error("Failed to log interaction: %s", e)

    def cleanup(self) -> None:
        """Release generator GPU/MPS resources."""
        self.generator.cleanup()


def _load_classifier(checkpoint_path: str) -> BaseClassifier:
    """
    Auto-detect classifier type from checkpoint metadata and load it.

    Reads meta.json to determine which class to instantiate.
    """
    import os
    meta_path = os.path.join(checkpoint_path, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No meta.json found at {checkpoint_path}. "
            "Train and save a classifier first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    clf_type = meta.get("type", "")

    if clf_type == "MinilmLRIntentClassifier":
        from .classifiers.intent import MinilmLRIntentClassifier
        return MinilmLRIntentClassifier.load(checkpoint_path)
    elif clf_type == "LoRAIntentClassifier":
        from .classifiers.intent import LoRAIntentClassifier
        return LoRAIntentClassifier.load(checkpoint_path)
    elif clf_type == "MinilmLRConcernClassifier":
        from .classifiers.concern import MinilmLRConcernClassifier
        return MinilmLRConcernClassifier.load(checkpoint_path)
    elif clf_type == "LoRAConcernClassifier":
        from .classifiers.concern import LoRAConcernClassifier
        return LoRAConcernClassifier.load(checkpoint_path)
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")

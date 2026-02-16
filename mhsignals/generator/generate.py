"""
Response generator: Flan-T5 text generation with quality controls.

Generates grounded, empathetic replies using retrieved KB snippets
as context for a seq2seq language model.
"""

import logging
import re
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..config import GeneratorConfig
from .prompt import PromptBuilder
from .safety import ResponseValidator

logger = logging.getLogger(__name__)

# Phrases to ban during generation
BANNED_PHRASES = [
    "i understand how you feel", "i've been there", "i can relate",
    "i went through", "stay strong", "you've got this",
    "sending prayers", "god bless", "bless you",
]

# Fallback response when generation fails
FALLBACK_RESPONSE = (
    "It sounds like you're going through a challenging time. "
    "While a specific response could not be generated right now, "
    "reaching out to a mental health professional or trusted person "
    "who can offer personalized support is encouraged. Your wellbeing matters."
)

HIGH_CONCERN_FALLBACK = (
    "What you're describing sounds very difficult and overwhelming. "
    "Given the intensity of what you're experiencing, it would be helpful to speak "
    "with a mental health professional who can provide personalized support. "
    "If you're in crisis, please reach out to a crisis helpline or emergency services."
)


class ResponseGenerator:
    """
    Generate grounded replies using Flan-T5 and retrieved KB snippets.

    Handles:
      - Model loading and device placement
      - Bad-words filtering during generation
      - Multi-attempt generation with retry logic
      - Response cleanup (truncation repair)
    """

    def __init__(self, config: GeneratorConfig):
        self._config = config
        self._prompt_builder = PromptBuilder()
        self._validator = ResponseValidator()

        self._device = self._resolve_device(config.device)

        logger.info("Loading generator model: %s", config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name, torch_dtype=torch.float32
        )
        self._model.to(self._device)
        self._model.eval()
        logger.info("Generator ready on device: %s", self._device)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Resolve the best available device."""
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            requested = "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            requested = "cpu"
        # Avoid Flan-T5 CPU segfault on macOS -- prefer MPS
        if requested == "cpu" and torch.backends.mps.is_available():
            logger.info("Using MPS instead of CPU (avoids macOS segfault)")
            requested = "mps"
        return requested

    def _build_bad_words_ids(self) -> List[List[int]]:
        """Build token-ID lists for banned phrases."""
        bad_words_ids = []
        for phrase in BANNED_PHRASES:
            try:
                ids = self._tokenizer(phrase.lower(), add_special_tokens=False).input_ids
                if len(ids) > 1:
                    bad_words_ids.append(ids)
            except Exception:
                pass
        return bad_words_ids

    def generate(
        self,
        post: str,
        snippets: List[Dict],
        intents: Optional[List[str]] = None,
        concern: Optional[str] = None,
        max_attempts: int = 3,
    ) -> str:
        """
        Generate a grounded response for a post using retrieved snippets.

        Args:
            post:         User's post text.
            snippets:     Retrieved KB snippets from the retriever.
            intents:      Predicted intent tags (for prompt context).
            concern:      Predicted concern level (for prompt context).
            max_attempts: Number of generation retries.

        Returns:
            Generated reply string (never None; returns fallback on failure).
        """
        if not snippets:
            logger.warning("No snippets provided -- returning fallback.")
            if concern and concern.lower() == "high":
                return HIGH_CONCERN_FALLBACK
            return FALLBACK_RESPONSE

        prompt = self._prompt_builder.build(
            post, snippets, intents, concern,
            max_chars=self._config.max_prompt_chars,
        )

        bad_words_ids = self._build_bad_words_ids()

        gen_kwargs = {
            "max_new_tokens": self._config.max_new_tokens,
            "min_length": 20,
            "max_length": 1024,
            "length_penalty": 1.2,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "num_beams": 4,
            "do_sample": False,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids

        if self._config.do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "num_beams": 1,
            })

        input_ids = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(self._device)

        best = None
        for attempt in range(max_attempts):
            try:
                gen_ids = self._model.generate(input_ids, **gen_kwargs)
                reply = self._tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
                best = reply

                if self._validator.is_valid(reply, post, snippets):
                    if self._validator.check_grounding(reply, snippets):
                        logger.info("Valid grounded response on attempt %d", attempt + 1)
                        return reply
                    else:
                        logger.warning("Low grounding on attempt %d -- accepting anyway", attempt + 1)
                        return reply

                logger.warning("Attempt %d: invalid response: %.150s...", attempt + 1, reply)

                # Add slight randomness for retries
                if attempt > 0:
                    gen_kwargs["temperature"] = 0.8 + (attempt * 0.15)
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["num_beams"] = 1

            except Exception as e:
                logger.error("Generation attempt %d failed: %s", attempt + 1, e)

        # Cleanup best attempt if available
        if best:
            cleaned = self._cleanup_truncated(best)
            if cleaned:
                return cleaned

        logger.error("All generation attempts failed, returning fallback")
        if concern and concern.lower() == "high":
            return HIGH_CONCERN_FALLBACK
        return FALLBACK_RESPONSE

    @staticmethod
    def _cleanup_truncated(text: str) -> Optional[str]:
        """Try to clean up a truncated response by finding the last full sentence."""
        if re.search(r"[.!?]\s*$", text):
            return text

        match = re.search(r"(.*(?<!\d)[.!?])", text, re.DOTALL)
        if match:
            return match.group(1)

        if "." in text:
            return text.rsplit(".", 1)[0] + "."

        return None

    def cleanup(self) -> None:
        """Release GPU/MPS memory."""
        try:
            del self._model
            del self._tokenizer
            import gc
            gc.collect()
            if self._device == "cuda":
                torch.cuda.empty_cache()
            elif self._device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

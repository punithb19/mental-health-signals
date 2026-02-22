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
    "i'm here", "i hope", "i wish", "i'm sorry",
    "thank you for", "thanks for", "thank you so",
    "keep me in", "thoughts and prayers",
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

POSITIVE_COPING_RESPONSE = (
    "It's wonderful to hear that you're taking such positive steps for your "
    "well-being. The strategies you're using -- like being mindful and reflective "
    "-- are evidence-based approaches that many people find genuinely helpful. "
    "The fact that you're already noticing improvements shows real commitment "
    "to your mental health. Keep building on what's working for you, and "
    "remember that consistency is key to lasting change."
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
        """Build token-ID lists for banned phrases and first-person tokens."""
        bad_words_ids = []
        for phrase in BANNED_PHRASES:
            try:
                ids = self._tokenizer(phrase.lower(), add_special_tokens=False).input_ids
                if len(ids) > 1:
                    bad_words_ids.append(ids)
            except Exception:
                pass

        # Ban standalone first-person pronoun "I" token to prevent
        # the model from generating any first-person text
        first_person_tokens = ["I", "I'm", "I've", "I'll", "I'd", "I am"]
        for token in first_person_tokens:
            try:
                ids = self._tokenizer(token, add_special_tokens=False).input_ids
                if ids:
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
        # Positive coping + low concern: the user is doing well; the KB has
        # problem-oriented advice that would be inappropriate here, so return
        # an affirming template directly.
        if (intents and "Positive Coping" in intents
                and concern and concern.lower() == "low"):
            logger.info("Positive coping detected -- returning affirming response")
            return POSITIVE_COPING_RESPONSE

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
            "min_new_tokens": 60,
            "length_penalty": 1.5,
            "repetition_penalty": 1.2,
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
            prompt, return_tensors="pt", truncation=True, max_length=768
        ).input_ids.to(self._device)

        # Decoder seeding: force the model to start addressing the user
        # directly, preventing echo/hallucination of the input
        DECODER_SEEDS = [
            "It sounds like you",
            "What you're going through",
            "Your feelings of",
        ]

        best = None
        for attempt in range(max_attempts):
            try:
                attempt_kwargs = gen_kwargs.copy()

                # Use decoder seeding on each attempt with different starters
                seed_text = DECODER_SEEDS[attempt % len(DECODER_SEEDS)]
                decoder_ids = self._tokenizer(
                    seed_text, add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(self._device)
                attempt_kwargs["decoder_input_ids"] = decoder_ids

                gen_ids = self._model.generate(input_ids, **attempt_kwargs)
                reply = self._tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

                # Normalize: capitalize first letter
                if reply and reply[0].islower():
                    reply = reply[0].upper() + reply[1:]
                best = reply

                # Clean up any instruction-leakage artifacts
                cleaned = self._cleanup_response(reply)
                if cleaned and len(cleaned) >= 40:
                    reply = cleaned

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
                    attempt_kwargs["temperature"] = 0.8 + (attempt * 0.15)
                    attempt_kwargs["do_sample"] = True
                    attempt_kwargs["num_beams"] = 1

            except Exception as e:
                logger.error("Generation attempt %d failed: %s", attempt + 1, e)

        # Cleanup best attempt if available
        if best:
            cleaned = self._cleanup_response(best)
            if cleaned:
                if cleaned[0].islower():
                    cleaned = cleaned[0].upper() + cleaned[1:]
                return cleaned

        logger.error("All generation attempts failed, returning fallback")
        if concern and concern.lower() == "high":
            return HIGH_CONCERN_FALLBACK
        if intents and "Positive Coping" in intents and concern and concern.lower() == "low":
            return POSITIVE_COPING_RESPONSE
        return FALLBACK_RESPONSE

    @staticmethod
    def _cleanup_response(text: str) -> Optional[str]:
        """Clean up a generated response: strip artifacts and find last full sentence."""
        if not text:
            return None

        # Strip common instruction-leakage and generic filler fragments
        leakage_phrases = [
            "Write a supportive", "Tell the person", "Use the advice",
            "Thank you for", "Thanks again", "We'll be in touch",
            "You're welcome", "We're here to",
            "Keep me in your", "thoughts and prayers",
            "Best of luck", "Good luck with",
            "a pleasure working with", "been a pleasure",
            "Let me know if you", "Feel free to",
            "Don't hesitate to", "If you have any questions",
        ]
        lines = text.split(". ")
        clean_lines = []
        for line in lines:
            if not any(lp.lower() in line.lower() for lp in leakage_phrases):
                clean_lines.append(line)
        text = ". ".join(clean_lines)

        # Ensure ends with proper punctuation
        text = text.strip()
        if not text:
            return None

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

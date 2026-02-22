"""
Prompt engineering for grounded RAG generation.

Uses T5-native task framing: summarization and paraphrase tasks that
T5 models handle well, even at the base (250M) size.
"""

from typing import Dict, List, Optional


MAX_SNIPPET_CHARS = 250
MAX_PROMPT_SNIPPETS = 3


class PromptBuilder:
    """Construct prompts for the Flan-T5 generator from snippets + post."""

    def build(
        self,
        post: str,
        snippets: List[Dict],
        intents: Optional[List[str]] = None,
        concern: Optional[str] = None,
        max_chars: int = 2000,
    ) -> str:
        """
        Build a T5-native grounded RAG prompt.

        Uses a summarization/paraphrase format that T5 handles well:
        "Summarize this advice for someone who [situation]: [advice]"

        This avoids complex multi-rule instructions that small T5 models
        fail to follow, and instead leverages T5's strong summarization.
        """
        # Extract key advice sentences from snippets
        use_snippets = snippets[:MAX_PROMPT_SNIPPETS]
        advice_parts = []
        for snip in use_snippets:
            txt = snip["text"][:MAX_SNIPPET_CHARS].replace("\n", " ").strip()
            # Trim to last full sentence
            if txt and txt[-1] not in ".!?":
                last_period = txt.rfind(".")
                if last_period > 50:
                    txt = txt[:last_period + 1]
            advice_parts.append(txt)

        advice_text = " ".join(advice_parts)

        # Concern-aware framing
        tone = "supportive"
        extra = ""
        if concern and concern.lower() == "high":
            tone = "gentle and supportive, encouraging professional help"
        elif concern and concern.lower() == "medium":
            tone = "warm and supportive, with practical suggestions"
        elif concern and concern.lower() == "low":
            tone = "encouraging and affirming"
            if intents and "Positive Coping" in intents:
                extra = " Acknowledge their positive efforts and encourage them to continue."

        post_text = post[:300].strip()

        prompt = (
            f"Summarize the following advice into a {tone} reply "
            f"for someone who says: \"{post_text}\"\n\n"
            f"Advice: {advice_text}\n\n"
            f"Write a 3-5 sentence reply addressing the person directly as 'you'.{extra}"
        )

        return prompt[:max_chars]

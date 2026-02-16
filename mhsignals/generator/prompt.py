"""
Prompt engineering for grounded RAG generation.

Builds structured prompts that ground Flan-T5 output in KB snippets.
"""

from typing import Dict, List, Optional


class PromptBuilder:
    """Construct prompts for the Flan-T5 generator from snippets + post."""

    INSTRUCTION = (
        "Instruction: You are a supportive AI mental health assistant. "
        "Your goal is to provide validation and support based ONLY on the provided advice perspectives.\n"
        "Read the user's situation and the advice perspectives carefully.\n"
        "Synthesize the relevant advice into a warm, supportive response (single paragraph).\n"
        "Validate the user's feelings but do NOT agree with negative self-talk.\n"
        "Do NOT offer personal opinions or advice not found in the perspectives.\n"
        "Do NOT use lists, bullet points, or numbered steps. Write in full sentences.\n"
        "Do NOT use 'I', 'me', 'my', or share personal experiences. Speak only as a supportive resource.\n"
        "Do NOT refer to the user as 'patient', 'client', or use clinical jargon.\n"
        "Do NOT pretend to be a counselor or refer to past sessions.\n\n"
    )

    def build(
        self,
        post: str,
        snippets: List[Dict],
        intents: Optional[List[str]] = None,
        concern: Optional[str] = None,
        max_chars: int = 2000,
    ) -> str:
        """
        Build a grounded RAG prompt for Flan-T5.

        Args:
            post:     User's post text.
            snippets: Retrieved KB snippets (each has a 'text' key).
            intents:  Predicted intent tags (currently used for context, not injected).
            concern:  Predicted concern level (currently used for context, not injected).
            max_chars: Maximum prompt length in characters.

        Returns:
            Formatted prompt string.
        """
        context = "Advice Perspectives:\n"
        for snip in snippets:
            txt = snip["text"][:250].replace("\n", " ")
            context += f"- {txt}\n"
        context += "\n"

        post_block = f"User Situation: {post}\n\n"
        task = "Assistant Response:\n"

        prompt = self.INSTRUCTION + context + post_block + task
        return prompt[:max_chars]

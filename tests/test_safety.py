"""
Tests for mhsignals.generator.safety — CrisisDetector and ResponseValidator.
"""


from mhsignals.generator.safety import CrisisDetector, ResponseValidator


class TestCrisisDetector:
    def setup_method(self):
        self.detector = CrisisDetector()

    # --- Keyword fast-path (explicit phrases) ---

    def test_immediate_crisis_keyword(self, crisis_post):
        """Explicit crisis keywords trigger immediate even without ML signals."""
        result = self.detector.detect(crisis_post)
        assert result.is_crisis is True
        assert result.level == "immediate"

    def test_suicide_keyword(self):
        result = self.detector.detect("I've been thinking about suicide.")
        assert result.is_crisis is True
        assert result.level == "immediate"

    # --- ML-driven detection ---

    def test_critical_risk_intent_high_concern(self):
        """Critical Risk intent + high concern = immediate."""
        result = self.detector.detect(
            "I don't see myself being here tomorrow.",
            intents=["Critical Risk"], concern="high",
        )
        assert result.is_crisis is True
        assert result.level == "immediate"

    def test_critical_risk_intent_alone(self):
        """Critical Risk intent alone = high."""
        result = self.detector.detect(
            "Everything feels pointless lately.",
            intents=["Critical Risk"], concern="medium",
        )
        assert result.is_crisis is True
        assert result.level == "high"

    def test_high_concern_with_distress(self):
        """High concern + Mental Distress = high."""
        result = self.detector.detect(
            "I can't stop crying and I feel trapped.",
            intents=["Mental Distress"], concern="high",
        )
        assert result.is_crisis is True
        assert result.level == "high"

    def test_high_concern_without_distress(self):
        """High concern without distress = medium (safety footer, not crisis)."""
        result = self.detector.detect(
            "I have been through a lot.",
            intents=["Miscellaneous"], concern="high",
        )
        assert result.is_crisis is False
        assert result.level == "medium"

    def test_medium_concern_with_distress(self):
        """Medium concern + Mental Distress = medium."""
        result = self.detector.detect(
            "I feel so hopeless and worthless every day.",
            intents=["Mental Distress", "Mood Tracking"], concern="medium",
        )
        assert result.is_crisis is False
        assert result.level == "medium"

    def test_no_crisis_benign_post(self):
        """Benign post with low concern = none."""
        result = self.detector.detect(
            "I went for a walk today and felt better.",
            intents=["Positive Coping"], concern="low",
        )
        assert result.is_crisis is False
        assert result.level == "none"

    def test_no_crisis_no_signals(self):
        """No ML signals and no keywords = none."""
        result = self.detector.detect("I had a regular day at work.")
        assert result.is_crisis is False
        assert result.level == "none"

    def test_paraphrased_crisis_caught_by_ml(self):
        """ML-driven detection catches paraphrased crisis language
        that keywords would miss, when classifier correctly tags it."""
        result = self.detector.detect(
            "I don't see myself being here tomorrow.",
            intents=["Critical Risk", "Mental Distress"], concern="high",
        )
        assert result.is_crisis is True
        assert result.level == "immediate"

    # --- Crisis response messages ---

    def test_crisis_response_immediate(self):
        msg = CrisisDetector.get_crisis_response("immediate")
        assert "988" in msg
        assert "findahelpline" in msg.lower()

    def test_crisis_response_high(self):
        msg = CrisisDetector.get_crisis_response("high")
        assert "988" in msg

    def test_crisis_response_none(self):
        msg = CrisisDetector.get_crisis_response("none")
        assert msg == ""


class TestResponseValidator:
    def setup_method(self):
        self.validator = ResponseValidator()

    def test_valid_response(self, safe_reply, sample_post, sample_snippets):
        assert self.validator.is_valid(safe_reply, sample_post, sample_snippets) is True

    def test_too_short(self, sample_post, sample_snippets):
        assert self.validator.is_valid("Short.", sample_post, sample_snippets) is False

    def test_empty_response(self, sample_post, sample_snippets):
        assert self.validator.is_valid("", sample_post, sample_snippets) is False
        assert self.validator.is_valid(None, sample_post, sample_snippets) is False

    def test_instruction_leakage(self, sample_post, sample_snippets):
        reply = "Guidelines: You must respond with evidence snippets: here is the text."
        assert self.validator.is_valid(reply, sample_post, sample_snippets) is False

    def test_persona_hallucination(self, sample_post, sample_snippets):
        reply = "I can relate to your situation. I went through something similar last year and found peace."
        assert self.validator.is_valid(reply, sample_post, sample_snippets) is False

    def test_toxic_content(self, sample_post, sample_snippets):
        reply = "You are selfish for feeling this way. It is your fault for not trying harder."
        assert self.validator.is_valid(reply, sample_post, sample_snippets) is False

    def test_hallucinated_phone(self, sample_post, sample_snippets):
        reply = "You can call me at (713) 853-8080 for help and support today."
        assert self.validator.is_valid(reply, sample_post, sample_snippets) is False

    def test_hallucinated_email(self, sample_post, sample_snippets):
        reply = "Reach out to support at helpdesk@fake.com for more assistance with your issues."
        assert self.validator.is_valid(reply, sample_post, sample_snippets) is False

    def test_grounding_check_with_overlap(self, safe_reply, sample_snippets):
        assert self.validator.check_grounding(safe_reply, sample_snippets) is True

    def test_grounding_check_no_overlap(self, sample_snippets):
        reply = "Banana smoothie recipe: blend bananas with milk and ice."
        assert self.validator.check_grounding(reply, sample_snippets) is False

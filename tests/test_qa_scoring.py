import unittest
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path

# Mock langchain imports to avoid dependency issues
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['langchain_groq'] = MagicMock()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.qa_scoring import (
    get_qa_scoring_prompt,
    run_qa_scoring,
    _format_timestamp,
    DIMENSION_WEIGHTS,
)
from src.graph.state import QAScoringResult


class TestFormatTimestamp(unittest.TestCase):
    """Test cases for timestamp formatting utility."""

    def test_format_timestamp_zero_seconds(self):
        """Test formatting 0 seconds."""
        self.assertEqual(_format_timestamp(0), "00:00")

    def test_format_timestamp_less_than_minute(self):
        """Test formatting seconds less than 60."""
        self.assertEqual(_format_timestamp(30), "00:30")
        self.assertEqual(_format_timestamp(45), "00:45")
        self.assertEqual(_format_timestamp(59), "00:59")

    def test_format_timestamp_exactly_one_minute(self):
        """Test formatting 60 seconds."""
        self.assertEqual(_format_timestamp(60), "01:00")

    def test_format_timestamp_multiple_minutes(self):
        """Test formatting multiple minutes."""
        self.assertEqual(_format_timestamp(125), "02:05")
        self.assertEqual(_format_timestamp(300), "05:00")
        self.assertEqual(_format_timestamp(605), "10:05")

    def test_format_timestamp_large_values(self):
        """Test formatting large timestamp values."""
        self.assertEqual(_format_timestamp(3661), "61:01")
        self.assertEqual(_format_timestamp(7325), "122:05")

    def test_format_timestamp_float_values(self):
        """Test formatting float timestamp values."""
        self.assertEqual(_format_timestamp(30.5), "00:30")
        self.assertEqual(_format_timestamp(125.9), "02:05")


class TestGetQAScoringPrompt(unittest.TestCase):
    """Test cases for QA scoring prompt generation."""

    def test_prompt_contains_scoring_philosophy(self):
        """Test that prompt includes scoring philosophy."""
        prompt = get_qa_scoring_prompt()
        self.assertIn("SCORING PHILOSOPHY", prompt)
        self.assertIn("baseline competent performance", prompt)

    def test_prompt_contains_all_dimensions(self):
        """Test that prompt includes all five dimensions."""
        prompt = get_qa_scoring_prompt()
        self.assertIn("Professionalism", prompt)
        self.assertIn("Empathy", prompt)
        self.assertIn("Problem Resolution", prompt)
        self.assertIn("Compliance", prompt)
        self.assertIn("Communication Clarity", prompt)

    def test_prompt_contains_dimension_rubrics(self):
        """Test that prompt includes dimension rubrics."""
        prompt = get_qa_scoring_prompt()
        self.assertIn("DIMENSION RUBRICS", prompt)

    def test_prompt_contains_scoring_scale(self):
        """Test that prompt explains 1-5 scoring scale."""
        prompt = get_qa_scoring_prompt()
        self.assertIn("1-5", prompt)
        self.assertIn("Scale 1-5", prompt)

    def test_prompt_contains_justification_guidelines(self):
        """Test that prompt includes justification guidelines."""
        prompt = get_qa_scoring_prompt()
        self.assertIn("JUSTIFICATION GUIDELINES", prompt)
        self.assertIn("MM:SS timestamp format", prompt)

    def test_prompt_contains_output_instructions(self):
        """Test that prompt includes output instructions."""
        prompt = get_qa_scoring_prompt()
        self.assertIn("OUTPUT INSTRUCTIONS", prompt)
        self.assertIn("compliance_flag", prompt)

    def test_prompt_is_string(self):
        """Test that prompt returns a string."""
        prompt = get_qa_scoring_prompt()
        self.assertIsInstance(prompt, str)

    def test_prompt_is_not_empty(self):
        """Test that prompt is not empty."""
        prompt = get_qa_scoring_prompt()
        self.assertGreater(len(prompt), 100)


class TestRunQAScoring(unittest.TestCase):
    """Test cases for run_qa_scoring function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_segment_1 = MagicMock()
        self.mock_segment_1.start = 0
        self.mock_segment_1.end = 10
        self.mock_segment_1.speaker = "Agent"
        self.mock_segment_1.text = "Hello, how can I help you?"

        self.mock_segment_2 = MagicMock()
        self.mock_segment_2.start = 10
        self.mock_segment_2.end = 20
        self.mock_segment_2.speaker = "Customer"
        self.mock_segment_2.text = "I have a problem with my account."

        self.mock_transcription = MagicMock()
        self.mock_transcription.call_id = "call_123"
        self.mock_transcription.segments = [self.mock_segment_1, self.mock_segment_2]

        self.mock_summary = MagicMock()
        self.mock_summary.summary = "Customer had account issue, agent resolved it."

        self.mock_scoring_result = MagicMock(spec=QAScoringResult)
        self.mock_scoring_result.professionalism = 4
        self.mock_scoring_result.empathy = 3
        self.mock_scoring_result.problem_resolution = 4
        self.mock_scoring_result.compliance = 5
        self.mock_scoring_result.communication_clarity = 4
        self.mock_scoring_result.justification = "Good overall performance."
        self.mock_scoring_result.overall_score = 0
        self.mock_scoring_result.call_id = None

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_success(self, mock_llm):
        """Test successful QA scoring."""
        mock_llm.with_structured_output.return_value.invoke.return_value = self.mock_scoring_result

        state = {
            "transcription": self.mock_transcription,
            "summary": self.mock_summary,
        }

        result = run_qa_scoring(state)

        self.assertIn("qa_score", result)
        self.assertEqual(result["qa_score"].call_id, "call_123")
        # Check that overall_score was computed
        self.assertNotEqual(result["qa_score"].overall_score, 0)

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_computes_overall_score(self, mock_llm):
        """Test that overall score is computed from dimension weights."""
        mock_llm.with_structured_output.return_value.invoke.return_value = self.mock_scoring_result

        state = {
            "transcription": self.mock_transcription,
            "summary": self.mock_summary,
        }

        result = run_qa_scoring(state)

        # Manually compute expected score
        expected_score = (
            4 * DIMENSION_WEIGHTS["professionalism"] +
            3 * DIMENSION_WEIGHTS["empathy"] +
            4 * DIMENSION_WEIGHTS["problem_resolution"] +
            5 * DIMENSION_WEIGHTS["compliance"] +
            4 * DIMENSION_WEIGHTS["communication_clarity"]
        )
        expected_score = round(expected_score, 2)

        self.assertEqual(result["qa_score"].overall_score, expected_score)

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_missing_transcription(self, mock_llm):
        """Test handling when transcription is missing."""
        state = {
            "summary": self.mock_summary,
        }

        result = run_qa_scoring(state)

        self.assertIn("qa_score", result)
        self.assertFalse(result["qa_score"]["is_valid"])
        self.assertIn("Transcription not available", result["qa_score"]["reason"])
        mock_llm.assert_not_called()

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_transcription_none(self, mock_llm):
        """Test handling when transcription is None."""
        state = {
            "transcription": None,
            "summary": self.mock_summary,
        }

        result = run_qa_scoring(state)

        self.assertFalse(result["qa_score"]["is_valid"])
        self.assertIn("Transcription not available", result["qa_score"]["reason"])
        mock_llm.assert_not_called()

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_missing_summary(self, mock_llm):
        """Test handling when summary is missing."""
        state = {
            "transcription": self.mock_transcription,
        }

        result = run_qa_scoring(state)

        self.assertFalse(result["qa_score"]["is_valid"])
        self.assertIn("Summary not available", result["qa_score"]["reason"])
        mock_llm.assert_not_called()

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_summary_none(self, mock_llm):
        """Test handling when summary is None."""
        state = {
            "transcription": self.mock_transcription,
            "summary": None,
        }

        result = run_qa_scoring(state)

        self.assertFalse(result["qa_score"]["is_valid"])
        self.assertIn("Summary not available", result["qa_score"]["reason"])
        mock_llm.assert_not_called()

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_with_dict_summary(self, mock_llm):
        """Test QA scoring when summary is a dictionary."""
        mock_llm.with_structured_output.return_value.invoke.return_value = self.mock_scoring_result

        summary_dict = {"summary": "Account issue resolved successfully."}
        state = {
            "transcription": self.mock_transcription,
            "summary": summary_dict,
        }

        result = run_qa_scoring(state)

        self.assertIn("qa_score", result)
        # Verify the invoke was called with the correct summary
        call_args = mock_llm.with_structured_output.return_value.invoke.call_args
        self.assertIn("Account issue resolved successfully", str(call_args))

    @patch('src.agents.qa_scoring.sleep')
    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_with_retries_success(self, mock_llm, mock_sleep):
        """Test that QA scoring retries on failure and eventually succeeds."""
        mock_llm.with_structured_output.return_value.invoke.side_effect = [
            Exception("LLM error"),
            Exception("LLM error"),
            self.mock_scoring_result,
        ]

        state = {
            "transcription": self.mock_transcription,
            "summary": self.mock_summary,
        }

        result = run_qa_scoring(state)

        # Should have called invoke 3 times
        self.assertEqual(mock_llm.with_structured_output.return_value.invoke.call_count, 3)
        # Should have succeeded on third attempt
        self.assertEqual(result["qa_score"].call_id, "call_123")

    @patch('src.agents.qa_scoring.sleep')
    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_exhausts_retries(self, mock_llm, mock_sleep):
        """Test that QA scoring fails after exhausting retries."""
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("LLM error")

        state = {
            "transcription": self.mock_transcription,
            "summary": self.mock_summary,
        }

        result = run_qa_scoring(state)

        # Should have attempted 3 times
        self.assertEqual(mock_llm.with_structured_output.return_value.invoke.call_count, 3)
        # Should have failed
        self.assertFalse(result["qa_score"]["is_valid"])
        self.assertIn("after 3 attempts", result["qa_score"]["reason"])

    @patch('src.agents.qa_scoring.sleep')
    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_retry_backoff(self, mock_llm, mock_sleep):
        """Test that retry backoff increases exponentially."""
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("Error")

        state = {
            "transcription": self.mock_transcription,
            "summary": self.mock_summary,
        }

        run_qa_scoring(state)

        # Check that sleep was called with exponential backoff
        sleep_calls = mock_sleep.call_args_list
        # First retry: min(2^0, 10) = 1
        # Second retry: min(2^1, 10) = 2
        self.assertEqual(len(sleep_calls), 2)
        self.assertEqual(sleep_calls[0][0][0], 1)  # First sleep
        self.assertEqual(sleep_calls[1][0][0], 2)  # Second sleep

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_formats_transcript(self, mock_llm):
        """Test that transcript is properly formatted with timestamps."""
        mock_llm.with_structured_output.return_value.invoke.return_value = self.mock_scoring_result

        state = {
            "transcription": self.mock_transcription,
            "summary": self.mock_summary,
        }

        run_qa_scoring(state)

        # Check that invoke was called with formatted transcript
        call_args = mock_llm.with_structured_output.return_value.invoke.call_args
        user_message = call_args[0][0][1]["content"]

        # Verify timestamp format
        self.assertIn("[00:00-00:10]", user_message)
        self.assertIn("[00:10-00:20]", user_message)

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_preserves_state(self, mock_llm):
        """Test that unrelated state fields are preserved."""
        mock_llm.with_structured_output.return_value.invoke.return_value = self.mock_scoring_result

        state = {
            "transcription": self.mock_transcription,
            "summary": self.mock_summary,
            "call_id": "call_123",
            "other_field": "should_be_preserved",
        }

        result = run_qa_scoring(state)

        self.assertIn("other_field", result)
        self.assertEqual(result["other_field"], "should_be_preserved")

    @patch('src.agents.qa_scoring.llm')
    def test_run_qa_scoring_all_dimensions_present(self, mock_llm):
        """Test that all five scoring dimensions are returned."""
        mock_llm.with_structured_output.return_value.invoke.return_value = self.mock_scoring_result

        state = {
            "transcription": self.mock_transcription,
            "summary": self.mock_summary,
        }

        result = run_qa_scoring(state)

        qa_score = result["qa_score"]
        self.assertTrue(hasattr(qa_score, "professionalism"))
        self.assertTrue(hasattr(qa_score, "empathy"))
        self.assertTrue(hasattr(qa_score, "problem_resolution"))
        self.assertTrue(hasattr(qa_score, "compliance"))
        self.assertTrue(hasattr(qa_score, "communication_clarity"))
        self.assertTrue(hasattr(qa_score, "overall_score"))


class TestDimensionWeights(unittest.TestCase):
    """Test cases for dimension weights configuration."""

    def test_weights_sum_to_one(self):
        """Test that dimension weights sum to 1.0."""
        total = sum(DIMENSION_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_weights_all_positive(self):
        """Test that all weights are positive."""
        for weight in DIMENSION_WEIGHTS.values():
            self.assertGreater(weight, 0)

    def test_weights_all_dimensions_present(self):
        """Test that all dimensions have weights."""
        expected_dimensions = {
            "professionalism",
            "empathy",
            "problem_resolution",
            "compliance",
            "communication_clarity",
        }
        self.assertEqual(set(DIMENSION_WEIGHTS.keys()), expected_dimensions)

    def test_weights_reasonable_values(self):
        """Test that weights are within reasonable range (0-1)."""
        for weight in DIMENSION_WEIGHTS.values():
            self.assertGreaterEqual(weight, 0)
            self.assertLessEqual(weight, 1)


if __name__ == '__main__':
    unittest.main()

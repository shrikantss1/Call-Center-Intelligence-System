from src.graph.edges import (
    route_after_intake,
    route_after_qa,
    route_after_transcription,
)


class TestRouteAfterIntake:
    def test_route_after_intake_validation_passed(self):
        """Test routing when validation passes."""
        state = {"validation_passed": True}
        assert route_after_intake(state) == "transcribe_step"

    def test_route_after_intake_validation_failed(self):
        """Test routing when validation fails."""
        state = {"state": "intake_failed"}
        assert route_after_intake(state) == "error_step"

    def test_route_after_intake_missing_validation_passed(self):
        """Test routing when state is not intake_failed."""
        state = {}
        assert route_after_intake(state) == "transcribe_step"


class TestRouteAfterTranscription:
    def test_route_after_transcription_always_summarize(self):
        """Test that transcription always routes to injection check."""
        state = {}
        assert route_after_transcription(state) == "injection_check_step"

    def test_route_after_transcription_with_data(self):
        """Test that transcription routes to injection check regardless of state content."""
        state = {"transcript": "some text", "duration": 300}
        assert route_after_transcription(state) == "injection_check_step"


class TestRouteAfterQA:
    def test_route_after_qa_with_critical_flag(self):
        """Test routing when critical compliance flag exists."""
        state = {
            "compliance_flags": [
                {"severity": "warning", "message": "minor issue"},
                {"severity": "critical", "message": "major issue"},
            ]
        }
        assert route_after_qa(state) == "supervisor_step"

    def test_route_after_qa_without_critical_flag(self):
        """Test routing when no critical compliance flags exist."""
        state = {
            "compliance_flags": [
                {"severity": "warning", "message": "minor issue"},
                {"severity": "info", "message": "informational"},
            ]
        }
        assert route_after_qa(state) == "report_step"

    def test_route_after_qa_empty_flags(self):
        """Test routing when compliance flags list is empty."""
        state = {"compliance_flags": []}
        assert route_after_qa(state) == "report_step"

    def test_route_after_qa_missing_flags(self):
        """Test routing when compliance_flags key is missing."""
        state = {}
        assert route_after_qa(state) == "report_step"

    def test_route_after_qa_first_critical_flag(self):
        """Test routing with critical flag in first position."""
        state = {
            "compliance_flags": [
                {"severity": "critical", "message": "critical issue"},
                {"severity": "warning", "message": "warning issue"},
            ]
        }
        assert route_after_qa(state) == "supervisor_step"

    def test_route_after_qa_multiple_critical_flags(self):
        """Test routing with multiple critical flags."""
        state = {
            "compliance_flags": [
                {"severity": "critical", "message": "issue 1"},
                {"severity": "critical", "message": "issue 2"},
            ]
        }
        assert route_after_qa(state) == "supervisor_step"

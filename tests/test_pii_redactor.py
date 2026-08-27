import pytest

from src.security.pii_redactor import PII_PATTERNS, get_pii_matches, redact_pii


class TestRedactPII:
    """Test PII redaction functionality."""

    def test_redact_ssn(self):
        """Test that SSN pattern is redacted."""
        text = "My SSN is 123-45-6789 and it should be redacted."
        result = redact_pii(text)
        assert "123-45-6789" not in result
        assert "[REDACTED]" in result

    def test_redact_credit_card_with_dashes(self):
        """Test that credit card numbers with dashes are redacted."""
        text = "My card is 1234-5678-9012-3456 please protect it."
        result = redact_pii(text)
        assert "1234-5678-9012-3456" not in result
        assert "[REDACTED]" in result

    def test_redact_credit_card_with_spaces(self):
        """Test that credit card numbers with spaces are redacted."""
        text = "Use this card: 1234 5678 9012 3456 for payment."
        result = redact_pii(text)
        assert "1234 5678 9012 3456" not in result
        assert "[REDACTED]" in result

    def test_redact_credit_card_without_formatting(self):
        """Test that credit card numbers without formatting are redacted."""
        text = "Card number 1234567890123456 should be redacted."
        result = redact_pii(text)
        assert "1234567890123456" not in result
        assert "[REDACTED]" in result

    def test_redact_email(self):
        """Test that email addresses are redacted."""
        text = "Contact me at john.doe@example.com for more info."
        result = redact_pii(text)
        assert "john.doe@example.com" not in result
        assert "[REDACTED]" in result

    def test_redact_email_with_numbers(self):
        """Test that email addresses with numbers are redacted."""
        text = "My email is test123@company.co.uk for registration."
        result = redact_pii(text)
        assert "test123@company.co.uk" not in result
        assert "[REDACTED]" in result

    def test_redact_phone_formatted_with_dashes(self):
        """Test that phone numbers with dashes are redacted."""
        text = "Call me at 555-123-4567 during business hours."
        result = redact_pii(text)
        assert "555-123-4567" not in result
        assert "[REDACTED]" in result

    def test_redact_phone_formatted_with_parentheses(self):
        """Test that phone numbers with parentheses are redacted."""
        text = "My number is (555) 123-4567 or (555) 234-5678."
        result = redact_pii(text)
        assert "(555) 123-4567" not in result
        assert "[REDACTED]" in result

    def test_redact_phone_with_plus_prefix(self):
        """Test that phone numbers with plus prefix are redacted."""
        text = "Call international: +1-555-123-4567 anytime."
        result = redact_pii(text)
        assert "+1-555-123-4567" not in result
        assert "[REDACTED]" in result

    def test_redact_phone_without_formatting(self):
        """Test that unformatted phone numbers are redacted."""
        text = "Reach me at 5551234567 for support."
        result = redact_pii(text)
        assert "5551234567" not in result
        assert "[REDACTED]" in result

    def test_redact_multiple_pii_items(self):
        """Test redacting multiple PII items in one text."""
        text = "Name: John Doe, Email: john@example.com, Phone: 555-123-4567, SSN: 123-45-6789"
        result = redact_pii(text)
        assert "john@example.com" not in result
        assert "555-123-4567" not in result
        assert "123-45-6789" not in result
        assert result.count("[REDACTED]") >= 3

    def test_redact_overlapping_patterns(self):
        """Test that overlapping PII patterns are handled correctly."""
        text = "SSN: 123-45-6789 and Card: 1234-5678-9012-3456"
        result = redact_pii(text)
        assert "123-45-6789" not in result
        assert "1234-5678-9012-3456" not in result

    def test_empty_string(self):
        """Test redaction on empty string."""
        result = redact_pii("")
        assert result == ""

    def test_no_pii_found(self):
        """Test that text without PII is unchanged."""
        text = "This is a normal message with no sensitive information."
        result = redact_pii(text)
        assert result == text

    def test_custom_replacement_string(self):
        """Test using a custom replacement string."""
        text = "SSN: 123-45-6789"
        result = redact_pii(text, replacement="[MASKED]")
        assert "123-45-6789" not in result
        assert "[MASKED]" in result

    def test_redact_preserves_non_pii_content(self):
        """Test that non-PII content is preserved."""
        text = "Contact John at john@example.com about the project deadline."
        result = redact_pii(text)
        assert "Contact" in result
        assert "about the project deadline" in result
        assert "john@example.com" not in result

    def test_redact_multiple_same_type_pii(self):
        """Test redacting multiple instances of the same PII type."""
        text = "First email: alice@example.com and second: bob@example.com"
        result = redact_pii(text)
        assert "alice@example.com" not in result
        assert "bob@example.com" not in result
        assert result.count("[REDACTED]") == 2


class TestGetPIIMatches:
    """Test PII detection and extraction."""

    def test_find_ssn(self):
        """Test finding SSN patterns."""
        text = "My SSN is 123-45-6789"
        matches = get_pii_matches(text)
        assert len(matches) >= 1
        ssn_match = next((m for m in matches if m['type'] == 'SSN'), None)
        assert ssn_match is not None
        assert ssn_match['value'] == "123-45-6789"

    def test_find_credit_card(self):
        """Test finding credit card patterns."""
        text = "Card: 1234-5678-9012-3456"
        matches = get_pii_matches(text)
        cc_match = next((m for m in matches if m['type'] == 'CREDIT_CARD'), None)
        assert cc_match is not None
        assert "1234" in cc_match['value'] and "3456" in cc_match['value']

    def test_find_email(self):
        """Test finding email patterns."""
        text = "Email me at test@example.com"
        matches = get_pii_matches(text)
        email_match = next((m for m in matches if m['type'] == 'EMAIL'), None)
        assert email_match is not None
        assert email_match['value'] == "test@example.com"

    def test_find_phone(self):
        """Test finding phone number patterns."""
        text = "Call 555-123-4567 for support"
        matches = get_pii_matches(text)
        phone_match = next((m for m in matches if m['type'] == 'PHONE'), None)
        assert phone_match is not None

    def test_find_multiple_different_pii(self):
        """Test finding multiple different PII types."""
        text = "SSN: 123-45-6789, Email: john@example.com, Phone: 555-123-4567"
        matches = get_pii_matches(text)
        assert len(matches) >= 3
        types = [m['type'] for m in matches]
        assert 'SSN' in types
        assert 'EMAIL' in types
        assert 'PHONE' in types

    def test_match_has_correct_positions(self):
        """Test that match positions are correct."""
        text = "Contact: john@example.com"
        matches = get_pii_matches(text)
        email_match = next((m for m in matches if m['type'] == 'EMAIL'), None)
        assert email_match is not None
        assert text[email_match['start']:email_match['end']] == "john@example.com"

    def test_no_pii_matches(self):
        """Test that no matches are found in clean text."""
        text = "This is a completely clean message with no sensitive data."
        matches = get_pii_matches(text)
        assert len(matches) == 0

    def test_multiple_same_type_pii(self):
        """Test finding multiple instances of same PII type."""
        text = "Contact alice@example.com or bob@example.com"
        matches = get_pii_matches(text)
        email_matches = [m for m in matches if m['type'] == 'EMAIL']
        assert len(email_matches) >= 2

    def test_pii_matches_sorted_by_position(self):
        """Test that matches are sorted by position."""
        text = "Email: john@example.com and phone: 555-123-4567"
        matches = get_pii_matches(text)
        positions = [m['start'] for m in matches]
        assert positions == sorted(positions)

    def test_match_details_complete(self):
        """Test that match details include all required fields."""
        text = "SSN: 123-45-6789"
        matches = get_pii_matches(text)
        assert len(matches) >= 1
        match = matches[0]
        assert 'type' in match
        assert 'value' in match
        assert 'start' in match
        assert 'end' in match


class TestPIIPatternsValidation:
    """Test the PII pattern definitions themselves."""

    def test_patterns_are_defined(self):
        """Test that PII_PATTERNS dictionary is properly defined."""
        assert isinstance(PII_PATTERNS, dict)
        assert len(PII_PATTERNS) > 0

    def test_required_patterns_exist(self):
        """Test that all required PII pattern types exist."""
        required_types = ['SSN', 'CREDIT_CARD', 'EMAIL', 'PHONE']
        for pattern_type in required_types:
            assert pattern_type in PII_PATTERNS
            assert isinstance(PII_PATTERNS[pattern_type], str)

    def test_patterns_are_valid_regex(self):
        """Test that pattern strings are valid regex."""
        import re
        for pattern_type, pattern in PII_PATTERNS.items():
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Invalid regex in {pattern_type}: {e}")

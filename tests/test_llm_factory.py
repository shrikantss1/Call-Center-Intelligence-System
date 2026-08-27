import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock langchain imports to avoid dependency issues
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['langchain_groq'] = MagicMock()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_factory import get_llm


class TestGetLLM(unittest.TestCase):
    """Test cases for the get_llm factory function."""

    def test_get_llm_openai_default(self):
        """Test creating OpenAI LLM with default model."""
        with patch('src.utils.llm_factory.ChatOpenAI') as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            result = get_llm('openai')

            mock_openai.assert_called_once_with(
                model='gpt-4o-mini',
                timeout=None,
                base_url='https://api.openai.com/v1'
            )
            self.assertEqual(result, mock_instance)

    def test_get_llm_openai_custom_model(self):
        """Test creating OpenAI LLM with custom model."""
        with patch('src.utils.llm_factory.ChatOpenAI') as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            result = get_llm('openai', model='gpt-4-turbo')

            mock_openai.assert_called_once_with(
                model='gpt-4-turbo',
                timeout=None,
                base_url='https://api.openai.com/v1'
            )

    def test_get_llm_openai_with_timeout(self):
        """Test creating OpenAI LLM with timeout."""
        with patch('src.utils.llm_factory.ChatOpenAI') as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            result = get_llm('openai', timeout=30.0)

            mock_openai.assert_called_once_with(
                model='gpt-4o-mini',
                timeout=30.0,
                base_url='https://api.openai.com/v1'
            )

    def test_get_llm_openai_fallback_on_typeerror(self):
        """Test OpenAI fallback when timeout is not supported."""
        with patch('src.utils.llm_factory.ChatOpenAI') as mock_openai:
            mock_instance = MagicMock()
            # First call raises TypeError, second succeeds
            mock_openai.side_effect = [TypeError('timeout not supported'), mock_instance]

            result = get_llm('openai', timeout=30.0)

            # Should be called twice: first with timeout, then with model_name
            self.assertEqual(mock_openai.call_count, 2)
            self.assertEqual(result, mock_instance)

    def test_get_llm_gemini_default(self):
        """Test creating Gemini LLM with default model."""
        with patch('src.utils.llm_factory.ChatGoogleGenerativeAI') as mock_gemini:
            mock_instance = MagicMock()
            mock_gemini.return_value = mock_instance

            result = get_llm('gemini')

            mock_gemini.assert_called_once_with(
                model='gemini-2.0-flash',
                timeout=None
            )

    def test_get_llm_gemini_custom_model(self):
        """Test creating Gemini LLM with custom model."""
        with patch('src.utils.llm_factory.ChatGoogleGenerativeAI') as mock_gemini:
            mock_instance = MagicMock()
            mock_gemini.return_value = mock_instance

            result = get_llm('gemini', model='gemini-1.5-pro')

            mock_gemini.assert_called_once_with(
                model='gemini-1.5-pro',
                timeout=None
            )

    def test_get_llm_gemini_with_timeout(self):
        """Test creating Gemini LLM with timeout."""
        with patch('src.utils.llm_factory.ChatGoogleGenerativeAI') as mock_gemini:
            mock_instance = MagicMock()
            mock_gemini.return_value = mock_instance

            result = get_llm('gemini', timeout=45.0)

            mock_gemini.assert_called_once_with(
                model='gemini-2.0-flash',
                timeout=45.0
            )

    def test_get_llm_groq_default(self):
        """Test creating Groq LLM with default model."""
        with patch('src.utils.llm_factory.ChatGroq') as mock_groq:
            mock_instance = MagicMock()
            mock_groq.return_value = mock_instance

            result = get_llm('groq')

            mock_groq.assert_called_once_with(
                model='llama-3.3-70b-versatile',
                timeout=None
            )

    def test_get_llm_groq_custom_model(self):
        """Test creating Groq LLM with custom model."""
        with patch('src.utils.llm_factory.ChatGroq') as mock_groq:
            mock_instance = MagicMock()
            mock_groq.return_value = mock_instance

            result = get_llm('groq', model='mixtral-8x7b-32768')

            mock_groq.assert_called_once_with(
                model='mixtral-8x7b-32768',
                timeout=None
            )

    def test_get_llm_case_insensitive(self):
        """Test that provider names are case-insensitive."""
        with patch('src.utils.llm_factory.ChatOpenAI') as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            get_llm('OpenAI')
            get_llm('OPENAI')
            get_llm('openAI')

            self.assertEqual(mock_openai.call_count, 3)

    def test_get_llm_with_whitespace(self):
        """Test that provider names with whitespace are handled."""
        with patch('src.utils.llm_factory.ChatOpenAI') as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            get_llm('  openai  ')

            mock_openai.assert_called_once()

    def test_get_llm_none_provider(self):
        """Test that None provider raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_llm(None)

        self.assertIn('provider is required', str(context.exception))

    def test_get_llm_unsupported_provider(self):
        """Test that unsupported provider raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_llm('claude')

        self.assertIn('Unsupported provider', str(context.exception))

    def test_get_llm_empty_provider(self):
        """Test that empty provider string raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_llm('')

        self.assertIn('Unsupported provider', str(context.exception))

    def test_get_llm_groq_fallback(self):
        """Test Groq fallback when timeout is not supported."""
        with patch('src.utils.llm_factory.ChatGroq') as mock_groq:
            mock_instance = MagicMock()
            mock_groq.side_effect = [TypeError('timeout not supported'), mock_instance]

            result = get_llm('groq', timeout=20.0)

            self.assertEqual(mock_groq.call_count, 2)
            self.assertEqual(result, mock_instance)


class TestLLMFactoryIntegration(unittest.TestCase):
    """Integration tests for LLM factory module initialization."""

    @patch('src.utils.llm_factory.load_config')
    @patch('src.utils.llm_factory.ChatOpenAI')
    def test_llm_initialization_success(self, mock_openai, mock_load_config):
        """Test successful LLM initialization on module load."""
        mock_config = MagicMock()
        mock_config.llm_provider = 'openai'
        mock_config.llm_timeout_seconds = 30
        mock_load_config.return_value = mock_config

        mock_openai_instance = MagicMock()
        mock_openai_instance.model_name = 'gpt-4o-mini'
        mock_openai.return_value = mock_openai_instance

        # This would test the initialization block at module level
        # For now we verify the get_llm function works with real config
        result = get_llm('openai', timeout=30)
        self.assertIsNotNone(result)

    @patch('src.utils.llm_factory.load_config')
    def test_llm_initialization_failure(self, mock_load_config):
        """Test handling of LLM initialization failure."""
        mock_load_config.side_effect = Exception('Config file not found')

        # The module should handle this gracefully
        # Testing get_llm directly with proper mocking
        with self.assertRaises(Exception):
            get_llm(None)


if __name__ == '__main__':
    unittest.main()

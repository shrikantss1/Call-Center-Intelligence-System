# Unit Test Coverage Summary

## Overview
Comprehensive unit tests have been created for two core modules:
- `src/utils/llm_factory.py` - LLM provider factory
- `src/agents/qa_scoring.py` - QA scoring agent

**Total Tests: 48** | **All Passing ✅**

---

## test_llm_factory.py (17 tests)

### TestGetLLM Class (14 tests)
Tests the `get_llm()` factory function with various configurations.

#### Provider Tests
- `test_get_llm_openai_default` - OpenAI with default model (gpt-4o-mini)
- `test_get_llm_openai_custom_model` - OpenAI with custom model override
- `test_get_llm_openai_with_timeout` - OpenAI with request timeout
- `test_get_llm_openai_fallback_on_typeerror` - OpenAI fallback when timeout param unsupported
- `test_get_llm_gemini_default` - Gemini with default model (gemini-2.0-flash)
- `test_get_llm_gemini_custom_model` - Gemini with custom model override
- `test_get_llm_gemini_with_timeout` - Gemini with request timeout
- `test_get_llm_groq_default` - Groq with default model (llama-3.3-70b-versatile)
- `test_get_llm_groq_custom_model` - Groq with custom model override
- `test_get_llm_groq_fallback` - Groq fallback when timeout param unsupported

#### Error Handling Tests
- `test_get_llm_none_provider` - Raises ValueError when provider is None
- `test_get_llm_unsupported_provider` - Raises ValueError for unknown providers
- `test_get_llm_empty_provider` - Raises ValueError for empty provider string
- `test_get_llm_case_insensitive` - Provider names are case-insensitive
- `test_get_llm_with_whitespace` - Whitespace is stripped from provider names

### TestLLMFactoryIntegration Class (2 tests)
Tests module-level initialization and global LLM instance.

- `test_llm_initialization_success` - Successful initialization with config
- `test_llm_initialization_failure` - Graceful handling of initialization errors

---

## test_qa_scoring.py (31 tests)

### TestFormatTimestamp Class (6 tests)
Tests the `_format_timestamp()` utility function.

- `test_format_timestamp_zero_seconds` - Formats 0 seconds as "00:00"
- `test_format_timestamp_less_than_minute` - Formats seconds < 60
- `test_format_timestamp_exactly_one_minute` - Formats 60 seconds as "01:00"
- `test_format_timestamp_multiple_minutes` - Formats multiple minutes correctly
- `test_format_timestamp_large_values` - Handles large timestamp values
- `test_format_timestamp_float_values` - Handles float input values

### TestGetQAScoringPrompt Class (8 tests)
Tests the `get_qa_scoring_prompt()` function.

- `test_prompt_contains_scoring_philosophy` - Includes scoring philosophy section
- `test_prompt_contains_all_dimensions` - Includes all 5 dimensions
- `test_prompt_contains_dimension_rubrics` - Includes detailed rubrics
- `test_prompt_contains_scoring_scale` - Explains the 1-5 scale
- `test_prompt_contains_justification_guidelines` - Includes feedback guidelines
- `test_prompt_contains_output_instructions` - Includes output format instructions
- `test_prompt_is_string` - Returns a string
- `test_prompt_is_not_empty` - Prompt is non-empty

### TestRunQAScoring Class (13 tests)
Tests the main `run_qa_scoring()` function with various scenarios.

#### Success Cases
- `test_run_qa_scoring_success` - Successfully evaluates a call
- `test_run_qa_scoring_computes_overall_score` - Correctly computes weighted overall score
- `test_run_qa_scoring_all_dimensions_present` - All 5 dimensions returned

#### Input Validation
- `test_run_qa_scoring_missing_transcription` - Handles missing transcription
- `test_run_qa_scoring_transcription_none` - Handles None transcription
- `test_run_qa_scoring_missing_summary` - Handles missing summary
- `test_run_qa_scoring_summary_none` - Handles None summary

#### Format & State Tests
- `test_run_qa_scoring_with_dict_summary` - Handles dict-format summaries
- `test_run_qa_scoring_formats_transcript` - Correctly formats transcript with timestamps
- `test_run_qa_scoring_preserves_state` - Preserves unrelated state fields

#### Retry & Error Handling
- `test_run_qa_scoring_with_retries_success` - Retries on failure and eventually succeeds
- `test_run_qa_scoring_exhausts_retries` - Fails gracefully after max retries
- `test_run_qa_scoring_retry_backoff` - Implements exponential backoff (1s, 2s, 4s...)

### TestDimensionWeights Class (4 tests)
Tests the `DIMENSION_WEIGHTS` configuration.

- `test_weights_sum_to_one` - Weights sum to 1.0
- `test_weights_all_positive` - All weights are positive
- `test_weights_all_dimensions_present` - All 5 dimensions have weights
- `test_weights_reasonable_values` - All weights are between 0-1

---

## Key Testing Strategies

### Mocking
- All external dependencies (LLM clients, config loaders) are mocked
- Enables fast, isolated unit tests without network calls
- Tests focus on logic, not integration

### Edge Cases
- None/missing inputs
- Invalid providers
- Case sensitivity
- Whitespace handling
- Retry exhaustion
- Timeout fallbacks

### Configuration Validation
- Dimension weights integrity
- Prompt structure completeness
- Scoring computation accuracy

---

## Running the Tests

### Run all tests:
```bash
python -m pytest tests/test_llm_factory.py tests/test_qa_scoring.py -v
```

### Run specific test file:
```bash
python -m pytest tests/test_llm_factory.py -v
python -m pytest tests/test_qa_scoring.py -v
```

### Run specific test class:
```bash
python -m pytest tests/test_llm_factory.py::TestGetLLM -v
python -m pytest tests/test_qa_scoring.py::TestRunQAScoring -v
```

### Run with coverage report:
```bash
python -m pytest tests/test_llm_factory.py tests/test_qa_scoring.py --cov=src.utils.llm_factory --cov=src.agents.qa_scoring
```

---

## Dependencies

### Test Framework
- `unittest` - Standard Python testing framework
- `unittest.mock` - Mocking and patching utilities
- `pytest` - Test runner and discovery

### Mocked Dependencies
- `langchain_openai.ChatOpenAI`
- `langchain_google_genai.ChatGoogleGenerativeAI`
- `langchain_groq.ChatGroq`
- `src.utils.config` functions
- `src.utils.llm_factory.llm` instance

---

## Notes for Maintenance

1. **Import Mocking**: Tests mock langchain imports at module level to avoid dependency resolution issues. This is done before importing the modules under test.

2. **LLM Mocking**: All LLM client calls are mocked with `MagicMock` instances. This ensures tests run instantly without actual API calls.

3. **State Management**: Tests use MagicMock for complex objects (transcription segments, summaries) to simplify setup and focus on logic.

4. **Weighted Scoring**: The overall score computation is verified numerically to ensure the weighted averaging formula is correct.

5. **Retry Logic**: Exponential backoff is tested to ensure resilience in case of transient LLM failures.

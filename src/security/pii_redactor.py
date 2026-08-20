import re
from typing import Dict, List

PII_PATTERNS = {
    'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
    'CREDIT_CARD': r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'PHONE': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
}


def redact_pii(text: str, replacement: str = '[REDACTED]') -> str:
    """
    Redact PII from text by finding all matches, deduplicating overlaps,
    and applying replacements right-to-left to avoid index shifts.
    """
    # Collect all matches from the original text using re.finditer()
    matches = []
    for pattern_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            matches.append((match.start(), match.end(), pattern_type))

    if not matches:
        return text

    # Sort by start position ascending for deduplication
    matches.sort(key=lambda x: x[0])

    # Deduplicate overlapping matches (keep the one starting earlier)
    deduplicated = []
    for start, end, pattern_type in matches:
        # Check if this match overlaps with the last kept match
        if deduplicated and start < deduplicated[-1][1]:
            # Overlap detected: skip this match, keep the earlier one already in deduplicated
            continue
        deduplicated.append((start, end, pattern_type))

    # Sort descending by start position for right-to-left replacement
    # This ensures earlier positions remain valid as we modify the string
    deduplicated.sort(key=lambda x: x[0], reverse=True)

    # Apply replacements right-to-left
    result = text
    for start, end, _ in deduplicated:
        result = result[:start] + replacement + result[end:]

    return result


def get_pii_matches(text: str) -> List[Dict]:
    """
    Find all PII matches in text and return their details.
    Returns list of dicts with 'type', 'value', 'start', 'end'.
    """
    matches = []
    for pattern_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            matches.append({
                'type': pattern_type,
                'value': match.group(),
                'start': match.start(),
                'end': match.end(),
            })

    # Sort by position for consistent ordering
    matches.sort(key=lambda x: x['start'])
    return matches

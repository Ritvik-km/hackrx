import re
from typing import List

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "without", "is", "are", "be",
    "shall", "will", "this", "that", "to", "of", "in", "on", "by", "policy"
}


def _extract_keywords(text: str) -> List[str]:
    """Extract potential keywords from the source text.

    Keywords are sequences of capitalised words or acronyms. The
    order is preserved and duplicates are removed. Longer phrases are
    returned before shorter ones to ensure proper substitution.
    """
    pattern = r"(?:(?:[A-Z][a-zA-Z0-9-]*|[A-Z]{2,})(?:\s+(?:[A-Z][a-zA-Z0-9-]*|[A-Z]{2,}))*)"
    found = re.findall(pattern, text)

    seen = set()
    keywords = []
    for kw in found:
        if kw.lower() in STOPWORDS:
            continue
        if kw not in seen:
            seen.add(kw)
            keywords.append(kw)

    keywords.sort(key=len, reverse=True)
    return keywords


def correct_with_keywords(answer: str, source_text: str) -> str:
    """Ensure important keywords from ``source_text`` appear in ``answer``.

    The function searches for capitalised keywords in ``source_text`` and
    replaces case-insensitive occurrences in ``answer`` with the exact
    form from ``source_text``. This helps preserve critical terminology
    such as policy clause names or defined terms.
    """
    if not answer or not source_text:
        return answer

    corrected = answer
    for keyword in _extract_keywords(source_text):
        regex = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
        corrected = regex.sub(keyword, corrected)

    return corrected
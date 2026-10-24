"""Output guardrails that run after citation validation.

A verified quote shows that the cited passage exists; it does not show that the
sentence around the marker says what the passage says. The number check closes
the commonest gap: a figure in the answer must occur in a passage the same
sentence cites. It is deliberately narrow (digits only, no paraphrase check), so
a flag is almost always a real mismatch and can be shown to the reader.
"""

import re

MARKER_RE = re.compile(r"\[(S\d+)\]")
# 12, 3.5, .003, 1,000, 87%, 25M -- not 3D or the digits of an [S2] marker.
NUMBER_RE = re.compile(r"(?<![\w.])(?:\d+(?:[.,]\d+)*|\.\d+)[%kKmMbB]?(?![\w])")


def numbers_in(text):
    """Comparable number tokens; single digits count only as decimals or percentages."""
    found = set()
    for token in NUMBER_RE.findall(MARKER_RE.sub(" ", text)):
        token = token.replace(",", "")
        plain = token.rstrip("%kKmMbB")
        if plain.startswith("."):
            plain = "0" + plain
        if plain.isdigit() and len(plain) == 1 and not token.endswith("%"):
            continue
        found.add(plain)
    return found


def unsupported_numbers(sentences, valid_sources, passages):
    """Numbers in a sentence that occur in none of the passages it cites.

    ``valid_sources``: source IDs with a verified quote. ``passages``: source_id -> passage text.
    A sentence with no verified marker is checked against all verified passages.
    """
    flagged = []
    for sentence in sentences:
        cited = set(MARKER_RE.findall(sentence)) & valid_sources or valid_sources
        support = set().union(*(numbers_in(passages[source]) for source in cited if source in passages)) if cited else set()
        missing = sorted(numbers_in(sentence) - support)
        if missing:
            flagged.append({"sentence": sentence, "numbers": missing})
    return flagged

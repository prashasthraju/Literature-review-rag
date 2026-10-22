"""Answer contract, prompt construction, parsing and citation validation.

The generator only ever sees the assembled context; its output is parsed into
the answer contract and every citation is checked against the sources of that
same request, so an unknown source ID or an invented quote can never be
returned as a valid citation.
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from src.evaluation.evidence import normalize

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "prompts"
DEFAULT_PROMPT = "answer_v2"
MARKER_RE = re.compile(r"\[(S\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
MAX_QUOTE_WORDS = 40
# Quotes may join fragments with an ellipsis or quote several table rows.
FRAGMENT_RE = re.compile(r"\s*(?:\.\.\.|…|\n)\s*")
WORD_RE = re.compile(r"[a-z0-9]+")
# A quote that drops or changes a few words (e.g. citation brackets) still
# identifies its passage; below this token similarity it does not.
APPROXIMATE_QUOTE_RATIO = 0.85


class Citation(BaseModel):
    source_id: str
    quote: str = ""


class AnswerDraft(BaseModel):
    """What the model must return."""

    status: Literal["answered", "partial", "insufficient_evidence"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ValidatedCitation(BaseModel):
    source_id: str
    quote: str
    valid: bool
    match: Literal["exact", "approximate", "none"] = "none"
    problem: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    section_title: str | None = None
    pages: list[int] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    """What the pipeline returns: the draft plus checked provenance."""

    question: str
    status: Literal["answered", "partial", "insufficient_evidence", "invalid_output"]
    answer: str
    citations: list[ValidatedCitation]
    caveats: list[str]
    problems: list[str]
    citation_coverage: float | None
    prompt_version: str
    model: str
    context: dict
    raw_output: str = ""


def load_aliases(path=ROOT / "configs" / "generation" / "paper_aliases.json"):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def named_documents(question, aliases):
    """document_ids of papers the question names explicitly (e.g. 'AniPortrait')."""
    text = normalize(question)
    return {document_id for document_id, names in aliases.items() if any(normalize(name) in text for name in names)}


def load_prompt(version=DEFAULT_PROMPT):
    return (PROMPT_DIR / f"{version}.md").read_text(encoding="utf-8").strip()


def build_messages(question, bundle, prompt_version=DEFAULT_PROMPT):
    sources = bundle.render() if bundle.passages else "(no sources were retrieved)"
    return [
        {"role": "system", "content": load_prompt(prompt_version)},
        {"role": "user", "content": f"<sources>\n{sources}\n</sources>\n\nQuestion: {question.strip()}"},
    ]


def parse_draft(raw):
    """Parse model output into an AnswerDraft; tolerate code fences and prose around the JSON."""
    match = JSON_RE.search(raw or "")
    if not match:
        raise ValueError("no JSON object in model output")
    try:
        return AnswerDraft.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"model output does not match the answer contract: {error}") from error


def quote_match(quote, source_text):
    """'exact' if every fragment (split at ellipses and line breaks) occurs verbatim
    in the source, 'approximate' if the quote nearly matches a window of the
    source with every number unchanged, else 'none'."""
    source = normalize(source_text)
    fragments = [normalize(part) for part in FRAGMENT_RE.split(quote) if normalize(part)]
    if fragments and all(fragment in source for fragment in fragments):
        return "exact"
    words = WORD_RE.findall(" ".join(FRAGMENT_RE.split(quote.lower())))
    tokens = WORD_RE.findall(source_text.lower())
    if len(words) < 4 or not tokens:
        return "none"
    size = len(words)
    numbers = {word for word in words if any(char.isdigit() for char in word)}
    for start in range(0, max(1, len(tokens) - size + 1)):
        window = tokens[start:start + size + size // 5]
        # A changed number changes the claim, so numbers must match exactly.
        if numbers - set(window):
            continue
        if SequenceMatcher(None, words, window, autojunk=False).ratio() >= APPROXIMATE_QUOTE_RATIO:
            return "approximate"
    return "none"


def answer_sentences(text):
    return [sentence.strip() for sentence in SENTENCE_RE.split(text.strip()) if sentence.strip()]


def validate(question, draft, bundle, prompt_version, model_name, raw_output="", required_documents=None):
    """Check every citation against this request's sources and enforce the contract.

    ``required_documents``: papers the question names; an answer must cite at
    least one of them, which catches a true quote attributed to the wrong paper.
    """
    problems = []
    known = {passage.source_id: passage for passage in bundle.passages}
    citations = []
    for citation in draft.citations:
        passage = known.get(citation.source_id)
        if passage is None:
            citations.append(ValidatedCitation(
                source_id=citation.source_id, quote=citation.quote, valid=False, problem="unknown source id"
            ))
            problems.append(f"citation to unknown source {citation.source_id}")
            continue
        problem = None
        match = quote_match(citation.quote, passage.text) if citation.quote.strip() else "none"
        if not citation.quote.strip():
            problem = "missing quote"
        elif match == "none":
            elsewhere = [
                other.source_id for other in bundle.passages
                if other.source_id != citation.source_id and quote_match(citation.quote, other.text) != "none"
            ]
            problem = f"quote is from {elsewhere[0]}, not {citation.source_id}" if elsewhere else "quote not found in source"
        if problem:
            problems.append(f"{citation.source_id}: {problem}")
        elif len(citation.quote.split()) > MAX_QUOTE_WORDS:
            problems.append(f"{citation.source_id}: quote longer than {MAX_QUOTE_WORDS} words")
        citations.append(ValidatedCitation(
            source_id=citation.source_id,
            quote=citation.quote,
            valid=problem is None,
            match=match,
            problem=problem,
            document_id=passage.document_id,
            document_title=passage.document_title,
            section_title=passage.section_title,
            pages=passage.pages,
            chunk_ids=passage.chunk_ids,
        ))

    markers = set(MARKER_RE.findall(draft.answer))
    for marker in sorted(markers - set(known)):
        problems.append(f"answer cites unknown source {marker}")
    valid_sources = {c.source_id for c in citations if c.valid}
    for marker in sorted((markers & set(known)) - valid_sources):
        problems.append(f"answer cites {marker} without a verified quote")

    status = draft.status
    coverage = None
    if status != "insufficient_evidence":
        sentences = answer_sentences(draft.answer)
        cited = [s for s in sentences if set(MARKER_RE.findall(s)) & valid_sources]
        coverage = len(cited) / len(sentences) if sentences else 0.0
        cited_documents = {c.document_id for c in citations if c.valid}
        if not valid_sources:
            # An answer with no verifiable support is not returned as an answer.
            problems.append("no verified citation; answer withheld")
            status = "insufficient_evidence"
        elif required_documents and not cited_documents & set(required_documents):
            problems.append("no verified citation from the paper the question names; answer withheld")
            status = "insufficient_evidence"
    elif draft.citations:
        problems.append("insufficient_evidence answer should not carry citations")

    answer_text = draft.answer.strip()
    if status == "insufficient_evidence" and (draft.status != "insufficient_evidence" or not answer_text):
        answer_text = "The retrieved sources do not verifiably support an answer to this question."

    return Answer(
        question=question,
        status=status,
        answer=answer_text,
        citations=citations,
        caveats=draft.caveats,
        problems=problems,
        citation_coverage=coverage,
        prompt_version=prompt_version,
        model=model_name,
        context=bundle.log(),
        raw_output=raw_output,
    )


def invalid_output(question, raw, error, bundle, prompt_version, model_name):
    return Answer(
        question=question,
        status="invalid_output",
        answer="The model did not return a valid answer.",
        citations=[],
        caveats=[],
        problems=[str(error), f"raw output: {(raw or '')[:500]}"],
        citation_coverage=None,
        prompt_version=prompt_version,
        model=model_name,
        context=bundle.log(),
        raw_output=raw or "",
    )

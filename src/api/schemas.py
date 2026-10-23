"""Request and response models of the chat API (the client contract)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_CHARS = 2000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list, max_length=50)
    # Adds the validator's problems, the raw model output and the context log.
    debug: bool = False

    @field_validator("message")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("message is empty")
        return value.strip()


class CitationView(BaseModel):
    source_id: str
    quote: str
    match: Literal["exact", "approximate", "none"]
    document_id: str | None
    document_title: str | None
    section_title: str | None
    pages: list[int]


class SourceView(BaseModel):
    """A passage the model was given, labelled as it saw it ([S1], [S2], ...)."""

    source_id: str
    document_id: str
    document_title: str
    section_title: str
    pages: list[int]
    text: str
    cited: bool


class AssistantMessage(BaseModel):
    message_id: str
    conversation_id: str
    request_id: str
    question: str
    standalone_question: str
    paper_ids: list[str]
    status: Literal["answered", "partial", "insufficient_evidence", "invalid_output"]
    answer: str
    citations: list[CitationView]
    sources: list[SourceView]
    caveats: list[str]
    unverified_citations: int
    citation_coverage: float | None
    model: str
    prompt_version: str
    timings_ms: dict[str, float]
    created_at: str
    feedback: int | None = None
    debug: dict | None = None


class MessageView(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    answer: AssistantMessage | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationDetail(ConversationSummary):
    messages: list[MessageView]


class PaperView(BaseModel):
    id: str
    title: str
    topic: str
    chunks: int
    pages: int
    sections: list[str]
    has_pdf: bool


class FeedbackRequest(BaseModel):
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=1000)


class Health(BaseModel):
    status: Literal["loading", "ready", "error"]
    detail: str | None = None
    generator: str
    retrieval_config: str
    prompt_version: str
    papers: int

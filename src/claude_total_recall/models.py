"""Pydantic data models for Claude Total Recall."""

from datetime import datetime

from pydantic import BaseModel, Field


class IndexedMessage(BaseModel):
    """A message indexed for search."""

    uuid: str
    session_id: str
    project_path: str
    timestamp: datetime
    role: str  # "user" or "assistant"
    searchable_text: str
    message_index: int  # Position in session for context retrieval
    is_subagent: bool = False


class SessionInfo(BaseModel):
    """Metadata about a conversation session."""

    session_id: str
    project_path: str
    first_prompt: str | None = None
    message_count: int = 0
    created: datetime | None = None
    modified: datetime | None = None

    @property
    def timestamp_fallback(self) -> datetime:
        """Get a timestamp for sorting, with fallback to epoch."""
        return self.modified or self.created or datetime.min


class MatchedMessage(BaseModel):
    """A message that matched a search query."""

    role: str
    content: str
    timestamp: datetime
    project: str
    session_id: str
    uuid: str


class ContextMessage(BaseModel):
    """A message in the context window around a match."""

    role: str
    content: str
    timestamp: datetime
    is_match: bool = False


class SearchResult(BaseModel):
    """A search result with context."""

    matched_message: MatchedMessage
    score: float
    context: list[ContextMessage] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response from search_conversations tool."""

    results: list[SearchResult]
    query: str
    total_matches: int
    offset: int = 0
    has_more: bool = False
    excluded_sessions: int = 0

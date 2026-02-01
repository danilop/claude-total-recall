"""Claude Total Recall - Search Claude Code conversation history."""

from .indexer import ConversationIndex, get_index
from .models import (
    ContextMessage,
    IndexedMessage,
    MatchedMessage,
    SearchResponse,
    SearchResult,
    SessionInfo,
)
from .query import search_conversations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ConversationIndex",
    "get_index",
    "search_conversations",
    "IndexedMessage",
    "SessionInfo",
    "SearchResult",
    "SearchResponse",
    "MatchedMessage",
    "ContextMessage",
]

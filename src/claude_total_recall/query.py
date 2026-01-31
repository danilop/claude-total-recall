"""Query engine for conversation search."""

from .indexer import get_index
from .models import (
    ContextMessage,
    IndexedMessage,
    MatchedMessage,
    SearchResponse,
    SearchResult,
)

# Configuration defaults
DEFAULT_CONTEXT_BEFORE_AFTER = 3
DEFAULT_THRESHOLD = 0.2
DEFAULT_MAX_RESULTS = 10
DEFAULT_OFFSET = 0


def search_conversations(
    query: str,
    project: str | None = None,
    context_before_after: int = DEFAULT_CONTEXT_BEFORE_AFTER,
    threshold: float = DEFAULT_THRESHOLD,
    max_results: int = DEFAULT_MAX_RESULTS,
    offset: int = DEFAULT_OFFSET,
    include_subagents: bool = True,
) -> SearchResponse:
    """
    Search conversations for messages matching the query.

    Args:
        query: Search string (keywords or sentence)
        project: Optional project path filter
        context_before_after: Number of messages to include before AND after each match
        threshold: Minimum cosine similarity (0-1)
        max_results: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        include_subagents: Include subagent conversations

    Returns:
        SearchResponse with results, query, total matches, and pagination info
    """
    index = get_index()

    # Search for matches (index auto-rebuilds if needed)
    # Get extra results to account for deduplication and pagination
    raw_results = index.search(
        query=query,
        project=project,
        threshold=threshold,
        max_results=(offset + max_results) * 3,
        include_subagents=include_subagents,
    )

    if not raw_results:
        return SearchResponse(results=[], query=query, total_matches=0)

    # Group by session and deduplicate overlapping windows
    deduplicated = _deduplicate_results(raw_results, context_before_after)
    total_after_dedup = len(deduplicated)

    # Apply pagination: skip offset, take max_results
    paginated = deduplicated[offset : offset + max_results]

    # Build search results with context
    results = []
    for msg, score in paginated:
        context_window = index.get_context_window(msg, context_before_after)

        context = []
        for ctx_msg in context_window:
            context.append(
                ContextMessage(
                    role=ctx_msg.role,
                    content=_truncate_content(ctx_msg.searchable_text),
                    timestamp=ctx_msg.timestamp,
                    is_match=(ctx_msg.uuid == msg.uuid),
                )
            )

        results.append(
            SearchResult(
                matched_message=MatchedMessage(
                    role=msg.role,
                    content=_truncate_content(msg.searchable_text),
                    timestamp=msg.timestamp,
                    project=msg.project_path,
                    session_id=msg.session_id,
                    uuid=msg.uuid,
                ),
                score=round(score, 4),
                context=context,
            )
        )

    return SearchResponse(
        results=results,
        query=query,
        total_matches=total_after_dedup,
        offset=offset,
        has_more=offset + len(results) < total_after_dedup,
    )


def _deduplicate_results(
    results: list[tuple[IndexedMessage, float]], context_before_after: int
) -> list[tuple[IndexedMessage, float]]:
    """
    Deduplicate results by merging overlapping context windows.

    When multiple matches are within 2*context_before_after of each other
    in the same session, keep only the highest-scoring match.
    """
    if not results:
        return []

    # Group by session
    by_session: dict[str, list[tuple[IndexedMessage, float]]] = {}
    for msg, score in results:
        if msg.session_id not in by_session:
            by_session[msg.session_id] = []
        by_session[msg.session_id].append((msg, score))

    # Deduplicate within each session
    dedup_distance = 2 * context_before_after
    deduplicated = []

    for _session_id, session_results in by_session.items():
        # Sort by message index
        session_results.sort(key=lambda x: x[0].message_index)

        # Merge overlapping windows
        kept: list[tuple[IndexedMessage, float]] = []
        for msg, score in session_results:
            if not kept:
                kept.append((msg, score))
                continue

            last_msg, last_score = kept[-1]
            distance = msg.message_index - last_msg.message_index

            if distance <= dedup_distance:
                # Overlapping - keep the higher score
                if score > last_score:
                    kept[-1] = (msg, score)
            else:
                kept.append((msg, score))

        deduplicated.extend(kept)

    # Sort by score descending
    deduplicated.sort(key=lambda x: x[1], reverse=True)

    return deduplicated


def _truncate_content(content: str, max_length: int = 2000) -> str:
    """Truncate content to a maximum length."""
    if len(content) <= max_length:
        return content
    return content[: max_length - 3] + "..."

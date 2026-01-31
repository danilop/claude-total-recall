"""FastMCP server for Claude Total Recall."""

import os

from mcp.server.fastmcp import FastMCP

from .query import (
    DEFAULT_CONTEXT_BEFORE_AFTER,
    DEFAULT_MAX_RESULTS,
    DEFAULT_OFFSET,
    DEFAULT_THRESHOLD,
    search_conversations,
)

# Create the MCP server
mcp = FastMCP("claude-total-recall")


def _get_current_project() -> str | None:
    """
    Get the current project path from environment.

    Claude Code sets CLAUDE_PROJECT_DIR when running plugins.
    Falls back to PWD/CWD if not available.
    """
    project = os.environ.get("CLAUDE_PROJECT_DIR")
    if project:
        return project
    return os.environ.get("PWD") or os.getcwd()


def _search(
    query: str,
    project: str | None,
    context_before_after: int,
    threshold: float,
    max_results: int,
    offset: int,
    include_subagents: bool,
) -> dict:
    """Common search implementation for both tools."""
    result = search_conversations(
        query=query,
        project=project,
        context_before_after=context_before_after,
        threshold=threshold,
        max_results=max_results,
        offset=offset,
        include_subagents=include_subagents,
    )
    return result.model_dump(mode="json")


@mcp.tool()
def search_project_history(
    query: str,
    context_before_after: int = DEFAULT_CONTEXT_BEFORE_AFTER,
    threshold: float = DEFAULT_THRESHOLD,
    max_results: int = DEFAULT_MAX_RESULTS,
    offset: int = DEFAULT_OFFSET,
    include_subagents: bool = True,
) -> dict:
    """
    Search conversation history for the CURRENT PROJECT only.

    Use this to find project-specific context: past decisions, implementation details,
    bugs discussed, architecture choices, and previous work done on this codebase.

    This searches previous Claude Code sessions for the project you're currently working on.

    Args:
        query: Keywords or sentence describing what to find
        context_before_after: Number of messages to include before AND after each match (default: 3)
        threshold: Minimum similarity score 0-1 (default: 0.2)
        max_results: Maximum number of results to return (default: 10)
        offset: Number of results to skip for pagination (default: 0)
        include_subagents: Include subagent conversations (default: true)

    Returns:
        Search results with matched messages, scores, session_id, surrounding context,
        and pagination info (total_matches, offset, has_more)
    """
    return _search(
        query=query,
        project=_get_current_project(),
        context_before_after=context_before_after,
        threshold=threshold,
        max_results=max_results,
        offset=offset,
        include_subagents=include_subagents,
    )


@mcp.tool()
def search_global_history(
    query: str,
    context_before_after: int = DEFAULT_CONTEXT_BEFORE_AFTER,
    threshold: float = DEFAULT_THRESHOLD,
    max_results: int = DEFAULT_MAX_RESULTS,
    offset: int = DEFAULT_OFFSET,
    include_subagents: bool = True,
) -> dict:
    """
    Search conversation history across ALL PROJECTS.

    Use this to find cross-project knowledge: user preferences, coding patterns,
    common solutions, global conventions, and insights from all previous work.

    This searches ALL previous Claude Code sessions regardless of project.

    Args:
        query: Keywords or sentence describing what to find
        context_before_after: Number of messages to include before AND after each match (default: 3)
        threshold: Minimum similarity score 0-1 (default: 0.2)
        max_results: Maximum number of results to return (default: 10)
        offset: Number of results to skip for pagination (default: 0)
        include_subagents: Include subagent conversations (default: true)

    Returns:
        Search results with matched messages, scores, project, session_id, surrounding context,
        and pagination info (total_matches, offset, has_more)
    """
    return _search(
        query=query,
        project=None,
        context_before_after=context_before_after,
        threshold=threshold,
        max_results=max_results,
        offset=offset,
        include_subagents=include_subagents,
    )


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

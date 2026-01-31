"""Load conversations from ~/.claude directory."""

import json
from datetime import datetime
from pathlib import Path

from .models import IndexedMessage, SessionInfo


def get_claude_dir() -> Path:
    """Get the Claude configuration directory."""
    return Path.home() / ".claude"


def get_projects_dir() -> Path:
    """Get the projects directory containing conversation history."""
    return get_claude_dir() / "projects"


def list_all_projects() -> list[str]:
    """List all projects with conversation history."""
    projects_dir = get_projects_dir()
    if not projects_dir.exists():
        return []

    projects = []
    for entry in projects_dir.iterdir():
        if entry.is_dir():
            # Decode escaped path (dashes become slashes)
            project_path = "/" + entry.name.replace("-", "/")
            # Remove double slashes that might occur
            while "//" in project_path:
                project_path = project_path.replace("//", "/")
            projects.append(project_path)
    return sorted(projects)


def _escape_project_path(project_path: str) -> str:
    """Convert a project path to its escaped directory name."""
    # Replace slashes with dashes (keeps leading dash from leading slash)
    return project_path.replace("/", "-")


def _get_project_dir(project_path: str) -> Path:
    """Get the directory for a specific project."""
    escaped = _escape_project_path(project_path)
    return get_projects_dir() / escaped


def load_sessions_index(project_path: str) -> list[SessionInfo]:
    """Load session metadata for a project."""
    project_dir = _get_project_dir(project_path)
    index_file = project_dir / "sessions-index.json"

    if not index_file.exists():
        return []

    try:
        with open(index_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    # Handle both old (list) and new (dict with entries) formats
    entries = data if isinstance(data, list) else data.get("entries", [])

    sessions = []
    for session in entries:
        sessions.append(
            SessionInfo(
                session_id=session.get("sessionId", ""),
                project_path=project_path,
                first_prompt=session.get("firstPrompt"),
                message_count=session.get("messageCount", 0),
                created=_parse_timestamp(session.get("created")),
                modified=_parse_timestamp(session.get("modified")),
            )
        )
    return sessions


def _parse_timestamp(ts: str | int | None) -> datetime | None:
    """Parse a timestamp from various formats."""
    if ts is None:
        return None
    if isinstance(ts, int):
        # Unix timestamp in milliseconds
        return datetime.fromtimestamp(ts / 1000)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _extract_searchable_text(message: dict) -> str:
    """Extract searchable text from a message."""
    content = message.get("message", {}).get("content")
    if content is None:
        return ""

    # User messages have string content
    if isinstance(content, str):
        return content

    # Assistant messages have array of content blocks
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)

    return ""


def load_session_messages(
    project_path: str, session_id: str, include_subagents: bool = True
) -> list[IndexedMessage]:
    """Load all messages from a session."""
    project_dir = _get_project_dir(project_path)
    session_file = project_dir / f"{session_id}.jsonl"

    messages = []

    # Load main session messages
    if session_file.exists():
        messages.extend(_load_jsonl_messages(session_file, project_path, session_id, False))

    # Load subagent messages if requested
    if include_subagents:
        subagents_dir = project_dir / session_id / "subagents"
        if subagents_dir.exists():
            for subagent_file in subagents_dir.glob("*.jsonl"):
                subagent_id = subagent_file.stem
                sub_messages = _load_jsonl_messages(
                    subagent_file, project_path, f"{session_id}/{subagent_id}", True
                )
                messages.extend(sub_messages)

    # Sort by timestamp and assign message indices
    messages.sort(key=lambda m: m.timestamp)
    for i, msg in enumerate(messages):
        msg.message_index = i

    return messages


def _load_jsonl_messages(
    filepath: Path, project_path: str, session_id: str, is_subagent: bool
) -> list[IndexedMessage]:
    """Load messages from a JSONL file."""
    messages = []

    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip non-message types
                msg_type = data.get("type")
                if msg_type not in ("user", "assistant"):
                    continue

                role = data.get("message", {}).get("role", msg_type)
                searchable_text = _extract_searchable_text(data)

                # Skip empty messages
                if not searchable_text.strip():
                    continue

                timestamp = _parse_timestamp(data.get("timestamp"))
                if timestamp is None:
                    timestamp = datetime.now()

                messages.append(
                    IndexedMessage(
                        uuid=data.get("uuid", ""),
                        session_id=session_id,
                        project_path=project_path,
                        timestamp=timestamp,
                        role=role,
                        searchable_text=searchable_text,
                        message_index=0,  # Will be set after sorting
                        is_subagent=is_subagent,
                    )
                )
    except OSError:
        pass

    return messages


def load_all_messages(include_subagents: bool = True) -> list[IndexedMessage]:
    """Load all messages from all projects."""
    all_messages = []

    for project_path in list_all_projects():
        sessions = load_sessions_index(project_path)
        for session in sessions:
            messages = load_session_messages(project_path, session.session_id, include_subagents)
            all_messages.extend(messages)

    return all_messages


def list_all_sessions() -> list[SessionInfo]:
    """List all sessions across all projects."""
    all_sessions = []

    for project_path in list_all_projects():
        sessions = load_sessions_index(project_path)
        all_sessions.extend(sessions)

    return sorted(all_sessions, key=lambda s: s.modified or datetime.min, reverse=True)

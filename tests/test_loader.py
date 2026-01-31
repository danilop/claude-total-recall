"""Tests for conversation loader."""

import json
from datetime import datetime
from unittest.mock import patch

from claude_total_recall.loader import (
    _escape_project_path,
    _extract_searchable_text,
    _parse_timestamp,
    list_all_projects,
    load_all_messages,
    load_session_messages,
    load_sessions_index,
)


class TestParseTimestamp:
    """Tests for timestamp parsing."""

    def test_parse_milliseconds(self):
        """Test parsing Unix timestamp in milliseconds."""
        ts = _parse_timestamp(1706789012345)
        assert ts is not None
        assert isinstance(ts, datetime)

    def test_parse_iso_string(self):
        """Test parsing ISO format string."""
        ts = _parse_timestamp("2024-01-15T10:30:00Z")
        assert ts is not None
        assert isinstance(ts, datetime)

    def test_parse_iso_with_timezone(self):
        """Test parsing ISO format with timezone."""
        ts = _parse_timestamp("2024-01-15T10:30:00+00:00")
        assert ts is not None

    def test_parse_none(self):
        """Test parsing None returns None."""
        ts = _parse_timestamp(None)
        assert ts is None

    def test_parse_invalid_string(self):
        """Test parsing invalid string returns None."""
        ts = _parse_timestamp("not a timestamp")
        assert ts is None


class TestEscapeProjectPath:
    """Tests for project path escaping."""

    def test_escape_simple_path(self):
        """Test escaping a simple path."""
        escaped = _escape_project_path("/Users/test/project")
        assert escaped == "-Users-test-project"

    def test_escape_nested_path(self):
        """Test escaping a deeply nested path."""
        escaped = _escape_project_path("/home/user/code/my-app/src")
        assert escaped == "-home-user-code-my-app-src"

    def test_escape_root(self):
        """Test escaping root path."""
        escaped = _escape_project_path("/")
        assert escaped == "-"


class TestExtractSearchableText:
    """Tests for extracting searchable text from messages."""

    def test_extract_string_content(self):
        """Test extracting from string content (user messages)."""
        message = {"message": {"content": "Hello world"}}
        text = _extract_searchable_text(message)
        assert text == "Hello world"

    def test_extract_array_content(self):
        """Test extracting from array content (assistant messages)."""
        message = {
            "message": {
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ]
            }
        }
        text = _extract_searchable_text(message)
        assert "First part" in text
        assert "Second part" in text

    def test_extract_mixed_content(self):
        """Test extracting from mixed content types."""
        message = {
            "message": {
                "content": [
                    {"type": "text", "text": "Text block"},
                    {"type": "tool_use", "name": "read_file"},  # Should be ignored
                    {"text": "Another text"},
                ]
            }
        }
        text = _extract_searchable_text(message)
        assert "Text block" in text
        assert "Another text" in text

    def test_extract_empty_message(self):
        """Test extracting from empty message."""
        message = {"message": {}}
        text = _extract_searchable_text(message)
        assert text == ""

    def test_extract_none_content(self):
        """Test extracting when content is None."""
        message = {"message": {"content": None}}
        text = _extract_searchable_text(message)
        assert text == ""


class TestLoadSessionsIndex:
    """Tests for loading sessions index."""

    def test_load_sessions_index(self, temp_claude_dir, sample_project):
        """Test loading sessions from index file."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            sessions = load_sessions_index("/Users/test/myproject")
            assert len(sessions) == 2
            assert sessions[0].session_id == "session-001"
            assert sessions[0].first_prompt == "Help me fix a bug"

    def test_load_sessions_index_missing_file(self, temp_claude_dir):
        """Test loading from non-existent index returns empty list."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            sessions = load_sessions_index("/Users/nonexistent/project")
            assert sessions == []

    def test_load_sessions_index_old_format(self, temp_claude_dir):
        """Test loading from old list format."""
        project_dir = temp_claude_dir / "projects" / "-Users-old-project"
        project_dir.mkdir(parents=True)

        # Old format: just a list
        old_format = [{"sessionId": "old-session", "messageCount": 5}]
        with open(project_dir / "sessions-index.json", "w") as f:
            json.dump(old_format, f)

        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            sessions = load_sessions_index("/Users/old/project")
            assert len(sessions) == 1
            assert sessions[0].session_id == "old-session"


class TestLoadSessionMessages:
    """Tests for loading session messages."""

    def test_load_session_messages(self, temp_claude_dir, sample_project):
        """Test loading messages from a session."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            messages = load_session_messages("/Users/test/myproject", "session-001")
            assert len(messages) == 4
            assert messages[0].role == "user"
            assert "bug" in messages[0].searchable_text.lower()

    def test_load_session_messages_ordering(self, temp_claude_dir, sample_project):
        """Test messages are ordered by timestamp."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            messages = load_session_messages("/Users/test/myproject", "session-001")
            for i in range(len(messages) - 1):
                assert messages[i].timestamp <= messages[i + 1].timestamp

    def test_load_session_messages_indices(self, temp_claude_dir, sample_project):
        """Test message indices are assigned correctly."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            messages = load_session_messages("/Users/test/myproject", "session-001")
            for i, msg in enumerate(messages):
                assert msg.message_index == i

    def test_load_session_with_subagents(self, temp_claude_dir, sample_project_with_subagents):
        """Test loading messages including subagents."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            messages = load_session_messages(
                "/Users/test/myproject", "session-001", include_subagents=True
            )
            subagent_msgs = [m for m in messages if m.is_subagent]
            assert len(subagent_msgs) == 2

    def test_load_session_without_subagents(self, temp_claude_dir, sample_project_with_subagents):
        """Test loading messages excluding subagents."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            messages = load_session_messages(
                "/Users/test/myproject", "session-001", include_subagents=False
            )
            subagent_msgs = [m for m in messages if m.is_subagent]
            assert len(subagent_msgs) == 0

    def test_load_missing_session(self, temp_claude_dir, sample_project):
        """Test loading non-existent session returns empty list."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            messages = load_session_messages("/Users/test/myproject", "nonexistent-session")
            assert messages == []


class TestListAllProjects:
    """Tests for listing all projects."""

    def test_list_all_projects(self, temp_claude_dir, sample_project):
        """Test listing all projects."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            projects = list_all_projects()
            assert len(projects) == 1
            assert "/Users/test/myproject" in projects

    def test_list_projects_empty(self, temp_claude_dir):
        """Test listing projects when none exist."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            projects = list_all_projects()
            assert projects == []

    def test_list_projects_sorted(self, temp_claude_dir):
        """Test projects are sorted alphabetically."""
        projects_dir = temp_claude_dir / "projects"
        (projects_dir / "-Users-z-project").mkdir(parents=True)
        (projects_dir / "-Users-a-project").mkdir(parents=True)

        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            projects = list_all_projects()
            assert projects[0] < projects[1]


class TestLoadAllMessages:
    """Tests for loading all messages."""

    def test_load_all_messages(self, temp_claude_dir, sample_project):
        """Test loading all messages from all projects."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            messages = load_all_messages()
            # 4 from session-001 + 2 from session-002 = 6
            assert len(messages) == 6

    def test_load_all_messages_with_subagents(self, temp_claude_dir, sample_project_with_subagents):
        """Test loading all messages including subagents."""
        with patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir):
            messages = load_all_messages(include_subagents=True)
            # 4 + 2 subagent from session-001 + 2 from session-002 = 8
            assert len(messages) == 8

"""Pytest fixtures for Claude Total Recall tests."""

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_claude_dir(tmp_path):
    """Create a temporary ~/.claude directory structure."""
    claude_dir = tmp_path / ".claude"
    projects_dir = claude_dir / "projects"
    projects_dir.mkdir(parents=True)
    return claude_dir


@pytest.fixture
def sample_project(temp_claude_dir):
    """Create a sample project with sessions."""
    project_name = "-Users-test-myproject"
    project_dir = temp_claude_dir / "projects" / project_name
    project_dir.mkdir(parents=True)

    # Create sessions-index.json
    sessions_index = {
        "entries": [
            {
                "sessionId": "session-001",
                "firstPrompt": "Help me fix a bug",
                "messageCount": 4,
                "created": 1706789000000,
                "modified": 1706789100000,
            },
            {
                "sessionId": "session-002",
                "firstPrompt": "Add authentication",
                "messageCount": 2,
                "created": 1706790000000,
                "modified": 1706790100000,
            },
        ]
    }
    with open(project_dir / "sessions-index.json", "w") as f:
        json.dump(sessions_index, f)

    # Create session-001.jsonl
    messages_001 = [
        {
            "type": "user",
            "uuid": "msg-001",
            "timestamp": 1706789010000,
            "message": {
                "role": "user",
                "content": "Help me fix a bug in the authentication module",
            },
        },
        {
            "type": "assistant",
            "uuid": "msg-002",
            "timestamp": 1706789020000,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'll help you fix the auth bug. Let me check the code.",
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "msg-003",
            "timestamp": 1706789030000,
            "message": {"role": "user", "content": "The login function returns None"},
        },
        {
            "type": "assistant",
            "uuid": "msg-004",
            "timestamp": 1706789040000,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The issue is in the validate_token function. It's not returning the user object.",
                    }
                ],
            },
        },
    ]
    with open(project_dir / "session-001.jsonl", "w") as f:
        for msg in messages_001:
            f.write(json.dumps(msg) + "\n")

    # Create session-002.jsonl
    messages_002 = [
        {
            "type": "user",
            "uuid": "msg-005",
            "timestamp": 1706790010000,
            "message": {"role": "user", "content": "Add JWT authentication to the API"},
        },
        {
            "type": "assistant",
            "uuid": "msg-006",
            "timestamp": 1706790020000,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'll implement JWT authentication. First, let's install the required packages.",
                    }
                ],
            },
        },
    ]
    with open(project_dir / "session-002.jsonl", "w") as f:
        for msg in messages_002:
            f.write(json.dumps(msg) + "\n")

    return project_dir


@pytest.fixture
def sample_project_with_subagents(sample_project):
    """Add subagent messages to the sample project."""
    subagents_dir = sample_project / "session-001" / "subagents"
    subagents_dir.mkdir(parents=True)

    # Create subagent messages
    subagent_messages = [
        {
            "type": "user",
            "uuid": "sub-msg-001",
            "timestamp": 1706789025000,
            "message": {"role": "user", "content": "Search for authentication patterns"},
        },
        {
            "type": "assistant",
            "uuid": "sub-msg-002",
            "timestamp": 1706789026000,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Found OAuth2 implementation in auth/oauth.py"}
                ],
            },
        },
    ]
    with open(subagents_dir / "agent-explore.jsonl", "w") as f:
        for msg in subagent_messages:
            f.write(json.dumps(msg) + "\n")

    return sample_project


@pytest.fixture
def mock_claude_dir(temp_claude_dir, sample_project):
    """Patch the claude directory to use temp directory."""
    with (
        patch("claude_total_recall.loader.get_claude_dir", return_value=temp_claude_dir),
        patch(
            "claude_total_recall.indexer.get_projects_dir",
            return_value=temp_claude_dir / "projects",
        ),
    ):
        yield temp_claude_dir


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / ".cache" / "claude-total-recall"
    cache_dir.mkdir(parents=True)
    return cache_dir

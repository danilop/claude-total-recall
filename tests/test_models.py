"""Tests for Pydantic models."""

from datetime import datetime

from claude_total_recall.models import (
    ContextMessage,
    IndexedMessage,
    MatchedMessage,
    SearchResponse,
    SearchResult,
    SessionInfo,
)


class TestIndexedMessage:
    """Tests for IndexedMessage model."""

    def test_create_basic_message(self):
        """Test creating a basic indexed message."""
        msg = IndexedMessage(
            uuid="test-uuid",
            session_id="session-001",
            project_path="/Users/test/project",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            role="user",
            searchable_text="Hello world",
            message_index=0,
        )
        assert msg.uuid == "test-uuid"
        assert msg.role == "user"
        assert msg.is_subagent is False

    def test_create_subagent_message(self):
        """Test creating a subagent message."""
        msg = IndexedMessage(
            uuid="test-uuid",
            session_id="session-001/agent-explore",
            project_path="/Users/test/project",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            role="assistant",
            searchable_text="Found the file",
            message_index=1,
            is_subagent=True,
        )
        assert msg.is_subagent is True

    def test_message_serialization(self):
        """Test message can be serialized to dict."""
        msg = IndexedMessage(
            uuid="test-uuid",
            session_id="session-001",
            project_path="/Users/test/project",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            role="user",
            searchable_text="Test content",
            message_index=0,
        )
        data = msg.model_dump()
        assert data["uuid"] == "test-uuid"
        assert data["searchable_text"] == "Test content"


class TestSessionInfo:
    """Tests for SessionInfo model."""

    def test_create_session_info(self):
        """Test creating session info."""
        session = SessionInfo(
            session_id="session-001",
            project_path="/Users/test/project",
            first_prompt="Help me fix a bug",
            message_count=10,
            created=datetime(2024, 1, 1, 10, 0, 0),
            modified=datetime(2024, 1, 1, 12, 0, 0),
        )
        assert session.session_id == "session-001"
        assert session.message_count == 10

    def test_session_info_optional_fields(self):
        """Test session info with optional fields."""
        session = SessionInfo(
            session_id="session-001",
            project_path="/Users/test/project",
        )
        assert session.first_prompt is None
        assert session.created is None
        assert session.modified is None
        assert session.message_count == 0


class TestMatchedMessage:
    """Tests for MatchedMessage model."""

    def test_create_matched_message(self):
        """Test creating a matched message."""
        msg = MatchedMessage(
            role="assistant",
            content="This is the answer",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            project="/Users/test/project",
            session_id="session-001",
            uuid="msg-001",
        )
        assert msg.role == "assistant"
        assert msg.content == "This is the answer"


class TestContextMessage:
    """Tests for ContextMessage model."""

    def test_create_context_message(self):
        """Test creating a context message."""
        msg = ContextMessage(
            role="user",
            content="What is this?",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            is_match=False,
        )
        assert msg.is_match is False

    def test_matched_context_message(self):
        """Test creating a matched context message."""
        msg = ContextMessage(
            role="assistant",
            content="This is the answer",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            is_match=True,
        )
        assert msg.is_match is True


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_search_result(self):
        """Test creating a search result."""
        matched = MatchedMessage(
            role="assistant",
            content="Found it",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            project="/Users/test/project",
            session_id="session-001",
            uuid="msg-001",
        )
        result = SearchResult(
            matched_message=matched,
            score=0.85,
            context=[],
        )
        assert result.score == 0.85
        assert result.matched_message.content == "Found it"

    def test_search_result_with_context(self):
        """Test search result with context messages."""
        matched = MatchedMessage(
            role="assistant",
            content="Answer",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            project="/Users/test/project",
            session_id="session-001",
            uuid="msg-002",
        )
        context = [
            ContextMessage(
                role="user",
                content="Question",
                timestamp=datetime(2024, 1, 1, 11, 59, 0),
                is_match=False,
            ),
            ContextMessage(
                role="assistant",
                content="Answer",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                is_match=True,
            ),
        ]
        result = SearchResult(
            matched_message=matched,
            score=0.9,
            context=context,
        )
        assert len(result.context) == 2
        assert result.context[1].is_match is True


class TestSearchResponse:
    """Tests for SearchResponse model."""

    def test_empty_response(self):
        """Test creating an empty search response."""
        response = SearchResponse(
            results=[],
            query="test query",
            total_matches=0,
        )
        assert len(response.results) == 0
        assert response.has_more is False

    def test_response_with_pagination(self):
        """Test search response with pagination info."""
        response = SearchResponse(
            results=[],
            query="test query",
            total_matches=25,
            offset=10,
            has_more=True,
        )
        assert response.offset == 10
        assert response.has_more is True
        assert response.total_matches == 25

    def test_response_serialization(self):
        """Test response can be serialized to JSON."""
        matched = MatchedMessage(
            role="assistant",
            content="Test",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            project="/test",
            session_id="s1",
            uuid="m1",
        )
        result = SearchResult(matched_message=matched, score=0.8, context=[])
        response = SearchResponse(
            results=[result],
            query="test",
            total_matches=1,
        )
        data = response.model_dump(mode="json")
        assert data["query"] == "test"
        assert len(data["results"]) == 1

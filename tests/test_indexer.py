"""Tests for conversation indexer."""

import pickle
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claude_total_recall.indexer import (
    BYTES_PER_MESSAGE,
    ConversationIndex,
    _get_sessions_fingerprint,
    get_index,
    get_memory_limit,
    get_physical_memory,
    select_sessions_within_limit,
)
from claude_total_recall.models import IndexedMessage, SessionInfo


class TestGetSessionsFingerprint:
    """Tests for session fingerprinting."""

    def test_empty_projects_dir(self, temp_claude_dir):
        """Test fingerprint for empty projects directory (no sessions-index files)."""
        with patch(
            "claude_total_recall.indexer.get_projects_dir",
            return_value=temp_claude_dir / "projects",
        ):
            fingerprint = _get_sessions_fingerprint()
            # Empty fingerprint_parts list produces MD5 of empty string
            assert len(fingerprint) == 32  # MD5 hash length

    def test_nonexistent_projects_dir(self, tmp_path):
        """Test fingerprint for non-existent directory."""
        with patch(
            "claude_total_recall.indexer.get_projects_dir", return_value=tmp_path / "nonexistent"
        ):
            fingerprint = _get_sessions_fingerprint()
            assert fingerprint == ""

    def test_fingerprint_changes_with_mtime(self, temp_claude_dir, sample_project):
        """Test fingerprint changes when file is modified."""
        import time

        with patch(
            "claude_total_recall.indexer.get_projects_dir",
            return_value=temp_claude_dir / "projects",
        ):
            fp1 = _get_sessions_fingerprint()

            # Touch the index file to change mtime
            time.sleep(0.1)
            index_file = sample_project / "sessions-index.json"
            index_file.touch()

            fp2 = _get_sessions_fingerprint()
            assert fp1 != fp2


class TestConversationIndex:
    """Tests for ConversationIndex class."""

    def test_init(self):
        """Test index initialization."""
        index = ConversationIndex()
        assert index._model is None
        assert index._messages == []
        assert index._embeddings is None

    def test_compute_text_hash(self):
        """Test text hashing is consistent."""
        index = ConversationIndex()
        hash1 = index._compute_text_hash("Hello world")
        hash2 = index._compute_text_hash("Hello world")
        hash3 = index._compute_text_hash("Different text")
        assert hash1 == hash2
        assert hash1 != hash3

    def test_load_cache_missing_file(self, tmp_path):
        """Test loading cache when file doesn't exist."""
        index = ConversationIndex()
        with patch("claude_total_recall.indexer.CACHE_FILE", tmp_path / "nonexistent.pkl"):
            cache = index._load_cache()
            assert cache == {}

    def test_load_cache_valid_file(self, tmp_path):
        """Test loading valid cache file."""
        cache_file = tmp_path / "cache.pkl"
        expected = {"hash1": np.array([1.0, 2.0, 3.0])}
        with open(cache_file, "wb") as f:
            pickle.dump(expected, f)

        index = ConversationIndex()
        with patch("claude_total_recall.indexer.CACHE_FILE", cache_file):
            cache = index._load_cache()
            assert "hash1" in cache
            np.testing.assert_array_equal(cache["hash1"], expected["hash1"])

    def test_load_cache_corrupted_file(self, tmp_path):
        """Test loading corrupted cache file returns empty dict."""
        cache_file = tmp_path / "cache.pkl"
        with open(cache_file, "wb") as f:
            f.write(b"not valid pickle data")

        index = ConversationIndex()
        with patch("claude_total_recall.indexer.CACHE_FILE", cache_file):
            cache = index._load_cache()
            assert cache == {}

    def test_save_and_load_cache(self, tmp_path):
        """Test saving and loading cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "embeddings.pkl"
        lock_file = cache_dir / "embeddings.lock"

        index = ConversationIndex()
        embeddings = {"hash1": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

        with (
            patch("claude_total_recall.indexer.CACHE_DIR", cache_dir),
            patch("claude_total_recall.indexer.CACHE_FILE", cache_file),
            patch("claude_total_recall.indexer.LOCK_FILE", lock_file),
        ):
            index._save_cache(embeddings)
            loaded = index._load_cache()
            assert "hash1" in loaded

    def test_needs_rebuild_initially(self):
        """Test needs_rebuild returns True initially."""
        index = ConversationIndex()
        assert index.needs_rebuild()

    def test_message_count(self):
        """Test message_count property."""
        index = ConversationIndex()
        assert index.message_count == 0

        index._messages = [MagicMock(), MagicMock()]
        assert index.message_count == 2

    def test_get_messages_by_session(self):
        """Test filtering messages by session."""
        index = ConversationIndex()
        index._messages = [
            IndexedMessage(
                uuid="m1",
                session_id="s1",
                project_path="/test",
                timestamp=datetime(2024, 1, 1),
                role="user",
                searchable_text="msg1",
                message_index=0,
            ),
            IndexedMessage(
                uuid="m2",
                session_id="s2",
                project_path="/test",
                timestamp=datetime(2024, 1, 1),
                role="user",
                searchable_text="msg2",
                message_index=0,
            ),
            IndexedMessage(
                uuid="m3",
                session_id="s1",
                project_path="/test",
                timestamp=datetime(2024, 1, 1),
                role="assistant",
                searchable_text="msg3",
                message_index=1,
            ),
        ]
        s1_messages = index.get_messages_by_session("s1")
        assert len(s1_messages) == 2
        assert all(m.session_id == "s1" for m in s1_messages)

    def test_get_context_window(self):
        """Test getting context window around a message."""
        index = ConversationIndex()
        messages = [
            IndexedMessage(
                uuid=f"m{i}",
                session_id="s1",
                project_path="/test",
                timestamp=datetime(2024, 1, 1, 12, i),
                role="user" if i % 2 == 0 else "assistant",
                searchable_text=f"msg{i}",
                message_index=i,
            )
            for i in range(10)
        ]
        index._messages = messages

        # Get context around message 5
        target = messages[5]
        context = index.get_context_window(target, context_before_after=2)

        # Should get messages 3, 4, 5, 6, 7
        assert len(context) == 5
        assert context[0].uuid == "m3"
        assert context[-1].uuid == "m7"

    def test_get_context_window_at_start(self):
        """Test context window at start of session."""
        index = ConversationIndex()
        messages = [
            IndexedMessage(
                uuid=f"m{i}",
                session_id="s1",
                project_path="/test",
                timestamp=datetime(2024, 1, 1, 12, i),
                role="user",
                searchable_text=f"msg{i}",
                message_index=i,
            )
            for i in range(5)
        ]
        index._messages = messages

        target = messages[0]
        context = index.get_context_window(target, context_before_after=3)

        # Should get messages 0, 1, 2, 3 (no messages before 0)
        assert len(context) == 4
        assert context[0].uuid == "m0"

    def test_get_context_window_at_end(self):
        """Test context window at end of session."""
        index = ConversationIndex()
        messages = [
            IndexedMessage(
                uuid=f"m{i}",
                session_id="s1",
                project_path="/test",
                timestamp=datetime(2024, 1, 1, 12, i),
                role="user",
                searchable_text=f"msg{i}",
                message_index=i,
            )
            for i in range(5)
        ]
        index._messages = messages

        target = messages[4]
        context = index.get_context_window(target, context_before_after=3)

        # Should get messages 1, 2, 3, 4 (no messages after 4)
        assert len(context) == 4
        assert context[-1].uuid == "m4"


class TestGetIndex:
    """Tests for global index singleton."""

    def test_get_index_returns_same_instance(self):
        """Test get_index returns the same instance."""
        # Reset global
        import claude_total_recall.indexer as indexer_module

        indexer_module._index = None

        index1 = get_index()
        index2 = get_index()
        assert index1 is index2

    def test_get_index_creates_conversation_index(self):
        """Test get_index creates a ConversationIndex."""
        import claude_total_recall.indexer as indexer_module

        indexer_module._index = None

        index = get_index()
        assert isinstance(index, ConversationIndex)


class TestGetPhysicalMemory:
    """Tests for physical memory detection."""

    def test_returns_positive_int_on_supported_platform(self):
        """Test that physical memory detection works on supported platforms."""
        memory = get_physical_memory()
        # On macOS and Linux, this should return a positive value
        # On unsupported platforms, it returns 0
        assert isinstance(memory, int)
        assert memory >= 0

    def test_linux_meminfo_parsing(self, tmp_path):
        """Test parsing /proc/meminfo on Linux."""
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal:       16384000 kB\nMemFree:        8192000 kB\n")

        with (
            patch("claude_total_recall.indexer.platform.system", return_value="Linux"),
            patch("builtins.open", return_value=meminfo.open()),
        ):
            memory = get_physical_memory()
            assert memory == 16384000 * 1024  # 16GB in bytes

    def test_macos_sysctl(self):
        """Test sysctl call on macOS."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "17179869184\n"  # 16GB in bytes

        with (
            patch("claude_total_recall.indexer.platform.system", return_value="Darwin"),
            patch("claude_total_recall.indexer.subprocess.run", return_value=mock_result),
        ):
            memory = get_physical_memory()
            assert memory == 17179869184

    def test_unsupported_platform_returns_zero(self):
        """Test unsupported platform returns 0."""
        with patch("claude_total_recall.indexer.platform.system", return_value="Windows"):
            memory = get_physical_memory()
            assert memory == 0


class TestGetMemoryLimit:
    """Tests for memory limit configuration."""

    def test_disabled_via_env(self):
        """Test memory limit can be disabled via environment variable."""
        with patch.dict("os.environ", {"TOTAL_RECALL_NO_MEMORY_LIMIT": "1"}):
            limit = get_memory_limit()
            assert limit == 0

    def test_explicit_override_via_env(self):
        """Test explicit memory limit override via environment variable."""
        with patch.dict("os.environ", {"TOTAL_RECALL_MEMORY_LIMIT_MB": "512"}, clear=True):
            limit = get_memory_limit()
            assert limit == 512 * 1024 * 1024  # 512MB in bytes

    def test_default_uses_fraction_of_physical(self):
        """Test default limit is 1/3 of physical memory."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("claude_total_recall.indexer.get_physical_memory", return_value=12 * 1024**3),
        ):
            limit = get_memory_limit()
            assert limit == 4 * 1024**3  # 1/3 of 12GB = 4GB

    def test_no_limit_when_detection_fails(self):
        """Test no limit applied when memory detection fails."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("claude_total_recall.indexer.get_physical_memory", return_value=0),
        ):
            limit = get_memory_limit()
            assert limit == 0

    def test_invalid_env_value_uses_default(self):
        """Test invalid environment value falls back to default."""
        with (
            patch.dict("os.environ", {"TOTAL_RECALL_MEMORY_LIMIT_MB": "not_a_number"}),
            patch("claude_total_recall.indexer.get_physical_memory", return_value=12 * 1024**3),
        ):
            limit = get_memory_limit()
            assert limit == 4 * 1024**3  # Falls back to 1/3 of 12GB


class TestSelectSessionsWithinLimit:
    """Tests for session selection with memory limits."""

    @pytest.fixture
    def sample_sessions(self) -> list[SessionInfo]:
        """Create sample sessions with different sizes and timestamps."""
        base_time = datetime(2024, 1, 1, 12, 0)
        return [
            SessionInfo(
                session_id="old",
                project_path="/test",
                message_count=100,
                modified=base_time - timedelta(days=10),
            ),
            SessionInfo(
                session_id="medium",
                project_path="/test",
                message_count=50,
                modified=base_time - timedelta(days=5),
            ),
            SessionInfo(
                session_id="new",
                project_path="/test",
                message_count=25,
                modified=base_time,
            ),
        ]

    def test_no_limit_includes_all(self, sample_sessions):
        """Test no limit (0) includes all sessions."""
        selected, excluded = select_sessions_within_limit(sample_sessions, 0)
        assert len(selected) == 3
        assert len(excluded) == 0

    def test_selects_newest_first(self, sample_sessions):
        """Test newest sessions are selected first."""
        # Limit to fit only the newest session
        limit = 25 * BYTES_PER_MESSAGE + 1
        selected, excluded = select_sessions_within_limit(sample_sessions, limit)

        assert len(selected) == 1
        assert selected[0].session_id == "new"
        assert len(excluded) == 2

    def test_excludes_oldest_when_limited(self, sample_sessions):
        """Test oldest sessions are excluded when limit is reached."""
        # Limit to fit newest + medium, but not oldest
        limit = (25 + 50) * BYTES_PER_MESSAGE + 1
        selected, excluded = select_sessions_within_limit(sample_sessions, limit)

        assert len(selected) == 2
        assert {s.session_id for s in selected} == {"new", "medium"}
        assert len(excluded) == 1
        assert excluded[0].session_id == "old"

    def test_all_sessions_fit(self, sample_sessions):
        """Test all sessions included when limit is large enough."""
        limit = (100 + 50 + 25) * BYTES_PER_MESSAGE + 1000
        selected, excluded = select_sessions_within_limit(sample_sessions, limit)

        assert len(selected) == 3
        assert len(excluded) == 0

    def test_handles_sessions_without_modified(self):
        """Test sessions without modified timestamp use created or fallback."""
        sessions = [
            SessionInfo(
                session_id="no_dates",
                project_path="/test",
                message_count=10,
            ),
            SessionInfo(
                session_id="has_created",
                project_path="/test",
                message_count=10,
                created=datetime(2024, 1, 1),
            ),
        ]
        # Should not raise, uses timestamp_fallback
        selected, excluded = select_sessions_within_limit(sessions, 10 * BYTES_PER_MESSAGE + 1)
        assert len(selected) == 1
        # The one with created date should be selected as "newer"
        assert selected[0].session_id == "has_created"


class TestConversationIndexMemoryManagement:
    """Tests for ConversationIndex memory management."""

    def test_excluded_session_count_initially_zero(self):
        """Test excluded_session_count starts at 0."""
        index = ConversationIndex()
        assert index.excluded_session_count == 0

    def test_excluded_session_count_property(self):
        """Test excluded_session_count property returns correct value."""
        index = ConversationIndex()
        index._excluded_session_count = 5
        assert index.excluded_session_count == 5

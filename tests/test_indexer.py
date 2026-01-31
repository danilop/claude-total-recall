"""Tests for conversation indexer."""

import pickle
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from claude_total_recall.indexer import (
    ConversationIndex,
    _get_sessions_fingerprint,
    get_index,
)
from claude_total_recall.models import IndexedMessage


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

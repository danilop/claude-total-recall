"""Semantic embedding index for conversation search."""

import contextlib
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
from filelock import FileLock

from .loader import get_projects_dir, load_all_messages
from .models import IndexedMessage

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CACHE_DIR = Path.home() / ".cache" / "claude-total-recall"
CACHE_FILE = CACHE_DIR / "embeddings.pkl"
LOCK_FILE = CACHE_DIR / "embeddings.lock"


def _get_sessions_fingerprint() -> str:
    """Get a fingerprint of all session files to detect changes."""
    projects_dir = get_projects_dir()
    if not projects_dir.exists():
        return ""

    fingerprint_parts = []
    for index_file in sorted(projects_dir.glob("*/sessions-index.json")):
        try:
            mtime = index_file.stat().st_mtime
            fingerprint_parts.append(f"{index_file}:{mtime}")
        except OSError:
            continue

    return hashlib.md5("\n".join(fingerprint_parts).encode()).hexdigest()


class ConversationIndex:
    """Index of conversation messages with semantic embeddings."""

    def __init__(self):
        self._model = None
        self._messages: list[IndexedMessage] = []
        self._embeddings: np.ndarray | None = None
        self._text_hashes: list[str] = []
        self._sessions_fingerprint: str = ""

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _compute_text_hash(self, text: str) -> str:
        """Compute a hash of the text for caching."""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_cache(self) -> dict[str, np.ndarray]:
        """
        Load cached embeddings as hash -> embedding dict.
        Returns empty dict on any failure (graceful degradation).
        """
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
                if isinstance(cache, dict):
                    return cache
        except (OSError, pickle.PickleError, EOFError):
            pass
        return {}

    def _save_cache(self, new_embeddings: dict[str, np.ndarray]):
        """
        Save embeddings atomically with file locking.

        Uses:
        - File lock to prevent concurrent read-modify-write races
        - Temp file + atomic rename to prevent corruption
        - Merge with existing cache to preserve parallel updates
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Use cross-platform file locking
        try:
            with FileLock(LOCK_FILE):
                # Now safely load, merge, and save
                existing = self._load_cache()
                merged = {**existing, **new_embeddings}

                # Atomic write: temp file then rename
                temp_file = CACHE_FILE.with_suffix(".tmp." + str(os.getpid()))
                try:
                    with open(temp_file, "wb") as f:
                        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temp_file, CACHE_FILE)
                except OSError:
                    with contextlib.suppress(OSError):
                        temp_file.unlink()
        except OSError:
            # Lock acquisition failed - skip caching (graceful degradation)
            pass

    def needs_rebuild(self) -> bool:
        """Check if the index needs to be rebuilt due to new conversations."""
        current_fingerprint = _get_sessions_fingerprint()
        return current_fingerprint != self._sessions_fingerprint

    def build_index(self, include_subagents: bool = True):
        """Build or update the search index."""
        self._sessions_fingerprint = _get_sessions_fingerprint()
        self._messages = load_all_messages(include_subagents)

        if not self._messages:
            self._embeddings = np.array([])
            self._text_hashes = []
            return

        # Load cached embeddings
        cache = self._load_cache()

        # Compute hashes and find what needs embedding
        self._text_hashes = []
        texts_to_embed = []
        indices_to_embed = []

        for i, msg in enumerate(self._messages):
            text_hash = self._compute_text_hash(msg.searchable_text)
            self._text_hashes.append(text_hash)

            if text_hash not in cache:
                texts_to_embed.append(msg.searchable_text)
                indices_to_embed.append(i)

        # Initialize embeddings array
        self._embeddings = np.zeros((len(self._messages), EMBEDDING_DIM), dtype=np.float32)

        # Copy cached embeddings
        for i, text_hash in enumerate(self._text_hashes):
            if text_hash in cache:
                self._embeddings[i] = cache[text_hash]

        # Generate new embeddings
        new_cache_entries = {}
        if texts_to_embed:
            batch_embeddings = self.model.encode(
                texts_to_embed,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for i, idx in enumerate(indices_to_embed):
                embedding = batch_embeddings[i]
                self._embeddings[idx] = embedding
                new_cache_entries[self._text_hashes[idx]] = embedding

        # Save new embeddings to cache
        if new_cache_entries:
            self._save_cache(new_cache_entries)

    def ensure_index(self, include_subagents: bool = True):
        """Ensure the index is built and up-to-date."""
        if self._embeddings is None or self.needs_rebuild():
            self.build_index(include_subagents)

    def search(
        self,
        query: str,
        project: str | None = None,
        threshold: float = 0.3,
        max_results: int = 100,
        include_subagents: bool = True,
    ) -> list[tuple[IndexedMessage, float]]:
        """Search for messages matching the query."""
        self.ensure_index(include_subagents)

        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        query_embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        similarities = np.dot(self._embeddings, query_embedding)
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            if score < threshold:
                break

            msg = self._messages[idx]

            if project and not msg.project_path.startswith(project):
                continue
            if not include_subagents and msg.is_subagent:
                continue

            results.append((msg, score))

            if len(results) >= max_results:
                break

        return results

    def get_messages_by_session(self, session_id: str) -> list[IndexedMessage]:
        """Get all messages for a session, sorted by index."""
        return sorted(
            [m for m in self._messages if m.session_id == session_id],
            key=lambda m: m.message_index,
        )

    def get_context_window(
        self, message: IndexedMessage, context_before_after: int = 3
    ) -> list[IndexedMessage]:
        """
        Get messages around a matched message for context.

        Args:
            message: The matched message
            context_before_after: Number of messages to include before AND after the match
        """
        session_messages = self.get_messages_by_session(message.session_id)

        msg_idx = None
        for i, m in enumerate(session_messages):
            if m.uuid == message.uuid:
                msg_idx = i
                break

        if msg_idx is None:
            return [message]

        start = max(0, msg_idx - context_before_after)
        end = min(len(session_messages), msg_idx + context_before_after + 1)

        return session_messages[start:end]

    @property
    def message_count(self) -> int:
        """Get the number of indexed messages."""
        return len(self._messages)


# Global index instance
_index: ConversationIndex | None = None


def get_index() -> ConversationIndex:
    """Get or create the global conversation index."""
    global _index
    if _index is None:
        _index = ConversationIndex()
    return _index

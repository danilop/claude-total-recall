"""Semantic embedding index for conversation search."""

import contextlib
import hashlib
import logging
import os
import pickle
import platform
import subprocess
from pathlib import Path

import numpy as np
from filelock import FileLock

from .loader import get_projects_dir, list_all_sessions, load_messages_for_sessions
from .models import IndexedMessage, SessionInfo

logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CACHE_DIR = Path.home() / ".cache" / "claude-total-recall"
CACHE_FILE = CACHE_DIR / "embeddings.pkl"
LOCK_FILE = CACHE_DIR / "embeddings.lock"

# Memory limit configuration
MEMORY_LIMIT_ENV = "TOTAL_RECALL_MEMORY_LIMIT_MB"
MEMORY_LIMIT_DISABLED_ENV = "TOTAL_RECALL_NO_MEMORY_LIMIT"
DEFAULT_MEMORY_FRACTION = 1 / 3
BYTES_PER_MESSAGE = 2600  # ~1KB message + 1.5KB embedding + overhead


def get_physical_memory() -> int:
    """
    Get physical memory in bytes using native Python.

    Returns:
        Physical memory in bytes, or 0 on failure (no limit applied).
    """
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Format: "MemTotal:       16384000 kB"
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass

    elif system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            pass

    return 0  # Fallback: no limit


def get_memory_limit() -> int:
    """
    Get memory limit in bytes for the in-memory index.

    Checks environment variables for overrides, otherwise uses 1/3 of physical RAM.

    Returns:
        Memory limit in bytes, or 0 for no limit.
    """
    # Check if limit is disabled
    if os.environ.get(MEMORY_LIMIT_DISABLED_ENV):
        return 0

    # Check for explicit override
    override = os.environ.get(MEMORY_LIMIT_ENV)
    if override:
        try:
            return int(override) * 1024 * 1024  # Convert MB to bytes
        except ValueError:
            logger.warning(f"Invalid {MEMORY_LIMIT_ENV} value: {override}, using default")

    # Default: 1/3 of physical memory
    physical = get_physical_memory()
    return int(physical * DEFAULT_MEMORY_FRACTION) if physical else 0


def select_sessions_within_limit(
    sessions: list[SessionInfo], memory_limit_bytes: int
) -> tuple[list[SessionInfo], list[SessionInfo]]:
    """
    Select newest sessions that fit within memory limit.

    Args:
        sessions: All available sessions.
        memory_limit_bytes: Maximum memory to use (0 = no limit).

    Returns:
        Tuple of (selected_sessions, excluded_sessions).
    """
    if memory_limit_bytes <= 0:
        return sessions, []

    # Sort by modification time, newest first
    sorted_sessions = sorted(
        sessions,
        key=lambda s: s.timestamp_fallback,
        reverse=True,
    )

    selected: list[SessionInfo] = []
    excluded: list[SessionInfo] = []
    current_bytes = 0

    for session in sorted_sessions:
        estimated_mem = session.message_count * BYTES_PER_MESSAGE
        if current_bytes + estimated_mem <= memory_limit_bytes:
            selected.append(session)
            current_bytes += estimated_mem
        else:
            excluded.append(session)

    return selected, excluded


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
        self._excluded_session_count: int = 0

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
        """Build or update the search index with memory limits."""
        self._sessions_fingerprint = _get_sessions_fingerprint()

        # Get all sessions and apply memory limit
        all_sessions = list_all_sessions()
        memory_limit = get_memory_limit()
        selected_sessions, excluded_sessions = select_sessions_within_limit(
            all_sessions, memory_limit
        )
        self._excluded_session_count = len(excluded_sessions)

        if excluded_sessions:
            logger.warning(
                f"Memory limit ({memory_limit / 1024 / 1024:.0f} MB) reached: "
                f"excluding {len(excluded_sessions)} oldest sessions from index. "
                f"Set {MEMORY_LIMIT_DISABLED_ENV}=1 to disable limit."
            )

        # Load messages only for selected sessions
        self._messages = load_messages_for_sessions(selected_sessions, include_subagents)

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

    @property
    def excluded_session_count(self) -> int:
        """Get the number of sessions excluded due to memory limits."""
        return self._excluded_session_count


# Global index instance
_index: ConversationIndex | None = None


def get_index() -> ConversationIndex:
    """Get or create the global conversation index."""
    global _index
    if _index is None:
        _index = ConversationIndex()
    return _index

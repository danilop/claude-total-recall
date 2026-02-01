"""Centralized configuration constants for Claude Total Recall."""

from pathlib import Path

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Cache paths
CACHE_DIR = Path.home() / ".cache" / "claude-total-recall"
CACHE_FILE = CACHE_DIR / "embeddings.pkl"
LOCK_FILE = CACHE_DIR / "embeddings.lock"

# Memory limits
MEMORY_LIMIT_ENV = "TOTAL_RECALL_MEMORY_LIMIT_MB"
MEMORY_LIMIT_DISABLED_ENV = "TOTAL_RECALL_NO_MEMORY_LIMIT"
DEFAULT_MEMORY_FRACTION = 1 / 3
BYTES_PER_MESSAGE = 2600

# Query defaults
DEFAULT_CONTEXT_BEFORE_AFTER = 3
DEFAULT_THRESHOLD = 0.2
DEFAULT_MAX_RESULTS = 10
DEFAULT_OFFSET = 0

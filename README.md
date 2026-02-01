# Claude Total Recall

Ever told Claude "like we discussed yesterday" only to realize... it has no idea?

**Total Recall** gives Claude Code the memory it's missing.

### 1. Compaction Erases Details

Long conversations hit context limits. When that happens, Claude Code **compacts**, summarizing earlier messages to make room. That brilliant debugging session from an hour ago? Reduced to "discussed authentication fixes." The specific error codes, the failed approaches, the final solution: gone from context.

*But the original messages still exist on disk.* Total Recall finds them.

### 2. Sessions Are Isolated

Each Claude Code session starts fresh. Yesterday you spent an hour explaining your project's architecture. Today, Claude has no idea. You're back to explaining that `UserService` talks to `AuthProvider` which validates against `TokenStore`.

*Every session is saved locally.* Total Recall searches across all of them.

### 3. Projects Don't Share Knowledge

You always use `uv` for Python projects. You prefer `pnpm` over `npm`. You like tests next to source files, not in a separate folder. But Claude asks you every single time, in every project.

*Your patterns are in past conversations.* Total Recall finds them across all your projects.

---

**Total Recall indexes every Claude Code conversation and provides semantic search.** Find discussions by *meaning*, not just keywords. Ask "how did we handle rate limiting?" and it finds the relevant conversation, even if you never used those exact words.

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Claude Code                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Session 1   │    │  Session 2   │    │  Session N   │               │
│  │  messages    │    │  messages    │    │  messages    │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             ▼                                           │
│              ~/.claude/projects/<project>/                              │
│              ├── sessions-index.json                                    │
│              ├── <session-id>.jsonl                                     │
│              └── <session-id>/subagents/*.jsonl                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Total Recall                                     │
│                                                                         │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌────────────────────┐   │
│  │ loader  │───▶│ indexer  │───▶│  query  │───▶│  MCP server        │   │
│  │         │    │          │    │         │    │  (FastMCP)         │   │
│  │ Reads   │    │ Embeds   │    │ Cosine  │    │                    │   │
│  │ JSONL   │    │ text     │    │ search  │    │ search_project_    │   │
│  │ files   │    │ (384-d)  │    │         │    │ search_global_     │   │
│  └─────────┘    └────┬─────┘    └─────────┘    └────────────────────┘   │
│                      │                                    ▲             │
│                      ▼                                    │             │
│         ~/.cache/claude-total-recall/              MCP Protocol         │
│         └── embeddings.pkl                                │             │
└───────────────────────────────────────────────────────────┼─────────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────┐
                                                  │   Claude Code    │
                                                  │   (via plugin)   │
                                                  └──────────────────┘
```

### The Data: Where Conversations Live

Claude Code stores all conversations in `~/.claude/projects/`. Each project gets its own directory:

```
~/.claude/projects/
├── -Users-alice-myapp/              # /Users/alice/myapp
│   ├── sessions-index.json          # Session metadata
│   ├── abc123.jsonl                 # Main session messages
│   └── abc123/
│       └── subagents/
│           └── agent-def456.jsonl   # Subagent conversations
├── -Users-alice-other-project/
│   └── ...
```

**Path escaping**: Project paths become directory names by replacing `/` with `-`. So `/Users/alice/myapp` becomes `-Users-alice-myapp`.

**Session files** (`.jsonl`) contain one JSON object per line:

```json
{"type": "user", "uuid": "msg-123", "timestamp": 1706789012345, "message": {"role": "user", "content": "How do I fix this auth bug?"}}
{"type": "assistant", "uuid": "msg-124", "timestamp": 1706789015678, "message": {"role": "assistant", "content": [{"type": "text", "text": "Let me check the auth module..."}]}}
```

### The Index: Making Search Fast

On first search, Total Recall:

1. **Loads** all messages from every project's JSONL files
2. **Embeds** each message using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dimensional vectors, ~80MB model)
3. **Caches** embeddings to `~/.cache/claude-total-recall/embeddings.pkl`

Subsequent searches are fast because:

- **Fingerprinting**: Computes MD5 of all `sessions-index.json` modification times. Only rebuilds when conversations change.
- **Incremental updates**: New messages get embedded; existing embeddings are loaded from cache.
- **Hash-based deduplication**: Each message text is hashed. Same text = same embedding (no recomputation).

**Concurrency safety**: The cache uses file locking (`fcntl.LOCK_EX`) and atomic writes (temp file + rename) to handle multiple Claude Code instances.

### The Search: Finding What Matters

When you search:

1. Your query gets embedded into the same 384-dimensional space
2. **Cosine similarity** finds the closest matches (dot product of normalized vectors)
3. Results above the threshold (default: 0.2) are returned with context

**Semantic matching** means "authentication issue" finds discussions about "login problems" or "JWT token errors". No exact keyword match needed.

**Context windows** include messages before and after each match, so you see the full conversation flow.

**Deduplication** merges overlapping windows: if messages 5, 6, and 7 all match, you get one result (the highest-scoring) with context, not three overlapping snippets.

### The Integration: MCP and Skills

Total Recall integrates with Claude Code through the **Model Context Protocol (MCP)**:

```
.mcp.json                           # Tells Claude Code how to start the server
  → uv run claude-total-recall      # Launches FastMCP server
    → Exposes search_project_history, search_global_history tools
```

The **agent skill** (`skills/conversation-recall/SKILL.md`) teaches Claude *when* to use these tools:

- "How did we fix that bug?" → triggers `search_project_history`
- "What's my preferred testing approach?" → triggers `search_global_history`
- After compaction → automatically searches to recover summarized details

## Features

- **Semantic Search**: Find by meaning, not just keywords
- **Context Windows**: See the full conversation around matches
- **Project Filtering**: Search current project or all projects
- **Incremental Indexing**: Only processes new conversations
- **Compaction Recovery**: Retrieve details lost when context is summarized
- **Agent Skill**: Triggers automatically on relevant questions

## Installation

### Prerequisites

- [Claude Code](https://claude.ai/download)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Option 1: Install from GitHub (Recommended)

```bash
# In Claude Code, add the marketplace source
/plugin marketplace add danilop/claude-total-recall

# Install the plugin
/plugin install claude-total-recall@claude-total-recall
```

**Restart Claude Code** after installation. MCP servers are only loaded at startup.

**Persistent**: The plugin is copied to `~/.claude/plugins/` and loads automatically every session.

### Option 2: Install from Local Directory

For development:

```bash
git clone https://github.com/danilop/claude-total-recall.git
cd claude-total-recall
uv sync
```

Then start Claude Code with the plugin:

```bash
claude --plugin-dir /path/to/claude-total-recall
```

**Not persistent**: You must pass `--plugin-dir` every time.

### Updating

To update to the latest version:

```bash
# Update the plugin files and install dependencies
cd ~/.claude/plugins/marketplaces/claude-total-recall
git pull
uv sync

# If installed from a local directory instead
cd /path/to/claude-total-recall
git pull
uv sync
```

**Restart Claude Code** after updating to reload the MCP server.

### Verify Installation

In Claude Code:

- `/mcp` should list `claude-total-recall`
- `/skills` should list `conversation-recall`

### Troubleshooting

If you see "Missing dependencies" errors, run:

```bash
cd ~/.claude/plugins/marketplaces/claude-total-recall
uv sync
```

Then restart Claude Code. (This is automatic on macOS/Linux but may be needed on Windows.)

## Usage

### Natural Language (via Skill)

Just ask naturally. The skill triggers automatically:

```
"How did we fix that auth bug?"
"What did we discuss about the database schema?"
"Find our React component discussions"
"What's my usual approach to error handling?"
```

### MCP Tools

| Tool | Scope | Use Case |
|------|-------|----------|
| `search_project_history` | Current project | Decisions, implementations, bugs in *this* codebase |
| `search_global_history` | All projects | User preferences, patterns across *all* work |

### Recovering Compacted Context

When Claude Code compacts a conversation (at ~95% context or via `/compact`), details get summarized. Total Recall retrieves the originals:

```
User: "Continue with the auth approach we discussed"
Claude: [searches for "auth approach implementation" to recover details]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | required | Keywords or sentence to search |
| `after` | none | Filter to messages on/after this date (inclusive). ISO 8601 format. |
| `before` | none | Filter to messages before this date (exclusive). ISO 8601 format. |
| `context_before_after` | 3 | Messages before AND after each match |
| `threshold` | 0.2 | Minimum similarity (0-1, higher = stricter) |
| `max_results` | 10 | Maximum results to return |
| `offset` | 0 | Skip results (for pagination) |
| `include_subagents` | true | Include agent/subagent conversations |

### Date Filtering Examples

```python
# Messages from a specific day
search_project_history(query="auth bug", after="2025-01-15", before="2025-01-16")

# Messages from the past week
search_project_history(query="refactoring", after="2025-01-25")

# Messages in January
search_project_history(query="database", after="2025-01-01", before="2025-02-01")
```

### Response Structure

```json
{
  "results": [
    {
      "matched_message": {
        "role": "assistant",
        "content": "To fix the authentication bug...",
        "timestamp": "2025-01-15T10:30:00Z",
        "project": "/Users/dev/myproject",
        "session_id": "abc123",
        "uuid": "msg-456"
      },
      "score": 0.8542,
      "context": [
        {"role": "user", "content": "How do I fix this auth bug?", "timestamp": "...", "is_match": false},
        {"role": "assistant", "content": "To fix the authentication bug...", "timestamp": "...", "is_match": true}
      ]
    }
  ],
  "query": "authentication bug fix",
  "total_matches": 25,
  "offset": 0,
  "has_more": true,
  "excluded_sessions": 0
}
```

### Pagination

```python
# First page
search_project_history(query="auth bug", max_results=10, offset=0)

# Next page
search_project_history(query="auth bug", max_results=10, offset=10)
```

## Memory Management

Total Recall limits in-memory index size to prevent excessive memory usage. By default, it uses 1/3 of physical RAM. When the limit is reached, the oldest sessions are excluded from the index (newest sessions are kept). All conversation files on disk are preserved.

When sessions are excluded, the `excluded_sessions` field in search responses indicates how many sessions were not indexed. A warning is also logged.

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TOTAL_RECALL_MEMORY_LIMIT_MB` | Override memory limit in MB | 1/3 of RAM |
| `TOTAL_RECALL_NO_MEMORY_LIMIT` | Set to any value to disable limit | - |

Examples:

```bash
# Limit to 256 MB
export TOTAL_RECALL_MEMORY_LIMIT_MB=256

# Disable limit (index all sessions)
export TOTAL_RECALL_NO_MEMORY_LIMIT=1
```

## Testing

### Without Claude Code

```bash
# Test server starts
uv run claude-total-recall
# Ctrl+C to exit

# Test search directly
uv run python -c "
from claude_total_recall.server import search_global_history
result = search_global_history(query='bug fix', max_results=3)
print(f'Found {result[\"total_matches\"]} matches')
"
```

### In Claude Code

1. Load the plugin
2. Run `/mcp` to verify
3. Ask: "Search my previous conversations for authentication"

## Project Structure

```
claude-total-recall/
├── .claude-plugin/
│   ├── plugin.json              # Plugin manifest
│   └── marketplace.json         # Distribution config
├── skills/
│   └── conversation-recall/
│       └── SKILL.md             # When to trigger searches
├── .mcp.json                    # MCP server config
├── src/claude_total_recall/
│   ├── server.py                # FastMCP server, tool definitions
│   ├── query.py                 # Search engine, deduplication
│   ├── indexer.py               # Embedding, caching, fingerprinting
│   ├── loader.py                # JSONL parsing, session loading
│   └── models.py                # Pydantic data models
├── pyproject.toml
└── LICENSE
```

## Technical Details

| Component | Technology |
|-----------|------------|
| Embedding model | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions) |
| Vector search | Cosine similarity via NumPy dot product |
| Cache format | Python pickle with file locking |
| MCP framework | [FastMCP](https://github.com/jlowin/fastmcp) |
| Package manager | [uv](https://docs.astral.sh/uv/) |

## License

[MIT License](LICENSE)

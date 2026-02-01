---
name: conversation-recall
description: |
  Search past Claude Code conversations. Trigger when:
  - After compaction: recover detailed context that was summarized
  - Cross-session: find discussions from previous sessions ("yesterday", "last week")
  - Cross-project: find user preferences and patterns across all projects
  - User says "recall", "remember", "how did we", "what did we discuss"
version: 0.1.0
---

# Conversation Recall Skill

Search past Claude Code conversation history using semantic search.

## When to Trigger

### 1. After Compaction

When context is compacted, earlier details are summarized. Search to recover originals:

- You see a `compact_boundary` marker
- User references something discussed earlier but details are missing
- User says "like we discussed", "as I mentioned", "continue with..."

```
User: "Continue implementing the auth system like we discussed"
→ search_project_history(query="auth system implementation")
```

### 2. Cross-Session

Find discussions from previous sessions (not just the current one):

- "How did we fix that bug yesterday?"
- "What approach did we decide on last week?"
- "Remember when we refactored the database?"

```
User: "How did we fix that auth bug last week?"
→ search_project_history(query="auth bug fix")
```

### 3. Cross-Project

Find user preferences and patterns across all projects:

- "How do I usually handle errors?"
- "What's my preferred testing approach?"
- "What package manager do I use for Python?"

```
User: "How do I usually structure React components?"
→ search_global_history(query="React component structure")
```

## Which Tool to Use

| Scope | Tool | Examples |
|-------|------|----------|
| **Current project** | `search_project_history` | Bugs, decisions, implementations in *this* codebase |
| **All projects** | `search_global_history` | User preferences, patterns, conventions |

## Parameters

Both tools accept:

- `query` (required): Keywords or sentence to search
- `after` (optional): Filter to messages on/after this date (inclusive). ISO 8601 format.
- `before` (optional): Filter to messages before this date (exclusive). ISO 8601 format.
- `context_before_after` (default: 3): Messages before/after each match
- `threshold` (default: 0.2): Minimum similarity (0-1)
- `max_results` (default: 10): Results to return
- `offset` (default: 0): Skip results for pagination
- `include_subagents` (default: true): Include agent conversations

## Date Filtering

### Converting Relative Time to Specific Dates

You know today's date from system context. Always convert relative time references to specific ISO 8601 dates:

| User says | Calculate (if today is 2025-02-01) |
|-----------|-------------------------------------|
| "yesterday" | `after="2025-01-31", before="2025-02-01"` |
| "last week" | `after="2025-01-25", before="2025-02-01"` |
| "in January" | `after="2025-01-01", before="2025-02-01"` |
| "last month" | `after="2025-01-01", before="2025-02-01"` |
| "recently" | `after="2025-01-25"` (past ~7 days) |
| "a while ago" | `after="2025-01-01"` (past ~30 days) |

### Date Parameter Semantics

- `after`: inclusive (>=) - messages ON or AFTER this date
- `before`: exclusive (<) - messages BEFORE this date
- Format: ISO 8601 - `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS` or `YYYY-MM-DDTHH:MM:SSZ`

### Examples

**Single day:**
```
User: "What did we discuss about auth yesterday?"
Today is 2025-02-01

-> search_project_history(
    query="auth authentication login",
    after="2025-01-31",
    before="2025-02-01"
  )
```

**Date range:**
```
User: "Find database discussions from last week"
Today is 2025-02-01

-> search_project_history(
    query="database schema migration",
    after="2025-01-25",
    before="2025-02-01"
  )
```

**Open-ended (recent):**
```
User: "What have we been working on lately?"
Today is 2025-02-01

-> search_project_history(
    query="implementation work progress",
    after="2025-01-25"
  )
```

### Tips

- Combine semantic query with date filters for best results
- When date is ambiguous, use broader range and refine based on result timestamps
- Results include timestamps - you can interpret and summarize temporal patterns

## Tips

- Search is **semantic**: "authentication issue" matches "login problem"
- Results include **context** (surrounding messages)
- Use `offset` to paginate when `has_more` is true
- Higher scores (closer to 1.0) = better matches

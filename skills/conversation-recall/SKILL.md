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
- `context_before_after` (default: 3): Messages before/after each match
- `threshold` (default: 0.2): Minimum similarity (0-1)
- `max_results` (default: 10): Results to return
- `offset` (default: 0): Skip results for pagination
- `include_subagents` (default: true): Include agent conversations

## Tips

- Search is **semantic**: "authentication issue" matches "login problem"
- Results include **context** (surrounding messages)
- Use `offset` to paginate when `has_more` is true
- Higher scores (closer to 1.0) = better matches

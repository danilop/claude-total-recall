"""Tests for query engine."""

from datetime import datetime

import pytest

from claude_total_recall.models import IndexedMessage
from claude_total_recall.query import (
    _deduplicate_results,
    _generate_hint,
    _truncate_content,
    parse_date_filter,
)


class TestParseDateFilter:
    """Tests for parse_date_filter function."""

    def test_parse_date_only(self):
        """'2025-01-15' parses correctly."""
        result = parse_date_filter("2025-01-15")
        assert result == datetime(2025, 1, 15)

    def test_parse_datetime(self):
        """'2025-01-15T10:30:00' parses correctly."""
        result = parse_date_filter("2025-01-15T10:30:00")
        assert result == datetime(2025, 1, 15, 10, 30, 0)

    def test_parse_datetime_with_z_suffix(self):
        """'2025-01-15T10:30:00Z' parses correctly and strips timezone."""
        result = parse_date_filter("2025-01-15T10:30:00Z")
        assert result == datetime(2025, 1, 15, 10, 30, 0)
        assert result.tzinfo is None

    def test_parse_datetime_with_timezone(self):
        """Datetime with timezone parses and strips timezone."""
        result = parse_date_filter("2025-01-15T10:30:00+05:00")
        assert result == datetime(2025, 1, 15, 10, 30, 0)
        assert result.tzinfo is None

    def test_parse_none(self):
        """None returns None."""
        result = parse_date_filter(None)
        assert result is None

    def test_parse_invalid_format(self):
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_date_filter("not-a-date")
        assert "Invalid date format" in str(exc_info.value)
        assert "ISO 8601" in str(exc_info.value)

    def test_parse_partial_date_invalid(self):
        """Partial date without full format raises ValueError."""
        with pytest.raises(ValueError):
            parse_date_filter("2025-01")

    def test_parse_with_microseconds(self):
        """Datetime with microseconds parses correctly."""
        result = parse_date_filter("2025-01-15T10:30:00.123456")
        assert result == datetime(2025, 1, 15, 10, 30, 0, 123456)


class TestTruncateContent:
    """Tests for content truncation."""

    def test_short_content_unchanged(self):
        """Test short content is not truncated."""
        content = "Short text"
        result = _truncate_content(content, max_length=100)
        assert result == content

    def test_long_content_truncated(self):
        """Test long content is truncated with ellipsis."""
        content = "A" * 100
        result = _truncate_content(content, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_exact_length_unchanged(self):
        """Test content at exact max length is not truncated."""
        content = "A" * 50
        result = _truncate_content(content, max_length=50)
        assert result == content
        assert "..." not in result

    def test_default_max_length(self):
        """Test default max length is 2000."""
        content = "A" * 2001
        result = _truncate_content(content)
        assert len(result) == 2000


class TestDeduplicateResults:
    """Tests for result deduplication."""

    def _make_message(self, uuid: str, session_id: str, message_index: int) -> IndexedMessage:
        """Helper to create a test message."""
        return IndexedMessage(
            uuid=uuid,
            session_id=session_id,
            project_path="/test",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            role="assistant",
            searchable_text=f"Message {uuid}",
            message_index=message_index,
        )

    def test_empty_results(self):
        """Test deduplicating empty results."""
        result = _deduplicate_results([], context_before_after=3)
        assert result == []

    def test_single_result(self):
        """Test deduplicating single result."""
        msg = self._make_message("m1", "s1", 0)
        results = [(msg, 0.9)]
        deduped = _deduplicate_results(results, context_before_after=3)
        assert len(deduped) == 1

    def test_non_overlapping_results(self):
        """Test results that don't overlap are kept."""
        msg1 = self._make_message("m1", "s1", 0)
        msg2 = self._make_message("m2", "s1", 10)  # Far apart
        results = [(msg1, 0.9), (msg2, 0.8)]
        deduped = _deduplicate_results(results, context_before_after=3)
        assert len(deduped) == 2

    def test_overlapping_results_keep_higher_score(self):
        """Test overlapping results keep higher score."""
        msg1 = self._make_message("m1", "s1", 0)
        msg2 = self._make_message("m2", "s1", 2)  # Close, overlapping context
        results = [(msg1, 0.7), (msg2, 0.9)]
        deduped = _deduplicate_results(results, context_before_after=3)
        assert len(deduped) == 1
        assert deduped[0][0].uuid == "m2"  # Higher score kept

    def test_different_sessions_not_merged(self):
        """Test messages from different sessions are not merged."""
        msg1 = self._make_message("m1", "s1", 0)
        msg2 = self._make_message("m2", "s2", 0)  # Different session
        results = [(msg1, 0.9), (msg2, 0.8)]
        deduped = _deduplicate_results(results, context_before_after=3)
        assert len(deduped) == 2

    def test_results_sorted_by_score(self):
        """Test deduplicated results are sorted by score descending."""
        msg1 = self._make_message("m1", "s1", 0)
        msg2 = self._make_message("m2", "s2", 0)
        msg3 = self._make_message("m3", "s3", 0)
        results = [(msg1, 0.5), (msg2, 0.9), (msg3, 0.7)]
        deduped = _deduplicate_results(results, context_before_after=3)
        assert deduped[0][1] == 0.9
        assert deduped[1][1] == 0.7
        assert deduped[2][1] == 0.5

    def test_context_window_size_affects_dedup(self):
        """Test context window size affects deduplication."""
        msg1 = self._make_message("m1", "s1", 0)
        msg2 = self._make_message("m2", "s1", 5)

        # With context=3, distance of 5 is NOT overlapping (2*3=6 > 5, so IS overlapping)
        results = [(msg1, 0.9), (msg2, 0.8)]
        deduped = _deduplicate_results(results, context_before_after=3)
        assert len(deduped) == 1  # Merged because 5 <= 6

        # With context=2, distance of 5 is NOT overlapping (2*2=4 < 5)
        deduped = _deduplicate_results(results, context_before_after=2)
        assert len(deduped) == 2  # Not merged because 5 > 4

    def test_multiple_overlaps_in_session(self):
        """Test multiple overlapping messages in same session."""
        msg1 = self._make_message("m1", "s1", 0)
        msg2 = self._make_message("m2", "s1", 2)
        msg3 = self._make_message("m3", "s1", 4)
        msg4 = self._make_message("m4", "s1", 20)  # Far away
        results = [
            (msg1, 0.5),
            (msg2, 0.9),  # Highest in first group
            (msg3, 0.6),
            (msg4, 0.8),
        ]
        deduped = _deduplicate_results(results, context_before_after=3)
        # msg1, msg2, msg3 should be merged (keeping msg2)
        # msg4 should be separate
        assert len(deduped) == 2
        uuids = {d[0].uuid for d in deduped}
        assert "m2" in uuids
        assert "m4" in uuids


class TestGenerateHint:
    """Tests for pagination hint generation."""

    def test_no_matches(self):
        """Test hint when no matches found."""
        hint = _generate_hint(
            total_matches=0,
            offset=0,
            results_count=0,
            max_results=10,
            has_more=False,
        )
        assert "No matches found" in hint
        assert "different search terms" in hint

    def test_has_more_results(self):
        """Test hint when more results available."""
        hint = _generate_hint(
            total_matches=25,
            offset=0,
            results_count=10,
            max_results=10,
            has_more=True,
        )
        assert "Showing 1-10 of 25" in hint
        assert "offset: 10" in hint
        assert "different search terms" in hint

    def test_has_more_with_offset(self):
        """Test hint when paginating through results."""
        hint = _generate_hint(
            total_matches=25,
            offset=10,
            results_count=10,
            max_results=10,
            has_more=True,
        )
        assert "Showing 11-20 of 25" in hint
        assert "offset: 20" in hint

    def test_final_page(self):
        """Test hint on final page of results."""
        hint = _generate_hint(
            total_matches=25,
            offset=20,
            results_count=5,
            max_results=10,
            has_more=False,
        )
        assert "Showing 21-25 of 25" in hint
        assert "final page" in hint
        assert "offset" not in hint.lower() or "offset:" not in hint

    def test_all_results_on_first_page(self):
        """Test hint when all results fit on first page."""
        hint = _generate_hint(
            total_matches=5,
            offset=0,
            results_count=5,
            max_results=10,
            has_more=False,
        )
        assert "Showing all 5 matches" in hint
        assert "different search terms" in hint

    def test_single_match(self):
        """Test hint when only one match found."""
        hint = _generate_hint(
            total_matches=1,
            offset=0,
            results_count=1,
            max_results=10,
            has_more=False,
        )
        assert "only match" in hint
        assert "different search terms" in hint

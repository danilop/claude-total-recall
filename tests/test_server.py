"""Tests for MCP server."""

import os
from unittest.mock import patch

from claude_total_recall.server import _get_current_project


class TestGetCurrentProject:
    """Tests for getting current project from environment."""

    def test_from_claude_project_dir(self):
        """Test getting project from CLAUDE_PROJECT_DIR."""
        with patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": "/test/project"}):
            project = _get_current_project()
            assert project == "/test/project"

    def test_fallback_to_pwd(self):
        """Test falling back to PWD when CLAUDE_PROJECT_DIR not set."""
        with patch.dict(os.environ, {"PWD": "/pwd/path"}, clear=True):
            project = _get_current_project()
            assert project == "/pwd/path"

    def test_fallback_to_cwd(self):
        """Test falling back to getcwd when nothing else available."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.getcwd", return_value="/cwd/path"),
        ):
            project = _get_current_project()
            assert project == "/cwd/path"

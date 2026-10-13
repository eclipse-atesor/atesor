#############################################################################
# Copyright (c) 2026 10xEngineers
#
# Author: Akif Ejaz <akif.ejaz@10xengineers.ai>
# This program and the accompanying materials are made available under the
# terms of the MIT License which is available at
# https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
#############################################################################

"""Deterministic repo-evidence collection for LLM prompts.

The agents can only "read and understand" a package if their prompts
contain the package's ACTUAL files — not just detection summaries. This
module gathers bounded excerpts of the real build files and docs
(zero LLM cost) so the analyst/scout reason from evidence, and extracts
source context around compile errors so the fixer sees the code it is
being asked to fix.

Everything here is size-capped: prompts run on free-tier models where
tokens are the scarce resource.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

from .config import to_host_path

logger = logging.getLogger(__name__)

# Build files worth showing to the LLM, in priority order. The head of
# each file carries the signal (project(), dependencies, targets).
_BUILD_FILES = (
    "go.mod",
    "Cargo.toml",
    "CMakeLists.txt",
    "configure.ac",
    "configure.in",
    "meson.build",
    "Makefile.am",
    "Makefile",
    "makefile",
    "GNUmakefile",
    "pyproject.toml",
    "setup.py",
    "package.json",
)

# Documentation that usually contains build instructions.
_DOC_FILES = (
    "README.md",
    "README.rst",
    "README",
    "README.txt",
    "INSTALL",
    "INSTALL.md",
    "BUILDING.md",
    "BUILD.md",
    "docs/BUILD.md",
)

# Per-file and total budgets (characters, ~4 chars/token).
_PER_FILE_CAP = 2500
_DOC_CAP = 2000
_TOTAL_CAP = 14000


def _read_capped(host_path: str, cap: int) -> Optional[str]:
    """Read up to ``cap`` characters of a file, or None if unreadable."""
    try:
        if not os.path.isfile(host_path):
            return None
        with open(host_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(cap + 1)
    except OSError:
        return None
    if not content.strip():
        return None
    if len(content) > cap:
        content = content[:cap] + "\n[... truncated ...]"
    return content


def _top_level_listing(host_repo: str, limit: int = 60) -> str:
    """Return a one-line-per-entry listing of the repo's top level."""
    try:
        entries = sorted(os.listdir(host_repo))
    except OSError:
        return "(repository not readable)"
    lines = []
    for name in entries[:limit]:
        suffix = "/" if os.path.isdir(os.path.join(host_repo, name)) else ""
        lines.append(f"{name}{suffix}")
    if len(entries) > limit:
        lines.append(f"[... {len(entries) - limit} more entries ...]")
    return "\n".join(lines)


def collect_build_evidence(repo_path: str) -> str:
    """Bundle the repo's real build files and docs for an LLM prompt.

    Args:
        repo_path: The repository path (container form is fine — it is
            translated to a host path for reading).

    Returns:
        A bounded markdown bundle: top-level listing, then excerpts of
        every build file that exists and of the first build-relevant
        doc that exists, in priority order until the total budget is
        spent.

    Raises:
        SandboxUnavailableError: On native, if the mirror copy fails.
    """
    host_repo = to_host_path(repo_path)
    sections: List[str] = [
        "### Top-level files\n" + _top_level_listing(host_repo)
    ]
    budget = _TOTAL_CAP - len(sections[0])

    for name in _BUILD_FILES:
        if budget <= 0:
            break
        content = _read_capped(
            os.path.join(host_repo, name), min(_PER_FILE_CAP, budget)
        )
        if content is None:
            continue
        section = f"### {name}\n```\n{content}\n```"
        sections.append(section)
        budget -= len(section)

    for name in _DOC_FILES:
        if budget <= 0:
            break
        content = _read_capped(
            os.path.join(host_repo, name), min(_DOC_CAP, budget)
        )
        if content is None:
            continue
        section = f"### {name} (excerpt)\n{content}"
        sections.append(section)
        budget -= len(section)
        break  # one doc excerpt is enough — build files carry the signal

    return "\n\n".join(sections)


# Matches gcc/clang/make/go style "path/to/file.c:123" references in
# error output. Path must look relative and source-ish to avoid pulling
# in /usr/include noise.
_ERROR_REF_RE = re.compile(
    r"(?:^|[\s'\"`(])"
    r"((?:[A-Za-z0-9_.\-]+/)*[A-Za-z0-9_.\-]+"
    r"\.(?:c|cc|cpp|cxx|h|hh|hpp|go|rs|py|s|S|asm|m4|am|ac|mk|cmake))"
    r":(\d+)",
    re.MULTILINE,
)

_EXCERPT_RADIUS = 12  # lines of context on each side of the error line
_MAX_ERROR_FILES = 3


def error_context_excerpts(error_text: str, repo_path: str) -> str:
    """Extract source context around ``file:line`` refs in an error.

    Gives the fixer the actual failing code instead of making it guess
    from the compiler message alone. Zero LLM cost.

    Args:
        error_text: The captured stderr/stdout of the failed command.
        repo_path: Repository path (container form accepted).

    Returns:
        Markdown excerpts (±12 lines around each referenced line) for
        up to 3 distinct in-repo files, or an empty string when the
        error references no readable repo files.

    Raises:
        SandboxUnavailableError: On native, if the mirror copy fails.
    """
    if not error_text:
        return ""
    host_repo = to_host_path(repo_path)
    seen: List[str] = []
    excerpts: List[str] = []

    for match in _ERROR_REF_RE.finditer(error_text):
        rel_path, line_no = match.group(1), int(match.group(2))
        if rel_path in seen:
            continue
        # Containment: skip absolute and ".." references. This check
        # does not stop a symlink in the repo that points outside it.
        if os.path.isabs(rel_path) or ".." in rel_path.split("/"):
            continue
        host_file = os.path.join(host_repo, rel_path)
        try:
            with open(host_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue
        seen.append(rel_path)
        start = max(0, line_no - 1 - _EXCERPT_RADIUS)
        end = min(len(lines), line_no + _EXCERPT_RADIUS)
        numbered = "".join(
            f"{i + 1:>5} | {lines[i]}" for i in range(start, end)
        )
        excerpts.append(
            f"### {rel_path} (around line {line_no})\n```\n{numbered}```"
        )
        if len(seen) >= _MAX_ERROR_FILES:
            break

    return "\n\n".join(excerpts)


__all__ = ["collect_build_evidence", "error_context_excerpts"]

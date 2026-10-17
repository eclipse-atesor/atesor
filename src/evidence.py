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
from typing import List, Optional, Tuple

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


def cut_text(text: str, limit: int) -> str:
    """Cut ``text`` to about ``limit`` characters at a line boundary.

    The cut falls on the last line end before ``limit``, or on the last
    word boundary for a text without one, so no word or line is broken.
    A marker says how much was left out.

    Args:
        text: The text to shorten.
        limit: The most characters to keep.

    Returns:
        ``text`` itself when it fits, else the kept part and a
        ``[... truncated ...] (N more lines left out)`` marker.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    newline = head.rfind("\n")
    if newline >= limit // 2:
        head = head[:newline]
    else:
        space = head.rfind(" ")
        if space >= limit // 2:
            head = head[:space]
    rest = text[len(head) :]
    lines = rest.count("\n")
    unit = f"{lines} more lines" if lines else f"{len(rest)} more characters"
    return f"{head}\n[... truncated ...] ({unit} left out)"


def _read_capped(host_path: str, cap: int) -> Optional[str]:
    """Read about ``cap`` characters of a file, or None if unreadable."""
    try:
        if not os.path.isfile(host_path):
            return None
        with open(host_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(cap * 4)
            more = f.read(1) != ""
    except OSError:
        return None
    if not content.strip():
        return None
    cut = cut_text(content, cap)
    if more and cut == content:
        cut += "\n[... truncated ...] (the rest of the file left out)"
    return cut


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


def top_level_listing(repo_path: str, limit: int = 60) -> str:
    """Return the repo's top-level entries, one per line (``dir/`` form).

    Args:
        repo_path: The repository path (container form is fine).
        limit: The most entries to list.

    Returns:
        The listing, with a ``[... N more entries ...]`` line when cut.

    Raises:
        SandboxUnavailableError: On native, if the mirror copy fails.
    """
    return _top_level_listing(to_host_path(repo_path), limit)


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


# Lines of a failed command's output that carry the cause. A word
# inside a hyphenated name (the package node-error-ex) does not count.
_KEY_LINE_RE = re.compile(
    r"(?i)((?<![\w-])(error|fatal|failed|killed|requires)(?![\w-])|"
    r"undefined reference|no such file|not found|cannot find|could not|"
    r"panic:|segmentation fault|permission denied|unable to|"
    r"no rule to make|\*\*\*)"
)
_MAX_KEY_LINES = 20
_KEY_LINE_CHARS = 300
_TAIL_CHARS = 2500

# The prefix the builder puts before a failed command's output:
# "Build command failed: <command> (exit <code>) - ".
_FAILURE_PREFIX_RE = re.compile(r"\A[^\n]*?\(exit -?\d+\) - ")


# make's generic line for a failed recipe; the cause is on an earlier
# line.
_MAKE_RECIPE_FAILED_RE = re.compile(
    r"^g?make(\[\d+\])?: \*\*\* \[[^\]]*\] Error \d+"
)


def first_error_line(text: str) -> str:
    """Return the line that best names the error, or "".

    The first line with the word "error" or "fatal" wins; otherwise the
    first line that matches another error pattern (make's "No rule to
    make target"). A "not found" warning before the error line does not
    win. The builder's "<command> (exit N) - " prefix and make's generic
    "*** [target] Error N" lines do not count.
    """
    lines = [
        line.strip()
        for line in strip_failure_prefix(text).splitlines()
        if line.strip() and not _MAKE_RECIPE_FAILED_RE.match(line.strip())
    ]
    for pattern in (_ERROR_WORD_RE, _KEY_LINE_RE):
        for line in lines:
            if pattern.search(line):
                return line
    return ""


def has_error_line(text: str) -> bool:
    """Return True when a line of ``text`` matches an error pattern."""
    return bool(_KEY_LINE_RE.search(text or ""))


def strip_failure_prefix(message: str) -> str:
    """Drop the builder's "<command> (exit N) - " prefix from a message."""
    return _FAILURE_PREFIX_RE.sub("", message or "", count=1)


def _tail_start(text: str, limit: int) -> int:
    """Return where the last ``limit`` characters of ``text`` start.

    The start moves to the next line start when one is near.
    """
    if len(text) <= limit:
        return 0
    start = len(text) - limit
    newline = text.find("\n", start)
    if 0 <= newline - start < 200:
        start = newline + 1
    return start


def failure_digest(
    output: str, tail_chars: int = _TAIL_CHARS
) -> Tuple[str, str]:
    """Split a failed command's output into earlier error lines and a tail.

    The tail is the end of the output. A build log can put the decisive
    error well before the end (parallel make, test runners), so the key
    lines are the distinct lines before the tail that match an error
    pattern (first 12 and last 8 when there are more than 20). No line
    is shown twice. The "<command> (exit N) - " prefix is dropped,
    because the prompt gives the command and the exit code.

    Args:
        output: The captured output (the recorded error message).
        tail_chars: The most characters of the tail to keep.

    Returns:
        ``(key_lines, tail)``. ``key_lines`` is empty when the tail
        holds the whole output or no earlier line matched. A cut tail
        starts with ``[... N earlier characters cut ...]``.
    """
    text = strip_failure_prefix(output)
    start = _tail_start(text, tail_chars)
    tail = text[start:]
    if start:
        tail = f"[... {start} earlier characters cut ...]\n{tail}"
    key: List[str] = []
    seen = set()
    for line in text[:start].splitlines():
        line = line.strip()
        if not line or line in seen or not _KEY_LINE_RE.search(line):
            continue
        seen.add(line)
        if len(line) > _KEY_LINE_CHARS:
            line = line[:_KEY_LINE_CHARS] + " [...]"
        key.append(line)
    if len(key) > _MAX_KEY_LINES:
        omitted = len(key) - _MAX_KEY_LINES
        key = (
            key[:12] + [f"[... {omitted} more matching lines ...]"] + key[-8:]
        )
    return "\n".join(key), tail


# Matches gcc/clang/make/go style "path/to/file.c:123" references in
# error output. Absolute paths into the repository are made relative
# first (see _repo_relative); /usr/include and other system paths stay
# out.
_ERROR_REF_RE = re.compile(
    r"(?:^|[\s'\"`(])"
    r"((?:[A-Za-z0-9_.\-]+/)*[A-Za-z0-9_.\-]+"
    r"\.(?:c|cc|cpp|cxx|h|hh|hpp|go|rs|py|s|S|asm|m4|am|ac|mk|cmake))"
    r":(\d+)",
    re.MULTILINE,
)

_EXCERPT_RADIUS = 12  # lines of context on each side of the error line
_MAX_ERROR_FILES = 3

# A line that reports an error; its file references are read first.
# A word inside a hyphenated name (node-error-ex) does not count.
_ERROR_WORD_RE = re.compile(r"(?i)(?<![\w-])(error|fatal)(?![\w-])")


def _repo_relative(error_text: str, repo_path: str, host_repo: str) -> str:
    """Rewrite absolute paths into the repository as repo-relative paths.

    CMake, Go and most compilers print absolute paths
    (``/workspace/repos/x/src/a.c:12``); the reference pattern only
    reads relative ones.
    """
    for root in {repo_path.rstrip("/"), host_repo.rstrip("/")}:
        if root:
            error_text = error_text.replace(root + "/", "")
    return error_text


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
    real_repo = os.path.realpath(host_repo)
    text = _repo_relative(error_text, repo_path, host_repo)
    error_lines = text.splitlines()
    # References on error lines first: warnings come before the error
    # in a build log, so the first reference is often only a warning.
    ordered = [
        ln for ln in error_lines if _ERROR_WORD_RE.search(ln)
    ] + error_lines
    seen: List[str] = []
    excerpts: List[str] = []

    for match in (m for ln in ordered for m in _ERROR_REF_RE.finditer(ln)):
        rel_path, line_no = match.group(1), int(match.group(2))
        # An out-of-tree build (`cd build && ../configure`) prints
        # `../src/a.c`; read it from the repository root.
        while rel_path.startswith("../"):
            rel_path = rel_path[3:]
        if rel_path in seen or os.path.isabs(rel_path):
            continue
        host_file = os.path.join(host_repo, rel_path)
        # Containment: the resolved file must stay inside the repo, also
        # through a symlink.
        if not os.path.realpath(host_file).startswith(real_repo + os.sep):
            continue
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


__all__ = [
    "collect_build_evidence",
    "error_context_excerpts",
    "cut_text",
    "failure_digest",
    "first_error_line",
    "has_error_line",
    "strip_failure_prefix",
    "top_level_listing",
]

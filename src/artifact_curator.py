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

"""Rank build artifacts as primary or secondary, and drop the noise.

The ArtifactScanner returns *every* executable / library / static-lib
in the build tree, which is noisy: CMake leaves compiler-probe binaries
(CMakeFiles/CompilerIdC/a.out, CheckTypeSize/OFF64_T.bin), autotools
leaves libtool shims (.libs/lt-*), and most projects ship test/example
binaries alongside the real deliverables. End users only care about the
*primary* outputs (final binary, final shared/static library, install
prefix).

This module asks the LLM to classify each path into:
    - primary: the package's headline deliverable(s).
    - secondary: still useful (tests, examples); rank below primary.
    - drop: build-system internals / intermediates / probe artifacts.

Failure modes (LLM unavailable, rate-limited, malformed JSON) fall back
to a deterministic rule-based filter, so the curator is always safe to
call.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Rule-based filter (fallback + sanitizer run BEFORE the LLM sees the list)

# Path substrings that are always build-system internals and should never be
# surfaced to the user, even if the LLM gets it wrong.
_HARD_NOISE_SUBSTRINGS = (
    "/CMakeFiles/",  # cmake internals (CompilerIdC, CheckTypeSize, etc.)
    "/_deps/",  # cmake FetchContent / ExternalProject scratch
    "/.libs/",  # libtool shims
    "/conftest",  # autoconf feature-probe binaries
    "/cmake-build-",  # CLion-style build dirs left in tree
    "/CompilerId",  # cmake compiler-id probes
    "/CheckTypeSize/",  # cmake type-size probes
    "/CheckFunctionExists",  # cmake function-exists probes
    "/CheckIncludeFile",
    "/TryCompile-",
    "/meson-private/",  # meson internals
    "/meson-info/",
    "/meson-logs/",
    "/build-aux/",  # autotools helpers
    "/autom4te.cache/",
)

# Path tokens that hint at non-primary status — kept as secondary, not dropped.
_SECONDARY_TOKENS = (
    "/test/",
    "/tests/",
    "/testing/",
    "/check/",
    "/checks/",
    "/example/",
    "/examples/",
    "/sample/",
    "/samples/",
    "/demo/",
    "/demos/",
    "/benchmark/",
    "/benchmarks/",
    "/bench/",
    "/tools/",  # often dev tools rather than the headline binary
    "/contrib/",
)


def _looks_like_noise(path: str) -> bool:
    return any(tok in path for tok in _HARD_NOISE_SUBSTRINGS)


def _looks_like_secondary(path: str) -> bool:
    p = path.lower()
    return any(tok in p for tok in _SECONDARY_TOKENS)


def _rule_based_curate(
    repo_name: str, raw_artifacts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Curate raw artifacts using deterministic rules.

    Drops obvious build-system noise, marks test/example binaries as
    secondary, and ranks everything else as primary. Libraries (type
    ``library_*``) and a binary named after the package are always
    primary. Output is primary-first.

    Args:
        repo_name: Package name, used to detect the primary binary.
        raw_artifacts: ArtifactScanner output entries.

    Returns:
        Artifacts annotated with a ``role`` field, primary-first.
    """
    primary: List[Dict[str, Any]] = []
    secondary: List[Dict[str, Any]] = []

    repo_tag = (repo_name or "").lower()

    for art in raw_artifacts:
        path = art.get("filepath") or art.get("path") or ""
        if not path or _looks_like_noise(path):
            continue

        a_type = art.get("type", "binary")
        is_library = a_type.startswith("library")
        bn = os.path.basename(path).lower()

        # A binary whose name matches the repo is almost certainly the
        # deliverable.
        is_named_match = bool(repo_tag) and (
            bn == repo_tag or bn.startswith(repo_tag + ".")
        )

        if is_library or is_named_match:
            primary.append({**art, "role": "primary"})
        elif _looks_like_secondary(path):
            secondary.append({**art, "role": "secondary"})
        else:
            primary.append({**art, "role": "primary"})

    return primary + secondary


# LLM-driven curator

_CURATOR_PROMPT = (
    "You are an artifact curator for a multi-distro RISC-V porting"
    " agent.\n"
    "Given the raw scan of a build tree, decide which paths a USER"
    " actually cares about.\n"
    "\n"
    "CLASSIFY EACH NUMBERED PATH AS:\n"
    '- "primary"   : the headline deliverable (final binary matching'
    " the package name, the main shared/static library, the installed"
    " CLI)\n"
    '- "secondary" : useful but not headline (test runners, example'
    " programs, dev tools)\n"
    '- "drop"      : build-system internals or probes (CMakeFiles/*,'
    " _deps/*, .libs/*, conftest, CompilerId*, CheckTypeSize,"
    " meson-private, autom4te.cache, libtool shims)\n"
    "\n"
    "RULES:\n"
    "- ALWAYS drop anything under CMakeFiles/, _deps/, .libs/,"
    " conftest, *CompilerId*, *CheckTypeSize*, meson-private/,"
    " meson-info/, autom4te.cache/, build-aux/\n"
    "- A binary whose basename equals the package name (or starts with"
    " it) is primary\n"
    "- Final libraries (libz.so, libz.a, libfoo.so.N.M) are primary\n"
    "- Binaries under test/ tests/ examples/ samples/ benchmark/ are"
    " secondary\n"
    "- If unsure between primary and secondary, prefer primary\n"
    "- Output STRICT JSON only, no prose\n"
    "\n"
    "PACKAGE: {repo_name}\n"
    "BUILD SYSTEM: {build_system}\n"
    "\n"
    "PATHS:\n"
    "{numbered_paths}\n"
    "\n"
    "OUTPUT JSON SCHEMA:\n"
    '{{"primary": [<ids>], "secondary": [<ids>], "drop": [<ids>]}}\n'
    "\n"
    "Every id from 1..{n} MUST appear in exactly one list."
)


def _format_numbered(raw_artifacts: List[Dict[str, Any]]) -> str:
    """Return the artifacts as numbered ``[type] path`` prompt lines."""
    lines = []
    for i, art in enumerate(raw_artifacts, start=1):
        path = art.get("filepath") or art.get("path") or ""
        a_type = art.get("type", "?")
        lines.append(f"{i}. [{a_type}] {path}")
    return "\n".join(lines)


def _parse_curator_json(
    text: str, n: int
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    """Extract {primary, secondary, drop} index lists from raw LLM output."""
    if not text:
        return None
    # Models often wrap the JSON in prose despite the strict-output
    # rule; take the span from the first "{" to the last "}".
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None

    def _coerce(key: str) -> List[int]:
        vals = data.get(key, []) or []
        out = []
        for v in vals:
            try:
                idx = int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= n:
                out.append(idx)
        return out

    return _coerce("primary"), _coerce("secondary"), _coerce("drop")


def curate_artifacts(
    raw_artifacts: List[Dict[str, Any]],
    repo_name: str,
    build_system: str = "unknown",
    llm: Any = None,
) -> List[Dict[str, Any]]:
    """Curate the raw artifact scan into a clean, user-facing list.

    Args:
        raw_artifacts: ArtifactScanner output (``{filepath, type, ...}``
            entries).
        repo_name: Package name, used as a hint for primary-binary
            detection.
        build_system: The detected build system, such as cmake, meson,
            make, autotools, go or cargo. The prompt shows it as a
            hint.
        llm: An instance returned by ``get_model_for_role()``. When
            ``None``, the deterministic rule-based curator is used.

    Returns:
        A list of artifact dicts (same shape as the input) with an added
        ``role`` field ("primary" or "secondary"), ordered primary-first.
        Noise paths are removed.
    """
    if not raw_artifacts:
        return []

    # Strip hard-noise up front so the LLM is not asked about CMake internals
    pre_filtered = [
        a
        for a in raw_artifacts
        if not _looks_like_noise(a.get("filepath") or a.get("path") or "")
    ]
    if not pre_filtered:
        logger.info(
            "Artifact curator: all %d entries were hard-noise;"
            " nothing to keep.",
            len(raw_artifacts),
        )
        return []

    if llm is None:
        logger.info(
            "Artifact curator: using rule-based curation (no LLM provided)."
        )
        return _rule_based_curate(repo_name, pre_filtered)

    n = len(pre_filtered)
    prompt = _CURATOR_PROMPT.format(
        repo_name=repo_name or "unknown",
        build_system=build_system or "unknown",
        numbered_paths=_format_numbered(pre_filtered),
        n=n,
    )

    try:
        # Lazy import so this module has no hard dependency on langchain when
        # running in pure rule-based mode (e.g. unit tests). Route through
        # invoke_llm (hard timeout — a hung provider must not stall the
        # finish path) and log_llm_call (audit trail), per project rules.
        from langchain_core.messages import HumanMessage

        from .graph import invoke_llm
        from .llm_helpers import (
            extract_content,
            response_cost,
            response_usage,
        )
        from .llm_logger import log_llm_call

        response = invoke_llm(llm, [HumanMessage(content=prompt)], timeout=60)
        # A list of parts (Claude thinking + text) keeps only the text.
        text = extract_content(getattr(response, "content", response))
        tokens_in, tokens_out = response_usage(response)
        log_llm_call(
            agent_role="curator",
            prompt=prompt,
            response=text,
            # ChatAnthropic names its model in `model`, not `model_name`.
            model=getattr(llm, "model_name", None)
            or getattr(llm, "model", "unknown"),
            cost_usd=response_cost(llm, response),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata={"repo": repo_name, "phase": "curate_artifacts"},
        )
    except Exception as e:
        logger.warning(
            "Artifact curator: LLM call failed (%s); falling back to rules.", e
        )
        return _rule_based_curate(repo_name, pre_filtered)

    parsed = _parse_curator_json(text, n)
    if not parsed:
        logger.warning(
            "Artifact curator: could not parse LLM JSON;"
            " falling back to rules."
        )
        return _rule_based_curate(repo_name, pre_filtered)

    primary_ids, secondary_ids, drop_ids = parsed

    # Reconcile coverage: anything the LLM forgot is treated as secondary so we
    # never silently lose a real deliverable.
    classified = set(primary_ids) | set(secondary_ids) | set(drop_ids)
    missing = [i for i in range(1, n + 1) if i not in classified]
    if missing:
        logger.info(
            "Artifact curator: LLM omitted %d ids;"
            " defaulting them to secondary.",
            len(missing),
        )
        secondary_ids = sorted(set(secondary_ids) | set(missing))

    drop_set = set(drop_ids)
    primary = [
        {**pre_filtered[i - 1], "role": "primary"}
        for i in primary_ids
        if 1 <= i <= n and i not in drop_set
    ]
    secondary = [
        {**pre_filtered[i - 1], "role": "secondary"}
        for i in secondary_ids
        if 1 <= i <= n
        and i not in drop_set
        and i not in {p for p in primary_ids}
    ]

    # If LLM dropped EVERYTHING, that's wrong (we already pre-filtered noise).
    # Fall back to rule-based so the user still sees something.
    if not primary and not secondary:
        logger.warning(
            "Artifact curator: LLM dropped all candidates; using rule-based."
        )
        return _rule_based_curate(repo_name, pre_filtered)

    curated = primary + secondary
    logger.info(
        "Artifact curator: %d raw -> %d kept"
        " (%d primary, %d secondary), %d dropped",
        len(raw_artifacts),
        len(curated),
        len(primary),
        len(secondary),
        len(raw_artifacts) - len(curated),
    )
    return curated

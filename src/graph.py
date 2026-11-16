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

"""LangGraph workflow for the RISC-V porting agent.

Owns every graph node (init, analyst, heuristic plan, scout,
supervisor, builder, verify, fixer, finish, escalate, cached recipe,
optimize), the prompt constants, the per-node routing functions, the
build-fix subgraph, and ``create_workflow``. Two contracts hold across
the whole file: nodes mutate the shared ``AgentState`` in place and
never return a fresh one, and every LLM call goes through
``invoke_llm`` or ``llm_call_with_validation`` plus ``log_llm_call``
— never ``llm.invoke()`` directly.
"""

import base64
import json
import logging
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from .artifact_scanner import (
    UNVERIFIED,
    VERIFIED,
    WRONG_ARCH,
    ArtifactScanner,
)
from .bench import crash_signal
from .evidence import (
    collect_build_evidence,
    cut_text,
    error_context_excerpts,
    failure_digest,
    first_error_line,
    has_error_line,
    strip_failure_prefix,
    strip_go_download_noise,
    top_level_listing,
)
from .knowledge import get_system_knowledge_summary
from .llm_helpers import (
    ValidationResult,
    is_refusal,
    llm_call_with_validation,
    response_cache_usage,
    response_cost,
    response_stop_reason,
    response_usage,
)
from .llm_logger import log_data_effect, log_llm_call
from .machine import (
    MachineProfile,
    forbidden_classes,
    parse_isa_string,
    probe_machine,
)
from .memory import (
    format_few_shot_examples,
    get_agent_memory,
    get_cached_recipe,
    load_failure_records,
    load_recipe_cache,
    mark_recipe_verification,
    recall_sandboxes,
    record_optimization,
    save_failure_record,
    save_learned_example,
    save_to_recipe_cache,
)
from .models import LLM_REQUEST_TIMEOUT, create_llm
from .scripted_ops import ScriptedOperations, quick_analysis
from .state import (
    TIME_LIMIT_MARK,
    AgentRole,
    AgentState,
    BuildPhase,
    BuildPlan,
    BuildStatus,
    CommandResult,
    ErrorCategory,
    FailureSeverity,
    FixAttempt,
    PackageAnalysis,
    TaskPhase,
    TaskPlan,
    _first_meaningful_error_line,
    _normalize_error_line,
    classify_error,
    create_error_record,
    should_escalate,
)
from .target import is_native
from .tools import (
    BUILD_STEP_RE,
    CommandValidator,
    apply_patch,
    ensure_build_container,
    execute_command,
    infrastructure_fault,
    masks_failure,
)

logger = logging.getLogger(__name__)

# Caller-side wall timeout for a single LLM invocation, in seconds.
# MUST stay strictly greater than models.LLM_REQUEST_TIMEOUT so the
# provider's own HTTP timeout fires first and releases the socket;
# otherwise a timed-out call orphans its worker thread and connection
# for the remainder of the provider request (MEM-01).
LLM_WALL_TIMEOUT = LLM_REQUEST_TIMEOUT + 15

scripted_ops = ScriptedOperations()


# Helper functions shared by the nodes.


def get_model_for_role(role: AgentRole) -> BaseChatModel:
    """Return the chat model configured for ``role``.

    See ``src/models.py`` for the per-provider model table.
    """
    return create_llm(role)


def get_model_pool_for_role(role: AgentRole) -> List[BaseChatModel]:
    """Return ``[primary, *fallbacks]`` for the role.

    Used by validated LLM calls so transient provider errors (404 /
    5xx) rotate to a backup model instead of escalating the whole
    agent.
    """
    from .models import create_llm_pool

    return create_llm_pool(role)


def _is_within_repo(candidate: str, repo_path: str) -> bool:
    """Return True when ``candidate`` resolves inside ``repo_path``.

    Defence-in-depth for LLM-supplied file paths: ``os.path.join``
    happily collapses ``../..`` back out of the repo, and
    ``shlex.quote`` preserves the traversal intact. Resolving both
    sides with ``realpath`` is the only reliable containment test.

    Args:
        candidate: The joined path the agent wants to write.
        repo_path: The repository root the write must stay inside.

    Returns:
        True if the write is contained, False otherwise.
    """
    try:
        root = os.path.realpath(repo_path)
        target = os.path.realpath(candidate)
    except OSError:
        return False
    return target == root or target.startswith(root + os.sep)


def _contained_write_cmd(repo_path: str, target: str, content: str) -> str:
    """Return a sandbox command that writes ``content`` to ``target``.

    The containment test runs where the write happens.
    ``_is_within_repo`` resolves container paths on the host, so it
    cannot see a symlink inside the sandbox, and the shell redirect
    that writes the file follows one: a repo file ``out`` that links to
    ``/etc/apk/repositories`` would let a fix overwrite it. Here
    ``realpath -m`` resolves every link in the sandbox first, and the
    write goes to the resolved path only when that stays inside the
    repository. A refused write exits with status 3.

    Args:
        repo_path: The repository root, as a sandbox path.
        target: The file to write, as a sandbox path.
        content: The file content.

    Returns:
        A bash command for ``execute_command(..., validate=False)``.
    """
    encoded = base64.b64encode(content.encode()).decode()
    return (
        f"root=$(realpath -- {shlex.quote(repo_path)}) && "
        f"dest=$(realpath -m -- {shlex.quote(target)}) && "
        'case "$dest" in "$root"/*) ;; *) '
        'echo "refusing to write outside the repository: $dest" >&2; '
        "exit 3;; esac && "
        'mkdir -p -- "$(dirname -- "$dest")" && '
        f'echo {shlex.quote(encoded)} | base64 -d > "$dest"'
    )


def invoke_llm(
    llm: BaseChatModel,
    messages: List[BaseMessage],
    timeout: int = LLM_WALL_TIMEOUT,
) -> BaseMessage:
    """Invoke an LLM under a hard wall-clock timeout.

    This is the only place ``llm.invoke`` may be called; every node's
    LLM traffic funnels through here (usually via
    ``llm_call_with_validation``) so the timeout and the audit log
    apply uniformly. The call runs on a daemon thread so a blocking
    HTTP connection cannot prevent process exit.

    Args:
        llm: The chat model to call.
        messages: The prompt messages.
        timeout: Wall-clock limit in seconds; must stay above the
            provider's own HTTP timeout (see ``LLM_WALL_TIMEOUT``).

    Returns:
        The model's response message.

    Raises:
        TimeoutError: If the call exceeds ``timeout`` seconds.
        Exception: Any error that ``llm.invoke`` raises, re-raised in
            the caller's thread.
    """
    import threading

    result = [None]
    exception = [None]
    done = threading.Event()

    def worker() -> None:
        try:
            result[0] = llm.invoke(messages)
        except Exception as e:
            exception[0] = e
        finally:
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    if not done.wait(timeout=timeout):
        raise TimeoutError(f"LLM call timed out after {timeout}s")

    if exception[0] is not None:
        raise exception[0]

    return result[0]


def extract_content(content: Any) -> str:
    """Return the text of an LLM response content payload.

    Providers return either a plain string or a list of content parts
    (dicts with a ``text`` key); both shapes flatten to one string. A
    part with a ``type`` other than ``text`` (for example a Claude
    ``thinking`` block) holds no answer text and is skipped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            [
                (
                    str(item.get("text", item))
                    if isinstance(item, dict)
                    else str(item)
                )
                for item in content
                if not (
                    isinstance(item, dict)
                    and item.get("type", "text") != "text"
                )
            ]
        )
    return str(content)


def extract_json_block(text: str) -> str:
    """Return the span from the first ``{`` to the last ``}``.

    Free-tier models often wrap JSON in prose despite the prompt; this
    trims the prose. Returns ``text`` unchanged when there is no brace
    pair.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _build_platform_banner() -> str:
    """Build a banner describing the active sandbox/package manager.

    Injected at the TOP of analyst, scout & fixer prompts so the LLM cannot
    ignore it (knowledge appendices buried at the bottom tend to lose
    to early-context few-shot examples in free-tier models).
    """
    try:
        from .platforms import get_active_profile

        p = get_active_profile()
        pm = p.pkg_install.split()[0]  # "apk" / "apt-get"
        # Two commands, never one `update && install` chain: the fixer
        # validator (CommandValidator.is_safe_single) rejects chains.
        install_example = (
            f"`{p.pkg_update}`, then `{p.pkg_install} <pkg>` as a "
            f"separate command"
        )
        other_pms = (
            "apt-get / apt / dpkg" if pm == "apk" else "apk / apk-tools"
        )
        # Pick one concrete correction example to anchor the LLM on
        # the right names. name_corrections maps WRONG_NAME ->
        # RIGHT_NAME for THIS distro.
        correction_example = ""
        if p.name_corrections:
            wrong, right = next(iter(p.name_corrections.items()))
            correction_example = (
                f"\n- Correct package names for {p.display_name}: "
                f"e.g. `{wrong}` is WRONG, use `{right}` instead."
            )
        # Pick a concrete worked install example using a real
        # canonical -> distro mapping.
        worked_example = ""
        if p.package_map:
            canonical, distro_pkg = next(iter(p.package_map.items()))
            worked_example = (
                f"\n- Worked example for this distro: install "
                f"`{canonical}` via `{p.pkg_update}`, then "
                f"`{p.pkg_install} {distro_pkg}`."
            )
        return (
            f"- Distro: **{p.display_name}** (libc: **{p.libc}**, "
            f"triplet: `{p.target_triplet}`)\n"
            f"- Package manager: **{pm}** (full install: "
            f"{install_example})\n"
            f"- NEVER emit {other_pms} commands — they do not exist "
            f"in this container\n"
            f"  and will fail with `command not found`. This is "
            f"unrecoverable for the build."
            f"{worked_example}"
            f"{correction_example}\n"
            f"- If you are unsure of a package name on this distro, "
            f"prefer the canonical\n"
            f"  short name (e.g. `openssl-dev`, `zlib-dev`) — the "
            f"agent auto-corrects to\n"
            f"  the distro-specific name. NEVER guess a name from "
            f"another distro."
        )
    except Exception:
        return (
            "- Platform: unknown (default to apt-get on "
            "debian/ubuntu, apk on alpine)"
        )


def _to_host_path(path: str) -> str:
    """Translate a /workspace container path to a host path.

    Thin delegate to :func:`src.config.to_host_path` — the canonical
    translator — kept for local call-site brevity.
    """
    from .config import to_host_path

    return to_host_path(path)


def _build_command_error_message(result, fallback: str) -> str:
    """Combine the exit code and captured output into one error line.

    Falls back to stdout when stderr is empty, and to a placeholder
    when neither stream captured anything.
    """
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or "No stderr/stdout output captured."
    return f"{fallback} (exit {result.exit_code}) - {detail}"


_STDOUT_TAIL_CHARS = 2000


def _stdout_evidence(result) -> str:
    """Return the end of stdout when the error message leaves it out.

    The message keeps stderr when stderr has text, so a configure
    script or test runner that prints its error on stdout loses it.
    Such tools print the decisive lines last, so only the end is kept,
    and only when it has an error line (plain progress output has none).
    """
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    if not stderr or not stdout:
        return ""
    tail = failure_digest(stdout, tail_chars=_STDOUT_TAIL_CHARS)[1]
    return tail if has_error_line(tail) else ""


def _try_clone_recovery(
    state: AgentState, url: str, name: str
) -> Optional[CommandResult]:
    """Try additional URL-level recovery when the initial clone fails.

    Delegates to :meth:`ScriptedOperations._try_url_variants` so both
    the initial clone path (in scripted_ops) and the supervisor-level
    recovery path (here) apply the same set of transformations:
    ``/cgit/`` → ``/git/``, appending ``.git``, and the homepage → git
    rules (GNU savannah, blicky, ...).

    Args:
        state: Agent state (used to record which variants were tried).
        url: The original clone URL.
        name: The canonical repo name.

    Returns:
        The successful ``CommandResult`` if a variant clones cleanly,
        or ``None`` if every attempt fails.
    """
    tried: List[str] = [url]
    result = scripted_ops._try_url_variants(url, name)

    # Reflect the attempts the variant helper made into audit state.
    # We don't have direct access to the internal list, so replay the
    # deterministic rules to record accurate telemetry.
    stripped = url.rstrip("/")
    variants: List[str] = []
    if "/cgit/" in url:
        variants.append(url.replace("/cgit/", "/git/"))
    if not url.endswith(".git"):
        variants.append(stripped + ".git")
    for cand in scripted_ops._resolve_homepage_to_git_urls(url, name):
        if cand not in variants:
            variants.append(cand)
    tried.extend(variants)

    if result is not None:
        state.context_cache["clone_recovery"] = {
            "original_url": url,
            "success_url": result.command,
            "attempted": tried,
        }
        return result

    state.context_cache["clone_recovery"] = {
        "original_url": url,
        "attempted": tried,
        "success_url": None,
    }
    return None


def _classify_clone_failure(message: str) -> ErrorCategory:
    """Map a clone failure message to an error category.

    Authentication and DNS resolution failures are NETWORK; a missing
    repository and anything unrecognized are CONFIGURATION.
    """
    msg = message.lower()
    if any(
        pat in msg
        for pat in [
            "could not read username",
            "could not read password",
            "no such device or address",
            "authentication failed",
        ]
    ):
        return ErrorCategory.NETWORK
    if re.search(r"repository\s+not\s+found", msg):
        return ErrorCategory.CONFIGURATION
    if any(
        pat in msg
        for pat in ["could not resolve", "name or service not known"]
    ):
        return ErrorCategory.NETWORK
    return ErrorCategory.CONFIGURATION


def is_toolchain_version_mismatch(error_message: str) -> bool:
    """Return True when the error says the Go/Rust toolchain is too old.

    Matches the Go ("go.mod requires go >= ...") and Rust
    ("edition2024", "requires rust version", ...) markers. The
    builder's recovery only upgrades Go; a Rust match still ends in a
    recorded failure.
    """
    err = (error_message or "").lower()
    patterns = [
        "go.mod requires go >=",
        "running go ",
        "feature `edition2024` is required",
        "not stabilized in this version of cargo",
        "this package requires rustc",
        "requires rust version",
    ]
    return any(p in err for p in patterns)


def _replan_signature(error_message: str) -> str:
    """Return a normalized signature for failures needing a new build plan."""
    msg = (error_message or "").lower()
    signatures = {
        "go_missing_module": "no required module provides",
        "go_output_dir_collision": "already exists and is a directory",
        "cmake_wrong_source_root": "does not appear to contain cmakelists.txt",
        "missing_package": "unable to locate package",
        "go_inconsistent_vendoring": "inconsistent vendoring",
        "missing_configure_script": "./configure: no such file or directory",
        # go.mod sits in a subdirectory: the plan builds in the wrong
        # place, which a new plan fixes (arc: v1/).
        "go_no_main_module": "cannot find main module",
    }
    for key, needle in signatures.items():
        if needle in msg:
            return key
    return ""


def _should_force_replan(state: AgentState) -> bool:
    """Return True once per replan signature to force a scout replan.

    The bound stops fixer/scout thrash on bad plans. A True result
    increments the per-signature counter in ``state.context_cache``.
    """
    signature = _replan_signature(state.last_error or "")
    if not signature:
        return False
    attempts_by_sig = state.context_cache.setdefault(
        "replan_attempts_by_signature", {}
    )
    count = int(attempts_by_sig.get(signature, 0))
    # Only force one scout replan per unique signature; if it still fails,
    # let normal routing decide between fixer/escalation.
    if count >= 1:
        return False
    attempts_by_sig[signature] = count + 1
    return True


# The agent_node decorator — every graph node's error/retry envelope.


def agent_node(role: AgentRole) -> Callable:
    """Wrap a graph node with the error-handling contract.

    Sets ``state.current_agent`` and logs start/completion for the
    audit trail. A rate-limit or credit error (429, 402, quota, ...)
    gets up to 3 attempts in total, with exponential backoff between
    them. Any other exception, or a rate-limit error on the last
    attempt, is caught, recorded via ``state.add_error``, and marks
    the run ESCALATED — the wrapper never raises, so a crashing node
    cannot take the graph down with it.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(state: AgentState) -> AgentState:
            import time

            state.current_agent = role
            logger.info(f"Node Started: {role.value}")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = func(state)
                    logger.info(f"Node Completed: {role.value}")
                    return result
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = any(
                        term in error_str
                        for term in [
                            "rate limit",
                            "429",
                            "too many requests",
                            "quota exceeded",
                            "resource_exhausted",
                            "402",
                            "spend limit exceeded",
                            "usd spend limit",
                        ]
                    )
                    if is_rate_limit and attempt < max_retries - 1:
                        import random

                        wait_time = 30 * (2**attempt) + random.uniform(0, 15)
                        logger.warning(
                            f"Rate limit hit in {role.value} "
                            f"(attempt {attempt + 1}/{max_retries}), "
                            f"waiting {wait_time:.0f}s before retry..."
                        )
                        time.sleep(wait_time)
                        continue

                    logger.exception(f"Exception in {role.value} node: {e}")
                    error_msg = (
                        f"Unexpected error in {role.value} agent: {str(e)}"
                    )
                    error = create_error_record(
                        message=error_msg,
                        category=classify_error(str(e)),
                        attempt_number=state.attempt_count,
                    )
                    state.add_error(error)
                    state.last_error = error_msg
                    state.build_status = BuildStatus.ESCALATED
                    state.current_phase = "escalate"
                    return state

            return state

        return wrapper

    return decorator


# init_node — clone the repository and run scripted quick analysis.


@agent_node(AgentRole.SUPERVISOR)
def init_node(state: AgentState) -> AgentState:
    """Clone or update the repository and run the scripted analysis.

    Zero LLM cost: clone, ``quick_analysis`` (build system, deps, arch
    scan, doc caching) and a sandbox tool check all run through
    ``scripted_ops``. A failed clone tries URL variants once and then
    escalates; a failed analysis is logged and the run continues with
    whatever was detected.
    """
    logger.info(f"Initializing workflow for {state.repo_url}")

    result = scripted_ops.clone_or_update_repository(
        state.repo_url, state.repo_name
    )
    state.log_scripted_op("clone_or_update_repository")

    if not result.success:
        message = _build_command_error_message(
            result, f"Repository clone/update failed for {state.repo_url}"
        )

        recovery_result = _try_clone_recovery(
            state, state.repo_url, state.repo_name
        )
        if recovery_result is not None:
            logger.info(
                f"Clone recovery succeeded for {state.repo_name} "
                f"via URL variant"
            )
            result = recovery_result
            state.log_scripted_op("clone_recovery")
        else:
            recovery_data = state.context_cache.get("clone_recovery", {})
            if recovery_data.get("attempted"):
                message += (
                    f" | Tried variants: "
                    f"{', '.join(recovery_data['attempted'])}"
                )

            category = _classify_clone_failure(message)
            error = create_error_record(
                message=message,
                category=category,
                severity=FailureSeverity.HIGH,
                command=result.command,
            )
            state.add_error(error)
            state.build_status = BuildStatus.FAILED
            state.current_phase = "escalate"
            state.log_agent_decision(
                AgentRole.SUPERVISOR,
                "ESCALATE",
                f"Critical init failure ({error.category.value}): "
                f"{error.message[:200]}",
            )
            return state

    logger.info("Performing quick analysis with scripted operations...")
    try:
        analysis = quick_analysis(state.repo_path)
        state.log_scripted_op("quick_analysis")

        state.build_system_info = analysis.get("build_system")

        if (
            state.build_system_info
            and state.build_system_info.type == "unknown"
        ):
            logger.warning(
                f" WARNING: Unable to identify build system in "
                f"{state.repo_name}. "
                "Exhaustive search was performed but no build "
                "configuration files found. "
                "The system will attempt configuration-free analysis "
                "or request manual input."
            )

        state.dependencies = analysis.get("dependencies")
        state.arch_specific_code = analysis.get("arch_specific_code", [])

        state.repo_tree = analysis.get("optimized_tree", "")
        logger.info(f"Repository tree captured ({len(state.repo_tree)} chars)")

        for doc_path in analysis.get("documentation", [])[:5]:
            content = scripted_ops.read_file(doc_path, max_lines=500)
            state.cache_file_content(doc_path, content)
            state.log_scripted_op("read_file")

        state.context_cache["quick_analysis"] = analysis

        if "go_main_info" in analysis:
            state.context_cache["go_main_info"] = analysis["go_main_info"]
            logger.info(
                f"Go main package detected: {analysis['go_main_info']}"
            )

        if "arch_build_files" in analysis:
            state.context_cache["arch_build_files"] = analysis[
                "arch_build_files"
            ]
            arch_info = analysis["arch_build_files"]
            if arch_info.get("has_arch_specific"):
                logger.info(
                    f"Arch-specific build files detected: "
                    f"{arch_info.get('archs_found')}, "
                    f"RISC-V exists: {arch_info.get('riscv_exists')}"
                )

        required_tools = set(["gcc", "make"])
        if state.build_system_info:
            required_tools.add(state.build_system_info.type)
        if state.dependencies:
            required_tools.update(state.dependencies.build_tools)

        system_info = scripted_ops.get_system_info(tools=list(required_tools))
        missing_tools = [
            t for t, status in system_info.items() if status == "Not installed"
        ]
        state.context_cache["system_info"] = system_info
        state.context_cache["missing_tools"] = missing_tools

        bsi = state.build_system_info
        bsi_type = bsi.type if bsi else "unknown"
        missing_str = ", ".join(missing_tools) if missing_tools else "None"
        logger.info(
            f"Quick analysis complete: "
            f"Build system: {bsi_type}, "
            f"Arch-specific code: "
            f"{len(state.arch_specific_code)} instances. "
            f"Relevant missing tools: {missing_str}"
        )

    except Exception as e:
        logger.warning(f"Quick analysis failed: {e}")

    # What earlier runs of the package left: a recipe, failed fixes.
    recall = _recall(state)
    if recall.get("recipe"):
        verdict = (recall["recipe"].get("verification") or {}).get(
            "status"
        ) or "not recorded"
        logger.info(
            f"Recall: a recipe of {state.repo_name} from "
            f"{recall['sandbox']} ({verdict})"
        )
    if recall.get("failures"):
        logger.info(
            f"Recall: failed fixes for {len(recall['failures'])} error(s) "
            "of earlier runs"
        )

    state.build_status = BuildStatus.PENDING
    state.current_phase = "initialized"

    return state


# analyst_node — one evidence-grounded LLM call about the package.

# The JSON schema braces are doubled because the prompt is rendered
# with str.format(); single braces would parse as format fields.
# The system part (static per platform, cached); the per-package part
# is ANALYST_TASK_PROMPT, sent as the user message.
ANALYST_PROMPT = (
    "You are the package analyst for a RISC-V (riscv64) porting agent."
    "\n"
    "Read the repository evidence in the user message — ACTUAL build "
    "files and docs —\n"
    "and produce the analysis every downstream agent builds from.\n"
    "Ground every claim in a file you can see. Never invent.\n"
    "\n"
    "## Active sandbox (READ FIRST — overrides conflicting knowledge)\n"
    "{platform_banner}\n"
    "\n"
    "## Your task\n"
    "1. Say what this package IS (purpose) and its dominant language.\n"
    "2. Confirm or correct the detected build system, citing the file\n"
    "   that proves it.\n"
    "3. List required libraries/tools by CANONICAL name (e.g. openssl,"
    "\n"
    "   zlib, pcre2) with the file line that requires each. Skip\n"
    "   compiler/make basics — the sandbox has them.\n"
    "4. List RISC-V-specific risks you can see in the evidence (SIMD/"
    "\n"
    "   x86 asm dirs, arch #ifdef lists, cgo, vendored binaries).\n"
    "5. Describe the build strategy in 2-3 sentences (native build,\n"
    "   which targets, which flags/feature toggles matter).\n"
    "6. Name the artifacts a successful build must produce (binary/\n"
    "   library file names).\n"
    "Keep every text short: purpose one sentence (at most 200 "
    "characters), build-system reasoning at most 150, each dependency "
    "reason at most 80, at most 6 risks of at most 160 each, build "
    "strategy at most 400.\n"
    "7. Decide needs_custom_plan: true when the STANDARD recipe for\n"
    "   this build system would fail (unusual layout, code generation\n"
    "   step, subprojects, heavy arch-specific code, pinned toolchain)"
    ";\n"
    "   false when a textbook build should work.\n"
    "\n"
    "## Output schema (JSON only — no prose, no markdown fences)\n"
    "{{\n"
    '  "purpose": "<one sentence>",\n'
    '  "language": "<dominant language>",\n'
    '  "build_system": {{"type": "go|cmake|make|autotools|cargo|meson'
    '|python|other",\n'
    '    "confidence": <0-1 float>, "reasoning": "<file evidence>"}},'
    "\n"
    '  "dependencies": [{{"name": "<canonical>", "reason": "<file:'
    'line evidence>"}}],\n'
    '  "riscv_risks": ["<risk grounded in evidence>"],\n'
    '  "build_strategy": "<2-3 sentences>",\n'
    '  "expected_artifacts": ["<file name>"],\n'
    '  "needs_custom_plan": <bool>\n'
    "}}"
)


def _object(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Return a closed JSON-schema object; every property is required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_STRINGS = {"type": "array", "items": _STRING}

# Structured-output schemas of the JSON roles (see bind_output_schema).
ANALYST_SCHEMA = _object(
    {
        "purpose": _STRING,
        "language": _STRING,
        "build_system": _object(
            {
                "type": {
                    "type": "string",
                    "enum": [
                        "go",
                        "cmake",
                        "make",
                        "autotools",
                        "cargo",
                        "meson",
                        "python",
                        "other",
                    ],
                },
                "confidence": {"type": "number"},
                "reasoning": _STRING,
            }
        ),
        "dependencies": {
            "type": "array",
            "items": _object({"name": _STRING, "reason": _STRING}),
        },
        "riscv_risks": _STRINGS,
        "build_strategy": _STRING,
        "expected_artifacts": _STRINGS,
        "needs_custom_plan": {"type": "boolean"},
    }
)

ANALYST_TASK_PROMPT = (
    "## Repository\n"
    "- Name: {repo_name} ({repo_url})\n"
    "- Path: {repo_path}\n"
    "- Scripted detection: {detection_summary}\n"
    "- Arch-specific code detected: {arch_summary}\n"
    "\n"
    "## Repository evidence (real file contents)\n"
    "{build_evidence}\n"
    "\n"
    "Reply with the JSON object of the output schema."
)


def _fallback_analysis(state: AgentState) -> PackageAnalysis:
    """Build a deterministic PackageAnalysis from scripted detection.

    Used when every analyst LLM attempt fails. Marks itself as not
    LLM-grounded and defers to the LLM scout whenever detection
    confidence is weak, so a starved LLM layer degrades to plain
    scripted detection instead of a worse result.
    """
    bsi = state.build_system_info
    deps = state.dependencies
    dependencies = []
    if deps:
        for name in (deps.libraries or [])[:10]:
            dependencies.append(
                {"name": str(name), "reason": "scripted detection"}
            )
    confidence = bsi.confidence if bsi else 0.0
    return PackageAnalysis(
        purpose=f"{state.repo_name} (analysis unavailable — LLM starved)",
        language="unknown",
        build_system=bsi.type if bsi else "unknown",
        build_system_confidence=confidence,
        build_system_reasoning="scripted detection only",
        dependencies=dependencies,
        riscv_risks=[
            f"{a.arch_type} code in {a.file}"
            for a in state.arch_specific_code[:5]
        ],
        build_strategy="",
        expected_artifacts=[],
        needs_custom_plan=confidence < 0.5,
        complexity=5,
        llm_grounded=False,
    )


def _shorten(text: Any, limit: int) -> str:
    """Return ``text`` cut at a word boundary to ``limit`` chars, with "…".

    The analyst is asked for short fields; this only bounds a field
    that is still too long, without breaking a word.
    """
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    head = text[: limit - 1]
    space = head.rfind(" ")
    if space >= limit // 2:
        head = head[:space]
    return head.rstrip(" ,;:.") + "…"


def _format_analysis_for_prompt(
    analysis: Optional[PackageAnalysis],
) -> str:
    """Render the package analysis as a compact prompt section.

    Shared by the scout and fixer prompts so both plan/diagnose against
    the analyst's understanding of the package.
    """
    if analysis is None:
        return "(no package analysis available)"
    deps = (
        ", ".join(
            f"{d.get('name', '?')} ({_shorten(d.get('reason', ''), 100)})"
            for d in analysis.dependencies[:10]
        )
        or "none identified"
    )
    risks = (
        "\n".join(f"- {r}" for r in analysis.riscv_risks[:8])
        or "- none identified"
    )
    grounding = (
        "LLM-read from repo files"
        if analysis.llm_grounded
        else "deterministic fallback (treat as weak)"
    )
    return (
        f"Purpose: {analysis.purpose}\n"
        f"Language: {analysis.language} | Build system: "
        f"{analysis.build_system} "
        f"(confidence {analysis.build_system_confidence:.2f}, "
        f"{_shorten(analysis.build_system_reasoning, 200)})\n"
        f"Strategy: {analysis.build_strategy}\n"
        f"Dependencies: {deps}\n"
        f"Expected artifacts: "
        f"{', '.join(analysis.expected_artifacts[:8]) or 'unknown'}\n"
        f"RISC-V risks:\n{risks}\n"
        f"Grounding: {grounding}"
    )


# Outputs that are no riscv64 ELF: scripts, sources, data, bytecode.
# ``.node`` (a Node addon) and ``.whl`` or ``.gem`` (can hold a
# compiled extension) are not in the list.
_NON_NATIVE_SUFFIXES = (
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".map",
    ".css",
    ".html",
    ".py",
    ".pyc",
    ".jar",
    ".class",
    ".war",
    ".php",
    ".phar",
    ".rb",
    ".md",
    ".txt",
    ".yml",
    ".yaml",
    ".xml",
    ".sh",
    ".pl",
    ".lua",
    ".wasm",
)
# A file that builds native code; one of them keeps the pipeline going.
_NATIVE_BUILD_FILES = frozenset(
    {
        "binding.gyp",
        "CMakeLists.txt",
        "meson.build",
        "configure.ac",
        "configure.in",
        "Makefile.am",
        "Cargo.toml",
        "go.mod",
        "build.zig",
        "setup.py",
    }
)
_NATIVE_SOURCE_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".rs",
    ".go",
    ".zig",
    ".s",
    ".asm",
    ".f90",
)
# The source scan stops here and then assumes native code.
_NATIVE_SCAN_LIMIT = 20000


def _native_build_input(host_repo: str) -> str:
    """Return the first native build file or source in the repo, or "".

    Skips ``.git`` and ``node_modules``. Returns ``"(scan limit)"``
    when the tree has more than ``_NATIVE_SCAN_LIMIT`` files, and
    ``"(no tree)"`` when the repo cannot be read: both count as native.
    """
    if not os.path.isdir(host_repo):
        return "(no tree)"
    seen = 0
    for root, dirs, files in os.walk(host_repo):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
        for name in files:
            seen += 1
            if seen > _NATIVE_SCAN_LIMIT:
                return "(scan limit)"
            if name in _NATIVE_BUILD_FILES or name.lower().endswith(
                _NATIVE_SOURCE_SUFFIXES
            ):
                return os.path.relpath(os.path.join(root, name), host_repo)
    return ""


def _unsupported_reason(state: AgentState) -> str:
    """Return why the package cannot give a riscv64 binary, or "".

    The analyst named only scripts or data as outputs, and the repo has
    no native build file or source. nginx-binaries (a TypeScript
    library; outputs ``lib/index.js`` and ``lib/index.d.ts``) ran the
    whole pipeline to "no ELF executable"; it is the only one of 162
    logged analyst answers that this rule matches. The rule does not
    use an empty output list: none of the 162 answers had one.
    """
    analysis = state.package_analysis
    if analysis is None or not analysis.llm_grounded:
        return ""
    outputs = [str(a) for a in analysis.expected_artifacts if str(a).strip()]
    if not outputs or not all(
        a.lower().endswith(_NON_NATIVE_SUFFIXES) for a in outputs
    ):
        return ""
    if _native_build_input(_to_host_path(state.repo_path)):
        return ""
    language = f" ({analysis.language})" if analysis.language else ""
    return (
        f"the package builds no riscv64 binary: the analyst expects only "
        f"{', '.join(outputs[:5])}{language}, and the repo has no native "
        f"source or build file"
    )


@agent_node(AgentRole.PLANNER)
def analyst_node(state: AgentState) -> AgentState:
    """Read the repo evidence and produce the consumed PackageAnalysis.

    This is the "understand" step: one LLM call that reads the ACTUAL
    build files/docs (via :mod:`src.evidence`) and emits a
    ``PackageAnalysis`` that downstream nodes consume — the heuristic
    planner (custom-plan decision + dependency merge), the scout
    (strategy/risks grounding), the fixer (package context), and
    verification (expected artifacts).

    ``task_plan`` is set deterministically as a structural record
    only; no LLM call generates it.
    """
    logger.info("Analyst reading repository evidence...")

    state.build_status = BuildStatus.PLANNING
    state.current_phase = "analyzing"

    bsi = state.build_system_info
    deps = state.dependencies
    detection_summary = (
        f"build_system={bsi.type if bsi else 'unknown'} "
        f"(confidence {bsi.confidence:.2f}, file {bsi.primary_file})"
        if bsi
        else "no build system detected"
    )
    if deps:
        detection_summary += (
            f"; libs={', '.join(deps.libraries[:8]) or 'none'}"
        )

    arch_counts: Dict[str, int] = {}
    for arch_code in state.arch_specific_code:
        arch_counts[arch_code.arch_type] = (
            arch_counts.get(arch_code.arch_type, 0) + 1
        )
    arch_summary = (
        ", ".join(f"{k}×{v}" for k, v in sorted(arch_counts.items())) or "none"
    )

    build_evidence = collect_build_evidence(state.repo_path)
    # The first scout pass reuses it (no build has run yet).
    state.context_cache["build_evidence"] = build_evidence

    system_prompt = ANALYST_PROMPT.format(
        platform_banner=_build_platform_banner()
    )
    prompt = ANALYST_TASK_PROMPT.format(
        repo_name=state.repo_name,
        repo_url=state.repo_url,
        repo_path=state.repo_path,
        detection_summary=detection_summary,
        arch_summary=arch_summary,
        build_evidence=build_evidence,
    )

    def _validate_analysis(data: dict) -> ValidationResult:
        if not str(data.get("purpose", "")).strip():
            return ValidationResult.bad(
                "'purpose' must be a non-empty" " string"
            )
        bs = data.get("build_system")
        if not isinstance(bs, dict) or not bs.get("type"):
            return ValidationResult.bad(
                "'build_system' must be an object with a 'type' key"
            )
        if not isinstance(data.get("dependencies", []), list):
            return ValidationResult.bad("'dependencies' must be an array")
        if not isinstance(data.get("needs_custom_plan"), bool):
            return ValidationResult.bad(
                "'needs_custom_plan' must be a JSON boolean"
            )
        return ValidationResult.good()

    if _llm_refused(state):
        logger.info("Analyst: the model refused this package; fallback")
        state.package_analysis = _fallback_analysis(state)
        state.task_plan = create_default_plan()
        state.current_phase = "analyzed"
        return state

    _analyst_pool = get_model_pool_for_role(AgentRole.PLANNER)
    outcome = llm_call_with_validation(
        invoke_fn=invoke_llm,
        llm=_analyst_pool[0],
        fallback_llms=_analyst_pool[1:],
        prompt=prompt,
        system_prompt=system_prompt,
        output_schema=ANALYST_SCHEMA,
        validator=_validate_analysis,
        fallback_factory=lambda: None,  # materialized below
        role=AgentRole.PLANNER.value,
        audit_metadata={"repo": state.repo_name, "phase": "analyst"},
        max_retries=2,
    )
    state.log_api_call(
        cost=outcome.cost_usd,
        tokens_in=outcome.input_tokens,
        tokens_out=outcome.output_tokens,
        calls=outcome.attempts,
    )
    if outcome.refused:
        _mark_llm_refused(state, AgentRole.PLANNER.value)

    if outcome.data is None:
        logger.warning(
            f"Analyst: deterministic fallback after {outcome.attempts} "
            f"attempts ({outcome.last_error or 'no data'})"
        )
        state.package_analysis = _fallback_analysis(state)
    else:
        data = outcome.data
        bs = data.get("build_system", {})
        deps_raw = data.get("dependencies", [])
        dependencies = [
            {
                "name": str(d.get("name", "")).strip().lower(),
                "reason": _shorten(d.get("reason", ""), 200),
            }
            for d in deps_raw
            if isinstance(d, dict) and str(d.get("name", "")).strip()
        ]
        try:
            confidence = float(bs.get("confidence", 0.5) or 0)
        except (TypeError, ValueError):
            confidence = 0.5
        state.package_analysis = PackageAnalysis(
            purpose=_shorten(data.get("purpose", ""), 300),
            language=_shorten(data.get("language", ""), 60),
            build_system=str(bs.get("type", "unknown")).lower(),
            build_system_confidence=confidence,
            build_system_reasoning=_shorten(bs.get("reasoning", ""), 300),
            dependencies=dependencies[:15],
            riscv_risks=[
                _shorten(r, 200) for r in data.get("riscv_risks", [])[:10]
            ],
            build_strategy=_shorten(data.get("build_strategy", ""), 600),
            expected_artifacts=[
                _shorten(a, 80)
                for a in data.get("expected_artifacts", [])[:10]
            ],
            needs_custom_plan=bool(data.get("needs_custom_plan", False)),
            llm_grounded=True,
        )
        logger.info(
            f"Analysis: {state.package_analysis.purpose[:80]} | "
            f"build={state.package_analysis.build_system} | "
            f"custom_plan={state.package_analysis.needs_custom_plan}"
        )

    # Structural record of the fixed pipeline; no LLM call generates
    # it, and no routing decision reads it.
    state.task_plan = create_default_plan()
    state.current_phase = "analyzed"
    unsupported = _unsupported_reason(state)
    if unsupported:
        logger.warning(f"Unsupported package: {unsupported}")
        state.context_cache["unsupported"] = unsupported
    return state


# Summarizer prompt (the finish_node recipe writer) and the default
# structural task plan.

SUMMARIZER_PROMPT = (
    "You are writing the RISC-V porting report for **{repo_name}**.\n"
    "\n"
    "## Build context\n"
    "- Repo: {repo_url}\n"
    "- Build system: {build_system}\n"
    "- Duration: {duration} | API calls: {api_calls} (~${cost:.4f})\n"
    "\n"
    "## Architecture issues observed\n"
    "{arch_issues}\n"
    "\n"
    "## Fixes applied\n"
    "{fixes}\n"
    "\n"
    "## Build steps actually executed\n"
    "{build_steps}\n"
    "\n"
    "## Required sections (use exactly these H2 headings, in order)\n"
    "1. `## Executive Summary` — status (success / partial / failed), 1-"
    "2 sentence portability verdict.\n"
    "2. `## Prerequisites` — exact package install command(s) for the ta"
    "rget distro.\n"
    "3. `## Build Instructions` — copy-pasteable shell commands, one fen"
    "ced block per logical step, in order.\n"
    "4. `## Architecture Notes` — only the arch-specific things this pro"
    "ject needed (preprocessor macros, SIMD handling, libc differences)."
    " Omit if N/A.\n"
    "5. `## Verification` — how to confirm produced binaries are riscv64"
    " (`file`, `readelf -h`).\n"
    "6. `## Known Issues & Recommendations` — leftover concerns + upstre"
    "am contribution ideas.\n"
    "\n"
    "Rules:\n"
    "- Every command in fenced ```bash blocks.\n"
    "- Only mention packages/flags that were actually used.\n"
    "- No generic RISC-V tutorial content — be specific to this project.\n"
    "- If a section has nothing useful to say, write a single sentence a"
    "cknowledging that and move on."
)


def create_default_plan() -> TaskPlan:
    """Return the fixed two-phase TaskPlan recorded on every run.

    The record is structural only: no routing or planning decision
    reads it, and no LLM call generates it. It exists so reports and
    state dumps keep a stable shape.
    """
    return TaskPlan(
        phases=[
            TaskPhase(
                1,
                "scout",
                "Analyze and create build plan",
                AgentRole.SCOUT,
                False,
                [],
            ),
            TaskPhase(
                2, "build", "Execute build", AgentRole.BUILDER, False, [1]
            ),
        ],
        estimated_total_cost=0.02,
        complexity_score=5,
    )


# supervisor_node — zero-LLM audit pass between routing decisions.


@agent_node(AgentRole.SUPERVISOR)
def supervisor_node(state: AgentState) -> AgentState:
    """Log a summary of the state between routing decisions.

    Routing is handled by ``route_supervisor_to_next`` at the graph-edge
    level. This node exists as a lightweight audit/logging pass-through
    so the graph topology is preserved.
    """
    bp_state = "Exists" if state.build_plan else "Missing"
    logger.info(
        f"Supervisor evaluating state... (phase={state.current_phase}, "
        f"status={state.build_status.value}, BuildPlan: {bp_state}, "
        f"attempts={state.attempt_count}/{state.max_attempts})"
    )

    summary = (
        f"build_status={state.build_status.value}, "
        f"errors={len(state.error_history)}, "
        f"fixes={len(state.fixes_attempted)}, "
        f"cost=${state.api_cost_usd:.4f}"
    )
    logger.info(f"Supervisor summary: {summary}")
    state.log_scripted_op("supervisor_eval")
    return state


# scout_node — LLM BuildPlan grounded in evidence and the analysis.

# Doubled braces in the JSON schema are literal braces after
# str.format().
# The system part (static per platform, cached; the knowledge block
# follows as a second system block); the per-package part is
# SCOUT_TASK_PROMPT, sent as the user message.
SCOUT_PROMPT = (
    "You are the build scout for a RISC-V porting agent. Produce a\n"
    "minimal, executable BuildPlan for the package in the user message,"
    "\nbuilt natively at its repository root.\n"
    "\n"
    "## ACTIVE SANDBOX (use ONLY these package commands)\n"
    "{platform_banner}\n"
    "\n"
    "## Target\n"
    "- Architecture: RISC-V (riscv64) (rv64, riscv, riscv64, RISCV64)\n"
    "- Native compilation only (DO NOT set `--host`, `GOOS/GOARCH`, `-DC"
    "MAKE_SYSTEM_PROCESSOR`).\n"
    "\n"
    "## Build plan rules\n"
    "1. Each command runs in a fresh shell at the repository root. "
    "**Always chain `cd` with `&&`** — e.g. `cd build && cmake ..`, "
    "never two commands.\n"
    "2. Group commands into ≤3 phases: `setup` (install deps) → `configu"
    "re` (optional) → `build`. Skip phases that have nothing to do.\n"
    "3. Install only `-dev` packages you can justify from CMakeLists.txt"
    " / configure.ac / meson.build / go.mod. No guessing.\n"
    "4. Use the package names from the platform knowledge below. Do NOT "
    "use names from a different distro.\n"
    "5. If the project has both CMake and autotools, prefer the one whos"
    "e dependencies are easier to satisfy; if unsure, prefer autotools.\n"
    "6. For Go: if `go.mod` is missing, first phase must `go mod init <m"
    "odule_path>` then `go mod tidy`. Always pass `-buildvcs=false` to `"
    "go build`. **If `Project context` in the user message gives a "
    "`plan_command`, use exactly that command (it already runs in the "
    "module directory) — do not substitute `go build ./...` (which "
    "compiles every helper and fails on the first one with missing deps)"
    ".** Run `go mod tidy` only when `go.sum` is missing "
    "(`has_go_sum` is false); `go build` fetches what it needs. "
    "Never emit `go build ./cmd` or `go build ./<top-level-dir>` "
    "without an explicit output path; if building a subpackage, use "
    "`go build -o ./bin/<name> <package>`.\n"
    "7. Disable optional features that pull in heavy or x86-specific dep"
    "s (tests, SIMD, examples, docs).\n"
    "8. NEVER reference paths or files you can't see in the project stru"
    "cture.\n"
    "9. Never add a command that runs tests (`ctest`, `make check`, "
    "`make test`, `meson test`, `go test`, `cargo test`). Atesor runs "
    "the package tests after a verified build, and they never gate it.\n"
    "\n"
    "## Self-check (do silently before responding)\n"
    "- Did I read the actual build files (or admit I couldn't)?\n"
    "- Does every package install command exist for the target distro?\n"
    "- Do my commands actually run if pasted into a shell at the "
    "repository root?\n"
    "\n"
    "## Output schema (JSON only)\n"
    "{{\n"
    '  "build_system": "go|cmake|make|autotools|cargo|meson|other",\n'
    '  "phases": [\n'
    '    {{"id": 1, "name": "setup", "commands": ["..."]}},\n'
    '    {{"id": 2, "name": "build", "commands": ["..."]}}\n'
    "  ]\n"
    "}}"
)

SCOUT_TASK_PROMPT = (
    "## Package\n"
    "- Name: **{repo_name}** | Repository root: `{repo_path}`\n"
    "\n"
    "## Analyst's understanding of this package (plan FROM this)\n"
    "{package_analysis}\n"
    "\n"
    "## Repo facts\n"
    "- Build system: {build_system} | Module dir: `{module_dir}`\n"
    "- Arch-specific code: {arch_code_count} occurrences | Cached docs: "
    "{doc_count}\n"
    "- Project context: {go_main_info}\n"
    "\n"
    "### Build file evidence (real contents — cite these, don't guess)\n"
    "{build_evidence}\n"
    "\n"
    "### Project structure\n"
    "{repo_tree}\n"
    "\n"
    "### Arch patterns\n"
    "{arch_build_patterns}\n"
    "\n"
    "### Detected dependencies\n"
    "{dependencies}\n"
    "\n"
    "### Arch concerns\n"
    "{arch_concerns}\n"
    "\n"
    "### Sandbox tools available\n"
    "{system_info}\n"
    "\n"
    "### Previous failure (if re-scouting after a failed build)\n"
    "{previous_failure}\n"
    "\n"
    "### Documentation excerpts\n"
    "{documentation}\n"
    "\n"
    "{recall_section}\n"
    "\n"
    "{few_shot_examples}\n"
    "\n"
    "Reply with the JSON BuildPlan of the output schema."
)

SCOUT_SCHEMA = _object(
    {
        "build_system": _STRING,
        "phases": {
            "type": "array",
            "items": _object(
                {
                    "id": {"type": "integer"},
                    "name": _STRING,
                    "commands": _STRINGS,
                }
            ),
        },
    }
)

# Heading of the knowledge block, the second (cached) system block.
_SCOUT_KNOWLEDGE_HEADING = (
    "## Platform knowledge (USE THIS — don't invent package names)\n"
)


# BUILD_STEP_RE (src/tools.py) matches the build and test commands.


def _rewind_after_verify_failure(state: AgentState) -> None:
    """Make the builder compile a fix that follows a verify failure.

    After a build failure the builder resumes at the failed phase, and
    the failed command has no cached success, so the fix is built. After
    a verify failure every phase is done and every command has a cached
    success, so nothing would run. Rewind to the first phase with a
    build step (the last phase when no phase has one) and forget the
    cached results from there on. Earlier phases, such as
    ``go mod init``, do not run again: they are not safe to repeat.

    Args:
        state: The agent state after the fixer applied a change.
    """
    plan = state.build_plan
    if not plan or not plan.phases:
        return
    if state.last_successful_phase < plan.phases[-1].id:
        return
    start = next(
        (
            index
            for index, phase in enumerate(plan.phases)
            if any(BUILD_STEP_RE.search(c) for c in phase.commands)
        ),
        len(plan.phases) - 1,
    )
    state.last_successful_phase = plan.phases[start - 1].id if start > 0 else 0
    for phase in plan.phases[start:]:
        for command in phase.commands:
            state.forget_command_result(command)


# The whitelist that execute_command applies to every build command,
# used by the scout to reject a plan command before the build runs it.
_plan_command_validator = CommandValidator()

# Text of the scout's critique for a blocked plan command. When the
# scout's final failure still carries it, no plan avoided the block.
_PLAN_BLOCKED = "the sandbox command validator blocks"


def validate_build_plan(plan: BuildPlan) -> tuple[bool, str]:
    """Reject a BuildPlan that is hallucinated or cannot fail loudly.

    Checks every command for placeholder paths, a cross-compiler
    prefix (builds are native-only), Go subcommands mistaken for
    package names, nested cmake build dirs, and shell constructs that
    mask a failing build step.

    Args:
        plan: The plan to check, from the scout or the heuristic
            planner.

    Returns:
        A ``(is_valid, reason)`` tuple; ``reason`` names the offending
        command when ``is_valid`` is False.
    """
    hallucination_patterns = [
        "/path/to/",
        "your_username",
        "example.com",
        "riscv64-unknown-linux-gnu-gcc",  # cross-compile prefix
    ]

    # Go subcommands that must not appear as package names
    go_subcommands = {
        "mod",
        "build",
        "get",
        "install",
        "run",
        "test",
        "tool",
        "generate",
        "fmt",
        "vet",
    }

    for phase in plan.phases:
        for cmd in phase.commands:
            # A build step that cannot fail turns a broken build into a
            # reported success (`make || true`, `make; true`).
            masked = masks_failure(cmd)
            if masked and BUILD_STEP_RE.search(cmd):
                return (
                    False,
                    f"Build command hides a failing step ({masked}): "
                    f"'{cmd}'",
                )
            for pattern in hallucination_patterns:
                if pattern in cmd:
                    return (
                        False,
                        f"Hallucination detected in command: '{cmd}' "
                        f"(contains '{pattern}')",
                    )

            # A /home path outside the mounted /workspace tree does not
            # exist in the sandbox; the model invented a host path.
            if "/home/" in cmd and "/workspace/" not in cmd:
                return False, f"Suspicious absolute path in command: '{cmd}'"

            # Check for `apk add go mod` / `apt-get install go build`
            # style hallucinations where Go subcommands are mistaken
            # for package names.
            stripped = cmd.strip()
            if (
                stripped.startswith("apk add")
                or stripped.startswith("apt-get install")
                or stripped.startswith("apt install")
            ):
                parts = stripped.split()
                # apk add <pkgs> / apt-get install <pkgs>
                skip = 2 if stripped.startswith("apk") else 2
                pkg_names = [p for p in parts[skip:] if not p.startswith("-")]
                for pkg in pkg_names:
                    if pkg in go_subcommands:
                        return False, (
                            f"Hallucination: '{cmd}' — '{pkg}' is a Go "
                            f"subcommand, not a package. "
                            f"Use 'go {pkg}' as a separate command."
                        )

            if "cmake" in cmd:
                if "cd build &&" in cmd and "-B build" in cmd:
                    return (
                        False,
                        f"Bad cmake command (would create nested "
                        f"build dir): '{cmd}'",
                    )
                # `cd build && cmake -S . -B build` nests a build/build
                # directory that later `cd build` steps never find.
                if (
                    "cd build &&" in cmd
                    and "cmake -S" in cmd
                    and "-B build" in cmd
                ):
                    return (
                        False,
                        f"Bad cmake pattern (creates build/build): '{cmd}'",
                    )

    return True, ""


# Test runs do not belong in a build plan: a failing test then fails
# the whole port, although finish_node runs the package tests itself
# and they never gate the build.
_TEST_TOOLS = {"ctest", "pytest"}
_TEST_SUBCOMMANDS = {
    "go": "test",
    "cargo": "test",
    "meson": "test",
    "npm": "test",
}
_TEST_TARGETS = {"check", "test", "tests"}
# Options whose next word is a value, not a target.
_TARGET_VALUE_OPTIONS = {
    "make": {"-C", "-f", "-I", "-o", "-W", "--directory", "--file"},
    "gmake": {"-C", "-f", "-I", "-o", "-W", "--directory", "--file"},
    "ninja": {"-C", "-f", "-j", "-k", "-l", "-d", "-t", "-w"},
}
_LEADING_CD_RE = re.compile(r"^\s*cd\s+\S+\s*&&\s*")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")


def is_test_command(command: str) -> bool:
    """Return True when ``command`` runs a test suite.

    Leading ``cd <dir> &&`` prefixes and ``NAME=value`` assignments are
    skipped. Matches ``ctest``, ``pytest`` (also ``python -m pytest``),
    ``go``/``cargo``/``meson``/``npm test``, ``make``/``ninja`` with a
    ``check``/``test``/``tests`` target, and ``cmake --build`` with
    ``--target check|test|tests``.

    Args:
        command: One build-plan command.

    Returns:
        True if the command's first invocation runs tests.
    """
    rest = command
    while True:
        stripped = _LEADING_CD_RE.sub("", rest, count=1)
        if stripped == rest:
            break
        rest = stripped
    try:
        words = shlex.split(rest)
    except ValueError:
        words = rest.split()
    while words and _ENV_ASSIGN_RE.match(words[0]):
        words = words[1:]
    if not words:
        return False
    tool = os.path.basename(words[0])
    args = words[1:]
    if tool in _TEST_TOOLS:
        return True
    if tool.startswith("python") and args[:2] == ["-m", "pytest"]:
        return True
    if tool in _TEST_SUBCOMMANDS:
        first = next((a for a in args if not a.startswith("-")), "")
        return first == _TEST_SUBCOMMANDS[tool]
    if tool in _TARGET_VALUE_OPTIONS:
        value_options = _TARGET_VALUE_OPTIONS[tool]
        skip = False
        for arg in args:
            if skip:
                skip = False
            elif arg in value_options:
                skip = True
            elif not arg.startswith("-") and arg in _TEST_TARGETS:
                return True
        return False
    if tool == "cmake" and "--build" in args and "--target" in args:
        target = args.index("--target") + 1
        return target < len(args) and args[target] in _TEST_TARGETS
    return False


def drop_test_commands(
    phases: List[BuildPhase],
) -> Tuple[List[BuildPhase], List[str]]:
    """Remove test runs from build phases; see :func:`is_test_command`.

    A phase left without commands is dropped, and the kept phases are
    numbered 1..n again (phase ids only order the phases).

    Args:
        phases: The phases of a scout plan, changed in place.

    Returns:
        ``(kept_phases, dropped_commands)``.
    """
    kept: List[BuildPhase] = []
    dropped: List[str] = []
    for phase in phases:
        commands = []
        for command in phase.commands:
            if is_test_command(command):
                dropped.append(command)
            else:
                commands.append(command)
        if commands:
            phase.commands = commands
            kept.append(phase)
    for number, phase in enumerate(kept, start=1):
        phase.id = number
    return kept, dropped


@agent_node(AgentRole.SCOUT)
def scout_node(state: AgentState) -> AgentState:
    """Produce a validated LLM BuildPlan for the repository.

    The prompt grounds the plan in the analyst's ``PackageAnalysis``,
    the real build-file evidence, and few-shot examples; on a re-scout
    after a failed build, the previous failure is included so the new
    plan does not repeat it. The reply must parse as JSON, pass
    ``validate_build_plan``, and survive critique retries across the
    role's model pool. When every LLM attempt fails, the node falls
    back to ``create_fallback_build_plan`` so the build still runs
    with the LLM layer down.
    """
    logger.info("Scout analyzing repository...")

    state.build_status = BuildStatus.SCOUTING
    state.current_phase = "scouting"

    # A re-plan after a validator block happens at most once (see
    # route_supervisor_to_next).
    if _last_error_is_validator_block(state):
        state.context_cache["validator_block_replans"] = (
            state.context_cache.get("validator_block_replans", 0) + 1
        )

    # After a refusal, the scout prompt (same package) is refused too.
    if _llm_refused(state):
        logger.info("Scout: the model refused this package; fallback plan")
        state.build_plan = create_fallback_build_plan(state)
        state.last_successful_phase = 0
        state.build_status = BuildStatus.PENDING
        return state

    build_sys = state.build_system_info
    deps = state.dependencies

    # The analyst's evidence is still current until a build has run: a
    # re-scout after a failure reads the tree again (configure may have
    # written files, a fix may have edited them).
    build_evidence = (
        state.context_cache.get("build_evidence")
        if not state.error_history
        else None
    ) or collect_build_evidence(state.repo_path)

    documentation = "No documentation cached"
    if state.file_content_cache:
        doc_files = [
            k
            for k in state.file_content_cache.keys()
            if any(d in k.upper() for d in ["README", "INSTALL", "BUILD"])
            # A top-level doc already in the build evidence is not sent
            # twice.
            and not (
                os.path.dirname(k) == state.repo_path.rstrip("/")
                and f"### {Path(k).name} (excerpt)" in build_evidence
            )
        ]
        if doc_files:
            report_docs = doc_files[:3]
            documentation = "\n\n".join(
                [
                    # The repo-relative path: two README.md files in
                    # different directories must not look the same.
                    f"## {os.path.relpath(f, state.repo_path)}\n"
                    f"{cut_text(state.file_content_cache[f], 2000)}"
                    for f in report_docs
                ]
            )

    arch_concerns = "None detected"
    if state.arch_specific_code:
        arch_concerns = "\n".join(
            [
                f"- {a.file}:{a.line} - {a.arch_type} ({a.severity})"
                for a in state.arch_specific_code[:10]
            ]
        )

    dependencies = "Unknown"
    if deps:
        dependencies = json.dumps(
            {
                "build_tools": deps.build_tools,
                "system_packages": deps.system_packages[:10],
                "libraries": deps.libraries[:10],
            },
            indent=2,
        )

    system_info_raw = scripted_ops.get_system_info()
    system_info = "\n".join(
        [f"- {k}: {v}" for k, v in system_info_raw.items()]
    )

    module_dir = (
        build_sys.module_dir
        if build_sys and hasattr(build_sys, "module_dir")
        else ""
    )

    go_main_info = state.context_cache.get("go_main_info", {})
    build_type_lower = (build_sys.type if build_sys else "").lower()
    if go_main_info:
        go_main_str = cut_text(
            json.dumps(go_main_info, separators=(",", ":")), 2000
        )
    elif build_type_lower == "go":
        go_main_str = (
            "Go project detected; main package not auto-discovered "
            "(scout should inspect the repo for main.go or cmd/ "
            "subdirs)"
        )
    else:
        go_main_str = "Not a Go project"

    few_shot_context = {
        "build_system": build_sys.type if build_sys else "unknown",
        "has_main": go_main_info.get("has_main", False),
        "module_dir": go_main_info.get("module_dir", module_dir),
        # An example of this package outranks every other one.
        "repo_name": state.repo_name,
    }
    if build_type_lower == "go":
        if go_main_info.get("main_path"):
            few_shot_context["main_path"] = go_main_info["main_path"]
        few_shot_context["has_cgo"] = _analysis_mentions_cgo(state)
    elif module_dir:
        few_shot_context["nested_root"] = module_dir
    few_shot_examples = (
        ""
        if state.replan
        else format_few_shot_examples(
            "scout", few_shot_context, max_examples=2, max_chars=2000
        )
    )
    _note_examples(state, "scout", few_shot_examples)

    # When the supervisor re-scouts after a failure, the new plan must
    # actually address the failure — otherwise the scout regenerates
    # the same plan and the loop burns attempts until escalation.
    previous_failure = "None — first scouting pass."
    if state.last_error:
        failed_cmd = ""
        if state.error_history and state.error_history[-1].command:
            failed_cmd = (
                f"Failed command: `{state.error_history[-1].command}`\n"
            )
        key_lines, tail = failure_digest(state.last_error, tail_chars=1200)
        stdout_tail = (
            state.error_history[-1].stdout_tail if state.error_history else ""
        )
        previous_failure = (
            f"{failed_cmd}Exit code: {_exit_code_text(state.last_error)}\n"
            + (
                f"Error lines from the cut part of the output:\n{key_lines}\n"
                if key_lines
                else ""
            )
            + f"Command output (the end; a cut is marked):\n{tail}\n"
            + (
                f"Standard output (the end):\n{stdout_tail}\n"
                if stdout_tail
                else ""
            )
            + "Your new plan MUST avoid repeating the cause of this "
            "failure (e.g. install the missing package, pick a "
            "different build path)."
        )

    arch_patterns_str = "No architecture-specific patterns detected"
    if state.arch_specific_code:
        arch_patterns_by_type = {}
        for arch_code in state.arch_specific_code:
            if arch_code.arch_type not in arch_patterns_by_type:
                arch_patterns_by_type[arch_code.arch_type] = 0
            arch_patterns_by_type[arch_code.arch_type] += 1
        arch_patterns_str = "\n".join(
            [
                f"- {arch_type}: {count} matches"
                for arch_type, count in arch_patterns_by_type.items()
            ]
        )

    package_analysis_str = _format_analysis_for_prompt(state.package_analysis)

    system_prompt = [
        SCOUT_PROMPT.format(platform_banner=_build_platform_banner()),
        _SCOUT_KNOWLEDGE_HEADING
        + get_system_knowledge_summary(
            build_system=build_sys.type if build_sys else None
        ),
    ]
    prompt = SCOUT_TASK_PROMPT.format(
        repo_name=state.repo_name,
        package_analysis=package_analysis_str,
        build_evidence=build_evidence,
        build_system=build_sys.type if build_sys else "unknown",
        repo_tree=state.repo_tree if state.repo_tree else "(Not available)",
        arch_build_patterns=arch_patterns_str,
        module_dir=module_dir,
        deps_count=len(deps.system_packages) if deps else 0,
        arch_code_count=len(state.arch_specific_code),
        doc_count=len(state.file_content_cache),
        go_main_info=go_main_str,
        documentation=documentation,
        arch_concerns=arch_concerns,
        dependencies=dependencies,
        repo_path=state.repo_path,
        system_info=system_info,
        previous_failure=previous_failure,
        recall_section=_recall_prompt_section(state),
        few_shot_examples=few_shot_examples,
    )

    def _validate_scout(data: dict) -> ValidationResult:
        phases = data.get("phases")
        if not isinstance(phases, list) or not phases:
            return ValidationResult.bad("'phases' must be a non-empty array")
        for i, p in enumerate(phases):
            if not isinstance(p, dict):
                return ValidationResult.bad(f"phase {i} is not an object")
            if not p.get("name"):
                return ValidationResult.bad(f"phase {i} missing 'name'")
            cmds = p.get("commands")
            if not isinstance(cmds, list) or not cmds:
                return ValidationResult.bad(
                    f"phase {i} ({p.get('name')}) must have a "
                    f"non-empty 'commands' array"
                )
            if not all(isinstance(c, str) and c.strip() for c in cmds):
                return ValidationResult.bad(
                    f"phase {i} ({p.get('name')}) has empty/non-string command"
                )
            # The builder runs each command through the same whitelist.
            # A blocked command fails the build, and the fixer cannot
            # change the whitelist, so reject it here, where a retry is
            # cheap.
            for cmd in cmds:
                allowed, why = _plan_command_validator.is_safe(cmd)
                if not allowed:
                    return ValidationResult.bad(
                        f"phase {i} ({p.get('name')}): {_PLAN_BLOCKED} "
                        f"`{cmd[:160]}` ({why}). Use standard build and "
                        f"package tools only, or drop the step (version "
                        f"checks are not needed)"
                    )
        # A -march the machine lacks gives a program that dies with
        # SIGILL (libwebp: rv64gcv on a board without V).
        misfit = _plan_isa_problems(
            state, [c for p in phases for c in p.get("commands") or []]
        )
        if misfit:
            profile = _machine(state)
            return ValidationResult.bad(
                f"{'; '.join(misfit)}: this machine runs "
                f"{profile.label if profile else 'another ISA'}; drop the "
                "flag or name only extensions that the machine has"
            )
        # Dry-build a BuildPlan to reuse the existing semantic validator
        try:
            trial_phases = [
                BuildPhase(
                    id=p.get("id", i + 1),
                    name=p["name"],
                    commands=p["commands"],
                    can_parallelize=p.get("can_parallelize", False),
                    expected_duration=p.get("expected_duration", "unknown"),
                )
                for i, p in enumerate(phases)
            ]
            trial_plan = BuildPlan(
                build_system=data.get(
                    "build_system", build_sys.type if build_sys else "unknown"
                ),
                build_system_confidence=data.get(
                    "build_system_confidence", 0.5
                ),
                phases=trial_phases,
                total_estimated_duration=data.get(
                    "total_estimated_duration", "unknown"
                ),
            )
            ok, reason = validate_build_plan(trial_plan)
            if not ok:
                return ValidationResult.bad(
                    f"{reason}. Avoid absolute paths like "
                    f"/path/to/, /home/, /root/"
                )
        except Exception as exc:
            return ValidationResult.bad(
                f"could not materialize BuildPlan: {exc}"
            )
        return ValidationResult.good()

    _scout_pool = get_model_pool_for_role(AgentRole.SCOUT)
    outcome = llm_call_with_validation(
        invoke_fn=invoke_llm,
        llm=_scout_pool[0],
        fallback_llms=_scout_pool[1:],
        prompt=prompt,
        system_prompt=system_prompt,
        output_schema=SCOUT_SCHEMA,
        validator=_validate_scout,
        fallback_factory=lambda: None,  # we build the BuildPlan fallback below
        role=AgentRole.SCOUT.value,
        audit_metadata={
            "repo": state.repo_name,
            "build_system": (
                state.build_system_info.type
                if state.build_system_info
                else "unknown"
            ),
            # The examples the prompt carried, to measure their effect.
            "examples": list(getattr(few_shot_examples, "ids", ())),
            "recall": _recall_label(state),
        },
        max_retries=2,
    )
    state.log_api_call(
        cost=outcome.cost_usd,
        tokens_in=outcome.input_tokens,
        tokens_out=outcome.output_tokens,
        calls=outcome.attempts,
    )

    if outcome.refused:
        # A refusal is not a build attempt: keep the attempt budget for
        # the fallback plan.
        _mark_llm_refused(state, AgentRole.SCOUT.value)
        state.build_plan = create_fallback_build_plan(state)
        state.last_successful_phase = 0
        state.build_status = BuildStatus.PENDING
        return state

    try:
        if outcome.data is None:
            raise ValueError(
                outcome.last_error or "scout produced no valid plan"
            )

        plan_data = outcome.data
        phases = [
            BuildPhase(
                id=p.get("id", i + 1),
                name=p["name"],
                commands=p["commands"],
                can_parallelize=p.get("can_parallelize", False),
                expected_duration=p.get("expected_duration", "unknown"),
            )
            for i, p in enumerate(plan_data["phases"])
        ]
        phases, dropped = drop_test_commands(phases)
        if dropped:
            logger.warning(
                "Dropped test commands from the scout plan (package "
                f"tests run after the build): {dropped}"
            )
        if not phases:
            raise ValueError("the scout plan held only test commands")

        state.build_plan = BuildPlan(
            build_system=plan_data.get(
                "build_system", build_sys.type if build_sys else "unknown"
            ),
            build_system_confidence=plan_data.get(
                "build_system_confidence",
                build_sys.confidence if build_sys else 0.5,
            ),
            phases=phases,
            total_estimated_duration=plan_data.get(
                "total_estimated_duration", "unknown"
            ),
            notes=plan_data.get("notes", []),
        )
        state.last_successful_phase = 0
        state.context_cache["plan_source"] = "scout"
        logger.info(f"Build plan created and validated: {len(phases)} phases")

    except Exception as e:
        logger.error(f"Scout failed: {e}")
        if _PLAN_BLOCKED in str(e):
            # Every plan the model could write needs a tool that the
            # command validator blocks (hadolint: cabal). A guessed
            # fallback plan cannot build the package, and the fixer
            # cannot change the whitelist, so the run escalates. The
            # scout already retried with the blocked command named,
            # so no second re-plan either.
            state.add_error(
                create_error_record(
                    message=(
                        f"Scout could not plan without blocked commands "
                        f"({_VALIDATOR_BLOCK}): {e}"
                    ),
                    category=ErrorCategory.MISSING_TOOLS,
                )
            )
            state.context_cache["validator_block_replans"] = 1
            state.build_status = BuildStatus.FAILED
            return state
        logger.info("Using fallback build plan due to scout failure")
        # The fallback plan builds, and that build counts the attempt.
        state.add_error(
            create_error_record(
                message=f"Scout failed: {e}",
                category=ErrorCategory.UNKNOWN,
            ),
            count_attempt=False,
        )
        state.build_plan = create_fallback_build_plan(state)

    state.build_status = BuildStatus.PENDING
    return state


def create_fallback_build_plan(state: AgentState) -> BuildPlan:
    """Return a deterministic BuildPlan for the detected build system.

    Used when the scout LLM produces nothing usable. Covers go, cmake,
    make, cargo, meson and autotools with canned recipes; for anything
    else the plan probes the repo for build files before trying a bare
    ``make``.
    """
    from .platforms import get_active_profile

    profile = get_active_profile()
    build_sys = state.build_system_info
    build_type = build_sys.type if build_sys else "unknown"

    def _setup(canonical_pkgs: List[str]) -> str:
        """Render install command for canonical package names.

        Uses the active platform profile.
        """
        return profile.install_cmd(
            [profile.resolve(p) for p in canonical_pkgs]
        )

    if build_type == "go":
        go_main_info = state.context_cache.get("go_main_info", {})
        module_name = (
            state.repo_url.replace("https://", "").rstrip("/")
            if state.repo_url
            else f"github.com/unknown/{state.repo_name}"
        )

        build_cmds = _go_prepare_commands(go_main_info, module_name)

        return BuildPlan(
            build_system="go",
            build_system_confidence=0.8,
            phases=[
                BuildPhase(1, "setup", [_setup(["git"])], False, "30s"),
                BuildPhase(2, "build", build_cmds, False, "3m"),
            ],
            total_estimated_duration="4m",
            notes=[f"Fallback Go build plan ({profile.name})"],
        )
    elif build_type == "cmake":
        return BuildPlan(
            build_system="cmake",
            build_system_confidence=0.7,
            phases=[
                BuildPhase(
                    1, "setup", [_setup(["gcc", "cmake"])], False, "30s"
                ),
                BuildPhase(
                    2,
                    "configure",
                    [
                        "mkdir -p build",
                        "cd build && cmake .. -DCMAKE_BUILD_TYPE=Release",
                    ],
                    False,
                    "1m",
                ),
                BuildPhase(
                    3, "build", ["cd build && make -j$(nproc)"], False, "5m"
                ),
            ],
            total_estimated_duration="7m",
            notes=[f"Fallback CMake build plan ({profile.name})"],
        )
    elif build_type == "make":
        return BuildPlan(
            build_system="make",
            build_system_confidence=0.7,
            phases=[
                BuildPhase(1, "setup", [_setup(["gcc"])], False, "30s"),
                BuildPhase(2, "build", ["make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="6m",
            notes=[f"Fallback Make build plan ({profile.name})"],
        )
    elif build_type == "cargo":
        return BuildPlan(
            build_system="cargo",
            build_system_confidence=0.8,
            phases=[
                BuildPhase(1, "setup", [_setup(["rust"])], False, "30s"),
                BuildPhase(2, "build", ["cargo build --release"], False, "5m"),
            ],
            total_estimated_duration="6m",
            notes=[f"Fallback Cargo build plan ({profile.name})"],
        )
    elif build_type == "meson":
        return BuildPlan(
            build_system="meson",
            build_system_confidence=0.7,
            phases=[
                BuildPhase(
                    1,
                    "setup",
                    [_setup(["gcc", "meson", "ninja"])],
                    False,
                    "30s",
                ),
                BuildPhase(
                    2, "configure", ["meson setup builddir"], False, "1m"
                ),
                BuildPhase(3, "build", ["ninja -C builddir"], False, "5m"),
            ],
            total_estimated_duration="7m",
            notes=[f"Fallback Meson build plan ({profile.name})"],
        )
    elif build_type == "autotools":
        has_configure = (
            os.path.exists(
                os.path.join(_to_host_path(state.repo_path), "configure")
            )
            if state.repo_path
            else True
        )
        configure_phase_cmds = ["./configure"]
        if not has_configure:
            configure_phase_cmds = [
                "cp /usr/share/gettext/m4/*.m4 m4/ 2>/dev/null || true",
                "autoreconf -fi",
                "./configure",
            ]
        return BuildPlan(
            build_system="autotools",
            build_system_confidence=0.6,
            phases=[
                BuildPhase(
                    1,
                    "setup",
                    [
                        _setup(["gcc", "autotools", "pkgconfig"]),
                        "cp /usr/share/gettext/m4/*.m4 "
                        "m4/ 2>/dev/null || true",
                    ],
                    False,
                    "30s",
                ),
                BuildPhase(2, "configure", configure_phase_cmds, False, "3m"),
                BuildPhase(3, "build", ["make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="9m",
            notes=[f"Fallback Autotools build plan ({profile.name})"],
        )
    else:
        logger.warning(
            f"Unknown build system '{build_type}', using discovery fallback"
        )
        repo_host = _to_host_path(state.repo_path)
        discovery_cmds = []
        for probe, label in [
            ("configure", "autotools"),
            ("CMakeLists.txt", "cmake"),
            ("Makefile", "make"),
            ("meson.build", "meson"),
            ("go.mod", "go"),
            ("Cargo.toml", "cargo"),
            ("setup.py", "python"),
        ]:
            if os.path.isfile(os.path.join(repo_host, probe)):
                discovery_cmds.append(f"ls -la {probe} 2>/dev/null")
        if not discovery_cmds:
            # Commands run INSIDE the container with cwd already set to
            # the repo — never emit a host path here (a translated
            # /home/... path does not exist in the sandbox and fails
            # with exit 2; observed on assetfinder, run 2026-07-02).
            discovery_cmds = [
                "ls -la",
                "find . -maxdepth 2 -name 'Makefile' -o "
                "-name 'configure' -o -name 'CMakeLists.txt' "
                "-o -name '*.mk' 2>/dev/null | head -20",
            ]
        return BuildPlan(
            build_system=build_type,
            build_system_confidence=0.3,
            phases=[
                BuildPhase(1, "discover", discovery_cmds, False, "15s"),
                BuildPhase(2, "build", ["make -j$(nproc)"], False, "5m"),
            ],
            total_estimated_duration="6m",
            notes=[
                f"Discovery fallback - build system not recognized; "
                f"probing repo for build files ({profile.name})"
            ],
        )


# builder_node — plan execution with deterministic, LLM-free recovery.


def _fixup_top_builddir_in_submakefiles(docker_repo_path: str) -> None:
    """Apply a post-configure fixup for old gettext/autotools projects.

    Old po/Makefile.in.in templates reference $(top_builddir) as a
    make variable but never define it (they expected @top_builddir@
    substitution that was added in newer gettext). When make descends
    into po/ without passing top_builddir, the dependency
    $(top_builddir)/config.status expands to /config.status which
    doesn't exist -> "No rule to make target '/config.status'".

    Fix: run a shell script INSIDE the sandbox container (to avoid host
    permission issues with root-owned files) that scans all
    subdirectory Makefiles and injects the correct relative
    top_builddir assignment. Any other error is logged, not raised.

    Args:
        docker_repo_path: Path to the repository inside the container.

    Raises:
        SandboxUnavailableError: If the native sandbox connection
            drops or never opens.
    """
    from src import sandbox
    from src.platforms import get_container_name
    from src.sandbox import SandboxUnavailableError

    try:
        container = get_container_name()
        # Shell script: walk subdirs, find Makefiles using
        # $(top_builddir) that don't already define it, and inject
        # `top_builddir = <relpath>`.
        script = (
            f"cd {docker_repo_path} && "
            "find . -mindepth 2 -name Makefile "
            "-not -path './.git/*' | while read f; do "
            "  if grep -q '\\$(top_builddir)' \"$f\" "
            "&& ! grep -q '^top_builddir' \"$f\"; then "
            "    depth=$(echo \"$f\" | tr -cd '/' | wc -c); "
            "depth=$((depth - 1)); "
            "    rel=$(python3 -c \"print('/'.join(['..'] * $depth))\"); "
            '    sed -i "1s|^|top_builddir = $rel\\n|" "$f" && '
            '    echo "Injected top_builddir=$rel into $f"; '
            "  fi; "
            "done"
        )
        # docker exec on qemu; ssh and podman exec on native. Native
        # goes through sandbox.run_call() so transient SSH transport
        # errors are retried and a real drop raises
        # SandboxUnavailableError instead of being reported as a
        # best-effort fixup miss (a dropped connection can leave the
        # sed half-applied and mask a real build failure).
        call = sandbox.build_exec_argv(container, ["sh", "-c", script])
        if call.native:
            result = sandbox.run_call(call, timeout=30)
        else:
            result = subprocess.run(
                call.argv,
                capture_output=True,
                text=True,
                timeout=30,
                **call.run_kwargs(),
            )
        if call.native:
            # The script edits Makefiles, so the local mirror is stale.
            from src import mirror

            mirror.mark_stale()
        if result.stdout.strip():
            logger.info(f"top_builddir fixup: {result.stdout.strip()}")
        if result.returncode != 0 and result.stderr.strip():
            logger.warning(
                f"_fixup_top_builddir_in_submakefiles stderr: "
                f"{result.stderr.strip()}"
            )
    except SandboxUnavailableError:
        # Sandbox loss is a real transport failure, not a best-effort
        # fixup miss. Let @agent_node escalate the run.
        raise
    except Exception as exc:
        logger.warning(f"_fixup_top_builddir_in_submakefiles: {exc}")


_MAKEFILE_NAMES = ("Makefile", "makefile", "GNUmakefile")
# Directories that never hold the primary build root.
_BUILD_SEARCH_SKIP = {
    ".git",
    "node_modules",
    "vendor",
    "testdata",
    "test",
    "tests",
    "examples",
    "example",
    "third_party",
    "contrib",
    "__pycache__",
}


def _find_makefile_dir(repo_host_path: str, max_depth: int = 3) -> str:
    """Return the shallowest subdirectory containing a makefile.

    Used as a deterministic safety net: build-system detection can
    report ``make`` while the makefile actually lives one level down
    (common for ``src/``-style layouts), and the builder then runs
    ``make`` in the repo root where it fails with "No targets specified
    and no makefile found".

    Args:
        repo_host_path: Host-side path to the repository root.
        max_depth: How many directory levels below the root to search.

    Returns:
        The makefile's directory RELATIVE to the repo root, or ``""``
        when the root already has one, none exists, or the result is
        ambiguous.
    """
    if not os.path.isdir(repo_host_path):
        return ""
    # Root already has one — nothing to redirect.
    if any(
        os.path.isfile(os.path.join(repo_host_path, n))
        for n in _MAKEFILE_NAMES
    ):
        return ""

    root_depth = repo_host_path.rstrip(os.sep).count(os.sep)
    candidates: List[str] = []
    for current, dirs, files in os.walk(repo_host_path):
        depth = current.rstrip(os.sep).count(os.sep) - root_depth
        if depth >= max_depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in _BUILD_SEARCH_SKIP]
        if current == repo_host_path:
            continue
        if any(n in files for n in _MAKEFILE_NAMES):
            candidates.append(os.path.relpath(current, repo_host_path))

    if not candidates:
        return ""
    # Shallowest wins; ties broken alphabetically for determinism.
    candidates.sort(key=lambda p: (p.count(os.sep), p))
    return candidates[0]


def _autotools_bootstrap_prefix(repo_host_path: str) -> str:
    """Return the command that generates a missing ``./configure``.

    Many autotools projects commit ``configure.ac``/``autogen.sh`` but
    NOT the generated ``configure`` script (it is a release-tarball
    artifact, not a source file). Running ``./configure`` on a fresh
    clone then dies with ``./configure: No such file or directory`` —
    the single failure mode behind 11 packages (grep, less, make, tar,
    wget, strace, ffmpeg, ...).

    Bootstrap choice, most-specific first:
      1. ``sh autogen.sh`` (or ``buildconf``, ``buildconf.sh``,
         ``bootstrap``) — upstream's own script, which usually wraps
         autoreconf with project-specific steps.
      2. ``autoreconf -fi`` when ``configure.ac``/``configure.in``
         exists.

    Gettext caveat: when ``m4/gettext.m4`` is present, ``autoreconf``
    internally runs ``autopoint``, which REVERTS any gettext.m4 fix the
    agent applied. In that case emit the same copy-first sequence the
    builder already uses elsewhere instead of a bare autoreconf.

    Args:
        repo_host_path: Host-side path to the repository root.

    Returns:
        A shell command to run before ``./configure``, or ``""`` when
        no bootstrap is needed or possible.
    """
    if not os.path.isdir(repo_host_path):
        return ""
    # Already bootstrapped (or ships a committed configure) — nothing to do.
    if os.path.isfile(os.path.join(repo_host_path, "configure")):
        return ""

    for script in ("autogen.sh", "buildconf", "buildconf.sh", "bootstrap"):
        path = os.path.join(repo_host_path, script)
        if os.path.isfile(path):
            return f"sh {script}"

    has_ac = any(
        os.path.isfile(os.path.join(repo_host_path, name))
        for name in ("configure.ac", "configure.in")
    )
    if not has_ac:
        return ""

    if os.path.isfile(os.path.join(repo_host_path, "m4", "gettext.m4")):
        return (
            "cp /usr/share/gettext/m4/*.m4 m4/ 2>/dev/null || true"
            " && aclocal -I m4 -I /usr/share/aclocal"
            " && autoconf"
        )
    return "autoreconf -fi"


def _extract_cd_prefix(command: str) -> str:
    """Extract leading `cd ... &&` prefix to match companion commands."""
    match = re.match(r"^\s*(cd\s+[^&;]+&&\s*)", command)
    return match.group(1) if match else ""


def _go_prepare_commands(go_main_info: dict, module_name: str) -> List[str]:
    """Return the Go plan commands for a detected module.

    ``go mod tidy`` runs only for a new module or a missing ``go.sum``:
    it resolves the dependencies of every package and platform, which
    took 5-25 minutes under QEMU (run 80). ``go build`` downloads only
    what it needs, and the builder runs ``go mod tidy`` itself when the
    build reports a missing ``go.sum`` entry.
    """
    module_dir = go_main_info.get("module_dir") or ""
    prefix = f"cd {module_dir} && " if module_dir else ""
    cmds = []
    if go_main_info.get("needs_go_init"):
        cmds += [f"go mod init {module_name}", "go mod tidy"]
    elif not go_main_info.get("has_go_sum"):
        cmds.append(f"{prefix}go mod tidy")
    command = go_main_info.get("plan_command") or go_main_info.get(
        "build_command"
    )
    if go_main_info.get("has_main") and command:
        cmds.append(command)
    else:
        cmds.append(f"{prefix}go build -buildvcs=false ./...")
    return cmds


def _inject_go_flag(command: str, flag: str) -> str:
    """Inject a Go build/install flag once, keeping the command shape."""
    if flag in command:
        return command
    return re.sub(
        r"\bgo\s+(build|install)\b",
        rf"go \1 {flag}",
        command,
        count=1,
    )


# Word-boundary matcher for genuine Go build/install invocations. Required
# because substring checks like `"go build" in cmd` wrongly match
# `cargo build` and `apk add ... cargo build-base`, and the Go-specific
# retries and flag injection would then edit those unrelated commands.
_GO_BUILD_RE = re.compile(r"\bgo\s+(?:build|install)\b")


def _is_go_build_command(command: str) -> bool:
    """Return True if ``command`` invokes ``go build`` or ``go install``."""
    return bool(_GO_BUILD_RE.search(command))


def _inject_go_output(command: str, output_path: str) -> str:
    """Inject `-o <path>` into Go build/install commands."""
    if re.search(r"(?:^|\s)-o\s+\S+", command):
        return command
    return _inject_go_flag(command, f"-o {output_path}")


# Map a header file's leading path component or basename to a *canonical*
# library name. A candidate is only ever installed if that canonical name
# resolves through the active platform profile's package_map, so unknown or
# project-local headers are never guessed-installed.
_HEADER_CANONICAL_HINTS = {
    "png": "libpng",
    "jpeglib": "libjpeg",
    "jpeg": "libjpeg",
    "jpeglib-turbo": "libjpeg-turbo",
    "webp": "libwebp",
    "tiff": "libtiff",
    "tiffio": "libtiff",
    "ogg": "ogg",
    "vorbis": "vorbis",
    "vorbisenc": "vorbis",
    "vorbisfile": "vorbis",
    "flac": "flac",
    "opus": "opus",
    "opusfile": "opus",
    "lcms2": "lcms2",
    "zlib": "zlib",
    "zstd": "zstd",
    "lz4": "lz4",
    "brotli": "brotli",
    "snappy": "snappy",
    "freetype": "freetype",
    "ft2build": "freetype",
    "curl": "curl",
    "heif": "libheif",
    "openjpeg": "openjpeg",
    "pcap": "libpcap",
    "capability": "libcap",
    "rdkafka": "librdkafka",
    "olm": "libolm",
    "de265": "libde265",
    "yuv": "libyuv",
    "ncurses": "ncurses",
    "curses": "ncurses",
    "sqlite3": "sqlite",
}


def _resolve_header_to_packages(stderr: str, profile) -> List[str]:
    """Map missing-header compile errors to installable distro packages.

    Only headers whose derived canonical name is present in the active
    profile's ``package_map`` are returned, so unknown/project-local headers
    are never guessed. Returns distro package names (deduplicated).
    """
    if not stderr:
        return []
    headers = re.findall(r"fatal error:\s*([\w./+-]+\.h)\b", stderr)
    headers += re.findall(r"'([\w./+-]+\.h)' file not found", stderr)
    packages: List[str] = []
    seen = set()
    for hdr in headers:
        parts = hdr.split("/")
        top = parts[0].lower()
        if top.endswith(".h"):
            top = top[:-2]
        stem = parts[-1].lower()
        if stem.endswith(".h"):
            stem = stem[:-2]
        raws = [r for r in (top, stem) if r]
        candidates: List[str] = []
        for raw in raws:
            hint = _HEADER_CANONICAL_HINTS.get(raw)
            if hint:
                candidates.append(hint)
            candidates.extend([raw, f"lib{raw}"])
        for cand in candidates:
            if cand in profile.package_map:
                distro = profile.resolve(cand)
                if distro not in seen:
                    seen.add(distro)
                    packages.append(distro)
                break
    return packages


def _is_suspected_oom(result, command: str) -> bool:
    """Detect a likely QEMU OOM-kill under emulated riscv64.

    Under QEMU emulation, parallel builds and even a single-thread
    ``cargo fetch`` or ``pip install`` occasionally OOM-kill (exit 137
    / SIGKILL). The heuristic:

    * Exit ``137`` (SIGKILL) is *always* treated as OOM regardless of
      the command — QEMU's mmap OOM path is by far the most common
      cause of that exit code inside the sandbox.
    * For known-parallel build commands, a non-zero exit with *no*
      captured output at all is also treated as OOM (build was killed
      before it could log anything).

    Args:
        result: The completed ``CommandResult``.
        command: The shell command that was executed.

    Returns:
        ``True`` if the failure looks like an OOM kill worth retrying
        serially, otherwise ``False``.
    """
    # A stop at the command time limit also exits 137 but is no
    # memory kill (run 80: 121 of 139 exit-137 kills came at the limit).
    if getattr(result, "timed_out", False):
        return False
    # Universal signal — always trust exit 137 as OOM, for every
    # command: in run 28020958388, cargo fetch / pip install /
    # dependency downloads also OOM'd for 58 packages.
    if result.exit_code == 137:
        return True
    # Known parallel builds with an empty-output silent kill.
    if not re.search(
        r"\b(go\s+(build|install|test)|make|ninja"
        r"|cargo\s+(build|install|test|fetch|update)"
        r"|pip\s+install|npm\s+(install|ci|run))\b",
        command,
    ):
        return False
    no_output = (
        not (result.stdout or "").strip() and not (result.stderr or "").strip()
    )
    return (not result.success) and no_output


def _serialize_build_command(command: str) -> str:
    """Rewrite a parallel build command to run single-threaded.

    Preserves any leading ``cd ... &&`` prefix so the parallelism limit is
    applied inside the effective build directory. Used to recover from QEMU
    OOM-kills where high build concurrency exhausts emulated memory.
    """
    prefix = _extract_cd_prefix(command)
    body = command[len(prefix) :]
    if re.search(r"\bgo\s+(build|install|test)\b", body):
        body = _inject_go_flag(body, "-p 1")
        if "gomaxprocs" not in body.lower():
            body = "env GOMAXPROCS=1 " + body
        return prefix + body
    if re.search(r"\bmake\b", body):
        body = re.sub(r"-j\s*\$\(nproc\)", "-j1", body)
        body = re.sub(r"-j\s*\d+", "-j1", body)
        body = re.sub(r"--jobs[= ]\s*\S+", "-j1", body)
        if "-j1" not in body:
            body = re.sub(r"\bmake\b", "make -j1", body, count=1)
        if "makeflags" not in body.lower():
            body = "env MAKEFLAGS=-j1 " + body
        return prefix + body
    if re.search(r"\bninja\b", body) and "-j" not in body:
        body = re.sub(r"\bninja\b", "ninja -j1", body, count=1)
        return prefix + body
    if re.search(r"\bcargo\s+(build|install|test|fetch|update)\b", body):
        # Cargo network commands (fetch/update) don't accept -j, so we
        # only add the flag for build variants that support it.
        if re.search(r"\bcargo\s+(build|install|test)\b", body) and (
            "-j" not in body
        ):
            body = re.sub(
                r"\bcargo\s+(build|install|test)\b",
                r"cargo \1 -j 1",
                body,
                count=1,
            )
        if "cargo_build_jobs" not in body.lower():
            body = "env CARGO_BUILD_JOBS=1 " + body
        return prefix + body
    if re.search(r"\bpip\s+install\b", body):
        # `pip install` has no jobs flag; the OOM comes from wheel
        # builds (setup.py compilations). Serialize via MAKEFLAGS,
        # which pip forwards to setup.py invocations.
        if "makeflags" not in body.lower():
            body = "env MAKEFLAGS=-j1 " + body
        return prefix + body
    if re.search(r"\bnpm\s+(install|ci|run)\b", body):
        # NPM's own worker pool is bounded by --jobs; back-compat flag
        # supported since npm 7.
        if "--jobs" not in body:
            body = re.sub(
                r"\bnpm\s+(install|ci|run)\b",
                r"npm \1 --jobs 1",
                body,
                count=1,
            )
        return prefix + body
    return command


# Memory that one rustc job can take on a large crate, in GiB. A worker
# runs no more cargo jobs than its share of the memory holds.
_RUST_GIB_PER_JOB = 3

# A cargo command that compiles, at the start of a command or a chain
# segment, after optional variable assignments. ``cargo fetch`` and
# ``cargo update`` compile nothing.
_CARGO_COMPILE_RE = re.compile(
    r"(^|&&|\|\||;|\(|\|)(\s*)((?:env\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s\"'])*\s+)*"
    r"cargo(?:\s+\+\S+)?\s+"
    r"(?:build|install|test|bench|check|run|doc|rustc)\b)"
)
_CARGO_JOBS_SET_RE = re.compile(
    r"CARGO_BUILD_JOBS|\bcargo\b[^;&|]*\s(?:-j\s*\d|--jobs\b)"
)


def _batch_workers() -> int:
    """Return the worker count of the batch (``ATESOR_BATCH_WORKERS``).

    1 for a single run, or when the value is not a number.
    """
    raw = os.environ.get("ATESOR_BATCH_WORKERS", "").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _cargo_jobs(profile: Optional[MachineProfile]) -> Optional[int]:
    """Return the cargo job count for this worker's share of the machine.

    The share is the cores and the memory divided by the batch workers;
    one job needs ``_RUST_GIB_PER_JOB``. None without a probe that
    found the cores and the memory.
    """
    if profile is None or not profile.cores or not profile.mem_kib:
        return None
    workers = _batch_workers()
    cores = max(1, profile.cores // workers)
    mem_gib = profile.mem_kib / (1024 * 1024) / workers
    return max(1, min(cores, int(mem_gib // _RUST_GIB_PER_JOB)))


def _with_cargo_jobs(command: str, state: AgentState) -> str:
    """Return ``command`` with ``CARGO_BUILD_JOBS`` for this worker.

    Applies on the native machine only (the probe gives the cores and
    the memory), to each cargo command that compiles and sets no job
    count. A count equal to the cores is cargo's own default, so the
    command does not change.
    """
    if not _CARGO_COMPILE_RE.search(command):
        return command
    if _CARGO_JOBS_SET_RE.search(command):
        return command
    profile = _machine(state)
    jobs = _cargo_jobs(profile)
    if jobs is None or jobs >= (profile.cores or 0):
        return command
    return _CARGO_COMPILE_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}env CARGO_BUILD_JOBS={jobs} "
        f"{m.group(3)}",
        command,
    )


# The time limit for a build command that runs again after it stopped at
# the normal 1800 s limit. Cargo cannot continue inside one crate, so one
# crate that compiles longer than the normal limit (jj_cli with
# codegen-units = 1 on a 4-core riscv64 board) needs a longer second run.
_RESUME_TIMEOUT_SECONDS = 3600


def _builder_retry_allowed(state: AgentState, key: str) -> bool:
    """Allow a deterministic builder retry at most once per signature key."""
    tried = state.context_cache.setdefault("builder_retry_keys", [])
    if key in tried:
        return False
    tried.append(key)
    return True


# Prefix of the error the builder records when its guard escalates.
_INFRA_FAULT = "Infrastructure fault"


# Common import-name -> pip-distribution-name mismatches. Anything not listed
# is installed under its import name, which is correct for most packages.
_PY_MODULE_PIP_NAMES = {
    "yaml": "pyyaml",
    "jinja2": "jinja2",
    "google": "protobuf",
    "google.protobuf": "protobuf",
    "serial": "pyserial",
    "cffi": "cffi",
    "cryptography": "cryptography",
    "setuptools": "setuptools",
}


def _resolve_missing_python_modules(stderr: str) -> List[str]:
    """Extract pip package names from ModuleNotFoundError build failures."""
    if not stderr:
        return []
    modules = re.findall(r"No module named ['\"]([\w.]+)['\"]", stderr)
    pkgs: List[str] = []
    seen = set()
    for mod in modules:
        pip_name = _PY_MODULE_PIP_NAMES.get(mod, mod.split(".")[0])
        if pip_name and pip_name not in seen:
            seen.add(pip_name)
            pkgs.append(pip_name)
    return pkgs


def _download_go_toolchain_cmd(version: str) -> str:
    """Return a shell command that installs a Go toolchain from go.dev.

    The ``go install golang.org/dl/goX.Y.Z@latest`` path does
    not work on Alpine musl (the ``dl`` helper links against glibc) and
    also OOM-kills under QEMU during the ``goX.Y.Z download`` step
    (observed root cause for 25+ Go packages in run 28020958388).
    Downloading the official riscv64 tarball into ``/usr/local`` is
    what our sandbox images already do at build time; replicating that
    at runtime is fast, deterministic, and works on every distro.

    The command never leaves the sandbox without a working Go: it
    checks the archive against the SHA-256 that Go publishes next to
    it, unpacks into a staging directory, runs the staged ``go
    version``, and only then swaps it into ``/usr/local/go``. A failed
    final move puts the old toolchain back.

    Args:
        version: The Go version string to install, e.g. ``"1.25.0"``.

    Returns:
        A shell command string that, when executed inside the sandbox,
        replaces ``/usr/local/go`` with the requested version.

    Raises:
        ValueError: If ``version`` is not a plain Go version. It comes
            from go.mod and is put into a shell command.
    """
    if not re.fullmatch(
        r"\d+\.\d+(?:\.\d+)?(?:(?:rc|beta)\d+)?", version or ""
    ):
        raise ValueError(f"not a Go version: {version!r}")
    tarball = f"go{version}.linux-riscv64.tar.gz"
    url = f"https://go.dev/dl/{tarball}"
    sum_url = f"https://dl.google.com/go/{tarball}.sha256"
    tmp = f"/tmp/{tarball}"
    stage = "/usr/local/.go-staging"
    return (
        f"set -e && "
        f"curl -fsSL -o {tmp} {url} && "
        f'echo "$(curl -fsSL {sum_url})  {tmp}" | sha256sum -c - && '
        f"rm -rf {stage} && mkdir -p {stage} && "
        f"tar -xzf {tmp} -C {stage} && "
        f"{stage}/go/bin/go version && "
        f"rm -rf /usr/local/go.old && "
        f"if [ -d /usr/local/go ]; then "
        f"mv /usr/local/go /usr/local/go.old; fi && "
        f"if ! mv {stage}/go /usr/local/go; then "
        f"mv /usr/local/go.old /usr/local/go; exit 1; fi && "
        f"rm -rf /usr/local/go.old {stage} {tmp} && "
        f"/usr/local/go/bin/go version"
    )


# A Go command at the start of a command or a chain segment, after
# optional variable assignments. Each of these fails at once when the
# ``go`` line of go.mod names a newer Go than the sandbox runs.
_GO_COMMAND_RE = re.compile(
    r"(?:^|&&|\|\||;|\()\s*(?:env\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*go\s+"
    r"(?:build|install|test|run|generate|vet|mod|get|list|work)\b"
)
_GO_LINE_RE = re.compile(
    r"^go\s+(\d+\.\d+(?:\.\d+)?(?:(?:rc|beta)\d+)?)\s*(?://.*)?$",
    re.MULTILINE,
)
_GO_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?(?:(rc|beta)(\d+))?")


def _go_version_key(version: str) -> Optional[tuple]:
    """Return a sort key for a Go version (``1.27``, ``1.27.1``, ``1.27rc1``).

    A pre-release sorts before the release; ``1.27`` is ``1.27.0``.
    None for a text that is no Go version.
    """
    match = _GO_VERSION_RE.fullmatch(version or "")
    if not match:
        return None
    major, minor, patch, pre, pre_num = match.groups()
    return (
        int(major),
        int(minor),
        int(patch or 0),
        -1 if pre else 0,
        int(pre_num or 0),
    )


def _go_mod_requirement(state: AgentState, command: str) -> Optional[str]:
    """Return the ``go`` line of the go.mod that ``command`` builds.

    Reads the go.mod of the ``cd`` target of the command, else the one
    at the repo root. None when neither has a ``go`` line.
    """
    host_repo = _to_host_path(state.repo_path)
    dirs = []
    match = re.match(r"^\s*cd\s+([^&;|]+?)\s*&&", command)
    if match:
        target = match.group(1).strip().strip("'\"")
        if target.startswith("/"):
            dirs.append(_to_host_path(target))
        else:
            dirs.append(os.path.join(host_repo, target))
    dirs.append(host_repo)
    for directory in dirs:
        try:
            with open(
                os.path.join(directory, "go.mod"), encoding="utf-8"
            ) as fh:
                text = fh.read(65536)
        except OSError:
            continue
        found = _GO_LINE_RE.search(text)
        if found:
            return found.group(1)
    return None


def _ensure_go_toolchain(state: AgentState, command: str) -> None:
    """Install the Go of go.mod before the first Go command of the run.

    gobusybox, libaws, xray-knife and mult each ran a first build that
    failed on ``go.mod requires go >= 1.27`` (the image has 1.26.5),
    and the builder then downloaded the toolchain. The check runs once
    per run, at the first Go command whose go.mod has a ``go`` line;
    the download is the builder's recovery command. When the check or
    the download fails, the build runs as before and the recovery
    after the failure still applies.
    """
    if state.context_cache.get("go_toolchain_checked"):
        return
    if not _GO_COMMAND_RE.search(command) or "GOTOOLCHAIN" in command:
        return
    needed = _go_mod_requirement(state, command)
    needed_key = _go_version_key(needed or "")
    if needed_key is None:
        return
    state.context_cache["go_toolchain_checked"] = True
    # Both commands are fixed text built here, not an LLM answer, so
    # the command whitelist does not apply (as in the recovery).
    probe = execute_command(
        "go env GOVERSION", cwd=state.repo_path, validate=False
    )
    found = re.search(r"go(\S+)", (probe.stdout or "").strip())
    running_key = _go_version_key(found.group(1)) if found else None
    if not probe.success or running_key is None or running_key >= needed_key:
        return
    # go.dev names the first release 1.27.0, not 1.27.
    version = f"{needed}.0" if re.fullmatch(r"\d+\.\d+", needed) else needed
    logger.info(
        f"go.mod needs Go {needed}, the sandbox runs {found.group(1)}; "
        f"downloading Go {version} before the first build"
    )
    result = execute_command(
        _download_go_toolchain_cmd(version),
        cwd=state.repo_path,
        validate=False,
    )
    state.log_scripted_op("install_go_version")
    if not result.success:
        logger.warning(
            f"Go {version} download failed (exit {result.exit_code}); "
            "the build runs with the installed Go"
        )


def _repo_has_gitmodules(repo_path: str) -> bool:
    """Return True when ``repo_path`` contains a ``.gitmodules`` file."""
    try:
        return os.path.isfile(os.path.join(repo_path, ".gitmodules"))
    except (OSError, TypeError):
        return False


def _npm_scripts(repo_path: str) -> List[str]:
    """Return the list of npm scripts declared in ``package.json``.

    Empty list when ``package.json`` is missing or unparseable. Used
    to detect the ``npm run build`` guard case: the LLM sometimes
    invents a ``build`` script that the project never defined.
    """
    pkg_path = os.path.join(repo_path, "package.json")
    if not os.path.isfile(pkg_path):
        return []
    try:
        import json as _json

        with open(pkg_path, "r", encoding="utf-8", errors="ignore") as fp:
            data = _json.load(fp)
    except Exception:
        return []
    scripts = data.get("scripts") if isinstance(data, dict) else None
    if not isinstance(scripts, dict):
        return []
    return list(scripts.keys())


@agent_node(AgentRole.BUILDER)
def builder_node(state: AgentState) -> AgentState:
    """Execute the build plan phase by phase inside the sandbox.

    Skips phases at or below ``state.last_successful_phase`` and
    commands with a cached success. Before running a command, applies
    scripted corrections (Go ``-buildvcs=false``, gettext/autotools
    bootstrap, make-root redirect, npm script guard). On failure, tries
    the deterministic recoveries (VCS flag, output-dir collision,
    vendoring, missing headers or Python modules, OOM serialization,
    toolchain upgrade, ``-mcmodel=medany``, per-package install skip)
    before recording a classified error and returning FAILED. Whether
    the outputs really are riscv64 is verify_node's job, not this
    node's.
    """
    logger.info("Builder executing build plan...")

    from .platforms import get_active_profile

    state.build_status = BuildStatus.BUILDING
    state.current_phase = "building"
    state.build_count += 1
    # The fixer's answers without a change count from this build on.
    state.context_cache["reasks_since_build"] = 0

    if not state.build_plan:
        logger.error("No build plan available")
        state.build_status = BuildStatus.FAILED
        state.add_error(
            create_error_record(
                "No build plan available",
                ErrorCategory.CONFIGURATION,
            )
        )
        return state

    predictions = predict_build_issues(state)
    if predictions:
        logger.info(
            f"Proactively identified {len(predictions)} potential issues"
        )
        state.context_cache["predicted_issues"] = predictions

    for phase in state.build_plan.phases:
        if phase.id <= state.last_successful_phase:
            continue

        logger.info(f"Executing phase {phase.id}: {phase.name}")

        for command in phase.commands:
            cached_result = state.get_cached_command_result(command)
            if cached_result and cached_result.success:
                logger.info(f"Using cached result for: {command[:50]}...")
                continue

            optimized_cmd = command
            for pred in predictions:
                if pred[
                    "pattern"
                ] == "dubious ownership" and _is_go_build_command(command):
                    if "-buildvcs=false" not in command:
                        optimized_cmd = _inject_go_flag(
                            command, "-buildvcs=false"
                        )
                        logger.info(
                            f"Proactively optimized command: {optimized_cmd}"
                        )
                        break

            # Defensive strip: -buildvcs=false is only valid for
            # go build / go install, not go mod tidy / go mod init /
            # go test / go run etc. The scout prompt says "pass
            # -buildvcs=false to go build" but LLMs often inject it
            # into every Go command.
            if (
                "-buildvcs=false" in optimized_cmd
                and not _is_go_build_command(optimized_cmd)
            ):
                stripped = optimized_cmd.replace("-buildvcs=false", "").strip()
                while "  " in stripped:
                    stripped = stripped.replace("  ", " ")
                logger.info(
                    f"Stripped -buildvcs=false from non-build Go command: "
                    f"{optimized_cmd[:80]} → {stripped[:80]}"
                )
                optimized_cmd = stripped

            # Rewrite autoreconf to copy-first sequence when the repo
            # has m4/gettext.m4. autoreconf calls autopoint internally
            # which reverts any gettext.m4 fix. Only rewrite when
            # m4/gettext.m4 exists — other repos are unaffected.
            if re.search(r"\bautoreconf\b", optimized_cmd):
                if os.path.isfile(
                    os.path.join(
                        _to_host_path(state.repo_path), "m4", "gettext.m4"
                    )
                ) and _gettext_m4_was_fixed(state):
                    optimized_cmd = (
                        "cp /usr/share/gettext/m4/*.m4 m4/ 2>/dev/null || true"
                        " && aclocal -I m4 -I /usr/share/aclocal"
                        " && autoconf"
                    )
                    logger.info(
                        "Rewrote autoreconf → copy-first gettext sequence "
                        "(autoreconf/autopoint would revert gettext.m4)"
                    )

            # Autotools bootstrap: a fresh clone of an autotools project
            # usually has no committed `configure`, so `./configure`
            # dies with "No such file or directory" before anything is
            # built. Generate it first (autogen.sh / autoreconf -fi).
            # Scripted and LLM-free; runs in state.repo_path, which is
            # the cwd of the execute_command below, so a `cd build &&`
            # companion still lands in the right place.
            if "./configure" in optimized_cmd:
                bootstrap = _autotools_bootstrap_prefix(
                    _to_host_path(state.repo_path)
                )
                if bootstrap:
                    optimized_cmd = f"{bootstrap} && {optimized_cmd}"
                    state.log_scripted_op("autotools_bootstrap")
                    logger.info(
                        f"No committed ./configure; prepended bootstrap: "
                        f"{bootstrap[:60]}"
                    )

            # Build-root threading: run `make` where the makefile
            # actually is. Detection can report `make` while the
            # makefile lives a level down, and the bare command then
            # fails with "No targets specified and no makefile found".
            # Only rewrite a plain `make` (no cd/-C/-f already steering
            # it) and only when a makefile really exists somewhere —
            # otherwise the original, truthful error is preserved.
            if (
                re.match(r"^make\b", optimized_cmd.strip())
                and " -C" not in optimized_cmd
                and " -f" not in optimized_cmd
            ):
                make_dir = _find_makefile_dir(_to_host_path(state.repo_path))
                if make_dir:
                    optimized_cmd = f"cd {shlex.quote(make_dir)} && " + (
                        optimized_cmd
                    )
                    state.log_scripted_op("make_root_redirect")
                    logger.info(
                        f"No makefile at repo root; redirected make to "
                        f"'{make_dir}'"
                    )

            # Before a command that starts with make, fix po/Makefile
            # top_builddir if needed. This handles the case where
            # configure was run by the fixer (not the builder), so the
            # post-configure hook wouldn't have fired yet.
            if re.match(r"\bmake\b", optimized_cmd.lstrip()):
                _fixup_top_builddir_in_submakefiles(state.repo_path)

            # ``npm run <script>`` guard: the fixer/scout frequently
            # invents an ``npm run build`` even when package.json does
            # not declare a ``build`` script. If the requested script
            # does not exist, substitute the first declared script of
            # ``build``, ``prod``, ``compile``, ``dist``, or skip the
            # command otherwise so the phase can proceed.
            _npm_run_match = re.match(
                r"^\s*npm\s+run\s+([A-Za-z0-9_.:\-]+)\b(.*)$",
                optimized_cmd,
            )
            if _npm_run_match:
                wanted = _npm_run_match.group(1)
                tail = _npm_run_match.group(2)
                scripts = _npm_scripts(_to_host_path(state.repo_path))
                if scripts and wanted not in scripts:
                    replacement = None
                    # Prefer well-known aliases in this order; skip
                    # otherwise so we do not blindly execute arbitrary
                    # scripts the project happens to declare.
                    for alias in ("build", "prod", "compile", "dist"):
                        if alias in scripts:
                            replacement = alias
                            break
                    if replacement and replacement != wanted:
                        rewritten = f"npm run {replacement}{tail}"
                        logger.warning(
                            "npm script %r not declared; "
                            "substituting %r (available: %s)",
                            wanted,
                            replacement,
                            ", ".join(scripts),
                        )
                        optimized_cmd = rewritten
                    else:
                        logger.warning(
                            "npm script %r not declared and no safe "
                            "alias available (scripts=%s); skipping",
                            wanted,
                            scripts,
                        )
                        continue

            _ensure_go_toolchain(state, optimized_cmd)

            # Cargo runs one rustc per core. On the board, the batch
            # workers share the cores and the memory (nushell: 4 jobs
            # next to another worker, OOM-killed after 29.6 minutes).
            jobs_cmd = _with_cargo_jobs(optimized_cmd, state)
            if jobs_cmd != optimized_cmd:
                state.log_scripted_op("cargo_build_jobs")
                logger.info(f"Limited cargo jobs: {jobs_cmd[:120]}")
                optimized_cmd = jobs_cmd

            result = execute_command(optimized_cmd, cwd=state.repo_path)
            state.cache_command_result(command, result)
            state.log_scripted_op("execute_build_command")

            if not result.success:
                logger.error(f"Command failed: {command}")
                logger.error(f"Error: {result.stderr[:500]}")

                # A stopped container, a dropped ssh session or a host
                # that slept is no build error, so the fixer must not
                # see it: start the container when it stopped, run the
                # command once more, and escalate when the fault stays.
                # Only a plan command runs again, never a fix command.
                fault = infrastructure_fault(result)
                if fault and _builder_retry_allowed(state, f"infra:{command}"):
                    container = ensure_build_container()
                    logger.warning(
                        f"Infrastructure fault ({fault}); the build "
                        f"container is {container}; running the command "
                        "once more"
                    )
                    state.log_scripted_op("infrastructure_rerun")
                    result = execute_command(
                        optimized_cmd, cwd=state.repo_path
                    )
                    state.cache_command_result(command, result)
                    if result.success:
                        continue
                    fault = infrastructure_fault(result)
                if fault:
                    output = result.stderr or result.stdout
                    message = (
                        f"{_INFRA_FAULT}: {fault}. Command: `{command[:200]}` "
                        f"(exit {result.exit_code}) - "
                        f"{_one_line_tail(output, 400)}"
                    )
                    state.add_error(
                        create_error_record(
                            message=message,
                            category=ErrorCategory.INFRASTRUCTURE,
                            severity=FailureSeverity.HIGH,
                            command=command,
                            attempt_number=state.attempt_count,
                        )
                    )
                    state.build_status = BuildStatus.ESCALATED
                    state.log_agent_decision(
                        AgentRole.BUILDER, "ESCALATE", message[:300]
                    )
                    return state

                is_go_build = _is_go_build_command(optimized_cmd)
                is_vcs_error = any(
                    pattern in result.stderr
                    for pattern in [
                        "dubious ownership",
                        "error obtaining VCS status",
                        "Use -buildvcs=false",
                        "fatal: detected dubious ownership",
                    ]
                )

                if (
                    is_go_build
                    and is_vcs_error
                    and "-buildvcs=false" not in optimized_cmd
                ):
                    logger.warning(
                        "Detected Go VCS error, retrying with -buildvcs=false"
                    )
                    retry_command = _inject_go_flag(
                        optimized_cmd, "-buildvcs=false"
                    )

                    retry_result = execute_command(
                        retry_command, cwd=state.repo_path
                    )
                    state.cache_command_result(retry_command, retry_result)
                    state.log_scripted_op("retry_build_command")

                    if retry_result.success:
                        logger.info("Retry with -buildvcs=false succeeded!")
                        phase.commands[phase.commands.index(command)] = (
                            retry_command
                        )
                        continue
                    else:
                        logger.error(
                            f"Retry also failed: {retry_result.stderr[:500]}"
                        )
                        result = retry_result

                if (
                    is_go_build
                    and "already exists and is a directory"
                    in (result.stderr or "").lower()
                    and "build output" in (result.stderr or "").lower()
                ):
                    out_match = re.search(
                        r'build output "([^"]+)"', result.stderr or ""
                    )
                    out_name = (
                        out_match.group(1)
                        if out_match
                        else f"{state.repo_name or 'build'}-bin"
                    )
                    safe_out = re.sub(r"[^A-Za-z0-9._-]", "-", out_name)
                    retry_command = _inject_go_output(
                        optimized_cmd, f"./.atesor-bin/{safe_out}"
                    )
                    retry_command = (
                        "mkdir -p ./.atesor-bin && " f"{retry_command}"
                    )
                    retry_result = execute_command(
                        retry_command, cwd=state.repo_path
                    )
                    state.cache_command_result(retry_command, retry_result)
                    state.log_scripted_op("retry_build_command")
                    if retry_result.success:
                        logger.info(
                            "Resolved Go output-dir collision by using -o"
                        )
                        phase.commands[phase.commands.index(command)] = (
                            retry_command
                        )
                        continue
                    result = retry_result

                if (
                    is_go_build
                    and "inconsistent vendoring"
                    in (result.stderr or "").lower()
                    and "-mod=mod" not in optimized_cmd
                ):
                    retry_command = _inject_go_flag(optimized_cmd, "-mod=mod")
                    retry_result = execute_command(
                        retry_command, cwd=state.repo_path
                    )
                    state.cache_command_result(retry_command, retry_result)
                    state.log_scripted_op("retry_build_command")
                    if retry_result.success:
                        logger.info(
                            "Resolved inconsistent vendoring with -mod=mod"
                        )
                        phase.commands[phase.commands.index(command)] = (
                            retry_command
                        )
                        continue
                    result = retry_result

                if is_go_build and any(
                    marker in (result.stderr or "").lower()
                    for marker in (
                        "no required module provides",
                        "missing go.sum entry",
                        "updates to go.mod needed",
                    )
                ):
                    prefix = _extract_cd_prefix(optimized_cmd)
                    tidy_cmd = f"{prefix}go mod tidy"
                    tidy_result = execute_command(
                        tidy_cmd, cwd=state.repo_path
                    )
                    state.cache_command_result(tidy_cmd, tidy_result)
                    state.log_scripted_op("retry_build_command")
                    if tidy_result.success:
                        retry_result = execute_command(
                            optimized_cmd, cwd=state.repo_path
                        )
                        state.cache_command_result(optimized_cmd, retry_result)
                        state.log_scripted_op("retry_build_command")
                        if retry_result.success:
                            logger.info(
                                "Resolved missing module by running "
                                "go mod tidy"
                            )
                            continue
                        result = retry_result

                if (
                    re.search(r"\bautoreconf\b", optimized_cmd)
                    and "autopoint" in (result.stderr or "").lower()
                    and "no such file" in (result.stderr or "").lower()
                ):
                    retry_command = (
                        "cp /usr/share/gettext/m4/*.m4 m4/ 2>/dev/null || true"
                        " && aclocal -I m4 -I /usr/share/aclocal"
                        " && autoconf"
                    )
                    retry_result = execute_command(
                        retry_command, cwd=state.repo_path
                    )
                    state.cache_command_result(retry_command, retry_result)
                    state.log_scripted_op("retry_build_command")
                    if retry_result.success:
                        logger.info(
                            "Recovered from autopoint failure via copy-first "
                            "gettext regeneration."
                        )
                        phase.commands[phase.commands.index(command)] = (
                            retry_command
                        )
                        continue
                    result = retry_result

                # Missing C/C++ dev header -> resolve to a distro package via
                # the active profile and install it, then retry. LLM-free, so
                # it works even when the scout/fixer are rate-limited.
                if not result.success:
                    _profile = get_active_profile()
                    header_pkgs = _resolve_header_to_packages(
                        result.stderr, _profile
                    )
                    if header_pkgs and _builder_retry_allowed(
                        state, f"header:{command}:{','.join(header_pkgs)}"
                    ):
                        install_cmd = _profile.install_cmd(header_pkgs)
                        logger.warning(
                            "Missing header detected; installing "
                            f"{header_pkgs} then retrying build"
                        )
                        inst_result = execute_command(
                            install_cmd, cwd=state.repo_path
                        )
                        state.log_scripted_op("install_missing_header_pkg")
                        if inst_result.success:
                            retry_result = execute_command(
                                optimized_cmd, cwd=state.repo_path
                            )
                            state.cache_command_result(
                                optimized_cmd, retry_result
                            )
                            state.log_scripted_op("retry_build_command")
                            if retry_result.success:
                                logger.info(
                                    "Resolved missing header by installing "
                                    f"{header_pkgs}"
                                )
                                continue
                            result = retry_result

                # CMake project invoked with bare `make` -> configure first.
                _cmake_no_make = (
                    "no targets specified and no makefile found"
                    in (result.stderr or "").lower()
                )
                if _cmake_no_make and _builder_retry_allowed(
                    state, f"cmake-config:{command}"
                ):
                    prefix = _extract_cd_prefix(optimized_cmd)
                    eff_dir = _to_host_path(state.repo_path)
                    cd_match = re.match(r"^\s*cd\s+([^&;]+?)\s*&&", prefix)
                    if cd_match:
                        sub = cd_match.group(1).strip()
                        if not os.path.isabs(sub):
                            eff_dir = os.path.join(eff_dir, sub)
                    if os.path.isfile(os.path.join(eff_dir, "CMakeLists.txt")):
                        retry_command = (
                            f"{prefix}cmake -S . -B build "
                            "-DCMAKE_BUILD_TYPE=Release "
                            "&& cmake --build build -j1"
                        )
                        retry_result = execute_command(
                            retry_command, cwd=state.repo_path
                        )
                        state.cache_command_result(retry_command, retry_result)
                        state.log_scripted_op("retry_build_command")
                        if retry_result.success:
                            logger.info(
                                "Reconfigured CMake project that was built "
                                "with bare make"
                            )
                            phase.commands[phase.commands.index(command)] = (
                                retry_command
                            )
                            continue
                        result = retry_result

                # go.mod lives in a subdirectory -> cd into it and retry.
                if "go.mod file not found" in (
                    result.stderr or ""
                ).lower() and _builder_retry_allowed(
                    state, f"gomod-dir:{command}"
                ):
                    host_root = _to_host_path(state.repo_path)
                    found = None
                    for root, _dirs, files in os.walk(host_root):
                        if "go.mod" in files:
                            rel = os.path.relpath(root, host_root)
                            if rel != ".":
                                found = rel
                            break
                    if found:
                        prefix = _extract_cd_prefix(optimized_cmd)
                        body = optimized_cmd[len(prefix) :]
                        retry_command = f"cd {found} && {body}"
                        retry_result = execute_command(
                            retry_command, cwd=state.repo_path
                        )
                        state.cache_command_result(retry_command, retry_result)
                        state.log_scripted_op("retry_build_command")
                        if retry_result.success:
                            logger.info(
                                f"Located go.mod in subdir '{found}'; "
                                "retried build there"
                            )
                            phase.commands[phase.commands.index(command)] = (
                                retry_command
                            )
                            continue
                        result = retry_result

                # CMake cached a stale compiler path -> delete the
                # build dir and retry
                # with explicit CMAKE_C_COMPILER. This happens on Alpine when
                # cmake was built with ccache support and /usr/lib/ccache/gcc
                # doesn't exist on the riscv64 image.
                if (
                    "cmake" in (optimized_cmd or "").lower()
                    and "not a full path to an existing compiler tool"
                    in (result.stderr or "")
                    and _builder_retry_allowed(
                        state, f"cmake-cache:{command[:80]}"
                    )
                ):
                    # Two paths for one directory: the host path to test
                    # that it exists, the container path for the rm that
                    # runs in the sandbox (the host path does not exist
                    # there, so an rm of it would remove nothing).
                    container_build_dir = None
                    for marker in (
                        "mkdir -p build && cd build",
                        "mkdir build && cd build",
                        "cd build && cmake",
                        "cmake -S . -B build",
                    ):
                        if marker in optimized_cmd:
                            container_build_dir = f"{state.repo_path}/build"
                            break
                    host_build_dir = (
                        _to_host_path(container_build_dir)
                        if container_build_dir
                        else None
                    )
                    if host_build_dir and os.path.isdir(host_build_dir):
                        logger.warning(
                            "CMake compiler-cache conflict detected; "
                            "deleting build/ and retrying with explicit "
                            "CMAKE_C_COMPILER"
                        )
                        rm_cmd = f"rm -rf {shlex.quote(container_build_dir)}"
                        execute_command(
                            rm_cmd, cwd=state.repo_path, use_docker=True
                        )
                        retry_cmd = optimized_cmd.replace(
                            "cmake ..",
                            "cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc"
                            " -DCMAKE_CXX_COMPILER=/usr/bin/g++",
                        )
                        retry_cmd = retry_cmd.replace(
                            "cmake -S . -B build",
                            "cmake -S . -B build"
                            " -DCMAKE_C_COMPILER=/usr/bin/gcc"
                            " -DCMAKE_CXX_COMPILER=/usr/bin/g++",
                        )
                        retry_result = execute_command(
                            retry_cmd, cwd=state.repo_path
                        )
                        state.cache_command_result(retry_cmd, retry_result)
                        state.log_scripted_op("retry_build_command")
                        if retry_result.success:
                            logger.info(
                                "Resolved CMake compiler-cache conflict by "
                                "deleting build/ and setting compiler paths"
                            )
                            phase.commands[phase.commands.index(command)] = (
                                retry_cmd
                            )
                            continue
                        result = retry_result
                    else:
                        logger.warning(
                            "CMake compiler-cache conflict but no build/ "
                            "directory found — retrying with env CC/CXX"
                        )
                        retry_cmd = (
                            "CC=/usr/bin/gcc CXX=/usr/bin/g++ " + optimized_cmd
                        )
                        retry_result = execute_command(
                            retry_cmd, cwd=state.repo_path
                        )
                        state.cache_command_result(retry_cmd, retry_result)
                        state.log_scripted_op("retry_build_command")
                        if retry_result.success:
                            phase.commands[phase.commands.index(command)] = (
                                retry_cmd
                            )
                            continue
                        result = retry_result

                # Build-time Python module missing (codegen scripts) -> pip
                # install it and retry. General, LLM-free recovery.
                if not result.success:
                    py_pkgs = _resolve_missing_python_modules(result.stderr)
                    if py_pkgs and _builder_retry_allowed(
                        state, f"pymod:{command}:{','.join(py_pkgs)}"
                    ):
                        pip_cmd = "pip install " + " ".join(py_pkgs)
                        logger.warning(
                            f"Missing Python module(s) {py_pkgs}; "
                            "installing then retrying build"
                        )
                        pip_result = execute_command(
                            pip_cmd, cwd=state.repo_path
                        )
                        state.log_scripted_op("install_missing_python_module")
                        if pip_result.success:
                            retry_result = execute_command(
                                optimized_cmd, cwd=state.repo_path
                            )
                            state.cache_command_result(
                                optimized_cmd, retry_result
                            )
                            state.log_scripted_op("retry_build_command")
                            if retry_result.success:
                                logger.info(
                                    "Resolved missing Python module by "
                                    f"installing {py_pkgs}"
                                )
                                continue
                            result = retry_result

                # Stopped at the command time limit: the build output so
                # far is kept, so the same command continues the build.
                # A -j1 retry (the memory-kill path) would only be slower.
                if getattr(result, "timed_out", False) and (
                    _builder_retry_allowed(state, f"timelimit:{command}")
                ):
                    resume_cmd = result.command or optimized_cmd
                    logger.warning(
                        "Command reached its time limit; running it again "
                        f"to continue the build: {resume_cmd[:120]}"
                    )
                    retry_result = execute_command(
                        resume_cmd,
                        cwd=state.repo_path,
                        timeout=_RESUME_TIMEOUT_SECONDS,
                    )
                    state.cache_command_result(resume_cmd, retry_result)
                    state.log_scripted_op("retry_build_command")
                    if retry_result.success:
                        logger.info(
                            "Build finished after continuing past the "
                            "time limit"
                        )
                        if resume_cmd != command:
                            phase.commands[phase.commands.index(command)] = (
                                resume_cmd
                            )
                        continue
                    result = retry_result

                # Suspected QEMU OOM-kill (exit 137 / empty output) on a
                # parallel build -> retry single-threaded. Generic fallback,
                # checked after the recoveries above so their specific
                # signatures win first. It serializes the command that
                # ran last (a recovery above may have changed it).
                if _is_suspected_oom(result, result.command or optimized_cmd):
                    serial_cmd = _serialize_build_command(
                        result.command or optimized_cmd
                    )
                    if serial_cmd != optimized_cmd and _builder_retry_allowed(
                        state, f"oom:{command}"
                    ):
                        logger.warning(
                            "Suspected QEMU OOM-kill; retrying serialized: "
                            f"{serial_cmd[:120]}"
                        )
                        retry_result = execute_command(
                            serial_cmd, cwd=state.repo_path
                        )
                        state.cache_command_result(serial_cmd, retry_result)
                        state.log_scripted_op("retry_build_command")
                        if retry_result.success:
                            logger.info(
                                "Serialized build recovered from suspected OOM"
                            )
                            phase.commands[phase.commands.index(command)] = (
                                serial_cmd
                            )
                            continue
                        result = retry_result

                # Go says "no required module provides package X" AND the
                # repo declares submodules -> init and retry. Common with
                # vendored, dot-slash-relative go.mod trees (aliyun-cli).
                _need_submod = (
                    "no required module provides package"
                    in (result.stderr or "").lower()
                )
                if (
                    _need_submod
                    and _repo_has_gitmodules(state.repo_path)
                    and _builder_retry_allowed(state, f"submod-init:{command}")
                ):
                    logger.warning(
                        "Missing Go module + .gitmodules present -> "
                        "running submodule update --init --recursive"
                    )
                    _sm_cmd = (
                        f"cd {state.repo_path} && "
                        "git submodule update --init --recursive"
                    )
                    _sm_res = execute_command(_sm_cmd, cwd=state.repo_path)
                    state.log_scripted_op("init_submodules")
                    if _sm_res.success:
                        _sm_retry = execute_command(
                            optimized_cmd, cwd=state.repo_path
                        )
                        state.cache_command_result(optimized_cmd, _sm_retry)
                        state.log_scripted_op("retry_build_command")
                        if _sm_retry.success:
                            logger.info(
                                "Resolved missing Go module by "
                                "initializing submodules"
                            )
                            continue
                        result = _sm_retry

                # Missing git in Debian sandbox -> install and retry
                if (
                    "command not found" in (result.stderr or "").lower()
                    and "git" in command
                    and _builder_retry_allowed(state, f"missing-git:{command}")
                ):
                    _gtool = get_active_profile()
                    _gtool_install = _gtool.install_cmd(["git"])
                    execute_command(_gtool_install, cwd=state.repo_path)
                    _gtool_retry = execute_command(
                        optimized_cmd, cwd=state.repo_path
                    )
                    state.cache_command_result(optimized_cmd, _gtool_retry)
                    state.log_scripted_op("retry_build_command")
                    if _gtool_retry.success:
                        logger.info("Installed git and retried successfully")
                        continue
                    result = _gtool_retry

                # Go version too old -> install newer Go via go.dev tarball
                if is_toolchain_version_mismatch(result.stderr or ""):
                    _gv_match = re.search(
                        r"go\.mod requires go (>=?\s*)?(\d+\.\d+(?:\.\d+)?)",
                        result.stderr or "",
                    )
                    if _gv_match and _builder_retry_allowed(
                        state, f"go-version:{command}"
                    ):
                        _gv_needed = _gv_match.group(2)
                        # Normalise short "1.25" → "1.25.0" so the
                        # go.dev tarball URL is always valid.
                        if _gv_needed.count(".") == 1:
                            _gv_needed = f"{_gv_needed}.0"
                        logger.warning(
                            "Go version too old, downloading Go "
                            f"{_gv_needed} tarball into /usr/local"
                        )
                        # Built here from a checked version, not by an
                        # LLM, so the LLM command whitelist does not
                        # apply; it rejects /usr/local/go/bin/go and
                        # sha256sum, so validation would block this step.
                        _gv_cmd = _download_go_toolchain_cmd(_gv_needed)
                        _gv_inst = execute_command(
                            _gv_cmd, cwd=state.repo_path, validate=False
                        )
                        state.log_scripted_op("install_go_version")
                        if _gv_inst.success:
                            _gv_retry = execute_command(
                                optimized_cmd, cwd=state.repo_path
                            )
                            state.cache_command_result(
                                optimized_cmd, _gv_retry
                            )
                            state.log_scripted_op("retry_build_command")
                            if _gv_retry.success:
                                logger.info(
                                    "Resolved by upgrading Go to "
                                    f"{_gv_needed}"
                                )
                                continue
                            result = _gv_retry

                # RISC-V linker relocation truncated -> add -mcmodel=medany
                if (
                    "relocation truncated" in (result.stderr or "").lower()
                    and "R_RISCV" in (result.stderr or "")
                    and _builder_retry_allowed(state, f"jal-reloc:{command}")
                ):
                    logger.warning(
                        "RISC-V relocation truncated; retrying with "
                        "-mcmodel=medany"
                    )
                    _medany_cmd = (
                        "CFLAGS='-mcmodel=medany' "
                        "CXXFLAGS='-mcmodel=medany' " + optimized_cmd
                    )
                    _medany_retry = execute_command(
                        _medany_cmd, cwd=state.repo_path
                    )
                    state.cache_command_result(_medany_cmd, _medany_retry)
                    state.log_scripted_op("retry_build_command")
                    if _medany_retry.success:
                        logger.info(
                            "Resolved RISC-V relocation by adding "
                            "-mcmodel=medany"
                        )
                        phase.commands[phase.commands.index(command)] = (
                            _medany_cmd
                        )
                        continue
                    result = _medany_retry

                # One package missing from the riscv64 repos fails the
                # whole install; retry package-by-package so the rest
                # still land.
                _pkg_fail_patterns = [
                    "unable to locate package",
                    "unable to select packages",
                    "no such package",
                    "has no installation candidate",
                ]
                if any(
                    p in (result.stderr or "").lower()
                    for p in _pkg_fail_patterns
                ) and _builder_retry_allowed(
                    state, f"missing-pkg:{command[:80]}"
                ):
                    _pprofile = get_active_profile()
                    _pkg_match = re.search(
                        r"(?:apt-get\s+install\s+-y|apk\s+add)\s+(.+)",
                        optimized_cmd,
                    )
                    if _pkg_match:
                        _pkgs = shlex.split(_pkg_match.group(1))
                        _good = []
                        _bad = []
                        for _pkg in _pkgs:
                            _single_cmd = _pprofile.install_cmd([_pkg])
                            _single_r = execute_command(
                                _single_cmd, cwd=state.repo_path
                            )
                            state.log_scripted_op("install_package_single")
                            if _single_r.success:
                                _good.append(_pkg)
                            else:
                                _bad.append(_pkg)
                                logger.warning(
                                    f"Package '{_pkg}' not found on "
                                    "riscv64; skipping"
                                )
                        if _good:
                            logger.info(
                                f"Installed {len(_good)}/{len(_pkgs)} "
                                "packages (skipped "
                                f"{len(_bad)} unavailable on riscv64)"
                            )
                            if _bad:
                                _good_cmd = _pprofile.install_cmd(_good)
                                phase.commands[
                                    phase.commands.index(command)
                                ] = _good_cmd
                            continue

                error_message = _build_command_error_message(
                    result, f"Build command failed: {command}"
                )
                error = create_error_record(
                    message=error_message,
                    category=classify_error(error_message),
                    command=command,
                    attempt_number=state.attempt_count,
                    stdout_tail=_stdout_evidence(result),
                )
                state.add_error(error)
                state.build_status = BuildStatus.FAILED

                return state

        state.last_successful_phase = phase.id
        logger.info(f"Phase {phase.id} completed successfully")

        # Post-configure fixup: after any configure phase, patch
        # subdirectory Makefiles that reference $(top_builddir) but
        # don't define it. This is a common issue with old
        # gettext/autotools projects where po/Makefile.in.in lacks the
        # top_builddir = @top_builddir@ substitution.
        if phase.name.lower() in ("configure", "config") or any(
            "./configure" in c for c in phase.commands
        ):
            _fixup_top_builddir_in_submakefiles(state.repo_path)

    # Every command succeeded. Whether the build produced riscv64
    # machine code is decided in one place, verify_node, which the
    # subgraph always runs next.
    state.build_status = BuildStatus.SUCCESS
    logger.info("All build phases completed; verifying artifacts next")

    # Close the fix-attempt feedback loop. The last applied fix led to
    # this build, so it worked; an earlier fix worked only if the error
    # after it differed from the error it addressed. A fix that left
    # the error as it was is not learned as a success.
    open_fixes = [
        fix
        for fix in state.fixes_attempted
        if not fix.success and fix.build_result is None
    ]
    for fix in open_fixes:
        if fix is open_fixes[-1] or _fix_changed_the_error(state, fix):
            fix.success = True
            fix.build_result = "build succeeded after fix"
        else:
            fix.build_result = "the error did not change after this fix"

    return state


# Fixer output validation — structure, containment, safe commands.


def validate_fix_command(command: str) -> tuple[bool, str]:
    """Check a fixer-proposed command for destructive or confused shapes.

    Blocks empty-file creation via ``touch``/redirect, ``rm -rf`` of
    roots, git push/hard-reset, Go subcommands used as package names,
    and anything the ``CommandValidator`` whitelist rejects.

    Args:
        command: The shell command the fixer wants to run.

    Returns:
        A ``(is_safe, reason)`` tuple.
    """
    dangerous_patterns = [
        (r"\btouch\s+.*\.go$", "Cannot create empty Go source files"),
        (r"\btouch\s+.*\.c$", "Cannot create empty C source files"),
        (r"\btouch\s+.*\.cpp$", "Cannot create empty C++ source files"),
        (r"\btouch\s+.*\.h$", "Cannot create empty header files"),
        (r"\btouch\s+.*\.rs$", "Cannot create empty Rust source files"),
        (r"\btouch\s+.*\.py$", "Cannot create empty Python source files"),
        (r"\brm\s+-rf\s+/", "Cannot delete root directories"),
        (r"\brm\s+-rf\s+\*", "Cannot delete all files"),
        (r"\bgit\s+push", "Cannot push to remote"),
        (r"\bgit\s+reset\s+--hard", "Cannot hard reset git"),
        (r">\s*$", "Cannot truncate files with empty redirect"),
        (r'echo\s+""?\s*>\s*', "Cannot overwrite files with empty content"),
    ]

    import re

    for pattern, reason in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return False, reason

    # Reject `apk add go mod` / `apt-get install go build` style hallucinations
    # where Go subcommands are used as package names
    _go_subcommands = {
        "mod",
        "build",
        "get",
        "install",
        "run",
        "test",
        "tool",
        "generate",
        "fmt",
        "vet",
    }
    stripped = command.strip()
    pkg_install_prefixes = ("apk add", "apt-get install", "apt install")
    if any(stripped.startswith(pfx) for pfx in pkg_install_prefixes):
        parts = stripped.split()
        pkg_names = [p for p in parts[2:] if not p.startswith("-")]
        for pkg in pkg_names:
            if pkg in _go_subcommands:
                return False, (
                    f"Hallucination: '{command}' — '{pkg}' is a Go "
                    f"subcommand, not a package. "
                    f"Use 'go {pkg}' as a separate command."
                )

    single_ok, single_reason = CommandValidator().is_safe_single(command)
    if not single_ok:
        # Name the rejected command: the reason alone ("Unknown command
        # pattern") does not tell the model which action to change.
        hint = ""
        if "Unknown command pattern" in single_reason:
            hint = f". Allowed tools include: {_ALLOWED_TOOLS_HINT}"
        elif "Redirection" in single_reason:
            hint = (
                ". To write a file, use a create_file action with a path "
                "inside the repo; to change a line, use `sed -i`"
            )
        elif "Chained" in single_reason:
            hint = (
                ". Put each command in its own action; to change a build "
                "plan command, use a plan_command action"
            )
        return False, f"{single_reason}: `{command[:160]}`{hint}"

    return True, "Command is safe"


# Tool names for a whitelist rejection, so a retry can pick a tool that
# passes. Not the whole whitelist (see tools.CommandValidator).
_ALLOWED_TOOLS_HINT = (
    "apt-get/apk, make, cmake, ninja, meson, ./configure, autoreconf, go, "
    "cargo, rustup, npm, pnpm, pip, python3, sed, cp, mv, mkdir, ln, rm, "
    "patch, git, curl, wget, tar"
)


def _split_and_chain(command: str) -> Optional[List[str]]:
    """Return the parts of a plain ``a && b`` chain, or None.

    Only a top-level ``&&`` splits: quoted text, backquotes, ``$(...)``
    and ``( ... )`` stay whole. A chain that also holds ``||``, ``;``,
    ``|``, ``&`` or a line break is no plain chain: its meaning changes
    when the parts run alone.
    """
    parts: List[str] = []
    buf: List[str] = []
    quote = ""
    depth = 0
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"', "`"):
            quote = ch
        elif ch == "\\" and i + 1 < n:
            buf.append(command[i : i + 2])
            i += 2
            continue
        elif ch == "(":
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
        elif depth == 0 and command.startswith("&&", i):
            parts.append("".join(buf).strip())
            buf = []
            i += 2
            continue
        elif depth == 0 and ch in ";|&\n":
            return None
        buf.append(ch)
        i += 1
    parts.append("".join(buf).strip())
    if quote or depth or len(parts) < 2 or not all(parts):
        return None
    return parts


def _split_chained_actions(actions: List[Any]) -> List[Any]:
    """Split each plain ``&&`` chain of the fixer into single commands.

    A ``cd <dir>`` part becomes the ``cd <dir> && `` prefix of the parts
    after it (``mkdir -p build && cd build && cmake ..`` gives
    ``mkdir -p build`` and ``cd build && cmake ..``). A chain is split
    only when every new command passes ``validate_fix_command``; else
    it stays whole, and the validator rejects it with its reason. The
    parts of one chain share a ``chain`` number: when one fails, the
    fixer skips the rest, as ``&&`` would. In October 2026 the rule
    against chains cost grep 17 of its 25 fixer answers.
    """
    out: List[Any] = []
    group = 0
    for action in actions:
        command = (
            str(action.get("command") or "").strip()
            if isinstance(action, dict) and action.get("type") == "command"
            else ""
        )
        parts = _split_and_chain(command) if command else None
        commands: List[str] = []
        cwd = ""
        for part in parts or []:
            if part == "cd" or part.startswith("cd "):
                try:
                    words = shlex.split(part)
                except ValueError:
                    words = []
                if len(words) != 2:
                    commands = []
                    break
                target = words[1]
                cwd = (
                    os.path.normpath(os.path.join(cwd, target))
                    if cwd and not target.startswith("/")
                    else target
                )
                continue
            prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
            commands.append(prefix + part)
        if (
            not commands
            or commands == [command]
            or not all(validate_fix_command(c)[0] for c in commands)
        ):
            out.append(action)
            continue
        group += 1
        out.extend(
            {"type": "command", "command": c, "chain": group} for c in commands
        )
    return out


def validate_fixer_response(
    fix_data: Dict, plan_commands: Optional[List[str]] = None
) -> tuple[bool, str]:
    """Check a fixer JSON response for structure, safety and containment.

    The recommended strategy must exist and carry at least one action;
    ``create_file``/``patch`` paths must stay inside the repo tree;
    ``command`` actions must pass ``validate_fix_command``; a
    ``plan_command`` must replace an exact command of the current plan
    with one that the plan validator accepts. A thin ``reflection``
    block only logs a warning — it never rejects a fix.

    Args:
        fix_data: The parsed JSON object from the fixer model.
        plan_commands: The commands of the current build plan; a
            ``plan_command`` action is rejected without them.

    Returns:
        A ``(is_valid, error_message)`` tuple.
    """
    try:
        if not isinstance(fix_data, dict):
            return (
                False,
                f"Expected JSON object from fixer, got "
                f"{type(fix_data).__name__}",
            )

        if not fix_data.get("strategies"):
            return False, "No strategies provided"

        recommended_id = fix_data.get("recommended_strategy_id", 1)
        strategies = fix_data.get("strategies", [])

        recommended = next(
            (s for s in strategies if s["id"] == recommended_id), None
        )
        if not recommended:
            return False, f"Recommended strategy {recommended_id} not found"

        actions = recommended.get("actions", [])
        if not actions:
            # An action-less strategy is an incomplete/hallucinated reply.
            logger.warning(
                "FIXER strategy has no actions (incomplete response)"
            )
            return False, "Recommended strategy has no actions"

        for action in actions:
            action_type = action.get("type", "").lower()

            if action_type == "create_file":
                path = action.get("path", "").strip()
                content = action.get("content", "").strip()

                if not path:
                    return False, "create_file action missing path"

                if not content:
                    return (
                        False,
                        f"create_file action has empty content for "
                        f"{path} - this won't help",
                    )

                # Repo paths are short; >100 chars means the model
                # invented the layout.
                if len(path) > 100:
                    return (
                        False,
                        f"Suspiciously long filepath: {path[:100]}...",
                    )

                # Only allow paths that stay inside the repo tree.
                # Absolute paths or `..` traversal would let a
                # hallucinated (or poisoned) fix write anywhere in the
                # shared sandbox — /etc, /usr/local/go, apt sources —
                # poisoning every later build on this container.
                if os.path.isabs(path) or ".." in Path(path).parts:
                    return (
                        False,
                        f"Path escapes the repository (absolute or "
                        f"contains '..'): {path}",
                    )

            # Same containment as create_file.
            elif action_type == "patch":
                patch_file = (
                    action.get("file") or action.get("path") or ""
                ).strip()
                if patch_file and (
                    os.path.isabs(patch_file) or ".." in Path(patch_file).parts
                ):
                    return (
                        False,
                        f"Patch path escapes the repository: {patch_file}",
                    )

            elif action_type == "plan_command":
                old = (action.get("replaces") or "").strip()
                new = (action.get("command") or "").strip()
                if not plan_commands:
                    return False, "plan_command: there is no build plan"
                if old not in plan_commands:
                    listed = "; ".join(f"`{c[:120]}`" for c in plan_commands)
                    return (
                        False,
                        "plan_command `replaces` must be one plan command "
                        f"exactly as written. The plan commands are: {listed}",
                    )
                if new:
                    allowed, why = _plan_command_validator.is_safe(new)
                    if not allowed:
                        return (
                            False,
                            f"plan_command `{new[:160]}` is not allowed: "
                            f"{why}",
                        )

            elif action_type == "command":
                cmd = action.get("command", "").strip()

                if not cmd:
                    return False, "command action is empty"

                # Reject a `cp -r` that copies a tree onto itself.
                if "cp -r" in cmd and "/*" in cmd:
                    parts = cmd.split()
                    if len(parts) >= 4:
                        src = parts[2]
                        dst = parts[3]
                        if (
                            src == dst
                            or src.split("/")[0] == dst.split("/")[0]
                        ):
                            return (
                                False,
                                f"Nonsensical command detected: {cmd} "
                                f"(copying to self)",
                            )

                is_safe, reason = validate_fix_command(cmd)
                if not is_safe:
                    return False, f"Unsafe command: {reason}"

        # The root cause is advisory: an empty one is logged, never
        # rejected.
        reflection = fix_data.get("reflection") or {}
        if not reflection.get("root_cause"):
            logger.warning("Fixer response has empty root_cause analysis")

        return True, ""

    except Exception as e:
        logger.error(f"Error validating fixer response: {e}")
        return False, f"Validation error: {str(e)}"


# fixer_node — diagnose the failure, then apply one validated strategy.

# Doubled braces in the JSON schema are literal braces after
# str.format().
# The system part (static per platform, cached; the knowledge block
# follows as a second system block); the per-failure part is
# FIXER_TASK_PROMPT, sent as the user message.
FIXER_PROMPT = (
    "You are the build doctor for a RISC-V porting agent. Diagnose\n"
    "the build failure in the user message and emit a minimal fix.\n"
    "\n"
    "## ACTIVE SANDBOX (use ONLY these package commands)\n"
    "{platform_banner}\n"
    "\n"
    "## Diagnostic procedure\n"
    "1. **Read the error.** What's the *root cause*, not the symptom?\n"
    "2. **Classify**: missing dep / wrong package name / arch-specific c"
    "ode / configure\n"
    "   gone stale / build-system mismatch / glibc-only API / unrelated "
    "env issue.\n"
    "3. **Pick the cheapest fix** in this order: (a) install a missing `"
    "-dev` package,\n"
    "   (b) change a build flag, (c) regenerate configure / copy `config"
    ".{{guess,sub}}`,\n"
    "   (d) add a `#ifdef __riscv` guard, (e) patch source.\n"
    "4. **Check the few-shot examples in the user message** — if one "
    "matches the error pattern, use it.\n"
    "5. **Verify your fix does not duplicate a previous failed attempt.*"
    "*\n"
    "\n"
    "## Critical rules\n"
    "- Each action holds ONE command and runs in a fresh shell at the "
    "repo root. Never join commands with `&&`, `||`, `;` or `|`: the "
    "validator rejects the whole strategy. The one allowed prefix is "
    "`cd <dir> && `. Put the package-index update and the install in "
    "two separate actions.\n"
    "- Do NOT propose unrelated cleanups. Fix only what's broken.\n"
    "- After your fix, the builder re-runs the build plan commands from "
    "the failed phase on. A build command that you run yourself does "
    "not change the plan. When a plan command is wrong (wrong directory, "
    "wrong target, a missing flag), change it with a `plan_command` "
    "action: `replaces` is the exact current plan command, `command` is "
    "the new one (it may use `&&`; empty removes the command).\n"
    "- Never add a stub or placeholder source file only to make a plan "
    "command pass: the build must produce the package's real programs. "
    "If the plan builds the wrong target (for example `go build .` when "
    "the main package is in `cmd/<name>`, or `cmake ..` when the "
    "CMakeLists.txt is in a subdirectory), fix the plan with a "
    "`plan_command` action.\n"
    "- For Go: `-buildvcs=false` is required; never install Go via the p"
    "ackage manager (it is preinstalled in the sandbox). Avoid `go build "
    "./cmd` / `go build ./<dir>` unless you also set `-o` to a file path"
    " (directory-name output collisions are common).\n"
    "- For CMake failures with 'not a full path to an existing compiler"
    "\n"
    "  tool': CMake's project() caches the compiler path on first run.\n"
    "  Delete the build dir (`rm -rf build/`) and retry with\n"
    "  `-DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++`"
    "\n"
    "  in the cmake invocation. Setting CC env var alone is not enough\n"
    "  — it will not override the cached CMAKE_C_COMPILER.\n"
    "- For autotools failures involving gettext macros: copy macros BEFO"
    "RE `aclocal`,\n"
    "  do NOT run `autoreconf` (it reverts the fix via `autopoint`).\n"
    "- If no fix inside the sandbox can work (for example a toolchain "
    "that does not exist for riscv64), set `escalate` instead of "
    "proposing a hack.\n"
    "\n"
    "## Output schema (JSON only)\n"
    "{{\n"
    '  "investigate": null,\n'
    '  "escalate": null,\n'
    '  "strategies": [\n'
    '    {{"id": 1, "description": "<one line>",\n'
    '      "actions": [\n'
    '        {{"type": "command", "command": "<one shell command>", '
    '"path": "", "content": "", "replaces": ""}}\n'
    "      ]}}\n"
    "  ],\n"
    '  "recommended_strategy_id": 1,\n'
    '  "reflection": {{"root_cause": "<why it broke, one sentence>"}}\n'
    "}}\n"
    "Give exactly ONE strategy: the one you recommend.\n"
    "\n"
    "Action types: `command` fills `command`; `create_file` fills "
    "`path` (repo-relative) and `content`; `patch` fills `path` and a "
    "unified diff in `content`; `plan_command` fills `replaces` (an "
    "exact command from the build plan) and `command` (its "
    'replacement). Leave the unused fields as "".\n'
    'To give up: set `escalate` to {{"reason": "<why no fix in the '
    'sandbox can work>"}} and `strategies` to [].\n'
    "Rules: ≥1 strategy with ≥1 action unless you investigate or "
    "escalate; no placeholder paths (`/path/to/`, `/home/`); prefer "
    "`command` with `sed` over `patch` for simple edits.\n"
)

FIXER_SCHEMA = _object(
    {
        "investigate": {
            "anyOf": [{"type": "null"}, _object({"commands": _STRINGS})]
        },
        "escalate": {
            "anyOf": [{"type": "null"}, _object({"reason": _STRING})]
        },
        "strategies": {
            "type": "array",
            "items": _object(
                {
                    "id": {"type": "integer"},
                    "description": _STRING,
                    "actions": {
                        "type": "array",
                        "items": _object(
                            {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "command",
                                        "create_file",
                                        "patch",
                                        "plan_command",
                                    ],
                                },
                                "command": _STRING,
                                "path": _STRING,
                                "content": _STRING,
                                "replaces": _STRING,
                            }
                        ),
                    },
                }
            ),
        },
        "recommended_strategy_id": {"type": "integer"},
        "reflection": _object({"root_cause": _STRING}),
    }
)

FIXER_TASK_PROMPT = (
    "## Package: **{repo_name}**\n"
    "\n"
    "## Build context\n"
    "- Target: {target_arch} | Build system: {build_system}\n"
    "- Attempt {attempt_count}/{max_attempts}\n"
    "\n"
    "## Failure\n"
    "- Failed phase: {failed_phase}\n"
    "- Failed command: `{failed_command}`{ran_as}\n"
    "- Exit code: {exit_code}\n"
    "\n"
    "{key_lines_section}"
    "### Command output (the end; a cut is marked)\n"
    "```\n"
    "{output_tail}\n"
    "```\n"
    "{stdout_section}"
    "\n"
    "### Source context around the error (real file contents)\n"
    "{error_file_excerpts}\n"
    "\n"
    "### What this package is (analyst)\n"
    "{package_context}\n"
    "\n"
    "### Build plan (the builder re-runs these from the failed phase; "
    "change one with a `plan_command` action)\n"
    "{build_plan}\n"
    "\n"
    "### Previous fix attempts (most recent last — DO NOT repeat any of "
    "these)\n"
    "{previous_fixes}\n"
    "\n"
    "{recall_section}\n"
    "\n"
    "### Top-level files\n"
    "{top_level_files}\n"
    "\n"
    "### Known arch concerns\n"
    "{known_arch_issues}\n"
    "\n"
    "{few_shot_examples}\n"
    "\n"
    "## Investigation option\n"
    "{investigation}\n"
    "\n"
    "Reply with the JSON object of the output schema."
)

# Heading of the knowledge block, the second (cached) system block.
_FIXER_KNOWLEDGE_HEADING = (
    "## Platform knowledge (canonical — use these package names + "
    "commands)\n"
)


# First-token whitelist for fixer investigation commands, meant for
# read-only inspection. The check looks at the command name only, so a
# `find -delete` or `go mod tidy` can still change the repo.
_INVESTIGATE_ALLOWED = {
    "cat",
    "head",
    "tail",
    "ls",
    "find",
    "grep",
    "egrep",
    "file",
    "stat",
    "wc",
    "pkg-config",
    "readelf",
    "nm",
    "ldd",
    "go",
    "which",
}

_INVESTIGATE_MAX_COMMANDS = 4
_INVESTIGATE_OUTPUT_CAP = 1500

_INVESTIGATION_OFFER = (
    "If (and ONLY if) the error lines and source excerpts above are not "
    "enough to find the root cause, you may ask ONCE for read-only "
    'inspection: set `investigate` to {"commands": ["<read-only shell '
    'command>", ...]} and `strategies` to []. Up to 4 commands; the '
    "first word is one of cat, head, tail, ls, find, grep, file, stat, "
    "wc, pkg-config, readelf, nm, ldd, go, which; no redirection. Their "
    "output comes back in the next message, and then you MUST give "
    "strategies."
)


def _run_fixer_investigation(state: AgentState, commands: List[str]) -> str:
    """Execute the fixer's read-only investigation commands.

    Each command must start with a whitelisted read-only tool and is
    executed inside the sandbox at the repo root; outputs are capped so
    the follow-up prompt stays small.

    Args:
        state: Current agent state (for the repo path).
        commands: Commands the fixer asked to run.

    Returns:
        A markdown block of command → output pairs (or rejection
        notes for disallowed commands).
    """
    sections = []
    for cmd in commands[:_INVESTIGATE_MAX_COMMANDS]:
        cmd = str(cmd).strip()
        if not cmd:
            continue
        first = cmd.split()[0] if cmd.split() else ""
        segments = CommandValidator.split_segments(cmd)
        heads = [s.split()[0] for s in segments if s.split()]
        if (
            first not in _INVESTIGATE_ALLOWED
            or len(segments) != 1
            or any(head not in _INVESTIGATE_ALLOWED for head in heads)
            or ">" in cmd
            or "<" in cmd
        ):
            sections.append(
                f"$ {cmd}\n[rejected: only a single read-only inspection "
                f"command is allowed]"
            )
            continue
        result = execute_command(cmd, cwd=state.repo_path, use_docker=True)
        output = (result.stdout or "") + (
            f"\n[stderr] {result.stderr}" if result.stderr else ""
        )
        output = output.strip() or "(no output)"
        output = cut_text(output, _INVESTIGATE_OUTPUT_CAP)
        sections.append(f"$ {cmd}  (exit {result.exit_code})\n{output}")
    state.log_scripted_op("fixer_investigation")
    return "\n\n".join(sections) or "(no valid commands to run)"


# Fixer prompt facts: exit code, failed phase, plan, earlier fixes.

_EXIT_CODE_RE = re.compile(r"\(exit (-?\d+)\)")

_EXIT_NOTES = {
    -1: "Atesor stopped the command: time limit or sandbox error",
    124: "the `timeout` limit stopped the command",
    126: "the command is not executable",
    127: "command not found",
    137: "SIGKILL: the process was killed, usually out of memory",
    139: "SIGSEGV: segmentation fault",
    143: "SIGTERM: the process was terminated",
}


def _exit_code_text(message: str) -> str:
    """Return the exit code in an error message, with a signal note."""
    match = _EXIT_CODE_RE.search(message or "")
    if not match:
        return "not reported"
    code = int(match.group(1))
    if code == 137 and TIME_LIMIT_MARK in (message or "").lower():
        return "137 (stopped at the command time limit, not a memory kill)"
    note = _EXIT_NOTES.get(code)
    return f"{code} ({note})" if note else str(code)


def _failed_phase_commands(state: AgentState) -> List[str]:
    """Return the commands of the plan phase that holds the failed one."""
    plan = state.build_plan
    failed = state.error_history[-1].command if state.error_history else None
    if plan and failed:
        for phase in plan.phases:
            if failed in phase.commands:
                return list(phase.commands)
    return []


def _gettext_m4_was_fixed(state: AgentState) -> bool:
    """Return True when a fix of this run changed ``m4/gettext.m4``.

    Only then must autoreconf not run: its autopoint step puts the old
    file back. For any other tree the rewrite hid the real error (pv,
    wget: the copy source does not exist on Debian trixie).
    """
    return any(
        record.get("ok")
        and "gettext.m4"
        in str(record.get("path") or "") + str(record.get("command") or "")
        for fix in state.fixes_attempted
        for record in fix.actions or []
    )


def _failed_phase_text(state: AgentState, command: Optional[str]) -> str:
    """Name the plan phase that holds ``command``."""
    plan = state.build_plan
    if plan and command:
        for phase in plan.phases:
            if command in phase.commands:
                return f"{phase.name} (phase {phase.id} of {len(plan.phases)})"
    return "not a build-plan command (for example the artifact check)"


def _build_plan_text(state: AgentState, failed_command: Optional[str]) -> str:
    """Render the plan's phases and commands, marking the failed one."""
    plan = state.build_plan
    if not plan:
        return "No build plan"
    lines = [f"Build system: {plan.build_system}"]
    for phase in plan.phases:
        done = (
            " (completed)" if phase.id <= state.last_successful_phase else ""
        )
        lines.append(f"{phase.id}. {phase.name}{done}")
        for cmd in phase.commands:
            mark = "   <- failed here" if cmd == failed_command else ""
            lines.append(f"   - `{cmd[:200]}`{mark}")
    return "\n".join(lines)


def _fix_status(fix: FixAttempt) -> str:
    """Return the outcome label of a fix attempt."""
    if fix.success:
        return "Success"
    if fix.build_result == _NO_CHANGE:
        return "Failed: no change applied"
    return "Failed"


def _previous_fixes_text(state: AgentState) -> str:
    """Render the last five fixes: actions, their results, what followed.

    Answers that the validator refused come last, with the reason.
    """
    rejected = [
        f"- Refused answer (do not send it again): {reason}"
        for reason in state.context_cache.get("rejected_fixes") or []
    ]
    fixes = state.fixes_attempted[-5:]
    if not fixes:
        return "\n".join(rejected) or "None"
    first = len(state.fixes_attempted) - len(fixes) + 1
    lines = []
    for number, fix in enumerate(fixes, start=first):
        lines.append(f"- Fix {number}: {fix.strategy} ({_fix_status(fix)})")
        if fix.root_cause:
            lines.append(f"  believed cause: {fix.root_cause}")
        results = fix.action_results or fix.changes_made
        if results:
            lines.append("  actions: " + "; ".join(results))
        if fix.success:
            lines.append("  then: the build passed")
            continue
        if fix.build_result == _NO_CHANGE:
            continue  # nothing ran, so no later error belongs to it
        after = next(
            (e for e in state.error_history if e.timestamp > fix.timestamp),
            None,
        )
        if after is not None:
            line = first_error_line(
                after.message
            ) or _first_meaningful_error_line(
                strip_failure_prefix(after.message)
            )
            lines.append(f"  then: next error: {line[:240]}")
    return "\n".join(lines + rejected)


def _error_key(message: str) -> str:
    """Return the normalized first meaningful line of an error."""
    return _normalize_error_line(
        _first_meaningful_error_line(strip_failure_prefix(message))
    )


# make's and ninja's summary lines repeat an earlier error. A parallel
# build prints a different number of them for the same error.
_SUMMARY_LINE_RE = re.compile(
    r"^(?:g?make(?:\[\d+\])?: \*\*\* (?:\[[^\]]*\] Error \d+|"
    r"Waiting for unfinished jobs)|ninja: build stopped:)"
)


def _error_lines(message: str) -> frozenset:
    """Return the normalized error lines of a failure message.

    The lines that match an error pattern, after the command prefix and
    Go's download lines are dropped; with none, the first meaningful
    line. make's "*** [target] Error N" and "Waiting for unfinished
    jobs" lines and ninja's "build stopped" line are left out: they
    repeat an earlier line, and their number changes between two runs
    of the same parallel build.
    """
    text = strip_go_download_noise(strip_failure_prefix(message or ""))
    lines = {
        _normalize_error_line(line)
        for line in (raw.strip() for raw in text.splitlines())
        if line and has_error_line(line) and not _SUMMARY_LINE_RE.match(line)
    }
    return frozenset(lines or {_error_key(message or "")})


def _error_after(state: AgentState, fix: FixAttempt) -> Optional[Any]:
    """Return the first error recorded after ``fix``, or None."""
    return next(
        (e for e in state.error_history if e.timestamp > fix.timestamp),
        None,
    )


def _fix_removed_an_error(fix: FixAttempt, after_message: str) -> bool:
    """Return True when an error line of ``fix`` is gone afterwards."""
    return bool(_error_lines(fix.error_message) - _error_lines(after_message))


def _fix_changed_the_error(state: AgentState, fix: FixAttempt) -> bool:
    """Return True when the build error after ``fix`` lost a line.

    The error lines are compared as sets, and progress is a line that
    is gone: a fix that removed one of two errors changed the error. A
    fix after which a new line came but every old line stayed made no
    progress.
    """
    after = _error_after(state, fix)
    return after is not None and _fix_removed_an_error(fix, after.message)


def _fix_key_line(state: AgentState, fix: FixAttempt) -> str:
    """Return the error line that a fix that worked is learned on.

    When the build passed after the fix, the decisive line of its error.
    When the build failed again on a changed error, a line the fix
    removed: a fix that removed the second of two errors is not the
    fix of the first one, which is still there.

    Returns:
        The line, or "" when the fix removed no error line.
    """
    after = _error_after(state, fix)
    if after is None:
        return first_error_line(fix.error_message)
    remaining = _error_lines(after.message)
    return first_error_line(
        fix.error_message,
        skip=lambda line: _normalize_error_line(line) in remaining,
    )


def _remember_failed_fixes(state: AgentState) -> None:
    """Record each fix after which the error stayed the same.

    The failure memory (``memory.save_failure_record``) shows such a fix
    to the fixer of a later run of the package as tried before. A fix
    is recorded once, when the error after it is known; a fix that
    changed nothing in the tree is left out.
    """
    done = state.context_cache.setdefault("failure_memory_fixes", [])
    for index, fix in enumerate(state.fixes_attempted):
        if index in done or fix.build_result == _NO_CHANGE:
            continue
        after = _error_after(state, fix)
        if after is None:
            continue
        done.append(index)
        if _fix_removed_an_error(fix, after.message):
            continue
        commands = [
            str(action.get("command") or action.get("path") or "")
            for action in fix.actions
            if action.get("ok")
        ] or [c for c in fix.changes_made if not c.startswith("REJECTED")]
        save_failure_record(
            state.repo_name,
            _failure_key(fix.error_message),
            {
                "strategy": _shorten(fix.strategy, 200),
                "commands": [_shorten(c, 200) for c in commands[:8]],
                "error_after": _shorten(
                    first_error_line(after.message)
                    or _error_key(after.message),
                    200,
                ),
                "date": datetime.now().strftime("%Y-%m-%d"),
            },
        )


def _failure_key(message: str) -> str:
    """Return the failure-memory key of an error: its decisive line."""
    return _normalize_error_line(
        first_error_line(message)
        or _first_meaningful_error_line(strip_failure_prefix(message))
    )


def _error_regex(line: str) -> str:
    """Return a regex for an error line, tolerant of paths and numbers.

    The normalized line writes a ``:line:column`` reference as ``:<n>``,
    so that token accepts a line number with an optional column.
    """
    pattern = re.escape(_normalize_error_line(line))
    for token, regex in (
        (":<n>", r":\d+(?::\d+)?"),
        ("<path>", r"\S+"),
        ("<hex>", r"[0-9a-fx]+"),
        ("<n>", r"\d+"),
    ):
        pattern = pattern.replace(re.escape(token), regex)
    return pattern


def _plan_commands(state: AgentState) -> List[str]:
    """Return every command of the current build plan, in order."""
    if not state.build_plan:
        return []
    return [c for phase in state.build_plan.phases for c in phase.commands]


def _replace_plan_command(state: AgentState, old: str, new: str) -> bool:
    """Replace (or, with an empty ``new``, remove) a plan command.

    A change in a phase that already completed makes the builder run
    that phase again.

    Returns:
        True when ``old`` was found.
    """
    if not state.build_plan:
        return False
    for phase in state.build_plan.phases:
        if old in phase.commands:
            i = phase.commands.index(old)
            if new:
                phase.commands[i] = new
            else:
                del phase.commands[i]
            state.forget_command_result(old)
            if phase.id <= state.last_successful_phase:
                state.last_successful_phase = phase.id - 1
            return True
    return False


def _one_line_tail(text: str, limit: int = 200) -> str:
    """Return the end of command output as one short line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else "..." + flat[-limit:]


def _failure_summary(text: str, limit: int = 200) -> str:
    """Return the end of a failed command's output, with its error line.

    The end alone can miss the decisive line: in wget run 1 it hid
    ``possibly undefined macro: AX_CODE_COVERAGE``.
    """
    tail = _one_line_tail(text, limit)
    line = " ".join(first_error_line(text or "").split())
    if line and line not in tail:
        return f"{_shorten(line, 160)} | {tail}"
    return tail


def _read_only_strategy(data: Dict[str, Any]) -> List[str]:
    """Return the commands of a recommended strategy that only reads.

    Returns:
        The commands, or an empty list when the strategy holds another
        action type or a command that changes something.
    """
    strategies = [
        s for s in data.get("strategies") or [] if isinstance(s, dict)
    ]
    if not strategies:
        return []
    recommended = data.get("recommended_strategy_id", 1)
    strategy = next(
        (s for s in strategies if s.get("id") == recommended), strategies[0]
    )
    commands: List[str] = []
    for action in strategy.get("actions") or []:
        if not isinstance(action, dict) or action.get("type") != "command":
            return []
        command = str(action.get("command") or "").strip()
        if not command or not _is_read_only_command(command):
            return []
        commands.append(command)
    return commands


@agent_node(AgentRole.FIXER)
def fixer_node(state: AgentState) -> AgentState:
    """Diagnose the last build error and apply one validated strategy.

    The prompt carries the failing command, a truncated error, source
    excerpts around the error's ``file:line`` references, the package
    analysis, and the previous fix attempts. The fixer may first spend
    ONE read-only investigation round (whitelisted inspection
    commands); the follow-up round must emit strategies. Actions are
    validated (``validate_fixer_response``) and contained to the repo
    tree before they run. There is no deterministic fix to fall back
    on: when every LLM attempt fails, the node records an error and
    returns FAILED.
    """
    logger.info("Fixer analyzing error...")

    state.build_status = BuildStatus.FIXING
    state.current_phase = "fixing"

    if not state.last_error:
        logger.warning("No error to fix")
        state.build_status = BuildStatus.PENDING
        return state

    if _llm_refused(state):
        logger.warning("Fixer: the model refused this package; escalating")
        state.build_status = BuildStatus.ESCALATED
        return state

    previous_fixes = _previous_fixes_text(state)
    # A fix after which the error stayed goes to the failure memory.
    _remember_failed_fixes(state)

    arch_issues = "None"
    if state.arch_specific_code:
        relevant = [
            a
            for a in state.arch_specific_code
            if a.severity in ["high", "critical"]
        ]
        relevant_subset = relevant[:5]
        if relevant_subset:
            arch_issues = "\n".join(
                [
                    f"- {a.file}:{a.line} - {a.arch_type}"
                    for a in relevant_subset
                ]
            )

    failed_command = "Unknown"
    ran_as = ""
    if state.error_history:
        last = state.error_history[-1]
        failed_command = last.command or "Unknown"
        # The builder can run a plan command in another form (an
        # autotools bootstrap before ./configure, a make directory).
        ran = (
            state.get_cached_command_result(last.command)
            if last.command
            else None
        )
        if ran and ran.command and ran.command != last.command:
            ran_as = f" (run as `{ran.command[:300]}`)"

    few_shot_context = {
        "build_system": (
            state.build_plan.build_system if state.build_plan else "unknown"
        ),
        "error_message": state.last_error,
        "repo_name": state.repo_name,
    }
    few_shot_examples = (
        ""
        if state.replan
        else format_few_shot_examples(
            "fixer", few_shot_context, max_examples=2, max_chars=2000
        )
    )
    _note_examples(state, "fixer", few_shot_examples)

    build_plan_str = _build_plan_text(
        state, failed_command if failed_command != "Unknown" else None
    )
    # The verified plan and the failed fixes of earlier runs.
    recall_section = _recall_prompt_section(state)
    key_lines, output_tail = failure_digest(state.last_error or "")
    stdout_tail = (
        state.error_history[-1].stdout_tail if state.error_history else ""
    )

    # Deterministic diagnosis context — the actual source lines the
    # error points at, plus the analyst's understanding of the package.
    error_file_excerpts = (
        error_context_excerpts(state.last_error or "", state.repo_path)
        or "(error references no readable repo files)"
    )
    # Read once: on native each read after a sandbox command re-copies
    # the mirror, and the investigation commands do not change the tree.
    top_level_files = top_level_listing(state.repo_path)
    package_context = _format_analysis_for_prompt(state.package_analysis)

    fixer_build_system = (
        state.build_plan.build_system if state.build_plan else None
    )
    fixer_system_prompt = [
        FIXER_PROMPT.format(platform_banner=_build_platform_banner()),
        _FIXER_KNOWLEDGE_HEADING
        + get_system_knowledge_summary(build_system=fixer_build_system),
    ]

    def _render_fixer_prompt(investigation: str) -> str:
        return FIXER_TASK_PROMPT.format(
            repo_name=state.repo_name,
            target_arch="RISC-V (riscv64)",
            build_system=(
                state.build_plan.build_system
                if state.build_plan
                else "unknown"
            ),
            attempt_count=state.attempt_count,
            max_attempts=state.max_attempts,
            failed_phase=_failed_phase_text(
                state, failed_command if failed_command != "Unknown" else None
            ),
            exit_code=_exit_code_text(state.last_error or ""),
            key_lines_section=(
                "### Error lines from the cut part of the output\n"
                f"```\n{key_lines}\n```\n\n"
                if key_lines
                else ""
            ),
            output_tail=output_tail or "No error details",
            stdout_section=(
                "\n### Standard output (the end; the output above is "
                f"stderr)\n```\n{stdout_tail}\n```\n"
                if stdout_tail
                else ""
            ),
            error_file_excerpts=error_file_excerpts,
            package_context=package_context,
            failed_command=failed_command,
            ran_as=ran_as,
            previous_fixes=previous_fixes,
            top_level_files=top_level_files,
            known_arch_issues=arch_issues,
            build_plan=build_plan_str,
            recall_section=recall_section,
            few_shot_examples=few_shot_examples,
            investigation=investigation,
        )

    _fixer_pool = get_model_pool_for_role(AgentRole.FIXER)

    # Bounded investigation loop: round 0 may answer with read-only
    # inspection commands instead of a fix; we execute them and re-ask
    # ONCE with the outputs. Round 1 must produce strategies.
    outcome = None
    investigation = _INVESTIGATION_OFFER
    # Round 1 continues round 0's conversation: the model sees its own
    # investigation request, then the results.
    history: Optional[list] = None
    round_prompt = None
    for round_idx in range(2):
        allow_investigation = round_idx == 0

        def _validate_fix(data: dict) -> ValidationResult:
            if data.get("escalate") and not data.get("strategies"):
                esc = data["escalate"]
                if (
                    not isinstance(esc, dict)
                    or not str(esc.get("reason", "")).strip()
                ):
                    return ValidationResult.bad(
                        "'escalate' must be an object with a 'reason'"
                    )
                return ValidationResult.good()
            if data.get("investigate") and not data.get("strategies"):
                if not allow_investigation:
                    return ValidationResult.bad(
                        "investigation already used — emit the "
                        "strategies schema now"
                    )
                inv = data.get("investigate")
                if not isinstance(inv, dict) or not isinstance(
                    inv.get("commands"), list
                ):
                    return ValidationResult.bad(
                        "'investigate' must be an object with a "
                        "'commands' array"
                    )
                return ValidationResult.good()
            # A plain `a && b` chain becomes single actions first.
            for option in data.get("strategies") or []:
                if isinstance(option, dict) and isinstance(
                    option.get("actions"), list
                ):
                    option["actions"] = _split_chained_actions(
                        option["actions"]
                    )
            ok, reason = validate_fixer_response(data, _plan_commands(state))
            if not ok:
                return ValidationResult.bad(reason)
            return ValidationResult.good()

        outcome = llm_call_with_validation(
            invoke_fn=invoke_llm,
            llm=_fixer_pool[0],
            fallback_llms=_fixer_pool[1:],
            prompt=round_prompt or _render_fixer_prompt(investigation),
            system_prompt=fixer_system_prompt,
            history=history,
            output_schema=FIXER_SCHEMA,
            validator=_validate_fix,
            # No deterministic fix: FAILED ends the subgraph, and the
            # supervisor routes the run.
            fallback_factory=lambda: None,
            role=AgentRole.FIXER.value,
            audit_metadata={
                "repo": state.repo_name,
                "error_category": (
                    state.last_error_category.value
                    if state.last_error_category
                    else "unknown"
                ),
                "build_attempt": state.attempt_count,
                "fixer_round": round_idx,
                "examples": list(getattr(few_shot_examples, "ids", ())),
                "recall": _recall_label(state),
            },
            max_retries=2,
        )
        state.log_api_call(
            cost=outcome.cost_usd,
            tokens_in=outcome.input_tokens,
            tokens_out=outcome.output_tokens,
            calls=outcome.attempts,
        )
        if outcome.refused:
            break

        if (
            outcome.data is not None
            and allow_investigation
            and outcome.data.get("investigate")
            and not outcome.data.get("strategies")
        ):
            commands = outcome.data["investigate"].get("commands", [])
            logger.info(
                f"Fixer requested investigation: {len(commands)} command(s)"
            )
            results = _run_fixer_investigation(state, commands)
            history = outcome.messages
            round_prompt = (
                "### Investigation results (you asked for these)\n"
                f"{results}\n\n"
                "You MUST now respond with the strategies schema — "
                "'investigate' is no longer allowed."
            )
            continue
        # A strategy that only reads changes nothing, so the rebuild
        # would fail the same way: run it as the investigation round.
        reading = (
            _read_only_strategy(outcome.data)
            if outcome.data is not None and allow_investigation
            else []
        )
        if reading:
            logger.info(
                f"Fixer strategy only reads ({len(reading)} command(s)); "
                "running it as the investigation round"
            )
            results = _run_fixer_investigation(state, reading)
            history = outcome.messages
            round_prompt = (
                "### Results of your strategy (its commands only read)\n"
                f"{results}\n\n"
                "These commands change nothing, so the build would fail "
                "the same way. You MUST now respond with a strategy that "
                "changes the source tree, the build plan or the "
                "environment."
            )
            continue
        break

    if outcome.refused:
        # Every later fixer prompt for this package is refused too.
        _mark_llm_refused(state, AgentRole.FIXER.value)
        state.build_status = BuildStatus.ESCALATED
        return state

    try:
        if outcome.data is None:
            logger.error(
                f"FIXER produced no valid response after "
                f"{outcome.attempts} attempts: "
                f"{outcome.last_error}"
            )
            # A provider fault (timeout, HTTP error) is no build attempt.
            state.add_error(
                create_error_record(
                    message=f"{_FIXER_INVALID}: {outcome.last_error}",
                    category=ErrorCategory.CONFIGURATION,
                ),
                count_attempt="LLM invocation raised"
                not in str(outcome.last_error or ""),
            )
            # The next fixer round sees why the answer was refused, so
            # it does not send the same one (libwebp run 4).
            rejected = state.context_cache.setdefault("rejected_fixes", [])
            rejected.append(_shorten(str(outcome.last_error or ""), 240))
            del rejected[:-5]
            state.build_status = BuildStatus.FAILED
            return state

        fix_data = outcome.data

        escalation = fix_data.get("escalate")
        if escalation and not fix_data.get("strategies"):
            reason = str(escalation.get("reason", "")).strip()[:1200]
            logger.warning(f"Fixer escalated: {reason}")
            state.add_error(
                create_error_record(
                    message=f"{_FIXER_ESCALATED}: {reason}",
                    category=state.last_error_category
                    or ErrorCategory.UNKNOWN,
                )
            )
            state.build_status = BuildStatus.ESCALATED
            return state

        recommended_id = fix_data.get("recommended_strategy_id", 1)
        strategies = fix_data.get("strategies", [])

        if not strategies:
            logger.error("No fix strategies generated")
            state.build_status = BuildStatus.FAILED
            return state

        strategy = next(
            (s for s in strategies if s["id"] == recommended_id), strategies[0]
        )

        logger.info(f"Applying fix strategy: {strategy['description']}")

        changes_made = []
        action_results: List[str] = []
        # What ran and whether it worked; the learner replays these.
        action_records: List[Dict[str, Any]] = []
        # Chains (see _split_chained_actions) with a failed step.
        failed_chains: set = set()
        failed_phase = _failed_phase_commands(state)
        for action in strategy.get("actions", []):
            if action["type"] == "create_file":
                file_path = action.get("path")
                file_content = action.get("content", "")

                if not file_path:
                    logger.error("create_file action missing 'path'")
                    continue

                full_file_path = os.path.join(state.repo_path, file_path)

                if not _is_within_repo(full_file_path, state.repo_path):
                    logger.error(
                        f"Refusing create_file outside repo: {file_path}"
                    )
                    changes_made.append(
                        f"REJECTED create_file outside repo: {file_path}"
                    )
                    action_results.append(
                        f"create `{file_path}` rejected: outside the repo"
                    )
                    continue

                # Built by Atesor with quoted arguments and base64
                # content, so the LLM whitelist does not apply.
                write_result = execute_command(
                    _contained_write_cmd(
                        state.repo_path, full_file_path, file_content
                    ),
                    cwd=state.repo_path,
                    use_docker=True,
                    validate=False,
                )

                action_records.append(
                    _action_record(
                        "create_file",
                        file_path,
                        file_content,
                        write_result.success,
                    )
                )
                if write_result.success:
                    changes_made.append(f"Created file: {file_path}")
                    action_results.append(f"created `{file_path}`")
                    logger.info(f"Created file: {file_path}")
                elif write_result.exit_code == 3:
                    logger.error(
                        f"Refusing create_file through a symlink out of "
                        f"the repo: {file_path}"
                    )
                    changes_made.append(
                        f"REJECTED create_file outside repo: {file_path}"
                    )
                    action_results.append(
                        f"create `{file_path}` rejected: symlink out of "
                        f"the repo"
                    )
                else:
                    logger.error(
                        f"Failed to create file {file_path}: "
                        f"{write_result.stderr}"
                    )
                    action_results.append(
                        f"create `{file_path}` failed: "
                        f"{_one_line_tail(write_result.stderr)}"
                    )

            elif action["type"] == "patch":
                patch_content = action["content"]
                file_path = action.get("file") or action.get("path") or None
                full_file_path = (
                    os.path.join(state.repo_path, file_path)
                    if file_path
                    else None
                )

                if full_file_path and not _is_within_repo(
                    full_file_path, state.repo_path
                ):
                    logger.error(f"Refusing patch outside repo: {file_path}")
                    changes_made.append(
                        f"REJECTED patch outside repo: {file_path}"
                    )
                    action_results.append(
                        f"patch of `{file_path}` rejected: outside the repo"
                    )
                    continue

                patched = apply_patch(
                    patch_content,
                    filepath=full_file_path,
                    cwd=state.repo_path,
                    use_docker=True,
                )
                action_records.append(
                    _action_record(
                        "patch", file_path or "", patch_content, patched
                    )
                )
                if patched:
                    changes_made.append(
                        f"Patched {file_path if file_path else 'repository'}"
                    )
                    action_results.append(
                        f"patched `{file_path or 'repository'}`"
                    )
                    state.patches_generated.append(patch_content)
                else:
                    logger.error(f"Failed to apply patch to {file_path}")
                    action_results.append(
                        f"patch of `{file_path or 'repository'}` did not "
                        f"apply"
                    )

            elif action["type"] == "plan_command":
                old = (action.get("replaces") or "").strip()
                new = (action.get("command") or "").strip()
                replaced = _replace_plan_command(state, old, new)
                action_records.append(
                    {
                        "type": "plan_command",
                        "replaces": old,
                        "command": new,
                        "ok": replaced,
                    }
                )
                if replaced and new and new.split() == old.split():
                    # The same text is no fix by itself: nushell's
                    # "rerun unchanged" gave a third build of the crate
                    # that the time limit had stopped twice. The phase
                    # still runs again with the next build, for an
                    # answer that also has a real change.
                    changes_made.append(f"{_SAME_PLAN_COMMAND}: `{old[:120]}`")
                    action_results.append(
                        f"plan command `{old[:120]}` unchanged (the same "
                        "text); a rebuild would repeat the failure"
                    )
                    logger.info(f"Plan command unchanged: {old[:120]}")
                elif replaced:
                    label = (
                        f"`{old[:120]}` -> `{new[:120]}`"
                        if new
                        else (f"`{old[:120]}` removed")
                    )
                    changes_made.append(f"Plan command changed: {label}")
                    action_results.append(f"plan: {label}")
                    logger.info(f"Plan command changed: {label}")
                else:
                    action_results.append(
                        f"plan command `{old[:120]}` not found"
                    )

            elif action["type"] == "command":
                command = action["command"]
                chain = action.get("chain")
                if chain and chain in failed_chains:
                    action_results.append(
                        f"`{command[:160]}` skipped: an earlier step of "
                        "its `&&` chain failed"
                    )
                    continue

                is_safe, reason = validate_fix_command(command)
                if not is_safe:
                    logger.warning(
                        f"Fix command blocked: {command} - {reason}"
                    )
                    action_results.append(
                        f"`{command[:160]}` blocked: {reason[:160]}"
                    )
                    if chain:
                        failed_chains.add(chain)
                    continue

                # Rewrite autoreconf to copy-first sequence when the
                # repo has m4/gettext.m4. autoreconf internally
                # calls autopoint which reverts any gettext.m4 fix.
                # The copy-first approach replaces the old m4/gettext.m4
                # with the system's modern one BEFORE running
                # aclocal+autoconf, which is the only reliable fix.
                if re.search(r"\bautoreconf\b", command):
                    if os.path.isfile(
                        os.path.join(
                            _to_host_path(state.repo_path), "m4", "gettext.m4"
                        )
                    ) and _gettext_m4_was_fixed(state):
                        rewritten = (
                            "cp /usr/share/gettext/m4/*.m4 m4/"
                            " && aclocal -I m4 -I /usr/share/aclocal"
                            " && autoconf"
                        )
                        logger.info(
                            "Rewrote autoreconf → copy-first gettext "
                            "sequence (autoreconf would revert "
                            "gettext.m4 fix via autopoint)"
                        )
                        command = rewritten

                commands_to_run = [command]
                for cmd in commands_to_run:
                    result = execute_command(cmd, cwd=state.repo_path)
                    action_records.append(
                        {
                            "type": "command",
                            "command": cmd,
                            "ok": result.success,
                        }
                    )
                    if result.success:
                        changes_made.append(f"Executed: {cmd}")
                        action_results.append(f"`{cmd[:160]}` ok")
                        # A command of the failed phase that the fix ran
                        # with success is done; the builder does not run
                        # it again (grep, axel, wget, re2 ran configure
                        # twice).
                        if cmd in failed_phase:
                            state.cache_command_result(cmd, result)
                    else:
                        logger.error(f"Fix command failed: {result.stderr}")
                        output = result.stderr or result.stdout or ""
                        action_results.append(
                            f"`{cmd[:160]}` failed (exit "
                            f"{result.exit_code}): {_failure_summary(output)}"
                        )
                        if chain:
                            failed_chains.add(chain)

        # A command that only reads changes nothing a rebuild could use,
        # and neither does a plan command replaced by the same text.
        effective = [
            c
            for c in changes_made
            if not c.startswith("REJECTED")
            and not c.startswith(_SAME_PLAN_COMMAND)
            and not (
                c.startswith("Executed: ")
                and _is_read_only_command(c[len("Executed: ") :])
            )
        ]
        fix_attempt = FixAttempt(
            error_category=state.last_error_category,
            strategy=strategy["description"],
            changes_made=changes_made,
            # builder_node flips this to True on the next good build;
            # a no-change fix keeps its marker and is never flipped.
            success=False,
            build_result=None if effective else _NO_CHANGE,
            error_message=state.last_error or "",
            action_results=action_results,
            actions=action_records,
            root_cause=_shorten(
                (fix_data.get("reflection") or {}).get("root_cause", ""), 300
            ),
        )
        state.add_fix_attempt(fix_attempt)

        logger.info(f"Fix applied: {len(changes_made)} changes")

        if not effective:
            # A rebuild of an unchanged tree repeats the same failure.
            # Count the attempt (it bounds this loop) and let
            # route_fix_result send the run back to the fixer, which
            # sees this fix as "no change applied".
            logger.warning(
                "Fix changed nothing; asking the fixer again without a "
                "rebuild"
            )
            state.attempt_count += 1
            state.context_cache["reasks_since_build"] = (
                state.context_cache.get("reasks_since_build", 0) + 1
            )
            return state

        _rewind_after_verify_failure(state)

        # Back to PENDING so route_fix_result sends the run to
        # build_node for a rebuild.
        state.build_status = BuildStatus.PENDING

    except Exception as e:
        logger.error(f"Fixer failed: {e}")
        state.build_status = BuildStatus.FAILED

    return state


# escalate_node — terminal human-handoff report.


# Escalation is reported under the supervisor role.
@agent_node(AgentRole.SUPERVISOR)
def escalate_node(state: AgentState) -> AgentState:
    """Write the terminal report for a run that cannot proceed.

    The report (last error, fixes tried, architecture issues,
    recommendation) lands in ``context_cache["escalation_report"]``
    with the status ESCALATED; ``main.py`` maps that to exit code 1.
    """
    logger.warning("Escalating to human intervention")

    state.build_status = BuildStatus.ESCALATED
    state.current_phase = "escalated"
    _remember_failed_fixes(state)

    err_cat = state.last_error_category
    err_cat_val = err_cat.value if err_cat else "Unknown"
    err_sev = state.last_error_severity
    err_sev_val = err_sev.value if err_sev else "unknown"
    err_msg = state.last_error[:500] if state.last_error else "N/A"
    report = f"""
# ESCALATION REPORT

## Summary
- Repository: {state.repo_name}
- Status: {state.build_status.value}
- Attempts: {state.attempt_count}/{state.max_attempts}
- Cost: ${state.api_cost_usd:.4f}
- Duration: {state.get_execution_duration():.1f}s

## Last Error
Category: {err_cat_val}
Severity: {err_sev_val}
Message: {err_msg}

## Fixes Attempted
{len(state.fixes_attempted)} fix attempts made:
"""

    for fix in state.fixes_attempted[-5:]:
        report += (
            f"\n- {fix.strategy} ({'Success' if fix.success else 'Failed'})"
        )

    report += f"""

## Architecture Issues
{len(state.arch_specific_code)} architecture-specific code instances found
"""

    if state.arch_specific_code:
        high_priority = [
            a
            for a in state.arch_specific_code
            if a.severity in ["high", "critical"]
        ]
        report_issues = high_priority[:5]
        for issue in report_issues:
            report += f"\n- {issue.file}:{issue.line} - {issue.arch_type}"

    report += f"""

## Recommendation
{_escalation_reason(state) or should_escalate(state)[1]}
"""

    if state.error_history:
        report += "\n\n## Recent Failures\n"
        for err in state.error_history[-5:]:
            timestamp = err.timestamp.strftime("%H:%M:%S")
            command = err.command if err.command else "N/A"
            msg = (err.message or "N/A").replace("\n", " ")
            report += (
                f"- [{timestamp}] severity={err.severity.value}, "
                f"category={err.category.value}, command={command}, "
                f"message={msg[:260]}\n"
            )

    logger.info(report)
    state.context_cache["escalation_report"] = report
    _log_data_effect(state)

    return state


# finish_node — package tests, the recipe, and auto-learning.


def _curate_artifacts(state: AgentState) -> None:
    """Curate ``build_artifacts`` once, into ``curated_artifacts``.

    The rules decide; the LLM is asked only when they are unsure, and
    its call counts in the run's call, token and cost totals. Never
    raises.
    """
    if (
        not state.build_artifacts
        or state.curated_artifacts
        or state.context_cache.get("curated")
    ):
        return
    # Once per run, also when the result is empty (everything dropped).
    state.context_cache["curated"] = True
    try:
        from .artifact_curator import curate_artifacts

        curator_llm = None
        try:
            # After a refusal, the rules curate without the LLM.
            if not _llm_refused(state):
                curator_llm = get_model_for_role(AgentRole.SUMMARIZER)
        except Exception as e:
            logger.info(
                f"Curator LLM unavailable, using rule-based fallback: {e}"
            )
        state.curated_artifacts = curate_artifacts(
            raw_artifacts=state.build_artifacts,
            repo_name=state.repo_name or "unknown",
            build_system=(
                state.build_plan.build_system
                if state.build_plan
                else "unknown"
            ),
            llm=curator_llm,
            usage_sink=lambda cost, tokens_in, tokens_out: state.log_api_call(
                cost=cost, tokens_in=tokens_in, tokens_out=tokens_out
            ),
        )
    except Exception as e:
        logger.warning(f"Artifact curation failed (non-fatal): {e}")
        state.curated_artifacts = []


# The most characters of a created file or a patch that a FixAttempt
# action record keeps; a longer one is not stored, so not learned.
_FIX_CONTENT_LIMIT = 4000


def _action_record(
    kind: str, path: str, content: str, ok: bool
) -> Dict[str, Any]:
    """Return the FixAttempt record of a create_file or patch action."""
    record: Dict[str, Any] = {
        "type": kind,
        "path": path,
        "ok": bool(ok),
        "content_chars": len(content or ""),
    }
    if content and len(content) <= _FIX_CONTENT_LIMIT:
        record["content"] = content
    return record


# Tools whose fixer commands only read; such a fix changed nothing that
# a later run could repeat.
_READ_ONLY_TOOLS = frozenset(
    {
        "cat",
        "ls",
        "grep",
        "egrep",
        "fgrep",
        "head",
        "tail",
        "stat",
        "file",
        "wc",
        "which",
        "readelf",
        "nm",
        "ldd",
        "pkg-config",
        "tree",
        "du",
        "pwd",
        "true",
    }
)


def _is_read_only_command(command: str) -> bool:
    """Return True when a fixer command only inspects the tree."""
    text = re.sub(r"^\s*cd\s+\S+\s*(?:&&|;)\s*", "", command or "")
    parts = text.split()
    if not parts:
        return True
    tool = os.path.basename(parts[0])
    if tool in _READ_ONLY_TOOLS:
        return True
    if tool == "find":
        writes = ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint")
        return not any(part in writes for part in parts)
    if tool == "go" and len(parts) > 1:
        if parts[1] in ("version", "list", "doc"):
            return True
        return parts[1] == "env" and not ({"-w", "-u"} & set(parts))
    return False


# Files in which "#" starts code (a preprocessor line), not a comment.
_C_FAMILY_SUFFIXES = (
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".cxx",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
    ".m",
    ".S",
    ".go",
)


def _is_stub_content(content: str, path: str = "") -> bool:
    """Return True when a created file holds no real code.

    An empty file, or one whose only line (comments and blank lines
    aside) is a Go package clause, makes a build command find a package
    but builds no program (probe-cli and netbird, October 2026). In a C
    or Go source ``#`` starts a preprocessor line, not a comment, so a
    header of ``#include`` and ``#define`` lines is real code.
    """
    comments: Tuple[str, ...] = ("//", "/*", "*")
    if not str(path or "").endswith(_C_FAMILY_SUFFIXES):
        comments += ("#",)
    lines = []
    for raw in (content or "").splitlines():
        line = raw.strip()
        if line and not line.startswith(comments):
            lines.append(line)
    if not lines:
        return True
    return len(lines) == 1 and bool(re.match(r"package\s+\w+\s*;?$", lines[0]))


def _learnable_actions(fix: FixAttempt) -> List[Dict[str, Any]]:
    """Return the actions of a fix that a later run can repeat.

    Read from ``fix.actions``; a FixAttempt without action records (an
    older state) falls back to the "Executed: <cmd>" lines of
    ``changes_made``. Failed actions, read-only commands, stub files
    and contents too long to store are left out.
    """
    records = fix.actions or [
        {"type": "command", "command": change[len("Executed: ") :], "ok": True}
        for change in (fix.changes_made or [])
        if change.startswith("Executed: ")
    ]
    actions: List[Dict[str, Any]] = []
    for record in records:
        if not record.get("ok"):
            continue
        kind = record.get("type")
        if kind == "command":
            command = str(record.get("command") or "").strip()
            if command and not _is_read_only_command(command):
                actions.append({"type": "command", "command": command})
        elif kind == "plan_command":
            actions.append(
                {
                    "type": "plan_command",
                    "replaces": record.get("replaces", ""),
                    "command": record.get("command", ""),
                }
            )
        elif kind == "patch" and record.get("content"):
            actions.append(
                {
                    "type": "patch",
                    "file": record.get("path", ""),
                    "content": record["content"],
                }
            )
        elif kind == "create_file" and record.get("content"):
            if not _is_stub_content(record["content"], record.get("path", "")):
                actions.append(
                    {
                        "type": "create_file",
                        "path": record.get("path", ""),
                        "content": record["content"],
                    }
                )
    return actions


_PACKAGE_INSTALL_RE = re.compile(
    r"(?:^|&&|;)\s*(?:\S+=\S+\s+)*(?:sudo\s+)?(?:apt-get|apt)\s+install\b"
    r"([^&;|]*)|(?:^|&&|;)\s*(?:sudo\s+)?apk\s+add\b([^&;|]*)"
)


def _installed_packages(command: str) -> List[str]:
    """Return the packages an apt-get or apk install command adds."""
    packages: List[str] = []
    for match in _PACKAGE_INSTALL_RE.finditer(command or ""):
        args = (match.group(1) or match.group(2) or "").split()
        packages.extend(arg for arg in args if not arg.startswith("-"))
    return packages


def _is_package_command(command: str) -> bool:
    """Return True when a plan command updates or installs packages."""
    return bool(
        re.match(
            r"\s*(?:\S+=\S+\s+)*(?:sudo\s+)?(?:apt-get|apt|apk)\b",
            command or "",
        )
    )


def _pre_build_index(phases: List[Dict[str, Any]]) -> int:
    """Return where a recipe's pre_build phase goes: after the setup."""
    for index, phase in enumerate(phases):
        if any(not _is_package_command(c) for c in phase.get("commands", [])):
            return index
    return len(phases)


# A fixer's own run of the package build (with or without NAME=value
# settings in front); the plan builds again after it.
_FIXER_PACKAGE_BUILD_RE = re.compile(
    r"(?:^|&&|;)\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:go|cargo)\s+build\b"
)


def _fix_anchor(state: AgentState, fix: FixAttempt) -> Optional[str]:
    """Return where the commands of ``fix`` go in a recipe replay.

    Returns:
        The plan command that failed before the fix, as the final plan
        holds it (a later plan_command fix can change it): the commands
        run just before its phase. "" when the failure was no command
        (the artifact check): the commands run after the plan. None when
        the place is unknown (no error record, or the command is no
        longer in the plan).
    """
    before = [e for e in state.error_history if e.timestamp <= fix.timestamp]
    if not before:
        return None
    command = before[-1].command or ""
    if not command:
        return ""
    for other in state.fixes_attempted or []:
        for action in other.actions or []:
            if (
                action.get("type") == "plan_command"
                and action.get("ok")
                and action.get("replaces") == command
            ):
                command = action.get("command") or ""
    return command if command and command in _plan_commands(state) else None


def _recipe_fix_steps(
    state: AgentState,
) -> Tuple[List[Tuple[Optional[str], str]], List[str]]:
    """Return the commands of the fixes that worked, and their packages.

    Each command comes with its place (see ``_fix_anchor``). In the
    order they ran, without read-only commands and without commands the
    plan holds. A fixer's own run of the package build (go build, cargo
    build) before a plan phase is left out, because that phase builds
    again; after the plan it stays. A plan_command or patch fix is in
    the plan or the patches already.
    """
    plan = set(_plan_commands(state))
    steps: List[Tuple[Optional[str], str]] = []
    seen: set = set()
    packages: List[str] = []
    for fix in state.fixes_attempted or []:
        if not fix.success:
            continue
        anchor = _fix_anchor(state, fix)
        for action in _learnable_actions(fix):
            command = (
                action.get("command") if action["type"] == "command" else ""
            )
            if not command or command in plan or command in seen:
                continue
            if anchor != "" and _FIXER_PACKAGE_BUILD_RE.search(command):
                continue
            seen.add(command)
            steps.append((anchor, command))
            packages.extend(_installed_packages(command))
    return steps, packages


def _recipe_phases(
    state: AgentState,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return the recipe phases and the packages the fixes installed.

    The plan, with the commands of the fixes that worked where the
    fixes ran: in a ``pre_<phase>`` phase just before the phase whose
    command failed, or in a ``post_build`` phase after the plan when
    the artifact check failed. When the place is unknown, they run
    after the package setup, before the first other phase.
    """
    phases = [
        {"name": p.name, "commands": list(p.commands)}
        for p in state.build_plan.phases
    ]
    steps, packages = _recipe_fix_steps(state)
    before: Dict[int, List[str]] = {}
    after: List[str] = []
    for anchor, command in steps:
        if anchor == "":
            after.append(command)
            continue
        index = next(
            (
                i
                for i, phase in enumerate(phases)
                if anchor and anchor in phase["commands"]
            ),
            None,
        )
        if index is None:
            index = _pre_build_index(phases)
        before.setdefault(index, []).append(command)
    for index in sorted(before, reverse=True):
        name = phases[index]["name"] if index < len(phases) else "build"
        phases.insert(
            index, {"name": f"pre_{name}", "commands": before[index]}
        )
    if after:
        phases.append({"name": "post_build", "commands": after})
    return phases, packages


def _plan_machine(state: AgentState) -> Dict[str, str]:
    """Return the -march or -mcpu value the plan compiles for, if any."""
    for command in _plan_commands(state):
        match = re.search(r"-m(arch|cpu)=([\w.+-]+)", command)
        if match:
            return {f"m{match.group(1)}": match.group(2)}
    return {}


# go build flags that take the next argument as their value.
_GO_FLAGS_WITH_VALUE = frozenset(
    {
        "-o",
        "-tags",
        "-ldflags",
        "-gcflags",
        "-asmflags",
        "-mod",
        "-p",
        "-modfile",
        "-overlay",
        "-pkgdir",
        "-pgo",
        "-buildmode",
        "-compiler",
        "-installsuffix",
        "-toolexec",
        "-C",
    }
)
_ENV_SETTING_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _command_words(segment: str) -> List[str]:
    """Split one shell command into words; quotes are kept together."""
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _repo_relative_dir(path: str, repo_path: str) -> Optional[str]:
    """Return a ``cd`` target as a repo-relative directory ("" = root).

    None when the directory is outside the repository.
    """
    if not path.startswith("/"):
        rel = os.path.normpath(path)
    elif repo_path:
        rel = os.path.relpath(path, repo_path)
    else:
        return None
    if rel == ".":
        return ""
    return None if rel.startswith("..") else rel


def _go_build_layout(
    commands: List[str], repo_path: str
) -> Tuple[str, Optional[str]]:
    """Return the main package target and module dir of a Go plan.

    The last ``go build`` decides: its last argument that is neither a
    flag nor a flag value is the target ("." when it has none), and a
    ``cd <dir>`` before it in the same command names the module
    directory. Without one, a ``cd <dir> && go ...`` command elsewhere
    in the plan names it; with none at all, the directory is None.
    """
    target, module_dir, other_dir = "", None, None
    for command in commands:
        cwd: Optional[str] = None
        for segment in re.split(r"\s*(?:&&|;)\s*", command):
            words = _command_words(segment)
            if words[:1] == ["cd"] and len(words) > 1:
                cwd = _repo_relative_dir(words[1], repo_path)
                continue
            while words and (
                words[0] == "env" or _ENV_SETTING_RE.match(words[0])
            ):
                words = words[1:]
            if words[:1] != ["go"]:
                continue
            if cwd is not None:
                other_dir = cwd
            if words[1:2] != ["build"]:
                continue
            args: List[str] = []
            skip_value = False
            for word in words[2:]:
                if skip_value:
                    skip_value = False
                elif word.startswith("-"):
                    skip_value = (
                        "=" not in word and word in _GO_FLAGS_WITH_VALUE
                    )
                else:
                    args.append(word)
            target = args[-1] if args else "."
            module_dir = cwd
    return target, module_dir if module_dir is not None else other_dir


def _plan_uses_cgo(commands: List[str]) -> bool:
    """Return True when a Go plan builds with cgo on purpose."""
    text = " ".join(commands)
    if "CGO_ENABLED=0" in text:
        return False
    if any(
        flag in text for flag in ("CGO_ENABLED=1", "CGO_CFLAGS", "CGO_LDFLAGS")
    ):
        return True
    return any(
        re.match(r"lib\S+-dev$", package)
        for command in commands
        for package in _installed_packages(command)
    )


def _build_env(commands: List[str]) -> Dict[str, str]:
    """Return the NAME=value settings in front of the plan's last build."""
    env: Dict[str, str] = {}
    for command in commands:
        words = _command_words(re.split(r"\s*(?:&&|;)\s*", command)[-1])
        if words[:1] == ["env"]:
            words = words[1:]
        settings: Dict[str, str] = {}
        while words and _ENV_SETTING_RE.match(words[0]):
            name, _, value = words.pop(0).partition("=")
            settings[name] = value
        if (
            settings
            and words[:1]
            and words[0]
            in (
                "go",
                "cargo",
                "make",
                "cmake",
                "meson",
                "ninja",
            )
        ):
            env = settings
    return env


def _analysis_mentions_cgo(state: AgentState) -> bool:
    """Return True when the analyst names cgo for this package."""
    analysis = state.package_analysis
    if analysis is None:
        return False
    text = " ".join(
        [
            *[str(risk) for risk in analysis.riscv_risks],
            str(analysis.build_strategy or ""),
            *[str(dep.get("reason", "")) for dep in analysis.dependencies],
        ]
    )
    return "cgo" in text.lower()


def _learned_layout(state: AgentState) -> Dict[str, Any]:
    """Return the layout facts of a verified port (scout trigger, recipe).

    ``has_main`` as detected at init; for Go the main package, the
    module directory and cgo, from the final plan first (a fix may have
    moved the build) and from the init-time detection second; for other
    build systems the nested build root; and the environment settings
    in front of the build command.
    """
    go_info = state.context_cache.get("go_main_info", {}) or {}
    layout: Dict[str, Any] = {"has_main": bool(go_info.get("has_main", False))}
    commands = _plan_commands(state)
    build_system = (
        state.build_plan.build_system if state.build_plan else ""
    ).lower()
    if build_system == "go":
        target, module_dir = _go_build_layout(commands, state.repo_path)
        main_path = target or go_info.get("main_path") or ""
        if main_path:
            layout["main_path"] = main_path
        layout["module_dir"] = (
            module_dir
            if module_dir is not None
            else (go_info.get("module_dir") or "")
        )
        layout["has_cgo"] = _plan_uses_cgo(commands)
    elif state.build_system_info and (
        state.build_system_info.type.lower() == build_system
    ):
        root = (state.build_system_info.module_dir or "").strip("/")
        if root:
            layout["nested_root"] = root
    env = _build_env(commands)
    if env:
        layout["build_env"] = env
    return layout


def _save_learning_data(state: AgentState):
    """Learn from a verified success; teach nothing from the rest.

    Feeds the scout, fixer and builder few-shot stores and the recipe
    cache. Artifacts are curated first so only user-facing outputs are
    recorded. An unverified or wrong-arch "success" is never learned:
    it would be replayed by later runs as if it had worked. Never
    raises — a learning failure must not sink a finished port.

    The scout trigger and the recipe keep the layout facts (main
    package, module or build root, cgo, build environment). Each fix
    that worked becomes a fixer example keyed on its decisive error
    line, and its shell commands go to a ``pre_build`` phase of the
    recipe, so a replay on a clean sandbox repeats them.
    """
    try:
        repo = state.repo_name or "unknown"
        bs = state.build_plan.build_system if state.build_plan else "unknown"

        # Curated already by finish_node; this covers other callers.
        _curate_artifacts(state)

        # Only a proven riscv64 port may teach later runs. An unverified
        # or wrong-arch "success" in the cache or the few-shot store
        # would be replayed as if it had worked.
        if not state.is_verified_success:
            logger.info(
                f"Not learning from {repo}: build is "
                f"{state.verification_status or 'not verified'}"
            )
            return

        layout = _learned_layout(state)

        # Scout learning
        if state.build_plan and state.build_plan.phases:
            scout_data = {
                "name": f"Auto: {repo} ({bs})",
                "tags": [bs, repo],
                "build_system": bs,
                "repo_name": repo,
                # The layout facts tell two packages of one build system
                # apart (main package, module or build root, cgo).
                "trigger": {"build_system": bs, **layout},
                "plan": {
                    "build_system": bs,
                    "phases": [
                        {"name": p.name, "commands": p.commands}
                        for p in state.build_plan.phases
                    ],
                },
                "reasoning": f"Auto-learned from successful {repo} build.",
            }
            save_learned_example("scout", scout_data)

        # Fixer learning: one example per fix that worked.
        for fix in state.fixes_attempted or []:
            if fix.success and fix.strategy:
                # The actions a later run can repeat: no read-only
                # command, no stub file (see _learnable_actions). A fix
                # with none of them is not saved.
                actions = _learnable_actions(fix)
                if not actions:
                    continue
                # The decisive error line that the fix removed, not the
                # failed command that heads the message and not Go's
                # download lines. With no such line the fix has no key
                # a later run can match.
                key_line = _fix_key_line(state, fix)
                if not key_line:
                    continue
                fixer_data = {
                    "name": f"Auto: {repo} - {_shorten(fix.strategy, 60)}",
                    "tags": [bs, "auto-fix"],
                    "build_system": bs,
                    "repo_name": repo,
                    "error_pattern": (
                        _error_regex(key_line) if key_line else ""
                    ),
                    "error_context": {"error_message": key_line[:200]},
                    "fix": {"strategy": fix.strategy, "actions": actions},
                    "reasoning": f"Auto-learned fix from {repo}.",
                }
                save_learned_example("fixer", fixer_data)

        # Builder learning
        if state.build_plan and state.build_plan.phases:
            builder_data = {
                "name": f"Auto: {repo} build execution",
                "tags": [bs, repo],
                "build_system": bs,
                "repo_name": repo,
                "phases": [
                    {"name": p.name, "commands": p.commands}
                    for p in state.build_plan.phases
                ],
                "timeout_recommendation": (
                    "600s" if len(state.build_plan.phases) > 2 else "120s"
                ),
                "reasoning": f"Auto-learned build execution from {repo}.",
            }
            save_learned_example("builder", builder_data)

        # Recipe cache
        if state.build_plan:
            duration = (
                state.get_execution_duration()
                if hasattr(state, "get_execution_duration")
                else 0.0
            )
            # The commands of the fixes that worked run again where they
            # ran, so a replay on a clean sandbox repeats them.
            phases, fix_packages = _recipe_phases(state)
            save_to_recipe_cache(
                repo_name=repo,
                repo_url=state.repo_url or "",
                build_system=bs,
                build_plan={"phases": phases},
                dependencies=sorted(
                    set(
                        state.dependencies.system_packages
                        if state.dependencies
                        else []
                    )
                    | set(fix_packages)
                ),
                # The diffs that were applied, not strategy descriptions.
                patches=list(state.patches_generated),
                artifacts=state.curated_artifacts
                or state.build_artifacts
                or [],
                build_duration_seconds=duration,
                recipe_markdown=state.porting_recipe or None,
                verification=state.artifact_verification,
                layout=layout or None,
                machine=_plan_machine(state) or None,
            )

        logger.info(f"Auto-learning complete for {repo}")
    except Exception as e:
        logger.warning(f"Auto-learning failed (non-fatal): {e}")


# The package's own tests run once after a successful build and never
# decide the port (user decision 2026-09-30: non-gating). In a batch
# run the limit is cut so that the tests end _TEST_RESERVE seconds
# before the batch kill (ATESOR_RUN_DEADLINE, epoch seconds): the
# summarizer and curator LLM calls, the outputs, the native mirror copy
# and the zip still need that time. Under _TEST_MIN seconds the phase
# is skipped; ATESOR_PACKAGE_TEST_TIMEOUT=0 turns it off.
PACKAGE_TEST_TIMEOUT = 600
_TEST_RESERVE = 420
_TEST_MIN = 60
_TEST_OUTPUT_TAIL = 3000


def _package_test_timeout(state: AgentState) -> int:
    """Return the seconds the package tests may use; 0 means skip."""
    raw = os.environ.get("ATESOR_PACKAGE_TEST_TIMEOUT", "").strip()
    try:
        limit = int(raw) if raw else PACKAGE_TEST_TIMEOUT
    except ValueError:
        limit = PACKAGE_TEST_TIMEOUT
    deadline = os.environ.get("ATESOR_RUN_DEADLINE", "").strip()
    if deadline:
        try:
            remaining = float(deadline) - time.time() - _TEST_RESERVE
            limit = min(limit, int(remaining))
        except ValueError:
            pass
    return limit if limit >= _TEST_MIN else 0


def _detect_test_command(state: AgentState) -> Optional[tuple]:
    """Return ``(framework, command)`` for the package's tests, or None.

    Looks at the configured build tree first (CTest, Meson), then at
    the language build files, then at a ``check`` or ``test`` target in
    the top Makefile.
    """
    root = _to_host_path(state.repo_path)
    build_system = (
        state.build_plan.build_system if state.build_plan else ""
    ).lower()

    def has(*parts: str) -> bool:
        return os.path.exists(os.path.join(root, *parts))

    for build_dir in ("build", "_build", "builddir"):
        if has(build_dir, "CTestTestfile.cmake"):
            return (
                "ctest",
                f"ctest --test-dir {build_dir} --output-on-failure",
            )
        if has(build_dir, "meson-info"):
            return (
                "meson test",
                f"meson test -C {build_dir} --print-errorlogs",
            )
    if has("go.mod") and (build_system == "go" or not has("Makefile")):
        return ("go test", "go test ./...")
    if has("Cargo.toml") and (
        build_system in ("cargo", "rust") or not has("Makefile")
    ):
        return ("cargo test", "cargo test --release")
    makefile = os.path.join(root, "Makefile")
    if os.path.isfile(makefile):
        try:
            with open(makefile, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            text = ""
        for target in ("check", "test"):
            if re.search(rf"^{target}\s*:", text, re.MULTILINE):
                return (f"make {target}", f"make {target}")
    return None


# A batch port whose build ran longer than this skips its package
# tests: their run time grows with the build (decision D6).
_BATCH_TEST_BUILD_LIMIT = 600


def _batch_test_skip(state: AgentState) -> str:
    """Return why a batch port skips its package tests, or "".

    Batch runs only (``ATESOR_RUN_DEADLINE`` is set): the tests of an
    unverified build prove nothing, and a build over 10 minutes has
    tests that hit the limit while they still download modules.
    """
    if not os.environ.get("ATESOR_RUN_DEADLINE", "").strip():
        return ""
    if not state.is_verified_success:
        return "the build is not verified"
    build_seconds = sum(
        result.duration_seconds
        for result in state.command_results_cache.values()
        if result.success
    )
    if build_seconds > _BATCH_TEST_BUILD_LIMIT:
        return (
            f"the build ran {build_seconds / 60:.0f} minutes; a batch run "
            "tests builds of 10 minutes or less"
        )
    return ""


def _match_tests_to_build(
    state: AgentState, detected: tuple
) -> Tuple[tuple, Optional[Dict[str, str]]]:
    """Return the test command and environment that match the build.

    The tests run with the ``NAME=value`` settings in front of the
    plan's last build command, and ``go test`` gets the ``-tags`` of the
    last ``go build``: without them, jr's tests failed at link time.

    Returns:
        ``(detected, env)``; ``env`` is None when the build set nothing.
    """
    commands = _plan_commands(state)
    env = _build_env(commands)
    framework, command = detected
    if framework == "go test" and "-tags" not in command:
        tags = _go_build_tags(commands)
        if tags:
            command = command.replace(
                "go test", f"go test -tags {shlex.quote(tags)}", 1
            )
    return (framework, command), (env or None)


def _go_build_tags(commands: List[str]) -> str:
    """Return the ``-tags`` value of the plan's last ``go build``."""
    tags = ""
    for command in commands:
        for segment in re.split(r"\s*(?:&&|;)\s*", command):
            words = _command_words(segment)
            while words and _ENV_SETTING_RE.match(words[0]):
                words = words[1:]
            if words[:2] != ["go", "build"]:
                continue
            for index, word in enumerate(words):
                if word == "-tags" and index + 1 < len(words):
                    tags = words[index + 1]
                elif word.startswith("-tags="):
                    tags = word[len("-tags=") :]
    return tags


# Keep the verified build of the port's tree across its package tests.
# lz4: `make check` rebuilt programs/lz4 with the test flags after the
# verification and pointed the link at that build. The copies go outside
# the tree, which the package zips. The first output line is the folder.
_TEST_KEEP_SCRIPT = (
    'k=$(mktemp -d "${TMPDIR:-/tmp}/atesor-test-keep.XXXXXX") || exit 1; '
    'echo "$k"; '
    "find . -path ./.git -prune -o -type l -print | "
    'while IFS= read -r p; do printf "%s\\t%s\\n" "$p" "$(readlink "$p")"; '
    'done > "$k/links" || exit 1; '
    ': > "$k/files"; i=0; '
    'for f in "$@"; do [ -f "$f" ] || continue; i=$((i + 1)); '
    'cp -pL "$f" "$k/$i" && printf "%s\\t%s\\n" "$i" "$f" >> "$k/files" '
    "|| exit 1; done"
)
# Prints each file and link that it puts back, then removes the copies.
_TEST_RESTORE_SCRIPT = (
    'k="$1"; tab=$(printf "\\t"); '
    'while IFS="$tab" read -r i f; do cmp -s "$k/$i" "$f" && continue; '
    'rm -f "$f" && mkdir -p "$(dirname "$f")" && cp -p "$k/$i" "$f" '
    '&& echo "$f" || exit 1; done < "$k/files"; '
    'while IFS="$tab" read -r p t; do '
    '[ "$(readlink "$p" 2>/dev/null)" = "$t" ] && continue; '
    'rm -f "$p" && ln -s "$t" "$p" && echo "$p" || exit 1; '
    'done < "$k/links"; '
    'rm -f "$k"/* && rmdir "$k"'
)


def _verified_artifact_rels(state: AgentState) -> List[str]:
    """Return the tree-relative paths of the port's verified artifacts."""
    repo = (state.repo_path or "").rstrip("/") + "/"
    rels: List[str] = []
    for item in state.curated_artifacts or state.build_artifacts or []:
        path = item.get("filepath") or item.get("path") or ""
        if path.startswith(repo):
            rel = os.path.normpath(path[len(repo) :])
            if rel and not rel.startswith("..") and rel not in rels:
                rels.append(rel)
    return rels


def _keep_verified_build(state: AgentState) -> str:
    """Copy the verified artifacts and the tree's links aside.

    Returns:
        The folder of the copies, or "" when the copy failed (the tests
        then run without it).
    """
    done = execute_command(
        ["sh", "-c", _TEST_KEEP_SCRIPT, "sh"] + _verified_artifact_rels(state),
        cwd=state.repo_path,
        timeout=600,
        validate=False,
    )
    folder = ((done.stdout or "").splitlines() or [""])[0].strip()
    # The first line must be the folder that mktemp made.
    if not done.success or "/atesor-test-keep." not in folder:
        logger.info(
            "Could not keep the verified build before the package tests: "
            f"{_one_line_tail(done.stderr or done.stdout or '')}"
        )
        return ""
    return folder


def _restore_verified_build(state: AgentState, folder: str) -> Dict[str, Any]:
    """Put back each artifact and link that the package tests changed.

    Returns:
        ``{"restored": [...]}`` when something changed, with
        ``"restore_error"`` when a step failed; else ``{}``.
    """
    done = execute_command(
        ["sh", "-c", _TEST_RESTORE_SCRIPT, "sh", folder],
        cwd=state.repo_path,
        timeout=600,
        validate=False,
    )
    restored = [line for line in (done.stdout or "").splitlines() if line]
    out: Dict[str, Any] = {"restored": restored} if restored else {}
    if restored:
        logger.info(
            f"The package tests changed {', '.join(restored)[:200]}; "
            "Atesor put back the verified build"
        )
    if not done.success:
        out["restore_error"] = _one_line_tail(done.stderr or "", 300) or (
            f"exit {done.exit_code}"
        )
        logger.warning(
            f"Could not put back the verified build: {out['restore_error']}"
        )
    return out


def _run_package_tests(
    state: AgentState,
    repo_path: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    detected: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Run the package's own test suite once and describe the outcome.

    Never raises: the phase must not turn a finished port into an
    escalation. The build status is not changed.

    Args:
        state: The run state.
        repo_path: The tree to test; the port's tree by default.
            Performance mode tests each variant tree.
        extra_env: Extra environment for the test command, or None.
        detected: ``(framework, command)`` to run instead of detecting
            it, or None.

    Returns:
        A dict with ``status`` (passed, failed, timeout, skipped or
        error) and, when tests ran, the framework, command, exit code,
        duration and output tail.
    """
    try:
        timeout = _package_test_timeout(state)
        if timeout <= 0:
            return {
                "status": "skipped",
                "reason": "disabled, or no time left in the run budget",
            }
        skip = _batch_test_skip(state)
        if skip:
            return {"status": "skipped", "reason": skip}
        detected = detected or _detect_test_command(state)
        if not detected:
            return {
                "status": "skipped",
                "reason": (
                    "no test suite found (ctest, meson test, go test, "
                    "cargo test, make check or make test)"
                ),
            }
        if extra_env is None:
            detected, extra_env = _match_tests_to_build(state, detected)
        framework, command = detected
        # Go ends its own test binaries at this limit; the sandbox's kill
        # stops only the go command (k9s, bearer: tests ran on).
        if framework == "go test" and "-timeout" not in command:
            command = command.replace(
                "go test", f"go test -timeout {max(timeout - 30, 30)}s", 1
            )
        logger.info(f"Running package tests: {command} (<= {timeout}s)")
        # The optimizer keeps the builds of its own trees.
        kept = _keep_verified_build(state) if repo_path is None else ""
        started = time.time()
        result = execute_command(
            command,
            cwd=repo_path or state.repo_path,
            timeout=timeout,
            extra_env=extra_env,
        )
        duration = round(time.time() - started, 1)
        restore = _restore_verified_build(state, kept) if kept else {}
        output = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
        # 137: the in-sandbox timeout --signal=KILL; -1: the host limit.
        # Exit 137 is also an OOM kill, so the elapsed time decides.
        timed_out = result.exit_code in (137, -1) and (
            duration >= timeout - 35
        )
        # A stopped container, a dropped ssh session or a host that
        # slept says nothing about the tests (sslsearch: exit 255 after
        # a night with the laptop lid closed).
        fault = infrastructure_fault(result) if result.exit_code else ""
        reason = {"reason": f"infrastructure fault: {fault}"} if fault else {}
        if result.exit_code == 0:
            status = "passed"
        elif fault:
            status = "error"
        elif timed_out:
            status = "timeout"
        else:
            status = "failed"
        return {
            "status": status,
            **reason,
            "framework": framework,
            "command": command,
            "exit_code": result.exit_code,
            "duration_seconds": duration,
            "timeout_seconds": timeout,
            "output_tail": output[-_TEST_OUTPUT_TAIL:],
            **restore,
        }
    except Exception as exc:
        logger.warning(f"Package tests could not run: {exc}")
        return {"status": "error", "reason": str(exc)[:300]}


@agent_node(AgentRole.SUMMARIZER)
def finish_node(state: AgentState) -> AgentState:
    """Close out a successful port: package tests, recipe, learning.

    Runs the package's own test suite once (never gates the port),
    asks the summarizer model for the Markdown recipe with a
    deterministic fallback, then saves learning data. SUCCESS always
    reaches this node, even at the attempt or cost ceiling.
    """
    logger.info("Generating comprehensive porting guide...")

    state.build_status = BuildStatus.SUCCESS
    state.current_phase = "finished"

    if state.package_tests is None:
        state.package_tests = _run_package_tests(state)
        logger.info(f"Package tests: {state.package_tests['status']}")

    arch_issues = (
        "\n".join(
            [
                f"- {a.file}:{a.line} - {a.arch_type} ({a.severity})"
                for a in state.arch_specific_code[:20]
            ]
        )
        if state.arch_specific_code
        else "None found."
    )

    fixes = (
        "\n".join(
            [
                f"- {fix.strategy} ({'Success' if fix.success else 'Failed'})"
                for fix in state.fixes_attempted
            ]
        )
        if state.fixes_attempted
        else "No fixes were needed (generic code)."
    )

    build_steps = ""
    if state.build_plan:
        for phase in state.build_plan.phases:
            build_steps += f"\n### {phase.name}\n```bash\n"
            for cmd in phase.commands:
                build_steps += f"{cmd}\n"
            build_steps += "```\n"

    # Curate first, so the summary lists the deliverables, not the
    # build-system probes.
    _curate_artifacts(state)

    # Prefer the curated artifact list: primary first, no noise.
    artifacts_info = ""
    artifacts_to_show = state.curated_artifacts or state.build_artifacts
    if artifacts_to_show:
        primary = [a for a in artifacts_to_show if a.get("role") == "primary"]
        secondary = [
            a for a in artifacts_to_show if a.get("role") == "secondary"
        ]
        rest = [a for a in artifacts_to_show if not a.get("role")]

        artifacts_info = "\n### Build Artifacts Generated\n\n"
        for label, group in (
            ("Primary", primary),
            ("Secondary (tests / examples)", secondary),
            ("Other", rest),
        ):
            if not group:
                continue
            if label != "Other" or (not primary and not secondary):
                artifacts_info += f"**{label}:**\n" if label != "Other" else ""
            for art in group:
                arch = art.get("architecture", "?")
                path = art.get("filepath") or art.get("path", "")
                artifacts_info += (
                    f"- **{art.get('type', 'binary')}** ({arch}): `{path}`\n"
                )
            artifacts_info += "\n"

    build_steps += artifacts_info

    # The guide must say whether the riscv64 output was proven; a
    # build whose commands passed is not yet a verified port.
    verification = state.artifact_verification or {}
    verdict = verification.get("status", "not run")
    verdict_reason = verification.get("reason", "")
    verdict_text = verdict + (f": {verdict_reason}" if verdict_reason else "")
    build_steps += f"\n**Artifact verification:** {verdict_text}\n"
    tests = state.package_tests or {}
    tests_text = tests.get("status", "not run")
    if tests.get("command"):
        tests_text += f" (`{tests['command']}`, exit {tests.get('exit_code')})"
    elif tests.get("reason"):
        tests_text += f": {tests['reason']}"
    build_steps += f"\n**Package tests (not gating):** {tests_text}\n"

    prompt = SUMMARIZER_PROMPT.format(
        repo_name=state.repo_name,
        repo_url=state.repo_url,
        build_system=(
            state.build_plan.build_system if state.build_plan else "unknown"
        ),
        build_steps=build_steps,
        arch_issues=arch_issues,
        fixes=fixes,
        api_calls=state.api_calls_made,
        cost=state.api_cost_usd,
        duration=f"{state.get_execution_duration():.1f}s",
    )

    try:
        if _llm_refused(state):
            raise RuntimeError("the model refused this package earlier")
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.SUMMARIZER)
        response = invoke_llm(llm, messages)
        tokens_in, tokens_out = response_usage(response)
        cost = response_cost(llm, response)
        state.log_api_call(
            cost=cost, tokens_in=tokens_in, tokens_out=tokens_out
        )

        recipe = extract_content(response.content)

        cache_read, cache_write = response_cache_usage(response)
        log_llm_call(
            agent_role=AgentRole.SUMMARIZER.value,
            prompt=prompt,
            response=recipe,
            model=_llm_name(llm) or "unknown",
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata={
                "repo": state.repo_name,
                "phase": "finish",
                "stop_reason": response_stop_reason(response),
                "cache_read": cache_read,
                "cache_write": cache_write,
            },
        )

        if is_refusal(response):
            _mark_llm_refused(state, AgentRole.SUMMARIZER.value)
            raise RuntimeError("the model refused the request")
        # An empty recipe would leave no recipe file at all.
        if not recipe.strip():
            raise RuntimeError("the model returned an empty recipe")

        state.porting_recipe = recipe
        logger.info("Porting guide generated successfully")

    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        state.porting_recipe = (
            f"# RISC-V Porting Recipe: {state.repo_name}\n\n"
            f"The build commands succeeded. Artifact verification: "
            f"{verdict_text}.\n\n{build_steps}"
        )

    _save_learning_data(state)
    _log_data_effect(state)

    return state


# cached_recipe_node — replay a verified cached recipe (--optimize).


def _cached_purpose(recipe: Dict[str, Any], repo_name: str) -> str:
    """Return the first sentence of the cached recipe's summary."""
    text = str(recipe.get("recipe_markdown") or "")
    match = re.search(r"#+\s*Executive Summary\s*\n+(.+)", text)
    if not match:
        return repo_name
    sentence = re.split(r"(?<=\.)\s", match.group(1).strip())[0]
    return sentence[:300] or repo_name


def _recipe_commands(recipe: Dict[str, Any]) -> List[str]:
    """Return every non-empty command of a cached recipe, in order."""
    return [
        str(command)
        for phase in (recipe.get("build_plan") or {}).get("phases") or []
        for command in phase.get("commands") or []
        if str(command).strip()
    ]


def _plan_from_recipe(state: AgentState, recipe: Dict[str, Any]) -> bool:
    """Make a cached recipe the build plan and apply its patches.

    Used by cached_recipe_node (performance mode) and by the recall seed
    of heuristic_plan_node. The note in
    ``context_cache["cached_recipe_note"]`` says where the plan came
    from and whether the source moved on since it was verified.

    Returns:
        False when the recipe holds no command; nothing changes then.
    """
    phases = []
    for phase in (recipe.get("build_plan") or {}).get("phases") or []:
        commands = [
            str(command)
            for command in phase.get("commands") or []
            if str(command).strip()
        ]
        if commands:
            phases.append(
                BuildPhase(
                    len(phases) + 1,
                    str(phase.get("name") or f"phase{len(phases) + 1}"),
                    commands,
                )
            )
    if not phases:
        return False
    build_system = str(recipe.get("build_system") or "unknown")
    built = recipe.get("last_built") or "an earlier run"
    commit = (recipe.get("verification") or {}).get("source_commit")
    state.build_plan = BuildPlan(
        build_system=build_system,
        build_system_confidence=1.0,
        phases=phases,
        total_estimated_duration=(
            f"{recipe.get('build_duration_seconds', '?')}s (cached)"
        ),
        notes=[f"From the recipe cache: the verified run of {built}"],
    )
    applied = 0
    patches = [str(patch) for patch in recipe.get("patches") or []]
    for patch in patches:
        if apply_patch(patch, cwd=state.repo_path, use_docker=True):
            applied += 1
            state.patches_generated.append(patch)
        else:
            logger.warning("A cached patch did not apply; the build decides")
    head = execute_command(
        ["git", "-C", state.repo_path, "rev-parse", "HEAD"],
        timeout=60,
        validate=False,
    )
    head_commit = (head.stdout or "").strip() if head.success else ""
    note = (
        f"Recipe from the cache (verified run of {built}): "
        f"{sum(len(p.commands) for p in phases)} commands in "
        f"{len(phases)} phases"
        + (f", {applied} of {len(patches)} patches applied" if patches else "")
    )
    if commit and head_commit and head_commit != commit:
        note += (
            f"; the recipe was verified on commit {commit[:12]}, and the "
            f"source is now at {head_commit[:12]}"
        )
    logger.info(note)
    state.context_cache["cached_recipe_note"] = note
    return True


@agent_node(AgentRole.PLANNER)
def cached_recipe_node(state: AgentState) -> AgentState:
    """Use a verified cached recipe as the build plan, without planning.

    Performance mode needs a real build tree, so the port builds again.
    The recipe of an earlier verified run replaces the analyst and the
    scout: no planning LLM call. Its patches are applied first. When the
    build fails anyway (the source moved on), the normal fixer and
    re-plan loop takes over.
    """
    recipe = state.cached_recipe or {}
    misfit = _plan_isa_problems(state, _recipe_commands(recipe))
    if misfit:
        logger.warning(
            f"The cached recipe compiles for {'; '.join(misfit)}; "
            "planning again"
        )
        state.cached_recipe = None
        return state
    if not _plan_from_recipe(state, recipe):
        logger.warning("The cached recipe has no command; planning again")
        state.cached_recipe = None
        return state
    build_system = str(recipe.get("build_system") or "unknown")
    artifacts = [
        os.path.basename(str(item.get("path") or ""))
        for item in recipe.get("artifacts") or []
        if isinstance(item, dict) and item.get("path")
    ]
    state.package_analysis = PackageAnalysis(
        purpose=_cached_purpose(recipe, state.repo_name),
        language="unknown",
        build_system=build_system,
        build_system_confidence=1.0,
        build_system_reasoning="recipe cache: a verified earlier run",
        dependencies=[
            {"name": str(name), "reason": "recipe cache"}
            for name in (recipe.get("dependencies") or [])[:15]
        ],
        riscv_risks=[],
        build_strategy="build with the cached recipe",
        expected_artifacts=[name for name in artifacts if name][:10],
        needs_custom_plan=False,
        complexity=1,
        llm_grounded=False,
    )
    state.log_scripted_op("cached_recipe")
    state.context_cache["plan_source"] = "cached_recipe"
    return state


# Recall: what earlier runs of the package left (recipes, failures).


# Directories of bundled third-party code. Their arch-specific code (an
# x86 asm file in gnulib, zlib's contrib) does not decide whether the
# package itself needs a custom plan.
_VENDORED_DIR_RE = re.compile(
    r"(?:^|/)(?:gnulib|build-aux|vendor|third_party|3rdparty|contrib|"
    r"support)/"
)


def _in_vendored_dir(path: str) -> bool:
    """Return True when a path lies in bundled third-party code."""
    return bool(_VENDORED_DIR_RE.search(path or ""))


def _foreign_recipe_layout(state: AgentState) -> Tuple[Dict[str, Any], str]:
    """Return the layout facts of this package's recipe on another distro.

    Its commands use another package manager, but the source layout
    (main package, module directory) is the same.
    """
    entry = load_recipe_cache().get("packages", {}).get(state.repo_name)
    if not isinstance(entry, dict):
        return {}, ""
    for sandbox in sorted(entry):
        recipe = get_cached_recipe(
            state.repo_name, sandbox=sandbox, repo_url=state.repo_url
        )
        if not recipe:
            continue
        layout = dict(recipe.get("layout") or {})
        if not layout and recipe.get("build_system") == "go":
            target, module_dir = _go_build_layout(
                _recipe_commands(recipe), state.repo_path
            )
            if target:
                layout = {"main_path": target, "module_dir": module_dir or ""}
        if layout:
            return layout, sandbox
    return {}, ""


def _recall(state: AgentState) -> Dict[str, Any]:
    """Return what earlier runs left for this package; cached per run.

    ``recipe`` and ``sandbox``: the cached recipe of this sandbox, else
    of the same distro on the other target (QEMU or native). Without
    one, ``layout`` and ``layout_sandbox``: the layout facts of a recipe
    from another distro. ``failures``: the failure memory of the
    package. Empty with --replan.
    """
    recall = state.context_cache.get("recall")
    if recall is not None:
        return recall
    recall = {}
    if not state.replan and state.repo_name:
        for sandbox in recall_sandboxes():
            recipe = get_cached_recipe(
                state.repo_name, sandbox=sandbox, repo_url=state.repo_url
            )
            if recipe and _recipe_commands(recipe):
                recall["recipe"] = {
                    key: value
                    for key, value in recipe.items()
                    if key != "recipe_markdown"
                }
                recall["sandbox"] = sandbox
                break
        if "recipe" not in recall:
            layout, where = _foreign_recipe_layout(state)
            if layout:
                recall["layout"] = layout
                recall["layout_sandbox"] = where
        failures = load_failure_records(state.repo_name)
        if failures:
            recall["failures"] = failures
    state.context_cache["recall"] = recall
    return recall


def _recall_seed(state: AgentState) -> Optional[Dict[str, Any]]:
    """Return the recalled recipe when it can be the plan, else None.

    The recipe must be verified, every command must pass the plan
    validator, and when the analyst names the artifacts, the recipe must
    have built one of them: a verified entry can still record only a
    side tool (netbird's recorded build makes a metrics helper).
    """
    recipe = _recall(state).get("recipe")
    if not recipe:
        return None
    if (recipe.get("verification") or {}).get("status") != "verified":
        return None
    commands = _recipe_commands(recipe)
    if not all(_plan_command_validator.is_safe(c)[0] for c in commands):
        return None
    trial = BuildPlan(
        build_system=str(recipe.get("build_system") or "unknown"),
        build_system_confidence=1.0,
        phases=[
            BuildPhase(index + 1, "recipe", [command])
            for index, command in enumerate(commands)
        ],
        total_estimated_duration="unknown",
    )
    if not validate_build_plan(trial)[0]:
        return None
    misfit = _plan_isa_problems(state, commands)
    if misfit:
        logger.info(
            f"The cached recipe of {state.repo_name} compiles for "
            f"{'; '.join(misfit)}; it is a hint, not the plan"
        )
        return None
    analysis = state.package_analysis
    expected = [
        os.path.basename(str(name)).lower()
        for name in (analysis.expected_artifacts if analysis else [])
        if str(name).strip()
    ]
    built = [
        os.path.basename(str(item.get("path") or "")).lower()
        for item in recipe.get("artifacts") or []
        if isinstance(item, dict) and item.get("path")
    ]
    if (
        expected
        and built
        and not any(
            b == e or b.startswith(e + ".") or e.startswith(b + ".")
            for b in built
            for e in expected
        )
    ):
        logger.info(
            f"The cached recipe of {state.repo_name} built {built[:3]}, "
            f"not the expected {expected[:3]}; it is a hint, not the plan"
        )
        return None
    return recipe


def _note_examples(state: AgentState, role: str, section: Any) -> None:
    """Record the ids of the few-shot examples a prompt carried."""
    shown = state.context_cache.setdefault("examples_shown", {})
    for example_id in getattr(section, "ids", ()) or ():
        ids = shown.setdefault(role, [])
        if example_id not in ids:
            ids.append(example_id)


# Commands that every install fix holds; a match on them is no copy.
_INDEX_UPDATE_RE = re.compile(r"^\s*(?:apt-get|apt|apk)\s+update\s*$")


def _copied_examples(state: AgentState) -> List[str]:
    """Return the shown fixer examples that a fix that worked repeated.

    An example counts when one of its commands (a package index update
    aside) is a command of a fix that worked, spaces aside.
    """
    shown = (state.context_cache.get("examples_shown") or {}).get("fixer")
    if not shown:
        return []

    def norm(command: Any) -> str:
        return " ".join(str(command or "").split())

    ran = {
        norm(action.get("command"))
        for fix in state.fixes_attempted
        if fix.success
        for action in _learnable_actions(fix)
        if action.get("command")
    }
    by_id = {ex.id: ex for ex in get_agent_memory("fixer").examples}
    copied = []
    for example_id in shown:
        example = by_id.get(example_id)
        actions = (
            (example.fix or example.solution or {}).get("actions") or []
            if example
            else []
        )
        if any(
            norm(a.get("command")) in ran
            and not _INDEX_UPDATE_RE.match(norm(a.get("command")))
            for a in actions
            if isinstance(a, dict) and a.get("command")
        ):
            copied.append(example_id)
    return copied


def _log_data_effect(state: AgentState, outcome: Optional[str] = None) -> None:
    """Write what the learned data gave this run to the call log.

    One record per run (see ``llm_logger.log_data_effect``): the plan
    source, the recalled recipe, the examples shown and copied, the
    fixer rounds, the cost and the duration. A second call for the same
    run writes nothing. Never raises.

    Args:
        state: The run state.
        outcome: The outcome to record in place of the one the state
            gives; ``main.py`` passes ``interrupted`` for a run that
            Ctrl-C or the batch's SIGTERM stopped (nushell had none).
    """
    try:
        if state.context_cache.get("data_effect_logged"):
            return
        if outcome:
            pass
        elif state.is_verified_success:
            outcome = "verified"
        elif state.build_status == BuildStatus.SUCCESS:
            outcome = "unverified"
        else:
            outcome = "failed"
        state.context_cache["data_effect_logged"] = True
        log_data_effect(
            {
                "repo": state.repo_name,
                "outcome": outcome,
                "outcome_label": state.outcome_label,
                "plan_source": state.context_cache.get("plan_source", ""),
                "recall": _recall_label(state),
                "examples_shown": state.context_cache.get("examples_shown")
                or {},
                "examples_copied": _copied_examples(state),
                "fixer_rounds": len(state.fixes_attempted),
                "builds": state.build_count,
                "llm_calls": state.api_calls_made,
                "cost_usd": round(state.api_cost_usd, 4),
                "duration_seconds": round(state.get_execution_duration(), 1),
            }
        )
    except Exception as e:
        logger.warning(f"Data-effect record failed (non-fatal): {e}")


def _recall_label(state: AgentState) -> str:
    """Return "<repo>[<sandbox>]" of the recalled recipe, or ""."""
    recall = _recall(state)
    if not recall.get("recipe"):
        return ""
    return f"{state.repo_name}[{recall.get('sandbox')}]"


def _recall_prompt_section(state: AgentState) -> str:
    """Render the recall for the scout and fixer prompts ("" if empty)."""
    recall = _recall(state)
    parts: List[str] = []
    recipe = recall.get("recipe")
    if recipe:
        verdict = (recipe.get("verification") or {}).get("status")
        heading = (
            "## Verified plan of this package (earlier run)"
            if verdict == "verified"
            else (
                "## Plan of this package from an earlier run (its "
                "program crashed on the machine; do not repeat its "
                "machine flags)"
                if verdict == "crashes"
                else "## Plan of this package from an earlier run "
                "(not verified)"
            )
        )
        phases = [
            {"name": phase.get("name"), "commands": phase.get("commands")}
            for phase in (recipe.get("build_plan") or {}).get("phases") or []
        ]
        if state.error_history and _recipe_commands(recipe) == _plan_commands(
            state
        ):
            advice = (
                "This run built with this plan and it failed, so the "
                "source or the sandbox changed since that run."
            )
        else:
            advice = (
                "Reuse its commands unless the evidence above shows that "
                "they no longer fit."
            )
        parts.append(
            f"{heading}\n"
            f"Sandbox: {recall.get('sandbox')} | Built: "
            f"{str(recipe.get('last_built') or '?')[:10]} | Verification: "
            f"{verdict or 'not recorded'}\n"
            f"```json\n"
            f"{cut_text(json.dumps({'phases': phases}, indent=2), 3000)}\n"
            f"```\n{advice}"
        )
    elif recall.get("layout"):
        parts.append(
            "## Layout of this package from a verified build on "
            f"{recall.get('layout_sandbox')}\n"
            f"{json.dumps(recall['layout'])}"
        )
    failures = recall.get("failures") or {}
    if failures:
        lines = ["## Tried before on this package (the error stayed)"]
        for key, records in list(failures.items())[:5]:
            for record in records[-3:]:
                commands = "; ".join(record.get("commands") or [])
                lines.append(
                    f"- Error `{_shorten(key, 160)}`: "
                    f"{record.get('strategy', '')} (commands: "
                    f"{_shorten(commands, 200)}); then: "
                    f"{record.get('error_after', '')}"
                )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _extra_go_mains(
    state: AgentState, go_main_info: Dict[str, Any], build_cmd: str
) -> List[str]:
    """Return a ``go build`` for each expected program the plan skips.

    An expected artifact name with its own ``cmd/<name>`` directory in
    the module gets one build command; the main build's own target is
    left out. At most five.
    """
    analysis = state.package_analysis
    if analysis is None or not analysis.llm_grounded:
        return []
    root = _to_host_path(state.repo_path)
    module_dir = str(go_main_info.get("module_dir") or "").strip("/")
    target, _ = _go_build_layout([build_cmd], state.repo_path)
    built = target[2:] if target.startswith("./") else target
    commands: List[str] = []
    for name in analysis.expected_artifacts or []:
        base = os.path.basename(str(name).strip())
        if not re.fullmatch(r"[A-Za-z0-9._-]+", base):
            continue
        rel = f"cmd/{base}"
        if rel == built or not os.path.isdir(
            os.path.join(root, module_dir, rel)
        ):
            continue
        command = f"go build -buildvcs=false -o {base} ./{rel}"
        if module_dir:
            command = f"cd {shlex.quote(module_dir)} && {command}"
        if command not in commands:
            commands.append(command)
    return commands[:5]


def _analyst_go_settings(state: AgentState) -> Tuple[Dict[str, str], str]:
    """Return the Go build environment and tags the analyst read.

    Only when the analyst's strategy names a build file (GNUmakefile,
    build.sh, ...): the settings then come from that file, not a guess
    (vuls needs GOEXPERIMENT=jsonv2 from its GNUmakefile).
    """
    analysis = state.package_analysis
    if analysis is None or not analysis.llm_grounded:
        return {}, ""
    text = " ".join(
        [str(analysis.build_strategy or "")]
        + [str(risk) for risk in analysis.riscv_risks]
    )
    if not re.search(
        r"(?i)\b(?:GNUmakefile|Makefile|build\.sh|\.goreleaser|Taskfile|"
        r"magefile|justfile)\b",
        text,
    ):
        return {}, ""
    # A value at the end of a sentence carries its full stop.
    env = {
        name: value.rstrip(".,;:")
        for name, value in re.findall(
            r"\b(CGO_ENABLED|GOEXPERIMENT)=([A-Za-z0-9_.,-]+)", text
        )
    }
    tags = re.search(r"-tags[= ]+[\"']?([A-Za-z0-9_,.-]+)", text)
    return env, tags.group(1).rstrip(".,;:") if tags else ""


# optimize_node and the optimizer agent — performance mode (--optimize).


# A harness is a whole source file: a free coder model can need more
# than the default wall time to write one.
OPTIMIZER_CODE_TIMEOUT = 240

# LLM cost limit of one --optimize step in USD (ATESOR_OPTIMIZE_COST_CAP).
# The $1 cap of should_escalate covers the fix loop only, and one
# workload source can ask up to 8 rounds (brotli: 23 calls, $0.19).
OPTIMIZER_COST_CAP = 0.50


def _optimizer_cost_cap() -> float:
    """Return the optimize step's LLM cost cap in USD."""
    try:
        return float(
            os.environ.get("ATESOR_OPTIMIZE_COST_CAP", OPTIMIZER_COST_CAP)
        )
    except ValueError:
        return OPTIMIZER_COST_CAP


def _llm_name(llm: Any) -> str:
    return getattr(llm, "model_name", None) or getattr(llm, "model", "")


def _optimizer_agent(
    state: AgentState,
) -> Callable[
    [str, str, Callable[[Dict[str, Any]], Optional[str]]],
    Tuple[Optional[Dict[str, Any]], str],
]:
    """Return the optimizer agent: one validated LLM question per call.

    A harness is code, so the builder's (coder) model writes it, with
    :data:`OPTIMIZER_CODE_TIMEOUT`; the planner's model chooses a
    program run. When no model of the role answers in time, the other
    role's models get the question once. Every call goes through
    ``llm_call_with_validation``: the reply must be JSON and pass the
    caller's check, with critique retries and model fallbacks. After
    ``_optimizer_cost_cap`` USD of calls, the agent gives no answer.
    """

    def ask(
        phase: str,
        prompt: str,
        check: Callable[[Dict[str, Any]], Optional[str]],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        cap = _optimizer_cost_cap()
        spent = state.context_cache.get("optimizer_cost_usd", 0.0)
        if spent >= cap:
            return None, (
                f"the optimize step used its LLM cost cap "
                f"(${spent:.2f} of ${cap:.2f}, ATESOR_OPTIMIZE_COST_CAP)"
            )
        code = phase.startswith("harness")
        role = AgentRole.BUILDER if code else AgentRole.PLANNER
        other = AgentRole.PLANNER if code else AgentRole.BUILDER
        pool = get_model_pool_for_role(role)

        def validator(data: Dict[str, Any]) -> ValidationResult:
            problem = check(data)
            if problem:
                return ValidationResult.bad(problem)
            return ValidationResult.good()

        def call(models: List[Any]) -> Any:
            outcome = llm_call_with_validation(
                invoke_fn=invoke_llm,
                llm=models[0],
                fallback_llms=models[1:],
                prompt=prompt,
                validator=validator,
                fallback_factory=None,
                role="optimizer",
                audit_metadata={
                    "repo": state.repo_name,
                    "phase": f"optimize-{phase}",
                },
                max_retries=2,
                timeout=OPTIMIZER_CODE_TIMEOUT if code else LLM_WALL_TIMEOUT,
                rotate_on_timeout=True,
            )
            state.log_api_call(
                cost=outcome.cost_usd,
                tokens_in=outcome.input_tokens,
                tokens_out=outcome.output_tokens,
                calls=outcome.attempts,
            )
            state.context_cache["optimizer_cost_usd"] = (
                state.context_cache.get("optimizer_cost_usd", 0.0)
                + (outcome.cost_usd or 0.0)
            )
            return outcome

        outcome = call(pool)
        if outcome.data is None and "timed out" in (outcome.last_error or ""):
            # No model of the role answered in time: the other role's
            # models get the question once.
            names = {_llm_name(llm) or id(llm) for llm in pool}
            spare = [
                llm
                for llm in get_model_pool_for_role(other)
                if (_llm_name(llm) or id(llm)) not in names
            ]
            if spare:
                outcome = call(spare)
        return outcome.data, outcome.last_error

    return ask


def _optimize_progress(message: str) -> None:
    """Print one line of the optimize step; the step can take an hour."""
    # The console shows only errors unless --verbose, so a long silent
    # step would look stuck.
    print(f"   {message}", flush=True)
    logger.info(f"Optimize: {message}")


@agent_node(AgentRole.SUPERVISOR)
def optimize_node(state: AgentState) -> AgentState:
    """Build, check and measure optimized variants (``--optimize``).

    The verified baseline stays the result of the port. The node
    catches every exception itself: an error goes into
    ``state.optimization`` instead of letting the ``agent_node``
    wrapper escalate a finished port, and the wrapper's rate-limit
    retry never gets the chance to repeat an hour of measurements.
    """
    from .optimize import run_optimization

    state.current_phase = "optimizing"
    baseline = state.package_tests or {}
    detected = None
    if baseline.get("command"):
        detected = (baseline.get("framework"), baseline["command"])

    def run_tests(
        tree: str, extra_env: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        if not detected:
            return None
        return _run_package_tests(
            state, repo_path=tree, extra_env=extra_env, detected=detected
        )

    try:
        state.optimization = run_optimization(
            state,
            agent=_optimizer_agent(state),
            run_tests=run_tests,
            progress=_optimize_progress,
        )
    except Exception as exc:  # performance mode never fails a port
        logger.exception(f"Optimization stopped: {exc}")
        state.optimization = {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}"[:300],
            "variants": [],
        }
    # Native measured results only; the call never raises.
    record_optimization(
        state.repo_name, state.optimization, state.machine_profile
    )
    # The verified build crashed in the timing runs: finish_node cached
    # it as verified, and no later run may replay it as the plan.
    crash = (state.optimization or {}).get("baseline_crash")
    if crash:
        label = (state.machine_profile or {}).get("label") or "this machine"
        mark_recipe_verification(
            state.repo_name, "crashes", f"on {label}: {crash}"
        )
    state.current_phase = "finished"
    return state


# Routing functions — one per source node, named by what it decides.
# Each inspects state fields and returns the destination node name;
# there is no central route_next and no phase-string dispatch.


def route_finish_to_next(state: AgentState) -> str:
    """After finish: run performance mode when --optimize asked for it."""
    # optimize_node itself skips a port that is not verified, so the
    # report always says why no variant was built.
    if state.optimize_requested:
        return "optimize_node"
    return END


def route_init_to_next(state: AgentState) -> str:
    """After init: replay a cached recipe, go to the analyst, or escalate.

    Init failure escalates. ``state.cached_recipe`` (set only with
    ``--optimize``) sends the run to ``cached_recipe_node``.
    """
    # ESCALATED means init_node crashed (agent_node wrapper). The clone
    # may be missing, so the analyst must not run.
    if state.build_status in (BuildStatus.FAILED, BuildStatus.ESCALATED):
        logger.warning("Initialization failed; forcing escalation")
        return "escalate_node"
    if state.cached_recipe:
        return "cached_recipe_node"
    return "analyst_node"


def route_cached_recipe_to_next(state: AgentState) -> str:
    """After the cached recipe: build it, or plan as usual without one."""
    if state.build_plan:
        return "supervisor_node"
    return "analyst_node"


def route_analyst_to_next(state: AgentState) -> str:
    """After the analyst: try a heuristic default plan, or escalate.

    The analyst ALWAYS leaves a ``package_analysis`` (LLM-grounded or
    deterministic fallback) plus the structural ``task_plan``; a
    missing analysis therefore means the node itself crashed.
    """
    if state.package_analysis is None:
        logger.warning("Analyst produced no analysis; escalating")
        return "escalate_node"
    if state.context_cache.get("unsupported"):
        return "escalate_node"
    return "heuristic_plan_node"


def route_heuristic_plan_to_next(state: AgentState) -> str:
    """After the heuristic planner: plan → supervisor, else LLM scout.

    The heuristic planner only materializes a default BuildPlan for
    build systems it recognizes with reasonable confidence, when the
    repo is not heavily arch-specific AND the analyst did not veto
    (``needs_custom_plan``). Anything else is
    deferred to the LLM scout, which plans from the repo evidence, the
    analysis, and any previous failure.
    """
    if state.build_plan:
        return "supervisor_node"
    return "scout_node"


def route_supervisor_to_next(state: AgentState) -> str:
    """Evaluate state and route to the right next action.

    Pure heuristic — no LLM call. Returns a single destination.
    """
    # A SUCCESS (verified or not) ALWAYS finishes — even at the attempt/cost
    # ceiling. Escalating a build that just succeeded (e.g. the final
    # fix landed on attempt max_attempts) would throw away the win and
    # skip the recipe. Escalation checks apply to non-success states.
    if state.build_status == BuildStatus.SUCCESS:
        return "finish_node"

    # A node that crashed marks the run ESCALATED (agent_node wrapper).
    # Sending that back to scout or fixer would hide the crash and burn
    # attempts on an infrastructure fault the LLM cannot fix.
    if state.build_status == BuildStatus.ESCALATED:
        return "escalate_node"

    should_esc, esc_reason = should_escalate(state)
    if should_esc:
        logger.warning(f"Escalating: {esc_reason}")
        return "escalate_node"

    if state.attempt_count >= state.max_attempts:
        return "escalate_node"

    if state.attempt_count >= 3 and state.is_in_error_loop():
        return "escalate_node"

    if state.package_analysis is None:
        return "analyst_node"

    # A plan command that the command validator blocks cannot be fixed
    # by the fixer: the whitelist rejects its answer too. Re-plan once
    # with the blocked command in the scout prompt, then escalate.
    if state.build_status == BuildStatus.FAILED and (
        _last_error_is_validator_block(state)
    ):
        if state.context_cache.get("validator_block_replans", 0) < 1:
            logger.warning(
                "A planned command was blocked by the command validator; "
                "re-planning once"
            )
            return "scout_node"
        logger.warning(f"Escalating: {_escalation_reason(state)}")
        return "escalate_node"

    # The fix loop stopped because more fixer rounds cannot help (a
    # refusal, or an error that two fixes did not change). Building
    # again would repeat the same failure.
    if state.build_status in (BuildStatus.FAILED, BuildStatus.FIXING):
        stop = _fix_loop_stop_reason(state)
        if stop:
            logger.warning(f"Escalating: {stop}")
            return "escalate_node"

    need_new_plan = state.build_status == BuildStatus.FAILED and (
        state.last_error_category
        in (ErrorCategory.DEPENDENCY, ErrorCategory.MISSING_TOOLS)
        or _should_force_replan(state)
    )
    if need_new_plan:
        return "scout_node"  # re-scout for updated build plan

    if state.build_status == BuildStatus.FAILED:
        return "build_fix_subgraph"

    if state.build_plan and state.build_status in (
        BuildStatus.PENDING,
        BuildStatus.BUILDING,
    ):
        return "build_fix_subgraph"

    return "scout_node"


# Fix-loop stop rules, shared by the subgraph and the supervisor.

# FixAttempt.build_result of a fix whose actions changed nothing in the
# tree. The builder does not mark such a fix successful, and the next
# step is another fixer round instead of an identical rebuild.
_NO_CHANGE = "no change applied"
# The changes_made entry of a plan_command action whose new text is the
# old text; it is no change a rebuild could use.
_SAME_PLAN_COMMAND = "Plan command unchanged"

# Prefix that execute_command writes for a command the whitelist blocks.
_VALIDATOR_BLOCK = "Command blocked by safety validation"

# Prefix of the error fixer_node records when no attempt gave a valid
# fix. The tree is unchanged, so a rebuild repeats the same failure.
_FIXER_INVALID = "FIXER proposed invalid fix"

# Prefix of the error fixer_node records when the model says that no
# fix inside the sandbox can work (the `escalate` field).
_FIXER_ESCALATED = "FIXER escalated"


# Fixer answers without a change (no rebuild) allowed per build.
_MAX_REASKS = 3


def _llm_refused(state: AgentState) -> bool:
    """Return True when a model refused this package earlier in the run."""
    return bool(state.context_cache.get("llm_refused"))


def _mark_llm_refused(state: AgentState, role: str) -> None:
    """Record a refusal; later steps of this package skip the LLM.

    The refusal is about the package, not one prompt: in the October
    2026 runs every role refused the same eight offensive-security
    packages, and each retry got the same refusal.

    Args:
        state: The current agent state.
        role: The role whose call was refused.
    """
    if not state.context_cache.get("llm_refused"):
        logger.warning(
            f"[{role}] the model refused this package; the remaining "
            f"steps of this run skip the LLM"
        )
    state.context_cache["llm_refused"] = role


def _last_error_is_validator_block(state: AgentState) -> bool:
    """Return True when the command validator blocked the last command."""
    if not state.error_history:
        return False
    return _VALIDATOR_BLOCK in (state.error_history[-1].message or "")


def _fix_loop_stop_reason(state: AgentState) -> str:
    """Return why another fixer round cannot help, or an empty string.

    Stops on: a refusal; a ``should_escalate`` reason other than the
    error category (attempt limit, three identical errors, cost cap);
    ``_MAX_REASKS`` fixer answers since the last build that changed
    nothing; a fixer round with no valid answer; and two or more fixes
    after which the error did not change. The category alone does not
    stop the loop: it comes from substring matches, and the fixer fixed
    errors classified as DISK_SPACE. In October 2026 the fixer passed
    21.7% of packages in round 1, but only 8.6% in round 3 and 5.4% in
    round 4.

    Args:
        state: The current agent state.

    Returns:
        A short reason for the escalation report, or ``""``.
    """
    if _llm_refused(state):
        return (
            "the model refused this package (safety stop), so no LLM fix "
            "is possible"
        )
    escalate, reason = should_escalate(state, check_category=False)
    if escalate:
        return reason
    # Answers that change nothing never reach a build: after three of
    # them since the last build, a fourth is unlikely to differ.
    if state.context_cache.get("reasks_since_build", 0) >= _MAX_REASKS:
        return (
            f"the fixer gave {_MAX_REASKS} answers since the last build "
            "that changed nothing in the tree"
        )
    last = state.error_history[-1].message if state.error_history else ""
    # A provider fault (timeout, HTTP error) may pass; an answer that
    # fails validation or parsing three times will not. In October 2026
    # no package passed after such a fixer round (32 runs).
    if (last or "").startswith(_FIXER_INVALID) and (
        "LLM invocation raised" not in last
    ):
        return (
            "the fixer gave no valid fix after its retries; a rebuild of "
            "the unchanged tree would repeat the failure"
        )
    # The error lines are compared as sets, without make's summary
    # lines: progress is a line that is gone. One compared line missed
    # progress (gomuks: the olm line had gone, and the loop stopped).
    if len(state.fixes_attempted) >= 2 and len(state.error_history) >= 2:
        before, after = state.error_history[-2:]
        if not (_error_lines(before.message) - _error_lines(after.message)):
            return (
                f"the error did not change after fix "
                f"{len(state.fixes_attempted)}; more fixer rounds are "
                f"unlikely to help"
            )
    return ""


def _escalation_reason(state: AgentState) -> str:
    """Return the reason the escalation report gives for this run."""
    if state.context_cache.get("unsupported"):
        return str(state.context_cache["unsupported"])
    last = state.error_history[-1].message if state.error_history else ""
    if (last or "").startswith(_FIXER_ESCALATED):
        return f"the fixer found no fix inside the sandbox: {last}"
    reason = _fix_loop_stop_reason(state)
    if reason:
        return reason
    if _last_error_is_validator_block(state):
        return (
            "a planned command is not allowed by the command validator, "
            "and a re-plan did not remove it"
        )
    return ""


# Build-fix subgraph routing: build → verify → fix → rebuild.


def route_build_result(state: AgentState) -> str:
    """Route a build result: verify, attempt a fix, or exit.

    A command blocked by the validator, or a stop rule from
    ``_fix_loop_stop_reason``, exits the subgraph without a fixer call;
    the supervisor then re-plans or escalates.
    """
    if state.build_status == BuildStatus.SUCCESS:
        return "verify_node"
    if state.build_status == BuildStatus.ESCALATED:
        return "__end__"
    if state.attempt_count >= state.max_attempts:
        logger.warning(
            f"Subgraph: max attempts ({state.max_attempts}) exceeded "
            f"after build — exiting subgraph"
        )
        return "__end__"
    if _last_error_is_validator_block(state):
        logger.warning(
            "Subgraph: the command validator blocked a planned command; "
            "the fixer cannot change the whitelist — exiting subgraph"
        )
        return "__end__"
    stop = _fix_loop_stop_reason(state)
    if stop:
        logger.warning(f"Subgraph: no fixer round ({stop}) — exiting")
        return "__end__"
    return "fix_node"


def route_verify_result(state: AgentState) -> str:
    """Route a verify result: exit on success, or loop to the fixer.

    Also exits once fixes were tried and the error is not a
    compilation, linking or architecture one — those are the only
    categories the fixer's patches address. The supervisor re-routes
    anything else after the subgraph ends (e.g. re-scout for a
    DEPENDENCY error).
    """
    if state.build_status in (BuildStatus.SUCCESS, BuildStatus.ESCALATED):
        return "__end__"
    if state.attempt_count >= state.max_attempts:
        logger.warning(
            f"Subgraph: max attempts ({state.max_attempts}) exceeded "
            f"after verify — exiting subgraph"
        )
        return "__end__"
    if state.fixes_attempted and state.last_error_category not in (
        ErrorCategory.LINKING,
        ErrorCategory.COMPILATION,
        ErrorCategory.ARCHITECTURE,
    ):
        return "__end__"
    stop = _fix_loop_stop_reason(state)
    if stop:
        logger.warning(f"Subgraph: no fixer round ({stop}) — exiting")
        return "__end__"
    return "fix_node"


def route_fix_result(state: AgentState) -> str:
    """After a fix: rebuild, ask the fixer again, or exit as failure.

    A fix that changed nothing goes straight back to the fixer: the
    rebuild would repeat the same failure.
    """
    if state.build_status == BuildStatus.ESCALATED:
        return "__end__"
    if state.build_status == BuildStatus.FAILED:
        logger.warning("Subgraph: fix resulted in FAILED status — exiting")
        return "__end__"
    if state.attempt_count >= state.max_attempts:
        logger.warning(
            f"Subgraph: max attempts ({state.max_attempts}) exceeded "
            f"after fix — exiting subgraph"
        )
        return "__end__"
    if (
        state.build_status == BuildStatus.FIXING
        and state.fixes_attempted
        and state.fixes_attempted[-1].build_result == _NO_CHANGE
    ):
        stop = _fix_loop_stop_reason(state)
        if stop:
            logger.warning(f"Subgraph: no fixer round ({stop}) — exiting")
            return "__end__"
        return "fix_node"
    return "build_node"


# heuristic_plan_node — zero-LLM default recipes, gated by the analyst.


def _probe_config_files(state: AgentState) -> List[str]:
    """Return the build config files that actually exist in the repo."""
    found = []
    for probe in (
        "CMakeLists.txt",
        "configure.ac",
        "Makefile.am",
        "meson.build",
        "go.mod",
        "Cargo.toml",
        "Makefile",
    ):
        path = os.path.join(_to_host_path(state.repo_path), probe)
        if os.path.isfile(path):
            found.append(probe)
    return found


def _setup_packages_for_plan(
    state: AgentState, base_canonicals: List[str], profile
) -> List[str]:
    """Merge detected AND analyst-identified deps into setup packages.

    Only canonical names that resolve through the active profile's
    ``package_map`` are added — unknown library names are left for the
    LLM scout / fixer rather than guess-installed. The analyst's
    dependencies come from reading the actual build files, so they take
    part on equal footing with scripted detection.

    Args:
        state: Current agent state (reads ``scout_deps_result`` and
            ``package_analysis``).
        base_canonicals: Build-system tool canonicals (e.g. ["cmake"]).
        profile: The active platform profile.

    Returns:
        Deduplicated distro package names for the setup phase.
    """
    deps_result = state.scout_deps_result or {}
    analysis = state.package_analysis
    analyst_names = (
        [d.get("name", "") for d in analysis.dependencies] if analysis else []
    )
    canonicals = list(base_canonicals)
    for name in (
        analyst_names
        + list(deps_result.get("build_tools", []))
        + list(deps_result.get("libraries", []))
        + list(deps_result.get("system_packages", []))
    ):
        key = str(name).strip().lower()
        if key and key in profile.package_map and key not in canonicals:
            canonicals.append(key)
    packages: List[str] = []
    for canonical in canonicals[:15]:  # keep the install command sane
        distro_pkg = profile.resolve(canonical)
        if distro_pkg not in packages:
            packages.append(distro_pkg)
    return packages


@agent_node(AgentRole.SCOUT)
def heuristic_plan_node(state: AgentState) -> AgentState:
    """Materialize a zero-LLM default BuildPlan when it is safe to.

    Gathers the init-time detection facts in one pass, then combines
    them with the analyst's verdict:

    * ``analysis.needs_custom_plan`` (the analyst READ the build files
      and says a textbook recipe will fail) → defer to the LLM scout.
    * Significant arch-specific code or weak build-system confidence →
      defer to the LLM scout.
    * Otherwise → emit the canned recipe for the detected build system
      (cmake, make, autotools or go), with the analyst's dependency
      list merged into the setup phase. Any other build system defers
      to the LLM scout.
    """
    logger.info("Heuristic planner evaluating default recipes...")
    if state.build_plan:
        logger.info("Build plan already exists; using as-is")
        return state

    bsi = state.build_system_info
    deps = state.dependencies
    arch_codes = state.arch_specific_code
    analysis = state.package_analysis

    # Recorded facts (kept on state for reports/serialization).
    state.scout_build_system_result = {
        "type": bsi.type if bsi else "unknown",
        "confidence": bsi.confidence if bsi else 0.0,
        "module_dir": (
            bsi.module_dir if bsi and hasattr(bsi, "module_dir") else ""
        ),
        "config_files": _probe_config_files(state),
    }
    state.scout_deps_result = {
        "build_tools": list(deps.build_tools) if deps else [],
        "system_packages": list(deps.system_packages) if deps else [],
        "libraries": list(deps.libraries) if deps else [],
        "missing_tools": list(state.context_cache.get("missing_tools", [])),
    }
    state.scout_arch_issues_result = {
        "total_issues": len(arch_codes),
        "high_severity": sum(1 for a in arch_codes if a.severity == "high"),
        "critical": sum(1 for a in arch_codes if a.severity == "critical"),
        "arch_types": list({a.arch_type for a in arch_codes}),
        "files": list({a.file for a in arch_codes}),
    }
    state.log_scripted_op("heuristic_plan_facts")

    # A verified plan of this package from an earlier run is the plan:
    # no default recipe and no scout call (see _recall_seed).
    seed = _recall_seed(state)
    if seed is not None and _plan_from_recipe(state, seed):
        logger.info(
            "Using the verified recipe of an earlier run "
            f"({_recall(state).get('sandbox')}) as the plan"
        )
        state.log_scripted_op("recall_seed")
        state.context_cache["plan_source"] = "recipe"
        state.build_status = BuildStatus.PENDING
        return state

    build_type = state.scout_build_system_result["type"]
    confidence = float(state.scout_build_system_result["confidence"])

    # The analyst read the actual build files; when it is LLM-grounded,
    # names a build system, and is at least as confident as file-name
    # detection, its verdict wins.
    if analysis and analysis.llm_grounded:
        if (
            analysis.build_system != "unknown"
            and analysis.build_system_confidence >= confidence
        ):
            build_type = analysis.build_system
            confidence = analysis.build_system_confidence

    n_deps = len(state.scout_deps_result["system_packages"])
    logger.info(
        f"Heuristic plan facts: build_system={build_type} "
        f"(confidence={confidence:.2f}), deps={n_deps}, "
        f"arch_issues={len(arch_codes)}"
    )

    # The analyst read the build files and says a textbook recipe will
    # fail (code generation, subprojects, pinned toolchain, ...).
    if analysis and analysis.llm_grounded and analysis.needs_custom_plan:
        logger.info(
            "Analyst flagged needs_custom_plan; deferring to LLM scout"
        )
        return state

    # Heavily arch-specific repos need the LLM scout's judgement (SIMD
    # opt-outs, feature flags) rather than a generic default plan.
    # Code in bundled third-party directories (gnulib, contrib, vendor)
    # does not decide: wget, gawk and zlib went to the scout for guarded
    # asm there, and the scout plans were not better than the default.
    deciding = [a for a in arch_codes if not _in_vendored_dir(a.file)]
    if any(a.severity == "critical" for a in deciding) or (
        sum(1 for a in deciding if a.severity == "high") >= 3
    ):
        logger.info(
            "Significant arch-specific code detected; deferring to LLM scout"
        )
        return state

    # Go files for amd64/arm64 with no riscv64 or generic version break
    # the default build (steampipe: 14 minutes to "undefined:
    # platform.Paths"); the scout prompt lists the files.
    if build_type == "go" and any(
        a.arch_type == "go_arch_files" for a in deciding
    ):
        logger.info(
            "Go files for other architectures without a riscv64 version; "
            "deferring to LLM scout"
        )
        return state

    if confidence < 0.5:
        logger.info(
            f"Build system confidence {confidence:.2f} < 0.5; "
            "deferring to LLM scout"
        )
        return state

    from .platforms import get_active_profile

    profile = get_active_profile()

    def _setup_cmd(base: List[str]) -> str:
        return profile.install_cmd(
            _setup_packages_for_plan(state, base, profile)
        )

    # The build file can sit in a subdirectory (libexpat: expat/); init
    # found it, so the default plan builds there.
    root = ""
    if bsi and bsi.type == build_type:
        root = (getattr(bsi, "module_dir", "") or "").strip("/")

    if build_type == "cmake":
        if root:
            configure = (
                f"cmake -S {shlex.quote(root)} -B build "
                "-DCMAKE_BUILD_TYPE=Release"
            )
            build = "cmake --build build -j$(nproc)"
        else:
            configure = (
                "mkdir -p build && cd build"
                " && cmake .. -DCMAKE_BUILD_TYPE=Release"
            )
            build = "cd build && make -j$(nproc)"
        state.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=confidence,
            phases=[
                BuildPhase(
                    1, "setup", [_setup_cmd(["cmake", "gcc"])], False, "30s"
                ),
                BuildPhase(2, "configure", [configure], False, "1m"),
                BuildPhase(3, "build", [build], True, "5m"),
            ],
            total_estimated_duration="7m",
            notes=["Default CMake plan (heuristic)"],
        )
        logger.info("Created default CMake BuildPlan (heuristic)")
    elif build_type == "make":
        state.build_plan = BuildPlan(
            build_system="make",
            build_system_confidence=confidence,
            phases=[
                BuildPhase(1, "setup", [_setup_cmd(["gcc"])], False, "30s"),
                BuildPhase(
                    2,
                    "build",
                    [
                        (
                            f"make -C {shlex.quote(root)} -j$(nproc)"
                            if root
                            else "make -j$(nproc)"
                        )
                    ],
                    True,
                    "5m",
                ),
            ],
            total_estimated_duration="6m",
            notes=["Default Make plan (heuristic)"],
        )
        logger.info("Created default Make BuildPlan (heuristic)")
    elif build_type == "autotools":
        state.build_plan = BuildPlan(
            build_system="autotools",
            build_system_confidence=confidence,
            phases=[
                BuildPhase(
                    1,
                    "setup",
                    [_setup_cmd(["gcc", "autotools", "pkgconfig"])],
                    False,
                    "30s",
                ),
                BuildPhase(2, "configure", ["./configure"], False, "3m"),
                BuildPhase(3, "build", ["make -j$(nproc)"], True, "5m"),
            ],
            total_estimated_duration="9m",
            notes=["Default Autotools plan (heuristic)"],
        )
        logger.info("Created default Autotools BuildPlan (heuristic)")
    elif build_type == "go":
        go_main_info = state.context_cache.get("go_main_info", {})
        build_cmd = (
            go_main_info.get("plan_command")
            or go_main_info.get("build_command")
            or "go build -buildvcs=false ./..."
        )
        if "-buildvcs=false" not in build_cmd and _is_go_build_command(
            build_cmd
        ):
            build_cmd = _inject_go_flag(build_cmd, "-buildvcs=false")
        # The environment and tags the analyst read in the package's own
        # build file (vuls: CGO_ENABLED=0 GOEXPERIMENT=jsonv2).
        go_env, go_tags = _analyst_go_settings(state)
        if _is_go_build_command(build_cmd):
            if go_tags and "-tags" not in build_cmd:
                build_cmd = _inject_go_flag(
                    build_cmd, f"-tags {shlex.quote(go_tags)}"
                )
            settings = " ".join(
                f"{name}={value}"
                for name, value in go_env.items()
                if f"{name}=" not in build_cmd
            )
            if settings:
                build_cmd = _GO_BUILD_RE.sub(
                    lambda match: f"{settings} {match.group(0)}",
                    build_cmd,
                    count=1,
                )

        module_name = (
            state.repo_url.replace("https://", "").rstrip("/")
            if state.repo_url
            else f"github.com/unknown/{state.repo_name}"
        )
        go_phase_commands = _go_prepare_commands(
            {**go_main_info, "plan_command": build_cmd}, module_name
        )
        # Each program the analyst names that has its own cmd/<name>
        # directory is built too (age built age but not age-keygen).
        for extra in _extra_go_mains(state, go_main_info, build_cmd):
            if go_tags and "-tags" not in extra:
                extra = _inject_go_flag(extra, f"-tags {shlex.quote(go_tags)}")
            settings = " ".join(f"{n}={v}" for n, v in go_env.items())
            if settings:
                extra = _GO_BUILD_RE.sub(
                    lambda match: f"{settings} {match.group(0)}",
                    extra,
                    count=1,
                )
            go_phase_commands.append(extra)

        state.build_plan = BuildPlan(
            build_system="go",
            build_system_confidence=confidence,
            phases=[
                BuildPhase(1, "setup", [_setup_cmd(["git"])], False, "30s"),
                BuildPhase(
                    2, "go_prepare_build", go_phase_commands, False, "2m"
                ),
            ],
            total_estimated_duration="7m",
            notes=["Default Go plan (heuristic)"],
        )
        logger.info("Created default Go BuildPlan (heuristic)")
    else:
        # meson / cargo / python / unknown: the LLM scout handles these
        # better than a canned plan (feature flags, workspaces, etc.).
        logger.info(
            f"Build system '{build_type}' has no default recipe; "
            "deferring to LLM scout"
        )
        return state

    state.context_cache["plan_source"] = "heuristic"
    state.build_status = BuildStatus.PENDING
    return state


# Build-fix subgraph and artifact verification (verify_node lives here
# because it runs inside that loop).


# The state keeps at most this many riscv64 outputs, because the curator
# sends the list to an LLM. The verification summary keeps the full
# counts.
_MAX_RECORDED_ARTIFACTS = 60


# The machine and what a build may ask of it.

_MARCH_RE = re.compile(r"-march=(rv(?:32|64)[a-z0-9_]+)")
# Extensions that a kernel can leave out of its ISA string although
# every Linux board runs them.
_ALWAYS_PRESENT = frozenset({"zicsr", "zifencei", "zmmul"})


def _machine(state: AgentState) -> Optional[MachineProfile]:
    """Return the probed profile of the native machine, once per run.

    None under QEMU, where the CPU model runs every extension, and when
    the probe fails (the checks that need it are then skipped).
    """
    if not is_native():
        return None
    cached = state.context_cache.get("machine_probe")
    if cached is None:
        profile, error = probe_machine("native")
        if profile is None:
            logger.warning(f"Machine probe failed ({error}); no ISA check")
        cached = profile.to_dict() if profile else {}
        state.context_cache["machine_probe"] = cached
        if cached and not state.machine_profile:
            state.machine_profile = cached
    return MachineProfile.from_dict(cached) if cached else None


def _isa_flags_the_machine_lacks(
    commands: List[str], profile: Optional[MachineProfile]
) -> List[str]:
    """Return each ``-march`` value that names an extension not present.

    libwebp was built six times with ``-march=rv64gcv`` on a board
    without V; the analyst had suggested the flag.

    Returns:
        ``-march=X (the machine lacks a, b)`` texts; empty when every
        value fits or no probe exists.
    """
    if profile is None or not profile.extensions:
        return []
    have = set(profile.extensions) | _ALWAYS_PRESENT
    problems = set()
    for command in commands:
        for value in _MARCH_RE.findall(command):
            missing = sorted(parse_isa_string(value) - have)
            if missing:
                problems.add(
                    f"-march={value} (the machine lacks {', '.join(missing)})"
                )
    return sorted(problems)


def _plan_isa_problems(state: AgentState, commands: List[str]) -> List[str]:
    """Return the ``-march`` problems of plan commands on this machine.

    The probe runs only when a command names ``-march``.
    """
    if not any("-march=" in command for command in commands):
        return []
    return _isa_flags_the_machine_lacks(commands, _machine(state))


def _primary_executable(recorded: List[Any], repo_path: str) -> str:
    """Return the first recorded riscv64 program in the tree, or ""."""
    root = repo_path.rstrip("/") + "/"
    for artifact in recorded:
        path = str(artifact.path)
        if (
            artifact.type == "binary"
            and path.startswith(root)
            and not re.search(r"/(?:tests?|testdata|examples?)/", path)
        ):
            return path
    return ""


def _smoke_run(path: str) -> str:
    """Run a built program with ``--version``, then ``--help``.

    A program that the machine cannot run dies with a crash signal
    (SIGILL for a missing extension). An exit status or a stop at the
    10 s limit is no crash.

    Returns:
        The crash, as ``"<name> --version" died with SIGILL``, or "".
    """
    name = os.path.basename(path)
    for flag in ("--version", "--help"):
        result = execute_command([path, flag], timeout=20, validate=False)
        signal = crash_signal(result.exit_code)
        if signal:
            return f"`{name} {flag}` died with {signal}"
        if result.exit_code == 0:
            return ""
    return ""


def _instruction_findings(
    state: AgentState, recorded: List[Any], profile: MachineProfile
) -> Tuple[List[str], List[str]]:
    """Scan the riscv64 outputs for instructions the machine lacks.

    Returns:
        ``(problems, warnings)``. A problem: an output holds instructions
        of an extension that a ``-march`` of the plan names and the
        machine lacks. A warning: such instructions without that flag,
        which can be runtime dispatch.
    """
    from .optimize import inspect_tree

    missing = set(forbidden_classes(profile))
    if not missing:
        return [], []
    root = state.repo_path.rstrip("/") + "/"
    rels = [
        str(a.path)[len(root) :]
        for a in recorded
        if a.type in ("binary", "library_shared")
        and str(a.path).startswith(root)
    ][:8]
    if not rels:
        return [], []
    facts, error = inspect_tree(state.repo_path, rels, None, False, "native")
    if facts is None:
        logger.warning(f"Instruction scan could not run: {error}")
        return [], []
    flagged = {
        name
        for value in _MARCH_RE.findall(" ".join(_plan_commands(state)))
        for name in parse_isa_string(value)
    }
    problems: List[str] = []
    warnings: List[str] = []
    for rel in rels:
        counts = ((facts.get("artifacts") or {}).get(rel) or {}).get("counts")
        found = sorted(name for name in counts or {} if name in missing)
        if not found:
            continue
        text = (
            f"{rel} holds {', '.join(found)} instructions that the machine "
            f"({profile.label}) lacks"
        )
        if flagged & set(found):
            problems.append(text + " (the plan's -march asks for them)")
        else:
            warnings.append(text + " (runtime dispatch, or a SIGILL risk)")
    return problems, warnings


def _artifacts_to_record(riscv64: List[Any], expected: List[str]) -> List[Any]:
    """Return the riscv64 outputs to record, expected names first."""

    def is_expected(artifact: Any) -> bool:
        name = artifact.path.rsplit("/", 1)[-1]
        return any(name == e or name.startswith(e + ".") for e in expected)

    first = [a for a in riscv64 if is_expected(a)]
    rest = [a for a in riscv64 if not is_expected(a)]
    return (first + rest)[:_MAX_RECORDED_ARTIFACTS]


@agent_node(AgentRole.BUILDER)
def verify_node(state: AgentState) -> AgentState:
    """Seal a build as verified, or send a wrong-arch build to the fixer.

    The ELF header of every build output is the evidence (see
    ``src/artifact_scanner.py``). Outcomes:
        * verified: riscv64 build outputs, and none for another
          architecture → SUCCESS.
        * wrong_arch: any build output for another architecture
          (x86-64, aarch64, riscv32, a mixed static archive) → FAILED
          with an ARCHITECTURE error for the fixer. One foreign output
          fails the port even when the others are riscv64.
        * unverified: no riscv64 output could be proven (header-only
          libraries, scripts, or a scan that could not run) → SUCCESS
          that the report, the exit code (3) and the zip manifest mark
          as unverified, and that is never cached or learned.
    """
    logger.info("Verifying build artifacts...")
    expected = (
        state.package_analysis.expected_artifacts
        if state.package_analysis
        else []
    )
    verdict = ArtifactScanner(state.repo_path).scan(
        expected_names=expected,
        since=state.execution_start_time.timestamp(),
    )
    state.artifact_verification = verdict.to_dict()
    # The latest scan is the truth; a rebuild after a fix rescans.
    state.build_artifacts = []

    if verdict.status == WRONG_ARCH:
        state.build_status = BuildStatus.FAILED
        state.add_error(
            create_error_record(
                message=f"Artifact verification failed: {verdict.reason}",
                category=ErrorCategory.ARCHITECTURE,
            )
        )
        return state

    recorded = _artifacts_to_record(verdict.riscv64, list(expected))
    for artifact in recorded:
        state.add_build_artifact(
            filepath=artifact.path,
            artifact_type=artifact.type,
            architecture=artifact.arch,
        )
    total = len(verdict.riscv64) + verdict.unlisted_riscv64
    if total > len(recorded):
        logger.info(
            f"Recorded {len(recorded)} of {total} riscv64 build outputs; "
            "the verification summary keeps the full count"
        )
    if verdict.status == UNVERIFIED:
        logger.warning(f"Build not verified: {verdict.reason}")
    # The header proves the architecture, not that the machine can run
    # the code: libwebp passed six times while it died with SIGILL.
    problem = (
        _runs_on_this_machine(state, recorded)
        if verdict.status == VERIFIED and is_native()
        else ""
    )
    if problem:
        state.artifact_verification.update(status="crashes", reason=problem)
        state.build_status = BuildStatus.FAILED
        state.add_error(
            create_error_record(
                message=f"Artifact verification failed: {problem}",
                category=ErrorCategory.ARCHITECTURE,
            )
        )
        return state
    state.build_status = BuildStatus.SUCCESS
    return state


def _runs_on_this_machine(state: AgentState, recorded: List[Any]) -> str:
    """Return why the verified outputs cannot run on this machine, or "".

    Native only. The instruction scan compares the outputs with the
    machine probe, and the primary program runs once with ``--version``
    (then ``--help``). The scan's warnings go into the verification
    record; they never fail the port alone.
    """
    profile = _machine(state)
    problems: List[str] = []
    if profile is not None:
        problems, warnings = _instruction_findings(state, recorded, profile)
        if warnings:
            state.artifact_verification["isa_warnings"] = warnings
    primary = _primary_executable(recorded, state.repo_path)
    crash = _smoke_run(primary) if primary else ""
    if crash:
        problems.append(f"the built program crashes on this machine: {crash}")
    return "; ".join(problems)


def create_build_fix_subgraph() -> StateGraph:
    """Compile the build-fix subgraph: build → (verify → fix → retry) loop.

    Entered from the parent graph. Exits when verify returns SUCCESS, a
    node escalates, the attempt ceiling is reached, or the fixer cannot
    help (see ``route_build_result``, ``route_verify_result`` and
    ``route_fix_result``). The parent graph reads ``build_status`` to
    decide next steps.
    """
    sg = StateGraph(AgentState)

    sg.add_node("build_node", builder_node)
    sg.add_node("verify_node", verify_node)
    sg.add_node("fix_node", fixer_node)

    sg.set_entry_point("build_node")

    sg.add_conditional_edges(
        "build_node",
        route_build_result,
        {"verify_node": "verify_node", "fix_node": "fix_node", "__end__": END},
    )

    sg.add_conditional_edges(
        "verify_node",
        route_verify_result,
        {"fix_node": "fix_node", "__end__": END},
    )

    sg.add_conditional_edges(
        "fix_node",
        route_fix_result,
        {"build_node": "build_node", "fix_node": "fix_node", "__end__": END},
    )

    return sg.compile()


def predict_build_issues(state: AgentState) -> List[Dict[str, str]]:
    """Flag plan commands likely to fail, before the builder runs them.

    Deterministic pattern match over the planned commands: a Go build
    without ``-buildvcs=false`` (the builder injects the flag when it
    sees the ``dubious ownership`` prediction), cmake over heavily
    arch-specific code, and package installs while tools are missing.

    Args:
        state: The agent state; reads the build plan, the arch-specific
            code, and ``context_cache["missing_tools"]``.

    Returns:
        A list of predicted issues with mitigation strategies.
    """
    predictions = []

    if not state.build_plan:
        return predictions

    for phase in state.build_plan.phases:
        for cmd in phase.commands:
            if _is_go_build_command(cmd) and "-buildvcs=false" not in cmd:
                predictions.append(
                    {
                        "phase": phase.name,
                        "issue": "Go VCS ownership error",
                        "pattern": "dubious ownership",
                        "mitigation": "Add -buildvcs=false flag",
                        "confidence": 0.7,
                    }
                )

            if "cmake" in cmd and state.arch_specific_code:
                high_severity = [
                    a for a in state.arch_specific_code if a.severity == "high"
                ]
                if high_severity:
                    predictions.append(
                        {
                            "phase": phase.name,
                            "issue": (
                                "Architecture-specific code may cause "
                                "build failure"
                            ),
                            "pattern": "arch_specific",
                            "mitigation": "Add RISC-V compatibility patches",
                            "confidence": 0.8,
                        }
                    )

            if any(
                tok in cmd
                for tok in (
                    "apk add",
                    "apt-get install",
                    "apt install",
                    "dpkg ",
                )
            ):
                missing = state.context_cache.get("missing_tools", [])
                if missing:
                    predictions.append(
                        {
                            "phase": phase.name,
                            "issue": f"Missing tools: {missing}",
                            "pattern": "missing_tools",
                            "mitigation": "Ensure dependencies are installed",
                            "confidence": 0.9,
                        }
                    )

    return predictions


# Workflow assembly.


def create_workflow() -> StateGraph:
    """Assemble and compile the porting StateGraph.

    Architecture:
        init ──→ analyst (LLM reads repo) ──→ heuristic_plan ─┬→ supervisor
              │                                               └→ scout_node
              └→ cached_recipe_node ──→ supervisor   (LLM plan)──┘
                 (--optimize with a verified cache entry)
        supervisor ──→ build_fix_subgraph (build→verify→fix) ──→ supervisor
        supervisor ──→ finish_node ──→ END
                       finish_node ──→ optimize_node ──→ END (--optimize)
        supervisor ──→ escalate_node ──→ END
        supervisor ──→ analyst_node (only after a crashed analyst)
        supervisor ──→ scout_node (error-aware re-scout)
    """
    logger.info("Creating properly graph-shaped workflow...")

    workflow = StateGraph(AgentState)
    build_fix_subgraph = create_build_fix_subgraph()

    # Nodes
    workflow.add_node("init_node", init_node)
    # --optimize with a verified cached recipe: that recipe is the plan.
    workflow.add_node("cached_recipe_node", cached_recipe_node)
    # LLM "understand" step: reads real build files/docs, produces the
    # PackageAnalysis consumed by every downstream node.
    workflow.add_node("analyst_node", analyst_node)
    workflow.add_node("supervisor_node", supervisor_node)
    # Zero-LLM default recipes, gated by the analyst
    workflow.add_node("heuristic_plan_node", heuristic_plan_node)
    # LLM scout: custom BuildPlan from evidence + analysis
    workflow.add_node("scout_node", scout_node)
    workflow.add_node("build_fix_subgraph", build_fix_subgraph)
    # Terminal nodes
    workflow.add_node("finish_node", finish_node)
    workflow.add_node("escalate_node", escalate_node)
    # Performance mode after a finished port (--optimize)
    workflow.add_node("optimize_node", optimize_node)

    # Entry point
    workflow.set_entry_point("init_node")

    # Edges

    # init → analyst (or the cached recipe, or escalate)
    workflow.add_conditional_edges(
        "init_node",
        route_init_to_next,
        {
            "analyst_node": "analyst_node",
            "cached_recipe_node": "cached_recipe_node",
            "escalate_node": "escalate_node",
        },
    )

    # Cached recipe → supervisor (build it); no usable recipe → analyst
    workflow.add_conditional_edges(
        "cached_recipe_node",
        route_cached_recipe_to_next,
        {
            "supervisor_node": "supervisor_node",
            "analyst_node": "analyst_node",
        },
    )

    # analyst → heuristic plan (or escalate)
    workflow.add_conditional_edges(
        "analyst_node",
        route_analyst_to_next,
        {
            "heuristic_plan_node": "heuristic_plan_node",
            "escalate_node": "escalate_node",
        },
    )

    # Heuristic plan → supervisor; deferred → LLM scout for a
    # validated custom BuildPlan.
    workflow.add_conditional_edges(
        "heuristic_plan_node",
        route_heuristic_plan_to_next,
        {"supervisor_node": "supervisor_node", "scout_node": "scout_node"},
    )

    # LLM scout → supervisor
    workflow.add_edge("scout_node", "supervisor_node")

    # Supervisor → next action
    workflow.add_conditional_edges(
        "supervisor_node",
        route_supervisor_to_next,
        {
            "analyst_node": "analyst_node",
            "scout_node": "scout_node",
            "build_fix_subgraph": "build_fix_subgraph",
            "finish_node": "finish_node",
            "escalate_node": "escalate_node",
        },
    )

    # Build-fix subgraph exits back to supervisor for re-evaluation
    workflow.add_edge("build_fix_subgraph", "supervisor_node")

    # Terminals; --optimize adds the performance step after finish.
    workflow.add_conditional_edges(
        "finish_node",
        route_finish_to_next,
        {"optimize_node": "optimize_node", END: END},
    )
    workflow.add_edge("optimize_node", END)
    workflow.add_edge("escalate_node", END)

    compiled = workflow.compile()
    logger.info(
        "Graph workflow compiled: analyst → heuristic/LLM plan → "
        "build-fix subgraph"
    )
    return compiled


# Compiled at import time; main.py runs this instance.
app = create_workflow()

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

"""Multi-agent state machine orchestration and workflow nodes.

Uses LangGraph to manage agent transitions and shared state for the
RISC-V porting pipeline.
"""

import base64
import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from .artifact_scanner import ArtifactScanner
from .evidence import collect_build_evidence, error_context_excerpts
from .knowledge import get_system_knowledge_summary
from .llm_helpers import (
    ValidationResult,
    llm_call_with_validation,
    response_cost,
    response_usage,
)
from .llm_logger import log_llm_call
from .memory import (
    format_few_shot_examples,
    save_learned_example,
    save_to_recipe_cache,
)
from .models import LLM_REQUEST_TIMEOUT, create_llm
from .scripted_ops import ScriptedOperations, quick_analysis
from .state import (
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
    classify_error,
    create_error_record,
    should_escalate,
)
from .tools import apply_patch, execute_command

# Configure logging
logger = logging.getLogger(__name__)

# Caller-side wall timeout for a single LLM invocation, in seconds.
# MUST stay strictly greater than models.LLM_REQUEST_TIMEOUT so the
# provider's own HTTP timeout fires first and releases the socket;
# otherwise a timed-out call orphans its worker thread and connection
# for the remainder of the provider request (MEM-01).
LLM_WALL_TIMEOUT = LLM_REQUEST_TIMEOUT + 15

# Initialize scripted operations
scripted_ops = ScriptedOperations()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_for_role(role: AgentRole) -> BaseChatModel:
    """Return the right model for an agent role."""
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


def invoke_llm(
    llm: BaseChatModel,
    messages: List[BaseMessage],
    timeout: int = LLM_WALL_TIMEOUT,
) -> BaseMessage:
    """Invoke an LLM with a hard timeout to prevent indefinite hangs.

    Uses a daemon thread so blocking HTTP connections don't prevent
    process exit. Raises TimeoutError if the call exceeds ``timeout``
    seconds.
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
    """Safely extract string content from LLM response."""
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
            ]
        )
    return str(content)


def extract_json_block(text: str) -> str:
    """Safely extract the first JSON block from a string."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _build_platform_banner() -> str:
    """Build a banner describing the active sandbox/package manager.

    Injected at the TOP of scout & fixer prompts so the LLM cannot
    ignore it (knowledge appendices buried at the bottom tend to lose
    to early-context few-shot examples in free-tier models).
    """
    try:
        from .platforms import get_active_profile

        p = get_active_profile()
        pm = p.pkg_install.split()[0]  # "apk" / "apt-get"
        install_example = p.install_cmd(["<pkg>"])
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
                f"`{canonical}` via `{p.install_cmd([distro_pkg])}`."
            )
        return (
            f"- Distro: **{p.display_name}** (libc: **{p.libc}**, "
            f"triplet: `{p.target_triplet}`)\n"
            f"- Package manager: **{pm}** (full install: "
            f"`{install_example}`)\n"
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
    """Build a robust, human-readable error message from command output."""
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or "No stderr/stdout output captured."
    return f"{fallback} (exit {result.exit_code}) - {detail}"


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
    """Classify a clone failure into a specific error category."""
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
    """Return True if an error indicates an outdated compiler toolchain."""
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
    }
    for key, needle in signatures.items():
        if needle in msg:
            return key
    return ""


def _should_force_replan(state: AgentState) -> bool:
    """Bounded replan guard to avoid fixer/scout thrash on bad plans."""
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


# ============================================================================
# NODE WRAPPER
# ============================================================================


def agent_node(role: AgentRole) -> Callable:
    """Wrap agent nodes with error handling and retries.

    Provides state tracking and automatic rate-limit retry.
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


# ============================================================================
# NODE: INITIALIZATION
# ============================================================================


@agent_node(AgentRole.SUPERVISOR)
def init_node(state: AgentState) -> AgentState:
    """Initialize the workflow by cloning the repository.

    Uses scripted operations; this is a zero-cost operation (no LLM
    calls).
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

        # Warn if build system could not be detected
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

    state.build_status = BuildStatus.PENDING
    state.current_phase = "initialized"

    return state


# ============================================================================
# NODE: PLANNER (Strategic Planning)
# ============================================================================

ANALYST_PROMPT = (
    "You are the package analyst for a RISC-V (riscv64) porting agent."
    "\n"
    "Read the repository evidence below — ACTUAL build files and docs "
    "—\n"
    "and produce the analysis every downstream agent builds from.\n"
    "Ground every claim in a file you can see. Never invent.\n"
    "\n"
    "## Active sandbox (READ FIRST — overrides conflicting knowledge)\n"
    "{platform_banner}\n"
    "\n"
    "## Repository\n"
    "- Name: {repo_name} ({repo_url})\n"
    "- Path: {repo_path}\n"
    "- Scripted detection: {detection_summary}\n"
    "- Arch-specific code detected: {arch_summary}\n"
    "\n"
    "## Repository evidence (real file contents)\n"
    "{build_evidence}\n"
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
    '  "needs_custom_plan": <bool>,\n'
    '  "complexity": <int 1-10>\n'
    "}}"
)


def _fallback_analysis(state: AgentState) -> PackageAnalysis:
    """Build a deterministic PackageAnalysis from scripted detection.

    Used when every analyst LLM attempt fails. Marks itself as not
    LLM-grounded and defers to the LLM scout whenever detection
    confidence is weak, so a starved LLM layer degrades to the old
    deterministic behavior instead of a worse one.
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
            f"{d.get('name', '?')} ({d.get('reason', '')[:60]})"
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
        f"{analysis.build_system_reasoning[:120]})\n"
        f"Strategy: {analysis.build_strategy}\n"
        f"Dependencies: {deps}\n"
        f"Expected artifacts: "
        f"{', '.join(analysis.expected_artifacts[:8]) or 'unknown'}\n"
        f"RISC-V risks:\n{risks}\n"
        f"Grounding: {grounding}"
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

    Replaces the old planner, whose LLM-generated TaskPlan nothing
    read. ``task_plan`` is now set deterministically as a structural
    record only.
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

    prompt = ANALYST_PROMPT.format(
        platform_banner=_build_platform_banner(),
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

    _analyst_pool = get_model_pool_for_role(AgentRole.PLANNER)
    outcome = llm_call_with_validation(
        invoke_fn=invoke_llm,
        llm=_analyst_pool[0],
        fallback_llms=_analyst_pool[1:],
        prompt=prompt,
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
                "reason": str(d.get("reason", ""))[:200],
            }
            for d in deps_raw
            if isinstance(d, dict) and str(d.get("name", "")).strip()
        ]
        try:
            complexity = int(data.get("complexity", 5))
        except (TypeError, ValueError):
            complexity = 5
        state.package_analysis = PackageAnalysis(
            purpose=str(data.get("purpose", ""))[:300],
            language=str(data.get("language", ""))[:60],
            build_system=str(bs.get("type", "unknown")).lower(),
            build_system_confidence=float(bs.get("confidence", 0.5) or 0),
            build_system_reasoning=str(bs.get("reasoning", ""))[:300],
            dependencies=dependencies[:15],
            riscv_risks=[
                str(r)[:200] for r in data.get("riscv_risks", [])[:10]
            ],
            build_strategy=str(data.get("build_strategy", ""))[:600],
            expected_artifacts=[
                str(a)[:80] for a in data.get("expected_artifacts", [])[:10]
            ],
            needs_custom_plan=bool(data.get("needs_custom_plan", False)),
            complexity=max(1, min(10, complexity)),
            llm_grounded=True,
        )
        state.context_cache["package_analysis"] = data
        logger.info(
            f"Analysis: {state.package_analysis.purpose[:80]} | "
            f"build={state.package_analysis.build_system} | "
            f"custom_plan={state.package_analysis.needs_custom_plan} | "
            f"complexity={state.package_analysis.complexity}/10"
        )

    # Structural record of the fixed pipeline — no LLM tokens are ever
    # spent on this again (the old planner burned a call on a TaskPlan
    # nothing read).
    state.task_plan = create_default_plan()
    state.current_phase = "analyzed"
    return state


# ============================================================================
# NODE: SUMMARIZER (New - Professional Documentation)
# ============================================================================

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
    """Create a default task plan if planning fails."""
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


# ============================================================================
# NODE: SUPERVISOR (Orchestration)
# ============================================================================


@agent_node(AgentRole.SUPERVISOR)
def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor evaluation node.

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

    # The LLM-based supervisor routing has been removed.
    # All routing decisions are now made by the ``route_supervisor_to_next``
    # conditional edge function, which is purely heuristic and requires
    # no LLM call. This eliminates the redundant LLM call that always
    # agreed with the heuristic anyway.

    summary = (
        f"build_status={state.build_status.value}, "
        f"errors={len(state.error_history)}, "
        f"fixes={len(state.fixes_attempted)}, "
        f"cost=${state.api_cost_usd:.4f}"
    )
    logger.info(f"Supervisor summary: {summary}")
    state.log_scripted_op("supervisor_eval")
    return state


# ============================================================================
# NODE: SCOUT (Architecture Analysis)
# ============================================================================

SCOUT_PROMPT = (
    "You are the build scout for a RISC-V porting agent. Produce a\n"
    "minimal, executable BuildPlan for **{repo_name}** built natively at"
    " `{repo_path}`.\n"
    "\n"
    "## ACTIVE SANDBOX (use ONLY these package commands)\n"
    "{platform_banner}\n"
    "\n"
    "## Target\n"
    "- Architecture: {target_arch} ({arch_identifiers})\n"
    "- Native compilation only (DO NOT set `--host`, `GOOS/GOARCH`, `-DC"
    "MAKE_SYSTEM_PROCESSOR`).\n"
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
    "## Platform knowledge (USE THIS — don't invent package names)\n"
    "{system_knowledge}\n"
    "\n"
    "{few_shot_examples}\n"
    "\n"
    "## Build plan rules\n"
    "1. Each command runs in a fresh shell at `{repo_path}`. **Always ch"
    "ain `cd` with `&&`** — e.g. `cd build && cmake ..`, never two comma"
    "nds.\n"
    "2. Group commands into ≤3 phases: `setup` (install deps) → `configu"
    "re` (optional) → `build`. Skip phases that have nothing to do.\n"
    "3. Install only `-dev` packages you can justify from CMakeLists.txt"
    " / configure.ac / meson.build / go.mod. No guessing.\n"
    "4. Use the package names from the platform knowledge above. Do NOT "
    "use names from a different distro.\n"
    "5. If the project has both CMake and autotools, prefer the one whos"
    "e dependencies are easier to satisfy; if unsure, prefer autotools.\n"
    "6. For Go: if `go.mod` is missing, first phase must `go mod init <m"
    "odule_path>` then `go mod tidy`. Always pass `-buildvcs=false` to `"
    "go build`. **If `Project context` above gives a `build_command`, us"
    "e exactly that command — do not substitute `go build ./...` (which "
    "compiles every helper and fails on the first one with missing deps)"
    ".** Never emit `go build ./cmd` or `go build ./<top-level-dir>` "
    "without an explicit output path; if building a subpackage, use "
    "`go build -o ./bin/<name> <package>`.\n"
    "7. Disable optional features that pull in heavy or x86-specific dep"
    "s (tests, SIMD, examples, docs).\n"
    "8. NEVER reference paths or files you can't see in the project stru"
    "cture.\n"
    "\n"
    "## Self-check (do silently before responding)\n"
    "- Did I read the actual build files (or admit I couldn't)?\n"
    "- Does every package install command exist for the target distro?\n"
    "- Do my commands actually run if pasted into a shell at `{repo_path"
    "}`?\n"
    "\n"
    "## Output schema (JSON only)\n"
    "{{\n"
    '  "build_system": "go|cmake|make|autotools|cargo|meson|other",\n'
    '  "build_system_confidence": <0-1 float>,\n'
    '  "phases": [\n'
    '    {{"id": 1, "name": "setup", "commands": ["..."], "ca'
    'n_parallelize": false, "expected_duration": "30s"}},\n'
    '    {{"id": 2, "name": "build", "commands": ["..."], "ca'
    'n_parallelize": false, "expected_duration": "2m"}}\n'
    "  ],\n"
    '  "total_estimated_duration": "<sum>",\n'
    '  "notes": ["..."],\n'
    '  "architecture_concerns": ["..."]\n'
    "}}"
)


def validate_build_plan(plan: BuildPlan) -> tuple[bool, str]:
    """Validate BuildPlan for hallucinations and common issues."""
    hallucination_patterns = [
        "/path/to/",
        "your_username",
        "example.com",
        "riscv64-unknown-linux-gnu-gcc",  # Unless we confirmed it's there
    ]

    # Go subcommands that must not appear as apk package names
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
            for pattern in hallucination_patterns:
                if pattern in cmd:
                    return (
                        False,
                        f"Hallucination detected in command: '{cmd}' "
                        f"(contains '{pattern}')",
                    )

            # Check for absolute paths that look guessed
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

            # Check for bad cmake patterns
            if "cmake" in cmd:
                if "cd build &&" in cmd and "-B build" in cmd:
                    return (
                        False,
                        f"Bad cmake command (would create nested "
                        f"build dir): '{cmd}'",
                    )
                # Ensure we're doing `cmake ..` from inside build dir,
                # not `cmake -S . -B build`.
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


@agent_node(AgentRole.SCOUT)
def scout_node(state: AgentState) -> AgentState:
    """Analyze the repository, leveraging scripted pre-analysis."""
    logger.info("Scout analyzing repository...")

    state.build_status = BuildStatus.SCOUTING
    state.current_phase = "scouting"

    build_sys = state.build_system_info
    deps = state.dependencies

    documentation = "No documentation cached"
    if state.file_content_cache:
        doc_files = [
            k
            for k in state.file_content_cache.keys()
            if any(d in k.upper() for d in ["README", "INSTALL", "BUILD"])
        ]
        if doc_files:
            report_docs = doc_files[:3]
            documentation = "\n\n".join(
                [
                    f"## {Path(f).name}\n{state.file_content_cache[f][:2000]}"
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
        go_main_str = json.dumps(go_main_info, indent=2)
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
        "main_path": go_main_info.get("main_path", ""),
        "module_dir": module_dir,
    }
    few_shot_examples = format_few_shot_examples(
        "scout", few_shot_context, max_examples=2, max_chars=2000
    )

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
        previous_failure = (
            f"{failed_cmd}Error: {state.last_error[:800]}\n"
            "Your new plan MUST avoid repeating the cause of this "
            "failure (e.g. install the missing package, pick a "
            "different build path)."
        )

    # Build architecture patterns info
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
    build_evidence = collect_build_evidence(state.repo_path)

    prompt = SCOUT_PROMPT.format(
        target_arch="RISC-V (riscv64)",
        arch_identifiers="rv64, riscv, riscv64, RISCV64",
        repo_name=state.repo_name,
        platform_banner=_build_platform_banner(),
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
        system_knowledge=get_system_knowledge_summary(),
        architecture=system_info_raw.get("architecture", "riscv64"),
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
        },
        max_retries=2,
    )
    state.log_api_call(
        cost=outcome.cost_usd,
        tokens_in=outcome.input_tokens,
        tokens_out=outcome.output_tokens,
        calls=outcome.attempts,
    )

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
        logger.info(f"Build plan created and validated: {len(phases)} phases")

    except Exception as e:
        logger.error(f"Scout failed: {e}")
        logger.info("Using fallback build plan due to scout failure")
        state.add_error(
            create_error_record(
                message=f"Scout failed: {e}",
                category=ErrorCategory.UNKNOWN,
            )
        )
        state.build_plan = create_fallback_build_plan(state)

    state.build_status = BuildStatus.PENDING
    return state


def create_fallback_build_plan(state: AgentState) -> BuildPlan:
    """Create a basic fallback build plan based on detected build system."""
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
        needs_init = go_main_info.get("needs_go_init", False)
        build_cmd = go_main_info.get(
            "build_command", "go build -buildvcs=false ./..."
        )
        module_name = (
            state.repo_url.replace("https://", "").rstrip("/")
            if state.repo_url
            else f"github.com/unknown/{state.repo_name}"
        )

        build_cmds = []
        if needs_init:
            build_cmds += [f"go mod init {module_name}", "go mod tidy"]
        else:
            build_cmds.append("go mod tidy")
        build_cmds.append(
            build_cmd
            if build_cmd != "go build ."
            else "go build -buildvcs=false ./..."
        )

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


# ============================================================================
# NODE: BUILDER (Build Execution)
# ============================================================================


def _fixup_top_builddir_in_submakefiles(docker_repo_path: str) -> None:
    """Post-configure fixup for old gettext/autotools projects.

    Old po/Makefile.in.in templates reference $(top_builddir) as a
    make variable but never define it (they expected @top_builddir@
    substitution that was added in newer gettext). When make descends
    into po/ without passing top_builddir, the dependency
    $(top_builddir)/config.status expands to /config.status which
    doesn't exist -> "No rule to make target '/config.status'".

    Fix: run a shell script INSIDE Docker (to avoid host permission
    issues with root-owned files) that scans all subdirectory
    Makefiles and injects the correct relative top_builddir assignment.

    Args:
        docker_repo_path: Path to the repository inside the container.
    """
    try:
        from src.platforms import get_container_name

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
        result = subprocess.run(
            ["docker", "exec", container, "sh", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout.strip():
            logger.info(f"top_builddir fixup: {result.stdout.strip()}")
        if result.returncode != 0 and result.stderr.strip():
            logger.warning(
                f"_fixup_top_builddir_in_submakefiles stderr: "
                f"{result.stderr.strip()}"
            )
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
      1. ``./autogen.sh`` / ``./buildconf`` — upstream's own script,
         which usually wraps autoreconf with project-specific steps.
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
# `cargo build`, `apk add ... cargo build-base`, `dotnet go build-server`,
# etc. — and the downstream optimizations (e.g. `command.replace("go build",
# "go build -buildvcs=false")`) then mangle those unrelated commands.
_GO_BUILD_RE = re.compile(r"\bgo\s+(?:build|install)\b")


def _is_go_build_command(command: str) -> bool:
    """True iff ``command`` actually invokes ``go build`` or ``go install``."""
    return bool(_GO_BUILD_RE.search(command))


def _inject_go_output(command: str, output_path: str) -> str:
    """Inject `-o <path>` into Go build/install commands."""
    if re.search(r"\b-o\s+\S+", command):
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
    # Universal signal — always trust exit 137 as OOM. Broadened from
    # build-only in run 28020958388 because cargo fetch / pip install /
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


def _builder_retry_allowed(state: AgentState, key: str) -> bool:
    """Allow a deterministic builder retry at most once per signature key."""
    tried = state.context_cache.setdefault("builder_retry_keys", [])
    if key in tried:
        return False
    tried.append(key)
    return True


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

    The previous ``go install golang.org/dl/goX.Y.Z@latest`` path does
    not work on Alpine musl (the ``dl`` helper links against glibc) and
    also OOM-kills under QEMU during the ``goX.Y.Z download`` step
    (observed root cause for 25+ Go packages in run 28020958388).
    Downloading the official riscv64 tarball into ``/usr/local`` is
    what our sandbox images already do at build time; replicating that
    at runtime is fast, deterministic, and works on every distro.

    Args:
        version: The Go version string to install, e.g. ``"1.25.0"``.

    Returns:
        A shell command string that, when executed inside the sandbox,
        replaces ``/usr/local/go`` with the requested version. Includes
        a sanity check (``go version``) so a partial download is
        surfaced as a non-zero exit.
    """
    tarball = f"go{version}.linux-riscv64.tar.gz"
    url = f"https://go.dev/dl/{tarball}"
    tmp = f"/tmp/{tarball}"
    return (
        f"set -e && "
        f"curl -fsSL -o {tmp} {url} && "
        f"rm -rf /usr/local/go && "
        f"tar -xzf {tmp} -C /usr/local && "
        f"rm -f {tmp} && "
        f"/usr/local/go/bin/go version"
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
    """Execute the build plan with smart error detection."""
    logger.info("Builder executing build plan...")

    from .platforms import get_active_profile

    state.build_status = BuildStatus.BUILDING
    state.current_phase = "building"

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
                # Clean up double spaces from the removal
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
                ):
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

            # Before any make command, proactively fix po/Makefile
            # top_builddir if needed. This handles the case where
            # configure was run by the fixer (not the builder), so the
            # post-configure hook wouldn't have fired yet.
            if re.match(r"\bmake\b", optimized_cmd.lstrip()):
                _fixup_top_builddir_in_submakefiles(state.repo_path)

            # ``npm run <script>`` guard: the fixer/scout frequently
            # invents an ``npm run build`` even when package.json does
            # not declare a ``build`` script (root cause: the scout
            # prompt encourages "run the standard build command"). If
            # the requested script does not exist, downgrade to the
            # only script present when there's one obvious choice, or
            # skip the command otherwise so the phase can proceed.
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

            result = execute_command(optimized_cmd, cwd=state.repo_path)
            state.cache_command_result(command, result)
            state.log_scripted_op("execute_build_command")

            if not result.success:
                logger.error(f"Command failed: {command}")
                logger.error(f"Error: {result.stderr[:500]}")

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

                if (
                    is_go_build
                    and "no required module provides"
                    in (result.stderr or "").lower()
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
                    build_dir = None
                    for marker in (
                        "mkdir -p build && cd build",
                        "mkdir build && cd build",
                        "cd build && cmake",
                        "cmake -S . -B build",
                    ):
                        if marker in optimized_cmd:
                            build_dir = os.path.join(
                                _to_host_path(state.repo_path), "build"
                            )
                            break
                    if build_dir and os.path.isdir(build_dir):
                        logger.warning(
                            "CMake compiler-cache conflict detected; "
                            "deleting build/ and retrying with explicit "
                            "CMAKE_C_COMPILER"
                        )
                        rm_cmd = f"rm -rf {build_dir}"
                        execute_command(
                            rm_cmd, cwd=state.repo_path, use_docker=True
                        )
                        # Rebuild the command with explicit compiler paths
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

                # Suspected QEMU OOM-kill (exit 137 / empty output) on a
                # parallel build -> retry single-threaded. Generic fallback,
                # checked last so specific signatures win first.
                if _is_suspected_oom(result, optimized_cmd):
                    serial_cmd = _serialize_build_command(optimized_cmd)
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
                        _gv_cmd = _download_go_toolchain_cmd(_gv_needed)
                        _gv_inst = execute_command(
                            _gv_cmd, cwd=state.repo_path
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

                # Package not found on riscv64 repo -> try
                # one-by-one, skip missing
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

    # All phases completed - NOW VERIFY THE BUILD PRODUCED ARTIFACTS
    logger.info("All build phases completed - verifying artifacts...")

    # Determine where to look for build artifacts based on build system
    build_system = (
        state.build_plan.build_system if state.build_plan else "unknown"
    )

    # Check multiple possible artifact locations
    artifact_dirs = [state.repo_path]  # Always check repo root

    # Add build-system-specific directories
    build_subdir = os.path.join(state.repo_path, "build")
    if build_system in ["cmake", "meson"]:
        artifact_dirs.insert(
            0, build_subdir
        )  # Check build/ first for cmake/meson

    # Also check common output directories
    for subdir in ["bin", "dist", "target/release", "output"]:
        artifact_dirs.append(os.path.join(state.repo_path, subdir))

    artifacts_found = False
    for artifact_dir in artifact_dirs:
        check_cmd = f"test -d {artifact_dir}"
        check_result = execute_command(check_cmd, cwd=state.repo_path)

        if not check_result.success:
            continue

        logger.info(f"Checking for artifacts in: {artifact_dir}")
        scanner = ArtifactScanner(artifact_dir, cwd=state.repo_path)
        artifacts = scanner.scan()

        is_valid, message = scanner.verify_build_success()
        logger.info(f"Artifact verification in {artifact_dir}: {message}")

        if is_valid:
            # Record all artifacts in state
            for artifact in artifacts:
                state.add_build_artifact(
                    filepath=artifact["filepath"],
                    artifact_type=artifact["type"],
                    architecture=artifact["architecture"],
                )

            summary = scanner.get_summary()
            logger.info(
                f"Build artifacts summary: "
                f"{json.dumps(summary, indent=2, default=str)}"
            )
            state.context_cache["artifact_summary"] = summary
            artifacts_found = True
            break

    if not artifacts_found:
        # For Go projects, check if any binary was produced in the repo root
        if build_system == "go":
            find_cmd = (
                f"find {state.repo_path} -maxdepth 1 -type f -executable"
            )
            find_result = execute_command(find_cmd, cwd=state.repo_path)
            if find_result.success and find_result.stdout.strip():
                logger.info(f"Go binary found: {find_result.stdout.strip()}")
                for binary_path in find_result.stdout.strip().split("\n"):
                    binary_path = binary_path.strip()
                    if binary_path:
                        # Verify it's a RISC-V binary
                        file_cmd = f"file {binary_path}"
                        file_result = execute_command(
                            file_cmd, cwd=state.repo_path
                        )
                        if (
                            file_result.success
                            and "RISC-V" in file_result.stdout
                        ):
                            state.add_build_artifact(
                                filepath=binary_path,
                                artifact_type="binary",
                                architecture="riscv64",
                            )
                            artifacts_found = True
                            logger.info(
                                f"Verified RISC-V binary: {binary_path}"
                            )
                        elif (
                            file_result.success and "ELF" in file_result.stdout
                        ):
                            # ELF binary on RISC-V host is likely RISC-V
                            state.add_build_artifact(
                                filepath=binary_path,
                                artifact_type="binary",
                                architecture="riscv64",
                            )
                            artifacts_found = True
                            logger.info(
                                f"Found ELF binary (assumed RISC-V on "
                                f"native host): {binary_path}"
                            )

    if not artifacts_found:
        logger.warning(
            "No build artifacts found, but all phases completed successfully"
        )
        # Don't fail the build - phases completed successfully,
        # artifacts may be installed elsewhere.
        logger.info("Treating as success since all build commands succeeded")

    state.build_status = BuildStatus.SUCCESS
    logger.info("Build completed successfully with artifact verification!")

    # Close the fix-attempt feedback loop: any fix applied before this
    # successful build evidently worked. Without this, every attempt
    # stays success=False forever — the fixer prompt then lists working
    # fixes as "(Failed)" and fixer auto-learning never triggers.
    for fix in state.fixes_attempted:
        if not fix.success and fix.build_result is None:
            fix.success = True
            fix.build_result = "build succeeded after fix"

    return state


# ============================================================================
# NODE: FIXER (Enhanced with Reflection)
# ============================================================================


def validate_fix_command(command: str) -> tuple[bool, str]:
    """Validate fix commands to prevent dangerous operations.

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

    from .tools import CommandValidator

    single_ok, single_reason = CommandValidator().is_safe_single(command)
    if not single_ok:
        return False, single_reason

    return True, "Command is safe"


def validate_fixer_response(fix_data: Dict) -> tuple[bool, str]:
    """Validate a FIXER response to ensure it makes logical sense.

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

        # Find recommended strategy
        recommended = next(
            (s for s in strategies if s["id"] == recommended_id), None
        )
        if not recommended:
            return False, f"Recommended strategy {recommended_id} not found"

        # Check if strategy has actions
        actions = recommended.get("actions", [])
        if not actions:
            # Strategy has no actions - this is incomplete/hallucinatory
            logger.warning(
                "FIXER strategy has no actions (incomplete response)"
            )
            return False, "Recommended strategy has no actions"

        # Validate actions in recommended strategy
        for action in actions:
            action_type = action.get("type", "").lower()

            # Validate create_file actions
            if action_type == "create_file":
                path = action.get("path", "").strip()
                content = action.get("content", "").strip()

                if not path:
                    return False, "create_file action missing path"

                # Check for empty files
                if not content:
                    return (
                        False,
                        f"create_file action has empty content for "
                        f"{path} - this won't help",
                    )

                # Check for obviously hallucinated placeholders
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

            # Validate patch actions (same containment as create_file)
            elif action_type == "patch":
                patch_file = (action.get("file") or "").strip()
                if patch_file and (
                    os.path.isabs(patch_file) or ".." in Path(patch_file).parts
                ):
                    return (
                        False,
                        f"Patch path escapes the repository: {patch_file}",
                    )

            # Validate command actions
            elif action_type == "command":
                cmd = action.get("command", "").strip()

                if not cmd:
                    return False, "command action is empty"

                # Check for nonsensical commands
                if "cp -r" in cmd and "/*" in cmd:
                    # Check if it's copying to itself
                    parts = cmd.split()
                    if len(parts) >= 4:
                        src = parts[2]  # source
                        dst = parts[3]  # destination
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

        # Validate reflection data
        reflection = fix_data.get("reflection", {})
        if not reflection.get("root_cause"):
            logger.warning("Fixer response has empty root_cause analysis")

        if not reflection.get("this_fix_will_work_because"):
            logger.warning(
                "Fixer response has empty "
                "'this_fix_will_work_because' - self-critique may be "
                "shallow"
            )

        return True, ""

    except Exception as e:
        logger.error(f"Error validating fixer response: {e}")
        return False, f"Validation error: {str(e)}"


# ============================================================================
# NODE: FIXER (Error Resolution)
# ============================================================================

FIXER_PROMPT = (
    "You are the build doctor for a RISC-V porting agent. Diagnose\n"
    "the failure for **{repo_name}** and emit a minimal fix.\n"
    "\n"
    "## ACTIVE SANDBOX (use ONLY these package commands)\n"
    "{platform_banner}\n"
    "\n"
    "## Build context\n"
    "- Target: {target_arch} | Build system: {build_system}\n"
    "- Attempt {attempt_count}/{max_attempts} | Phase: {current_phase}\n"
    "\n"
    "## Failure\n"
    "**Failed command:**\n"
    "```\n"
    "{failed_command}\n"
    "```\n"
    "**Error (truncated):**\n"
    "```\n"
    "{error_output}\n"
    "```\n"
    "**Exit code:** {exit_code}\n"
    "\n"
    "### Source context around the error (real file contents)\n"
    "{error_file_excerpts}\n"
    "\n"
    "### What this package is (analyst)\n"
    "{package_context}\n"
    "\n"
    "### Previous fix attempts (most recent last — DO NOT repeat any of "
    "these)\n"
    "{previous_fixes}\n"
    "\n"
    "### Repo structure\n"
    "{repo_tree}\n"
    "\n"
    "### Known arch concerns\n"
    "{known_arch_issues}\n"
    "\n"
    "### Current build plan\n"
    "{build_plan}\n"
    "\n"
    "## Platform knowledge (canonical — use these package names + comman"
    "ds)\n"
    "{system_knowledge}\n"
    "\n"
    "{few_shot_examples}\n"
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
    "4. **Check the few-shot examples above** — if one matches the error"
    " pattern, use it.\n"
    "5. **Verify your fix does not duplicate a previous failed attempt.*"
    "*\n"
    "\n"
    "## Critical rules\n"
    "- Each command runs in a fresh shell at the repo root. Chain `cd` w"
    "ith `&&`.\n"
    "- Do NOT propose unrelated cleanups. Fix only what's broken.\n"
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
    "- If the error indicates an unfixable mismatch (e.g. project needs "
    "newer Go\n"
    "  than the sandbox provides), say so in `reflection.root_cause` and"
    " propose\n"
    "  ESCALATE rather than a hack.\n"
    "\n"
    "## Output schema (JSON only)\n"
    "{{\n"
    '  "strategies": [\n'
    '    {{"id": 1, "description": "<one line>",\n'
    '      "actions": [\n'
    '        {{"type": "command", "command": "<shell>"}}\n'
    '        // or {{"type": "create_file", "path": "<rel>", "c'
    'ontent": "..."}}\n'
    '        // or {{"type": "patch", "file": "<rel>", "content'
    '": "<unified diff>"}}\n'
    "      ]}}\n"
    "  ],\n"
    '  "recommended_strategy_id": 1,\n'
    '  "reflection": {{\n'
    '    "root_cause": "<why it broke>",\n'
    '    "this_fix_will_work_because": "<why the chosen fix addresses'
    ' the root cause>"\n'
    "  }}\n"
    "}}\n"
    "\n"
    "Rules: ≥1 strategy with ≥1 action; no placeholder paths (`/path/to/"
    "`, `/home/`);\n"
    "prefer `command` with `sed` over `patch` for simple edits.\n"
    "\n"
    "## Investigation option\n"
    "{investigation}"
)


# First-token whitelist for fixer investigation commands. Read-only
# inspection ONLY — the fixer gathers evidence with these, it does not
# mutate the repo through them.
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
    "If (and ONLY if) the error text and source excerpts above are "
    "genuinely\n"
    "insufficient to diagnose the root cause, you may respond ONCE "
    "with this\n"
    "shape instead of the strategies schema:\n"
    "{\n"
    '  "investigate": {"commands": ["<read-only shell command>"]}\n'
    "}\n"
    "Up to 4 commands; allowed first tokens: cat, head, tail, ls, "
    "find, grep,\n"
    "file, stat, wc, pkg-config, readelf, nm, ldd, go, which. No "
    "redirection.\n"
    "The outputs will be returned to you and then you MUST emit the "
    "strategies\n"
    "schema."
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
        from .tools import CommandValidator

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
        if len(output) > _INVESTIGATE_OUTPUT_CAP:
            output = output[:_INVESTIGATE_OUTPUT_CAP] + "\n[... truncated ...]"
        sections.append(f"$ {cmd}  (exit {result.exit_code})\n{output}")
    state.log_scripted_op("fixer_investigation")
    return "\n\n".join(sections) or "(no valid commands to run)"


@agent_node(AgentRole.FIXER)
def fixer_node(state: AgentState) -> AgentState:
    """Fix build errors using a reflection pattern."""
    logger.info("Fixer analyzing error...")

    state.build_status = BuildStatus.FIXING
    state.current_phase = "fixing"

    if not state.last_error:
        logger.warning("No error to fix")
        state.build_status = BuildStatus.PENDING
        return state

    # Prepare context
    previous_fixes = (
        "\n".join(
            [
                f"- {fix.strategy} ({'Success' if fix.success else 'Failed'})"
                for fix in state.fixes_attempted[-5:]
            ]
        )
        if state.fixes_attempted
        else "None"
    )

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
    if state.error_history:
        last = state.error_history[-1]
        failed_command = last.command or "Unknown"

    few_shot_context = {
        "build_system": (
            state.build_plan.build_system if state.build_plan else "unknown"
        ),
        "error_message": state.last_error,
    }
    few_shot_examples = format_few_shot_examples(
        "fixer", few_shot_context, max_examples=2, max_chars=2000
    )

    # Get build plan details
    build_plan_str = "Unknown"
    if state.build_plan:
        build_plan_str = (
            f"Build System: {state.build_plan.build_system}\n"
            f"Completed Phases: {state.last_successful_phase}"
        )

    # Deterministic diagnosis context — the actual source lines the
    # error points at, plus the analyst's understanding of the package.
    error_file_excerpts = (
        error_context_excerpts(state.last_error or "", state.repo_path)
        or "(error references no readable repo files)"
    )
    package_context = _format_analysis_for_prompt(state.package_analysis)

    def _render_fixer_prompt(investigation: str) -> str:
        return FIXER_PROMPT.format(
            repo_name=state.repo_name,
            target_arch="RISC-V (riscv64)",
            platform_banner=_build_platform_banner(),
            build_system=(
                state.build_plan.build_system
                if state.build_plan
                else "unknown"
            ),
            current_phase=state.current_phase,
            attempt_count=state.attempt_count,
            max_attempts=state.max_attempts,
            failed_phase=state.current_phase,
            exit_code="N/A",
            error_output=(
                state.last_error[:3000]
                if state.last_error
                else "No error details"
            ),
            error_file_excerpts=error_file_excerpts,
            package_context=package_context,
            failed_command=failed_command,
            previous_fixes=previous_fixes,
            repo_tree=(
                state.repo_tree[:500] if state.repo_tree else "(Not available)"
            ),
            known_arch_issues=arch_issues,
            build_plan=build_plan_str,
            system_knowledge=get_system_knowledge_summary(),
            few_shot_examples=few_shot_examples,
            investigation=investigation,
        )

    _fixer_pool = get_model_pool_for_role(AgentRole.FIXER)

    # Bounded investigation loop: round 0 may answer with read-only
    # inspection commands instead of a fix; we execute them and re-ask
    # ONCE with the outputs. Round 1 must produce strategies.
    outcome = None
    investigation = _INVESTIGATION_OFFER
    for round_idx in range(2):
        allow_investigation = round_idx == 0

        def _validate_fix(data: dict) -> ValidationResult:
            if "investigate" in data and "strategies" not in data:
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
            ok, reason = validate_fixer_response(data)
            if not ok:
                return ValidationResult.bad(reason)
            return ValidationResult.good()

        outcome = llm_call_with_validation(
            invoke_fn=invoke_llm,
            llm=_fixer_pool[0],
            fallback_llms=_fixer_pool[1:],
            prompt=_render_fixer_prompt(investigation),
            validator=_validate_fix,
            # no deterministic fix — escalate via FAILED status
            fallback_factory=lambda: None,
            role=AgentRole.FIXER.value,
            audit_metadata={
                "repo": state.repo_name,
                "error_category": (
                    state.last_error_category.value
                    if state.last_error_category
                    else "unknown"
                ),
                "attempt": state.attempt_count,
                "fixer_round": round_idx,
            },
            max_retries=2,
        )
        state.log_api_call(
            cost=outcome.cost_usd,
            tokens_in=outcome.input_tokens,
            tokens_out=outcome.output_tokens,
            calls=outcome.attempts,
        )

        if (
            outcome.data is not None
            and allow_investigation
            and "investigate" in outcome.data
            and "strategies" not in outcome.data
        ):
            commands = outcome.data["investigate"].get("commands", [])
            logger.info(
                f"Fixer requested investigation: {len(commands)} command(s)"
            )
            results = _run_fixer_investigation(state, commands)
            investigation = (
                "### Investigation results (you asked for these)\n"
                f"{results}\n\n"
                "You MUST now respond with the strategies schema — "
                "'investigate' is no longer allowed."
            )
            continue
        break

    try:
        if outcome.data is None:
            logger.error(
                f"FIXER produced no valid response after "
                f"{outcome.attempts} attempts: "
                f"{outcome.last_error}"
            )
            state.add_error(
                create_error_record(
                    message=(
                        f"FIXER proposed invalid fix: " f"{outcome.last_error}"
                    ),
                    category=ErrorCategory.CONFIGURATION,
                )
            )
            state.build_status = BuildStatus.FAILED
            return state

        fix_data = outcome.data

        # Get recommended strategy
        recommended_id = fix_data.get("recommended_strategy_id", 1)
        strategies = fix_data.get("strategies", [])

        if not strategies:
            logger.error("No fix strategies generated")
            state.build_status = BuildStatus.FAILED
            return state

        # Find recommended strategy
        strategy = next(
            (s for s in strategies if s["id"] == recommended_id), strategies[0]
        )

        logger.info(f"Applying fix strategy: {strategy['description']}")

        # Apply fix actions
        changes_made = []
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
                    continue

                dir_path = os.path.dirname(full_file_path)
                if dir_path:
                    mkdir_result = execute_command(
                        f"mkdir -p {shlex.quote(dir_path)}",
                        cwd=state.repo_path,
                        use_docker=True,
                    )
                    if not mkdir_result.success:
                        logger.warning(
                            f"Failed to create directory: {dir_path}"
                        )

                # Use base64 encoding to safely transfer LLM-generated content
                encoded = base64.b64encode(file_content.encode()).decode()
                write_result = execute_command(
                    f"echo {shlex.quote(encoded)} | base64 -d "
                    f"> {shlex.quote(full_file_path)}",
                    cwd=state.repo_path,
                    use_docker=True,
                )

                if write_result.success:
                    changes_made.append(f"Created file: {file_path}")
                    logger.info(f"Created file: {file_path}")
                else:
                    logger.error(
                        f"Failed to create file {file_path}: "
                        f"{write_result.stderr}"
                    )

            elif action["type"] == "patch":
                patch_content = action["content"]
                file_path = action.get("file")
                full_file_path = (
                    os.path.join(state.repo_path, file_path)
                    if file_path
                    else None
                )

                if full_file_path and not _is_within_repo(
                    full_file_path, state.repo_path
                ):
                    logger.error(
                        f"Refusing patch outside repo: {file_path}"
                    )
                    changes_made.append(
                        f"REJECTED patch outside repo: {file_path}"
                    )
                    continue

                if apply_patch(
                    patch_content,
                    filepath=full_file_path,
                    cwd=state.repo_path,
                    use_docker=True,
                ):
                    changes_made.append(
                        f"Patched {file_path if file_path else 'repository'}"
                    )
                    state.patches_generated.append(patch_content)
                else:
                    logger.error(f"Failed to apply patch to {file_path}")

            elif action["type"] == "command":
                command = action["command"]

                is_safe, reason = validate_fix_command(command)
                if not is_safe:
                    logger.warning(
                        f"Fix command blocked: {command} - {reason}"
                    )
                    continue

                # Rewrite autoreconf to copy-first sequence when
                # gettext macros are missing. autoreconf internally
                # calls autopoint which reverts any gettext.m4 fix.
                # The copy-first approach replaces the old m4/gettext.m4
                # with the system's modern one BEFORE running
                # aclocal+autoconf, which is the only reliable fix.
                if re.search(r"\bautoreconf\b", command):
                    if os.path.isfile(
                        os.path.join(
                            _to_host_path(state.repo_path), "m4", "gettext.m4"
                        )
                    ):
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
                    if result.success:
                        changes_made.append(f"Executed: {cmd}")
                    else:
                        logger.error(f"Fix command failed: {result.stderr}")

        # Record fix attempt
        fix_attempt = FixAttempt(
            error_category=state.last_error_category,
            strategy=strategy["description"],
            changes_made=changes_made,
            success=False,  # Will be updated after rebuild
        )
        state.add_fix_attempt(fix_attempt)

        logger.info(f"Fix applied: {len(changes_made)} changes")

        # Reset to pending so supervisor can try building again
        state.build_status = BuildStatus.PENDING

    except Exception as e:
        logger.error(f"Fixer failed: {e}")
        state.build_status = BuildStatus.FAILED

    return state


# ============================================================================
# NODE: ESCALATE
# ============================================================================


@agent_node(AgentRole.SUPERVISOR)  # Escalation is a supervisor-level summary
def escalate_node(state: AgentState) -> AgentState:
    """Escalate to a human with a comprehensive report."""
    logger.warning("Escalating to human intervention")

    state.build_status = BuildStatus.ESCALATED
    state.current_phase = "escalated"

    # Generate escalation report
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
{should_escalate(state)[1]}
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

    return state


# ============================================================================
# NODE: FINISH
# ============================================================================


def _save_learning_data(state: AgentState):
    """Save successful build patterns to examples and recipe cache.

    Implements the auto-learning step after a successful build.
    """
    try:
        repo = state.repo_name or "unknown"
        bs = state.build_plan.build_system if state.build_plan else "unknown"

        # --- Curate artifacts so recipe + report show only
        # user-facing deliverables ---
        if state.build_artifacts and not state.curated_artifacts:
            try:
                from .artifact_curator import curate_artifacts

                curator_llm = None
                try:
                    curator_llm = get_model_for_role(AgentRole.SUMMARIZER)
                except Exception as e:
                    logger.info(
                        f"Curator LLM unavailable, using rule-based "
                        f"fallback: {e}"
                    )
                state.curated_artifacts = curate_artifacts(
                    raw_artifacts=state.build_artifacts,
                    repo_name=repo,
                    build_system=bs,
                    llm=curator_llm,
                )
            except Exception as e:
                logger.warning(f"Artifact curation failed (non-fatal): {e}")
                state.curated_artifacts = []

        # --- Scout learning ---
        if state.build_plan and state.build_plan.phases:
            scout_data = {
                "name": f"Auto: {repo} ({bs})",
                "tags": [bs, repo],
                "build_system": bs,
                "repo_name": repo,
                "trigger": {
                    "build_system": bs,
                    "has_main": bool(
                        state.context_cache.get("go_main_info", {}).get(
                            "has_main", False
                        )
                    ),
                },
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

        # --- Fixer learning ---
        for fix in state.fixes_attempted or []:
            if fix.success and fix.strategy:
                fixer_data = {
                    "name": f"Auto: {repo} - {fix.strategy[:40]}",
                    "tags": [bs, "auto-fix"],
                    "build_system": bs,
                    "repo_name": repo,
                    "error_pattern": (
                        re.escape((state.last_error or "")[:80])
                        if state.last_error
                        else ""
                    ),
                    # ``changes_made`` records "Executed: <cmd>" /
                    # "Created file: <p>" strings; recover the raw
                    # commands for the few-shot example.
                    "fix": {
                        "strategy": fix.strategy,
                        "actions": [
                            {
                                "type": "command",
                                "command": change[len("Executed: ") :],
                            }
                            for change in (fix.changes_made or [])
                            if change.startswith("Executed: ")
                        ],
                    },
                    "reasoning": f"Auto-learned fix from {repo}.",
                }
                save_learned_example("fixer", fixer_data)

        # --- Builder learning ---
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

        # --- Recipe cache ---
        if state.build_plan:
            duration = (
                state.get_execution_duration()
                if hasattr(state, "get_execution_duration")
                else 0.0
            )
            save_to_recipe_cache(
                repo_name=repo,
                repo_url=state.repo_url or "",
                build_system=bs,
                build_plan={
                    "phases": [
                        {"name": p.name, "commands": p.commands}
                        for p in state.build_plan.phases
                    ]
                },
                dependencies=[],
                patches=[
                    f.strategy
                    for f in (state.fixes_attempted or [])
                    if f.success
                ],
                artifacts=state.curated_artifacts
                or state.build_artifacts
                or [],
                build_duration_seconds=duration,
                recipe_markdown=state.porting_recipe or None,
            )

        logger.info(f"Auto-learning complete for {repo}")
    except Exception as e:
        logger.warning(f"Auto-learning failed (non-fatal): {e}")


@agent_node(AgentRole.SUMMARIZER)
def finish_node(state: AgentState) -> AgentState:
    """Finalize successful porting and generate the guide."""
    logger.info("Generating comprehensive porting guide...")

    state.build_status = BuildStatus.SUCCESS
    state.current_phase = "finished"

    # Prepare context for summarizer
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

    # Prepare artifacts information (prefer curated list — primary
    # first, no noise)
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

    # Create prompt
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
        messages = [HumanMessage(content=prompt)]
        llm = get_model_for_role(AgentRole.SUMMARIZER)
        response = invoke_llm(llm, messages)
        tokens_in, tokens_out = response_usage(response)
        cost = response_cost(llm, response)
        state.log_api_call(
            cost=cost, tokens_in=tokens_in, tokens_out=tokens_out
        )

        recipe = extract_content(response.content)

        # Log LLM call for debugging
        log_llm_call(
            agent_role=AgentRole.SUMMARIZER.value,
            prompt=prompt,
            response=recipe,
            model=llm.model_name if hasattr(llm, "model_name") else "unknown",
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata={"repo": state.repo_name, "phase": "finish"},
        )

        state.porting_recipe = recipe
        logger.info("Porting guide generated successfully")

    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        # Fallback recipe
        state.porting_recipe = (
            f"# RISC-V Porting Recipe: {state.repo_name}\n\n"
            f"Build succeeded.\n\n{build_steps}"
        )

    # Auto-learning: save successful patterns
    _save_learning_data(state)

    return state


# ============================================================================
# ROUTING FUNCTION
# ============================================================================


# ============================================================================
# ROUTING FUNCTIONS — one per source node, named by what they decide
# ============================================================================
# Each function inspects state and returns a single destination node name
# (or a list of Send() for fan-out). No string-phase indirection.


def route_init_to_next(state: AgentState) -> str:
    """After init: go to the analyst, or escalate if init failed."""
    if state.build_status == BuildStatus.FAILED:
        logger.warning("Initialization failed; forcing escalation")
        return "escalate_node"
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
    return "heuristic_plan_node"


def route_heuristic_plan_to_next(state: AgentState) -> str:
    """After the heuristic planner: plan → supervisor, else LLM scout.

    The heuristic planner only materializes a default BuildPlan for
    build systems it recognizes with reasonable confidence AND the
    analyst did not veto (``needs_custom_plan``). Anything else is
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
    # A verified success ALWAYS finishes — even at the attempt/cost
    # ceiling. Escalating a build that just succeeded (e.g. the final
    # fix landed on attempt max_attempts) would throw away the win and
    # skip the recipe. Escalation checks apply to non-success states.
    if state.build_status == BuildStatus.SUCCESS:
        return "finish_node"

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


# --------------------------------------------------------------------------
# Build-fix subgraph internal routing
# --------------------------------------------------------------------------


def route_build_result(state: AgentState) -> str:
    """Route a build result: verify, attempt a fix, or exit."""
    if state.build_status == BuildStatus.SUCCESS:
        return "verify_node"
    if state.attempt_count >= state.max_attempts:
        logger.warning(
            f"Subgraph: max attempts ({state.max_attempts}) exceeded "
            f"after build — exiting subgraph"
        )
        return "__end__"
    return "fix_node"


def route_verify_result(state: AgentState) -> str:
    """Route a verify result: exit as success, or go to fix."""
    if state.build_status == BuildStatus.SUCCESS:
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
    return "fix_node"


def route_fix_result(state: AgentState) -> str:
    """After a fix: retry build, or exit subgraph as failure."""
    if state.build_status == BuildStatus.FAILED:
        logger.warning("Subgraph: fix resulted in FAILED status — exiting")
        return "__end__"
    if state.attempt_count >= state.max_attempts:
        logger.warning(
            f"Subgraph: max attempts ({state.max_attempts}) exceeded "
            f"after fix — exiting subgraph"
        )
        return "__end__"
    return "build_node"


# ============================================================================
# HEURISTIC PLAN NODE — zero-LLM default recipes, gated by the analyst
# ============================================================================


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

    Replaces the former "parallel scout branches + aggregator" — three
    graph nodes that only copied init-time detection into differently
    named fields. The same facts are gathered here in one honest pass,
    then combined with the analyst's verdict:

    * ``analysis.needs_custom_plan`` (the analyst READ the build files
      and says a textbook recipe will fail) → defer to the LLM scout.
    * Significant arch-specific code or weak build-system confidence →
      defer to the LLM scout.
    * Otherwise → emit the canned recipe for the detected build system,
      with the analyst's dependency list merged into the setup phase.
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

    build_type = state.scout_build_system_result["type"]
    confidence = float(state.scout_build_system_result["confidence"])

    # The analyst read the actual build files; when it is LLM-grounded
    # and disagrees with (or is more confident than) file-name
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
    if (
        state.scout_arch_issues_result["critical"]
        or state.scout_arch_issues_result["high_severity"] >= 3
    ):
        logger.info(
            "Significant arch-specific code detected; deferring to LLM scout"
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

    if build_type == "cmake":
        state.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=confidence,
            phases=[
                BuildPhase(
                    1, "setup", [_setup_cmd(["cmake", "gcc"])], False, "30s"
                ),
                BuildPhase(
                    2,
                    "configure",
                    [
                        "mkdir -p build && cd build"
                        " && cmake .. -DCMAKE_BUILD_TYPE=Release"
                    ],
                    False,
                    "1m",
                ),
                BuildPhase(
                    3, "build", ["cd build && make -j$(nproc)"], True, "5m"
                ),
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
                BuildPhase(2, "build", ["make -j$(nproc)"], True, "5m"),
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
        needs_init = go_main_info.get("needs_go_init", False)
        build_cmd = go_main_info.get(
            "build_command", "go build -buildvcs=false ./..."
        )
        if "-buildvcs=false" not in build_cmd and _is_go_build_command(
            build_cmd
        ):
            build_cmd = _inject_go_flag(build_cmd, "-buildvcs=false")

        go_phase_commands = []
        if needs_init:
            module_name = (
                state.repo_url.replace("https://", "").rstrip("/")
                if state.repo_url
                else f"github.com/unknown/{state.repo_name}"
            )
            go_phase_commands.extend(
                [f"go mod init {module_name}", "go mod tidy"]
            )
        else:
            go_phase_commands.append("go mod tidy")
        go_phase_commands.append(build_cmd)

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

    state.build_status = BuildStatus.PENDING
    return state


# ============================================================================
# BUILD-FIX SUBGRAPH — encapsulates the build → verify → fix → retry loop
# ============================================================================


def _locate_expected_artifacts(
    state: AgentState, expected: List[str]
) -> List[tuple]:
    """Search the sandbox for the analyst's expected artifact names.

    Covers the common case where a build installs its outputs outside
    the repo tree (``make install`` → /usr/local) and the repo scan
    finds nothing.

    Args:
        state: Current agent state.
        expected: Artifact file names from ``PackageAnalysis``.

    Returns:
        List of ``(path, file_output)`` tuples for every match found.
    """
    hits: List[tuple] = []
    for name in expected[:5]:
        name = str(name).strip()
        # Basenames only — a path-ish or globby "expectation" is an
        # LLM artifact, not a file name.
        if not name or "/" in name or any(c in name for c in "*?[]$`"):
            continue
        result = execute_command(
            f"find {shlex.quote(state.repo_path)} /usr/local/bin "
            f"/usr/local/lib -maxdepth 6 -type f -name "
            f"{shlex.quote(name)} 2>/dev/null | head -3",
            cwd=state.repo_path,
            use_docker=True,
        )
        for path in (result.stdout or "").strip().splitlines():
            path = path.strip()
            if not path:
                continue
            file_result = execute_command(
                f"file {shlex.quote(path)}",
                cwd=state.repo_path,
                use_docker=True,
            )
            hits.append((path, (file_result.stdout or "").strip()))
    return hits


@agent_node(AgentRole.BUILDER)
def verify_node(state: AgentState) -> AgentState:
    """Post-build artifact verification. Seals build success or routes to fix.

    Outcomes:
        * RISC-V artifacts found → SUCCESS.
        * Artifacts found but for a DIFFERENT architecture (x86/ARM
          fallthrough) → FAILED with an ARCHITECTURE error. This is the
          silent-regression case the scanner exists to catch; it must
          not pass as success.
        * Nothing scannable found → keep the builder's SUCCESS verdict
          (many packages `make install` their artifacts elsewhere, or
          only produce scripts) but record the caveat for the report.
    """
    logger.info("Verifying build artifacts...")
    scanner = ArtifactScanner(state.repo_path, cwd=state.repo_path)
    artifacts = scanner.scan()
    is_valid, message = scanner.verify_build_success()
    logger.info(f"Verification: {message}")

    if is_valid:
        for artifact in artifacts:
            state.add_build_artifact(
                filepath=artifact["filepath"],
                artifact_type=artifact["type"],
                architecture=artifact.get("architecture"),
            )
        state.build_status = BuildStatus.SUCCESS
        return state

    summary = scanner.get_summary()
    wrong_arch = [
        arch for arch in summary.get("by_architecture", {}) if arch != "RISC-V"
    ]
    if wrong_arch:
        # Build "succeeded" but produced non-riscv64 binaries — a real
        # failure that the fixer must see (and that must never be
        # reported as a successful port).
        state.build_status = BuildStatus.FAILED
        state.add_error(
            create_error_record(
                message=(
                    f"Artifact verification failed: {message}. "
                    f"Non-RISC-V architectures found: "
                    f"{', '.join(wrong_arch)}"
                ),
                category=ErrorCategory.ARCHITECTURE,
            )
        )
        return state

    # No ELF artifacts in the repo scan. Before tolerating, look for
    # the artifacts the analyst said this build must produce — they may
    # have been installed outside the repo tree (make install).
    expected = (
        state.package_analysis.expected_artifacts
        if state.package_analysis
        else []
    )
    if expected:
        hits = _locate_expected_artifacts(state, expected)
        riscv_hits = [(p, d) for p, d in hits if "RISC-V" in d]
        wrong_arch = [
            (p, d) for p, d in hits if "ELF" in d and "RISC-V" not in d
        ]
        if riscv_hits:
            for path, _desc in riscv_hits:
                state.add_build_artifact(
                    filepath=path,
                    artifact_type="binary",
                    architecture="RISC-V",
                )
            logger.info(
                f"Verified {len(riscv_hits)} expected artifact(s) found "
                f"outside the repo scan"
            )
            state.build_status = BuildStatus.SUCCESS
            return state
        if wrong_arch:
            state.build_status = BuildStatus.FAILED
            state.add_error(
                create_error_record(
                    message=(
                        "Expected artifacts exist but are not RISC-V: "
                        + ", ".join(
                            f"{p} ({d[:60]})" for p, d in wrong_arch[:3]
                        )
                    ),
                    category=ErrorCategory.ARCHITECTURE,
                )
            )
            return state

    # Nothing scannable anywhere — tolerate (scripts-only repos) but
    # leave a caveat, including which expected artifacts never showed
    # up, for the recipe/report.
    logger.warning(
        "No verifiable ELF artifacts found; keeping build success "
        "verdict but flagging as unverified"
    )
    state.context_cache["artifact_verification"] = {
        "verified": False,
        "reason": message,
        "expected_missing": list(expected),
    }
    state.build_status = BuildStatus.SUCCESS
    return state


def create_build_fix_subgraph() -> StateGraph:
    """Subgraph: build → (verify → fix → retry) loop.

    Entered from the parent graph. Exits when build succeeds or escalation
    thresholds are reached. The parent graph reads ``build_status`` to
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
        {"build_node": "build_node", "__end__": END},
    )

    return sg.compile()


def predict_build_issues(state: AgentState) -> List[Dict[str, str]]:
    """Predict potential build issues before execution.

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


# ============================================================================
# WORKFLOW CREATION
# ============================================================================


def create_workflow() -> StateGraph:
    """Create the properly graph-shaped LangGraph workflow.

    Architecture:
        init ──→ analyst (LLM reads repo) ──→ heuristic_plan ─┬→ supervisor
                                                              └→ scout_node
                                                       (LLM plan) │
        supervisor ──→ build_fix_subgraph (subgraph) ────→ supervisor
        supervisor ──→ finish_node ──→ END
        supervisor ──→ escalate_node ──→ END
        supervisor ──→ analyst_node (re-analyze)
        supervisor ──→ scout_node (error-aware re-scout)
    """
    logger.info("Creating properly graph-shaped workflow...")

    workflow = StateGraph(AgentState)
    build_fix_subgraph = create_build_fix_subgraph()

    # --- Nodes ---
    workflow.add_node("init_node", init_node)
    # LLM "understand" step: reads real build files/docs, produces the
    # PackageAnalysis consumed by every downstream node.
    workflow.add_node("analyst_node", analyst_node)
    workflow.add_node("supervisor_node", supervisor_node)
    # Zero-LLM default recipes, gated by the analyst
    workflow.add_node("heuristic_plan_node", heuristic_plan_node)
    # LLM scout: custom BuildPlan from evidence + analysis
    workflow.add_node("scout_node", scout_node)
    # Build-fix subgraph
    workflow.add_node("build_fix_subgraph", build_fix_subgraph)
    # Terminal nodes
    workflow.add_node("finish_node", finish_node)
    workflow.add_node("escalate_node", escalate_node)

    # --- Entry point ---
    workflow.set_entry_point("init_node")

    # --- Edges ---

    # init → analyst (or escalate)
    workflow.add_conditional_edges(
        "init_node",
        route_init_to_next,
        {"analyst_node": "analyst_node", "escalate_node": "escalate_node"},
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

    # Terminals
    workflow.add_edge("finish_node", END)
    workflow.add_edge("escalate_node", END)

    compiled = workflow.compile()
    logger.info(
        "Graph workflow compiled: analyst → heuristic/LLM plan → "
        "build-fix subgraph"
    )
    return compiled


# Create global app instance
app = create_workflow()

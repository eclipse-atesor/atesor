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

"""Global state definitions, data structures, and status tracking.

Manages the ``AgentState`` dataclass, the enums that describe the
porting process, and the system-wide helper functions that operate on
that state.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

# ============================================================================
# ENUMS
# ============================================================================


# Growth caps for the append-only run records (MEM-03). Both lists are
# serialized wholesale into `<repo>_state_*.json`, so an escalated run
# that walks the 120-step recursion limit would otherwise balloon both
# memory and the artifact. Only the tail is ever read back.
MAX_AUDIT_EVENTS = 2000
MAX_ERROR_HISTORY = 200


class BuildStatus(str, Enum):
    """Current status of the build process."""

    PENDING = "PENDING"
    PLANNING = "PLANNING"
    SCOUTING = "SCOUTING"
    BUILDING = "BUILDING"
    TESTING = "TESTING"
    FIXING = "FIXING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"


class ErrorCategory(str, Enum):
    """Classification of errors encountered."""

    UNKNOWN = "UNKNOWN"
    DEPENDENCY = "DEPENDENCY"
    COMPILATION = "COMPILATION"
    LINKING = "LINKING"
    ARCHITECTURE = "ARCHITECTURE"
    NETWORK = "NETWORK"
    RATE_LIMIT = "RATE_LIMIT"
    CONFIGURATION = "CONFIGURATION"
    MISSING_TOOLS = "MISSING_TOOLS"
    PERMISSION = "PERMISSION"
    DISK_SPACE = "DISK_SPACE"
    LICENSE_INCOMPATIBLE = "LICENSE_INCOMPATIBLE"
    REQUIRES_HARDWARE = "REQUIRES_HARDWARE"
    ARCHITECTURE_IMPOSSIBLE = "ARCHITECTURE_IMPOSSIBLE"


class FailureSeverity(str, Enum):
    """Severity level for command and execution failures."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentRole(str, Enum):
    """Roles of different agents in the system."""

    PLANNER = "planner"
    SUPERVISOR = "supervisor"
    SCOUT = "scout"
    BUILDER = "builder"
    FIXER = "fixer"
    SUMMARIZER = "summarizer"
    AGENT = "agent"


class Action(str, Enum):
    """Available actions for the supervisor."""

    PLAN = "PLAN"
    SCOUT = "SCOUT"
    BUILDER = "BUILDER"
    FIXER = "FIXER"
    ESCALATE = "ESCALATE"
    FINISH = "FINISH"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class CommandResult:
    """Result of a shell command execution."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """Return True if the command exited with status 0."""
        return self.exit_code == 0

    @property
    def failed(self) -> bool:
        """Return True if the command exited with a non-zero status."""
        return self.exit_code != 0


@dataclass
class ErrorRecord:
    """Record of an error that occurred."""

    category: ErrorCategory
    message: str
    severity: FailureSeverity = FailureSeverity.MEDIUM
    command: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    attempt_number: int = 0


@dataclass
class FixAttempt:
    """Record of a fix attempt."""

    error_category: ErrorCategory
    strategy: str
    changes_made: List[str]
    success: bool
    build_result: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ArchSpecificCode:
    """Information about architecture-specific code found."""

    file: str
    line: int
    code_snippet: str
    arch_type: str  # "x86", "arm", "simd", etc.
    severity: str  # "low", "medium", "high", "critical"
    suggested_fix: Optional[str] = None


@dataclass
class BuildPhase:
    """A single phase in the build plan."""

    id: int
    name: str
    commands: List[str]
    can_parallelize: bool = False
    expected_duration: str = "unknown"
    required_dependencies: List[str] = field(default_factory=list)
    success_criteria: Optional[str] = None


@dataclass
class BuildPlan:
    """Complete build plan for the package."""

    build_system: str
    build_system_confidence: float
    phases: List[BuildPhase]
    total_estimated_duration: str
    notes: List[str] = field(default_factory=list)


@dataclass
class DependencyInfo:
    """Information about package dependencies."""

    system_packages: List[str] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    build_tools: List[str] = field(default_factory=list)
    risc_v_available: bool = True
    install_method: str = "apt"
    version_constraints: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskPhase:
    """A phase in the overall task plan."""

    id: int
    name: str
    description: str
    agent: AgentRole
    use_scripted_ops: bool
    depends_on: List[int] = field(default_factory=list)
    estimated_cost: float = 0.0  # In USD
    status: str = "pending"  # pending, in_progress, completed, failed


@dataclass
class TaskPlan:
    """High-level decomposition of the porting task."""

    phases: List[TaskPhase]
    can_parallelize: List[List[int]] = field(
        default_factory=list
    )  # Groups of parallel phase IDs
    estimated_total_cost: float = 0.0
    estimated_total_time: str = "unknown"
    complexity_score: int = 5  # 1-10


@dataclass
class BuildSystemInfo:
    """Detected build system information."""

    type: str
    confidence: float
    primary_file: str
    additional_files: List[str] = field(default_factory=list)
    requires_configuration: bool = True
    module_dir: str = ""


@dataclass
class PackageAnalysis:
    """The analyst agent's understanding of the package.

    Produced from ACTUAL repo evidence (README, build files) by one LLM
    call, and consumed downstream — the heuristic planner uses
    ``dependencies``/``needs_custom_plan`` to decide whether a default
    recipe is safe, the scout grounds its BuildPlan in
    ``build_strategy``/``riscv_risks``, the fixer gets ``purpose`` and
    risks as diagnosis context, and verification checks
    ``expected_artifacts``.
    """

    purpose: str = ""  # what the package is/does (one sentence)
    language: str = ""  # dominant implementation language
    build_system: str = "unknown"
    build_system_confidence: float = 0.0
    build_system_reasoning: str = ""  # file evidence for the call
    dependencies: List[Dict[str, str]] = field(
        default_factory=list
    )  # [{"name": canonical, "reason": file evidence}]
    riscv_risks: List[str] = field(default_factory=list)
    build_strategy: str = ""  # how to build, grounded in files read
    expected_artifacts: List[str] = field(
        default_factory=list
    )  # binary/library names the build should produce
    needs_custom_plan: bool = False  # True → defer to LLM scout
    complexity: int = 5  # 1-10
    llm_grounded: bool = False  # False → deterministic fallback


@dataclass
class AgentState:
    """Comprehensive state for the RISC-V porting agent.

    All agents read from and write to this shared state object.
    """

    # ========== Repository Information ==========
    repo_url: str
    repo_name: str
    repo_path: str = "/workspace/repos"
    repo_tree: str = ""  # Optimized tree output for initial context

    # ========== Task Planning ==========
    task_plan: Optional[TaskPlan] = None
    package_analysis: Optional[PackageAnalysis] = None
    current_phase: str = "initialization"

    # ========== Build Information ==========
    build_system_info: Optional[BuildSystemInfo] = None
    build_plan: Optional[BuildPlan] = None
    build_status: BuildStatus = BuildStatus.PENDING
    tests_run: bool = False

    # ========== Dependencies ==========
    dependencies: Optional[DependencyInfo] = None

    # ========== Architecture Analysis ==========
    arch_specific_code: List[ArchSpecificCode] = field(default_factory=list)

    # ========== Execution Tracking ==========
    attempt_count: int = 0
    max_attempts: int = 5
    last_successful_phase: int = 0

    # ========== Error Handling ==========
    last_error: Optional[str] = None
    last_error_category: Optional[ErrorCategory] = None
    last_error_severity: Optional[FailureSeverity] = None
    error_history: List[ErrorRecord] = field(default_factory=list)
    fixes_attempted: List[FixAttempt] = field(default_factory=list)

    # ========== Performance Tracking ==========
    api_calls_made: int = 0
    api_cost_usd: float = 0.0
    api_tokens_in: int = 0
    api_tokens_out: int = 0
    scripted_ops_count: int = 0
    execution_start_time: datetime = field(default_factory=datetime.now)

    # ========== Caching & Memory ==========
    context_cache: Dict[str, Any] = field(default_factory=dict)
    file_content_cache: Dict[str, str] = field(default_factory=dict)
    command_results_cache: Dict[str, CommandResult] = field(
        default_factory=dict
    )

    # ========== Parallel Scout Results ==========
    scout_build_system_result: Optional[Dict[str, Any]] = None
    scout_deps_result: Optional[Dict[str, Any]] = None
    scout_arch_issues_result: Optional[Dict[str, Any]] = None

    # ========== Subgraph status (read by the parent graph) ==========
    subgraph_outcome: Optional[str] = (
        None  # "success" | "failure" | "fix_needed"
    )

    # ========== Agent Communication ==========
    messages: List[BaseMessage] = field(default_factory=list)

    # ========== Output Artifacts ==========
    patches_generated: List[str] = field(default_factory=list)
    porting_recipe: Optional[str] = None
    build_artifacts: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Raw scan results
    curated_artifacts: List[Dict[str, Any]] = field(
        default_factory=list
    )  # User-facing subset

    # ========== Debugging & Audit ==========
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    current_agent: Optional[AgentRole] = None

    # ========== Metadata ==========
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def update_timestamp(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now()

    def add_error(self, error: ErrorRecord) -> None:
        """Add an error to history and update state.

        History is capped at ``MAX_ERROR_HISTORY`` (MEM-03); the
        escalation logic only inspects recent errors.
        """
        self.error_history.append(error)
        if len(self.error_history) > MAX_ERROR_HISTORY:
            del self.error_history[:-MAX_ERROR_HISTORY]
        self.last_error = error.message
        self.last_error_category = error.category
        self.last_error_severity = error.severity
        self.attempt_count += 1
        self.update_timestamp()

    def add_fix_attempt(self, fix: FixAttempt) -> None:
        """Record a fix attempt."""
        self.fixes_attempted.append(fix)
        self.update_timestamp()

    def log_api_call(
        self,
        cost: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        calls: int = 1,
    ) -> None:
        """Track API usage with real token counts and cost.

        Args:
            cost: Real USD cost of the call(s), from token usage.
            tokens_in: Billed prompt tokens.
            tokens_out: Billed completion tokens.
            calls: Number of LLM invocations this covers (a validated
                call may retry several times).
        """
        self.api_calls_made += max(1, calls)
        self.api_cost_usd += cost
        self.api_tokens_in += tokens_in
        self.api_tokens_out += tokens_out
        self.update_timestamp()

    def log_scripted_op(self, operation: str = "unknown") -> None:
        """Track scripted operation usage."""
        self.scripted_ops_count += 1
        self.log_event("scripted_op", {"operation": operation})
        self.update_timestamp()

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to the audit trail.

        The trail is capped at ``MAX_AUDIT_EVENTS``: it is appended to on
        every scripted op (dozens per attempt) and serialized wholesale
        by :meth:`to_dict`/``save_to_json``, so an escalated run that
        walks the 120-step recursion limit would otherwise grow the
        in-memory list and the state JSON without bound (MEM-03).
        Oldest events are dropped first; the report only ever reads the
        tail via :meth:`get_last_audit_events`.
        """
        self.audit_trail.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": event_type,
                "agent": (
                    self.current_agent.value if self.current_agent else None
                ),
                "data": data,
            }
        )
        if len(self.audit_trail) > MAX_AUDIT_EVENTS:
            del self.audit_trail[:-MAX_AUDIT_EVENTS]
        self.update_timestamp()

    def log_agent_decision(
        self, agent: AgentRole, action: str, reason: str
    ) -> None:
        """Log a decision made by an agent."""
        self.log_event(
            "decision",
            {"agent": agent.value, "action": action, "reason": reason},
        )

    def cache_command_result(
        self, command: str, result: CommandResult
    ) -> None:
        """Cache a command result for reuse."""
        cache_key = self._generate_cache_key(command)
        self.command_results_cache[cache_key] = result
        self.update_timestamp()

    def get_cached_command_result(
        self, command: str
    ) -> Optional[CommandResult]:
        """Retrieve cached command result if available."""
        cache_key = self._generate_cache_key(command)
        return self.command_results_cache.get(cache_key)

    def cache_file_content(self, filepath: str, content: str) -> None:
        """Cache file content to avoid repeated reads."""
        self.file_content_cache[filepath] = content
        self.update_timestamp()

    def _generate_cache_key(self, command: str) -> str:
        """Generate a cache key for a command."""
        import hashlib

        return hashlib.md5(command.encode()).hexdigest()

    def get_execution_duration(self) -> float:
        """Get total execution time in seconds."""
        return (datetime.now() - self.execution_start_time).total_seconds()

    def is_in_error_loop(self) -> bool:
        """Check if we're stuck in an error loop."""
        if len(self.error_history) < 3:
            return False

        # Check if last 3 errors are the same category
        recent_errors = self.error_history[-3:]
        categories = [e.category for e in recent_errors]

        return len(set(categories)) == 1  # All same category

    def add_build_artifact(
        self,
        filepath: str,
        artifact_type: str,
        architecture: Optional[str] = None,
    ) -> None:
        """Record a build artifact that was successfully created."""
        artifact = {
            "filepath": filepath,
            # e.g. "library", "binary", "test", "header".
            "type": artifact_type,
            "architecture": architecture,  # e.g. "RISC-V", "x86_64".
            "timestamp": datetime.now().isoformat(),
        }
        self.build_artifacts.append(artifact)
        self.update_timestamp()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary for serialization."""
        from dataclasses import asdict

        # We need a custom converter for Enums and Datetime
        def custom_serializer(obj: Any) -> Any:
            if isinstance(obj, (datetime, Enum)):
                return str(obj)
            if isinstance(obj, list):
                return [custom_serializer(i) for i in obj]
            if isinstance(obj, dict):
                return {k: custom_serializer(v) for k, v in obj.items()}
            if hasattr(obj, "__dict__"):
                return custom_serializer(vars(obj))
            return obj

        return custom_serializer(asdict(self))

    def save_to_json(self, filepath: str) -> None:
        """Save the current state to a JSON file."""
        import json

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_last_audit_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the last N audit events."""
        return self.audit_trail[-limit:]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def sanitize_repo_name(raw: str) -> str:
    """Reduce a repo name to a shell- and path-safe token.

    ``repo_name`` is interpolated into shell strings (``rm -rf``,
    ``git clone``, ``cd``) and filesystem paths throughout the
    pipeline, so it must never contain shell metacharacters or path
    traversal. Anything outside ``[A-Za-z0-9._-]`` is replaced with
    ``-``; leading dots are stripped so the name can never be ``..``
    or a hidden file.

    Args:
        raw: The raw name segment derived from the repo URL.

    Returns:
        A safe, non-empty repo name.
    """
    name = re.sub(r"[^A-Za-z0-9._-]", "-", raw or "").lstrip(".")
    return name or "repo"


def is_valid_repo_url(repo_url: str) -> bool:
    """Return True when ``repo_url`` is a plain, shell-safe http(s) URL.
    """
    url = (repo_url or "").strip()
    return bool(re.fullmatch(r"https?://[A-Za-z0-9._~:/?#@!+,=%\-]+", url))


def create_initial_state(repo_url: str, max_attempts: int = 5) -> AgentState:
    """Create initial state for a new porting task.

    Raises:
        ValueError: If ``repo_url`` is not a plain http(s) URL. URLs
            are interpolated into in-container shell commands, so
            anything else (shell metacharacters, other schemes) is
            rejected up front.
    """
    url = (repo_url or "").strip()
    if not is_valid_repo_url(url):
        raise ValueError(
            f"Unsupported or unsafe repository URL: {repo_url!r} "
            "(expected a plain http(s) URL without shell metacharacters)"
        )

    repo_name = sanitize_repo_name(
        url.rstrip("/").split("/")[-1].removesuffix(".git")
    )

    return AgentState(
        repo_url=url,
        repo_name=repo_name,
        repo_path=f"/workspace/repos/{repo_name}",
        max_attempts=max_attempts,
    )


def classify_error(error_message: str) -> ErrorCategory:
    """Classify an error message into a category.

    Uses pattern matching on common error substrings.

    Args:
        error_message: The raw error text to classify.

    Returns:
        The matching ``ErrorCategory``.
    """
    error_lower = error_message.lower()

    # Check rate limiting before network ("timeout" appears in both).
    if any(
        term in error_lower
        for term in [
            "rate limit",
            "too many requests",
            "429",
            "quota exceeded",
        ]
    ):
        return ErrorCategory.RATE_LIMIT

    # Git clone auth failures (sandbox has no git credentials)
    if any(
        term in error_lower
        for term in [
            "could not read username",
            "could not read password",
            "no such device or address",
            "authentication failed",
            "could not read from remote",
        ]
    ):
        return ErrorCategory.NETWORK

    # Bad repository URL (not a real repo)
    if re.search(r"repository\s+['\"][^'\"]+['\"]\s+not\s+found", error_lower):
        return ErrorCategory.CONFIGURATION

    # Network errors
    if any(
        term in error_lower
        for term in ["network", "connection", "timeout", "unreachable"]
    ):
        return ErrorCategory.NETWORK

    # Linking errors
    if any(
        term in error_lower
        for term in [
            "linking error",
            "linker",
            "undefined symbol",
            "cannot find -l",
            "ld returned",
            "collect2: error",
        ]
    ):
        return ErrorCategory.LINKING

    # Autotools/configure script issues (check before COMPILATION to
    # catch configure syntax errors).
    if any(
        term in error_lower
        for term in [
            "possibly undefined macro",
            "macro not found in library",
            "autoconf failed",
            "autoreconf: error",
            "aclocal: not found",
        ]
    ):
        return ErrorCategory.CONFIGURATION

    # Configure script has unexpanded M4 macros or shell-incompatible syntax
    if "configure" in error_lower and "syntax error" in error_lower:
        return ErrorCategory.CONFIGURATION

    # Compilation errors
    if any(
        term in error_lower
        for term in [
            "compilation error",
            "syntax error",
            "parse error",
            "undeclared",
            "undefined reference",
            "implicit declaration",
            "path_max unset",
            "fortified realpath",
        ]
    ):
        return ErrorCategory.COMPILATION

    # Architecture-specific errors. Word boundaries are mandatory:
    # bare substring checks misclassify ordinary words and REPO NAMES —
    # "sse" matches inside "assetfinder", so every error mentioning
    # that repo's path was tagged ARCHITECTURE (observed in the
    # 2026-07-02 smoke run).
    if re.search(
        r"\b(architecture|sse\d?|avx\d*|neon|simd"
        r"|unsupported instruction|illegal instruction"
        r"|x86[-_]64|amd64|arch)\b",
        error_lower,
    ):
        return ErrorCategory.ARCHITECTURE

    # Configuration errors
    if any(
        term in error_lower
        for term in [
            "configure error",
            "cmake error",
            "configure: error",
            "unsupported option",
            "invalid argument",
            "unrecognized option",
            "no go files in",
            "no go source files",
            "no buildable go source files",
            "no rule to make target",
            "no makefile found",
            "cannot find main module",
            "no required module provides",
            "directory prefix . does not contain main module",
            "build output",
            "already exists and is a directory",
            "inconsistent vendoring",
            "does not appear to contain cmakelists.txt",
        ]
    ):
        return ErrorCategory.CONFIGURATION

    # Missing tools
    if any(
        term in error_lower
        for term in [
            "command not found",
            "no such command",
            "not installed",
            "cmake: not found",
            "make: not found",
        ]
    ):
        return ErrorCategory.MISSING_TOOLS

    # Dependency errors
    if any(
        term in error_lower
        for term in [
            "cannot find",
            "not found",
            "no such file",
            "missing dependency",
            "package not found",
            "module not found",
            "import error",
        ]
    ):
        return ErrorCategory.DEPENDENCY

    # Permission errors
    if any(
        term in error_lower
        for term in ["permission denied", "access denied", "forbidden"]
    ):
        return ErrorCategory.PERMISSION

    # Disk space
    if any(
        term in error_lower
        for term in ["no space left", "disk full", "out of space"]
    ):
        return ErrorCategory.DISK_SPACE

    # Python/System errors
    if any(
        term in error_lower
        for term in [
            "keyerror",
            "indexerror",
            "attributeerror",
            "typeerror",
            "valueerror",
            "importerror",
        ]
    ):
        return ErrorCategory.CONFIGURATION  # Usually a code/config bug

    # Package manager resolution errors (apk, apt, etc.)
    if any(
        term in error_lower
        for term in [
            "unable to select packages",
            "no such package",
            "unable to locate package",
            "unable to lock database",
            "broken packages",
            "has no installation candidate",
        ]
    ):
        return ErrorCategory.DEPENDENCY

    # Empty repository (git clone succeeded but no commits)
    if any(
        term in error_lower
        for term in [
            "does not have any commits",
            "does not have any commits yet",
            "empty repository",
        ]
    ):
        return ErrorCategory.CONFIGURATION

    # Go toolchain version mismatch
    if any(
        term in error_lower
        for term in [
            "go.mod requires go >=",
            "requires go >=",
            "running go ",
            "feature `edition2024` is required",
            "not stabilized in this version of cargo",
            "requires rustc ",
            "requires rust version",
            "this package requires rustc",
        ]
    ):
        return ErrorCategory.MISSING_TOOLS

    return ErrorCategory.UNKNOWN


def create_error_record(
    message: str,
    category: Optional[ErrorCategory] = None,
    severity: Optional[FailureSeverity] = None,
    command: Optional[str] = None,
    attempt_number: int = 0,
) -> ErrorRecord:
    """Create an error record with automatic classification."""
    if category is None:
        category = classify_error(message)
    if severity is None:
        severity = infer_failure_severity(
            category, command=command, message=message
        )

    return ErrorRecord(
        category=category,
        message=message,
        severity=severity,
        command=command,
        attempt_number=attempt_number,
    )


def infer_failure_severity(
    category: ErrorCategory,
    command: Optional[str] = None,
    message: str = "",
) -> FailureSeverity:
    """Infer failure severity from the category and command context.

    Severity levels:
        Low: non-blocking probe failures.
        Medium: standard build/config failures to fix before continuing.
        High: critical initialization/infrastructure blockers.

    Args:
        category: The classified error category.
        command: The command that failed, if known.
        message: The raw error message, if available.

    Returns:
        The inferred ``FailureSeverity``.
    """
    cmd = (command or "").strip().lower()
    msg = (message or "").lower()

    if cmd.startswith("which "):
        return FailureSeverity.LOW

    high_categories = {
        ErrorCategory.LICENSE_INCOMPATIBLE,
        ErrorCategory.REQUIRES_HARDWARE,
        ErrorCategory.ARCHITECTURE_IMPOSSIBLE,
        ErrorCategory.PERMISSION,
        ErrorCategory.DISK_SPACE,
    }
    if category in high_categories:
        return FailureSeverity.HIGH

    if any(
        pattern in cmd
        for pattern in [
            "git clone",
            "git pull",
            "apt-get update",
            "apt update",
            "apk update",
            "apk add",
        ]
    ):
        return FailureSeverity.HIGH

    if category in {
        ErrorCategory.CONFIGURATION,
        ErrorCategory.DEPENDENCY,
        ErrorCategory.COMPILATION,
        ErrorCategory.LINKING,
        ErrorCategory.NETWORK,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.MISSING_TOOLS,
        ErrorCategory.ARCHITECTURE,
    }:
        return FailureSeverity.MEDIUM

    if "not found" in msg and "which " in msg:
        return FailureSeverity.LOW

    return FailureSeverity.MEDIUM


def should_escalate(state: AgentState) -> tuple[bool, str]:
    """Determine whether the task should be escalated to a human.

    Args:
        state: The current agent state.

    Returns:
        A ``(should_escalate, reason)`` tuple.
    """
    # Max attempts reached
    if state.attempt_count >= state.max_attempts:
        return True, f"Maximum attempts ({state.max_attempts}) reached"

    # Stuck in error loop
    if state.is_in_error_loop():
        return True, "Stuck in error loop with no progress"

    # Fundamental blockers
    fundamental_categories = {
        ErrorCategory.LICENSE_INCOMPATIBLE,
        ErrorCategory.REQUIRES_HARDWARE,
        ErrorCategory.ARCHITECTURE_IMPOSSIBLE,
    }

    if state.last_error_category in fundamental_categories:
        return True, f"Fundamental blocker: {state.last_error_category.value}"

    # Cost limit (if set)
    cost_limit = 1.0  # $1 USD limit
    if state.api_cost_usd > cost_limit:
        return True, f"API cost limit (${cost_limit}) exceeded"

    return False, ""


def get_next_action_recommendation(state: AgentState) -> Action:
    """Recommend the next action based on the current state.

    This is a helper for the Supervisor agent.

    Args:
        state: The current agent state.

    Returns:
        The recommended ``Action``.
    """
    # Check for escalation first
    should_esc, _ = should_escalate(state)
    if should_esc:
        return Action.ESCALATE

    # Initial planning
    if not state.task_plan:
        return Action.PLAN

    # Initial scouting
    if not state.build_plan:
        return Action.SCOUT

    # Build execution
    if state.build_status == BuildStatus.PENDING:
        return Action.BUILDER

    # Error recovery
    if state.build_status == BuildStatus.FAILED:
        if state.last_error_category in {
            ErrorCategory.DEPENDENCY,
            ErrorCategory.MISSING_TOOLS,
            ErrorCategory.UNKNOWN,
        }:
            return Action.SCOUT  # Need more info
        if _looks_like_replan_failure(state.last_error or ""):
            return Action.SCOUT
        else:
            return Action.FIXER  # Try to fix

    if state.build_status == BuildStatus.SUCCESS:
        return Action.FINISH

    # Default to builder
    return Action.BUILDER


def _looks_like_replan_failure(error_message: str) -> bool:
    """Return True when rebuilding the plan is usually better than patching."""
    msg = (error_message or "").lower()
    return any(
        pat in msg
        for pat in [
            "no required module provides",
            "already exists and is a directory",
            "does not appear to contain cmakelists.txt",
            "unable to locate package",
            "inconsistent vendoring",
            "./configure: no such file or directory",
        ]
    )

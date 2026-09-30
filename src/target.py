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

"""Execution target and SSH settings for native RISC-V runs.

Atesor builds under QEMU on the local computer (the ``qemu`` target,
the default) or on a real riscv64 machine that the user owns (the
``native`` target). This module reads the target and the SSH settings
from the environment, which ``main.py`` fills from ``.env``, validates
them, and builds the ``ssh`` argv that every native command uses.

Each Atesor process opens its own ssh connection and closes it with
``close_connection()`` when the command ends.

The module imports nothing else from ``src``, so ``src.config`` can use
it at import time.
"""

import glob
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import List, MutableMapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

QEMU = "qemu"
NATIVE = "native"
TARGETS = (QEMU, NATIVE)

FIELD_MODE = "field"
ALIAS_MODE = "alias"

ALIAS_KEY = "ATESOR_SSH_HOST"
FIELD_KEYS = (
    "ATESOR_SSH_HOSTNAME",
    "ATESOR_SSH_PORT",
    "ATESOR_SSH_USER",
    "ATESOR_SSH_IDENTITY_FILE",
)
NATIVE_PLATFORMS = ("alpine", "debian", "ubuntu")
DEFAULT_REMOTE_WORKDIR = "~/atesor-ai"
DEFAULT_SSH_PORT = 22
FIELD_MODE_HOST = "atesor-native"

# Options for every native ssh call. None of them holds a secret, so
# they may appear in argv. ControlPath is added separately because it
# needs the per-user cache directory.
SSH_OPTIONS = (
    "BatchMode=yes",
    "ConnectTimeout=10",
    "ServerAliveInterval=15",
    "ServerAliveCountMax=4",
    "ControlMaster=auto",
    "ControlPersist=10m",
)
# A Unix socket path holds at most 107 bytes, and ssh adds about 17
# characters while it creates the control socket. "%C" expands to 40.
MAX_CONTROL_PATH = 90

# One ssh destination: an alias, a host name, an IPv4 address or
# user@host. No leading "-", so the value can never become an option.
_DESTINATION_RE = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9._-]*(@[A-Za-z0-9_][A-Za-z0-9._-]*)?$"
)
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_PORT_RE = re.compile(r"^[0-9]{1,5}$")
# No whitespace, quotes or "%": ssh would split the line or expand
# "%" tokens in the generated config file.
_PATH_VALUE_RE = re.compile(r"^[^\s\"'%]+$")
_WORKDIR_RE = re.compile(r"^[A-Za-z0-9._/~-]+$")
_SAFE_PHYSICAL_WORKDIR_RE = re.compile(r"^/[A-Za-z0-9._/+-]+$")
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

_cached_config: Optional["TargetConfig"] = None
_resolved_workdir: Optional[str] = None
_machine_name: Optional[str] = None
_written_ssh_config: Optional[str] = None


class TargetConfigError(RuntimeError):
    """Raised when a native-only helper runs without a valid config."""


@dataclass(frozen=True)
class TargetConfig:
    """Validated execution target and SSH settings.

    Attributes:
        name: ``"qemu"`` or ``"native"``. An invalid ``ATESOR_TARGET``
            keeps its raw value here and adds an error.
        errors: Problems found during validation. Preflight rung 1
            reports all of them at once.
        platform: ``ATESOR_PLATFORM``, lower case (native only).
        mode: ``"field"`` or ``"alias"`` on native, else empty.
        ssh_host: The alias-mode destination, as typed after ``ssh``.
        hostname: The field-mode address.
        port: The field-mode SSH port.
        user: The field-mode login user, or empty for the ssh default.
        identity_file: The field-mode private key path, or empty.
        remote_workdir: The work directory on the machine, as given.
    """

    name: str = QEMU
    errors: Tuple[str, ...] = ()
    platform: str = ""
    mode: str = ""
    ssh_host: str = ""
    hostname: str = ""
    port: int = DEFAULT_SSH_PORT
    user: str = ""
    identity_file: str = ""
    remote_workdir: str = DEFAULT_REMOTE_WORKDIR

    @property
    def is_native(self) -> bool:
        """Return True when the native target is selected."""
        return self.name == NATIVE

    @property
    def is_valid(self) -> bool:
        """Return True when validation found no problem."""
        return not self.errors


def load_target(
    environ: Optional[MutableMapping[str, str]] = None,
) -> TargetConfig:
    """Read and validate the execution target from the environment.

    Args:
        environ: Mapping to read instead of ``os.environ``.

    Returns:
        A ``TargetConfig``. Problems go into its ``errors`` field
        instead of an exception, so the preflight can list all of them.
    """
    env = os.environ if environ is None else environ
    raw = env.get("ATESOR_TARGET", "").strip().lower()
    if raw in ("", QEMU):
        return TargetConfig(name=QEMU)
    if raw != NATIVE:
        return TargetConfig(
            name=raw,
            errors=(
                f"ATESOR_TARGET={raw!r} is not valid. Use qemu or native.",
            ),
        )
    return _load_native(env)


def get_target() -> TargetConfig:
    """Return the execution target, read from the environment once.

    Returns:
        The cached ``TargetConfig``.
    """
    global _cached_config
    if _cached_config is None:
        _cached_config = load_target()
    return _cached_config


def reset_target_cache() -> None:
    """Forget the cached target and what the preflight found.

    ``main.py`` calls this after it puts the CLI overrides into the
    environment. Tests call it through an autouse fixture.
    """
    global _cached_config, _resolved_workdir, _written_ssh_config
    global _machine_name
    _cached_config = None
    _resolved_workdir = None
    _written_ssh_config = None
    _machine_name = None


def is_native() -> bool:
    """Return True when the native target is selected."""
    return get_target().is_native


def target_name_from_env(
    environ: Optional[MutableMapping[str, str]] = None,
) -> str:
    """Return ``"native"`` or ``"qemu"`` from ``ATESOR_TARGET``.

    Unlike ``get_target()``, this never validates and never caches, so
    ``src.config`` can call it at import time. Any value other than
    ``native`` gives ``qemu``; preflight rung 1 reports a bad value.

    Args:
        environ: Mapping to read instead of ``os.environ``.

    Returns:
        The target name.
    """
    env = os.environ if environ is None else environ
    value = env.get("ATESOR_TARGET", "").strip().lower()
    return NATIVE if value == NATIVE else QEMU


def apply_target_flag(
    argv: Sequence[str],
    environ: Optional[MutableMapping[str, str]] = None,
) -> None:
    """Copy a ``--target`` flag from argv into ``ATESOR_TARGET``.

    ``main.py`` calls this before it imports ``src.config``, because
    that module computes its paths at import time. The flag wins over
    ``.env``. The last ``--target`` wins, as in argparse.

    Args:
        argv: The command-line arguments, without the program name.
        environ: Mapping to write instead of ``os.environ``.
    """
    env = os.environ if environ is None else environ
    value = None
    for index, arg in enumerate(argv):
        if arg == "--":
            break
        if arg == "--target" and index + 1 < len(argv):
            value = argv[index + 1]
        elif arg.startswith("--target="):
            value = arg.split("=", 1)[1]
    if value is not None:
        env["ATESOR_TARGET"] = value


def cache_dir() -> str:
    """Return the per-user cache directory for the native SSH files.

    Returns:
        ``$XDG_CACHE_HOME/atesor-ai``, or ``~/.cache/atesor-ai``.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "atesor-ai")


def control_dir() -> str:
    """Return the directory for the ssh control sockets.

    ``$XDG_RUNTIME_DIR`` is short and private to the user, so it holds
    the sockets when it exists. Otherwise the cache directory does.

    Returns:
        ``$XDG_RUNTIME_DIR/atesor-ai``, or ``cache_dir()``.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if runtime and os.path.isdir(runtime):
        return os.path.join(runtime, "atesor-ai")
    return cache_dir()


def control_path() -> str:
    """Return the ssh ``ControlPath`` value, with its ``%C`` token.

    The name holds the process ID, so each Atesor process has its own
    connection. One process can then close its connection without
    stopping the commands of another process, such as a batch worker.
    """
    return os.path.join(control_dir(), f"cm-{os.getpid()}-%C")


def close_connection() -> None:
    """Close the ssh connections that this process opened.

    Only the sockets of this process are closed. The call never raises,
    because it runs while the command ends.
    """
    pattern = os.path.join(control_dir(), f"cm-{os.getpid()}-*")
    for path in glob.glob(pattern):
        # -F /dev/null: the user's ssh config plays no part in a close.
        argv = ["ssh", "-F", os.devnull, "-o", f"ControlPath={path}"]
        argv.extend(["-O", "exit", FIELD_MODE_HOST])
        try:
            result = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(f"Could not close the ssh connection: {exc}")
            continue
        if result.returncode != 0:
            logger.warning(
                "Could not close the ssh connection: "
                f"{redact(result.stderr.strip())}"
            )


def control_path_fits() -> bool:
    """Return True when the expanded control socket path fits.

    Returns:
        True when the path, with ``%C`` expanded to 40 characters, is at
        most ``MAX_CONTROL_PATH`` bytes long.
    """
    expanded = len(control_path().encode()) - len("%C") + 40
    return expanded <= MAX_CONTROL_PATH


def render_ssh_config(config: TargetConfig) -> str:
    """Return the text of the generated field-mode SSH config.

    Args:
        config: A valid field-mode configuration.

    Returns:
        One ``Host`` block for ``atesor-native``.
    """
    lines = [
        f"Host {FIELD_MODE_HOST}",
        f"    HostName {config.hostname}",
        f"    Port {config.port}",
    ]
    if config.user:
        lines.append(f"    User {config.user}")
    if config.identity_file:
        lines.append(f"    IdentityFile {config.identity_file}")
        lines.append("    IdentitiesOnly yes")
    lines.append("    StrictHostKeyChecking accept-new")
    return "\n".join(lines) + "\n"


def write_ssh_config(config: TargetConfig) -> str:
    """Write the field-mode SSH config with mode 0600.

    The directory gets mode 0700. The file is rewritten each time, so
    a changed ``.env`` takes effect on the next run.

    Args:
        config: A valid field-mode configuration.

    Returns:
        The path of the written file.
    """
    directory = _ensure_private_dir(cache_dir())
    path = os.path.join(directory, "ssh_config")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(render_ssh_config(config))
    return path


def ssh_argv() -> List[str]:
    """Return ``ssh`` and its options for every native call.

    The list holds no destination: append ``ssh_alias()`` and the
    remote command. Field mode adds ``-F`` with the generated config,
    which the first call writes.

    Returns:
        The argv prefix.

    Raises:
        TargetConfigError: If the native target is not selected or its
            configuration is not valid.
    """
    global _written_ssh_config
    config = _require_valid_native()
    _ensure_private_dir(control_dir())
    argv = ["ssh"]
    for option in SSH_OPTIONS:
        argv.extend(["-o", option])
    argv.extend(["-o", "ControlPath=" + control_path()])
    if config.mode == FIELD_MODE:
        if _written_ssh_config is None:
            _written_ssh_config = write_ssh_config(config)
        argv.extend(["-F", _written_ssh_config])
    return argv


def ssh_alias() -> str:
    """Return the ssh destination for native calls.

    Returns:
        The alias-mode value, or ``atesor-native`` in field mode.

    Raises:
        TargetConfigError: If the native target is not selected or its
            configuration is not valid.
    """
    config = _require_valid_native()
    if config.mode == ALIAS_MODE:
        return config.ssh_host
    return FIELD_MODE_HOST


def remote_workdir() -> Optional[str]:
    """Return the absolute work directory that preflight rung 10 found.

    Returns:
        The physical path on the machine, or None before rung 10 runs.
    """
    return _resolved_workdir


def set_remote_workdir(path: str) -> None:
    """Record the absolute work directory that preflight rung 10 found.

    Args:
        path: The physical path on the machine.
    """
    global _resolved_workdir
    _resolved_workdir = path


def is_safe_remote_workdir(path: str) -> bool:
    """Return True when a physical remote workdir is shell-safe."""
    return bool(_SAFE_PHYSICAL_WORKDIR_RE.match(path))


def set_machine_name(name: str) -> None:
    """Record the host name that preflight rung 5 found.

    Args:
        name: The output of ``uname -n`` on the machine.
    """
    global _machine_name
    _machine_name = name


def machine_label() -> str:
    """Return a name for the native machine, for console messages.

    Alias mode shows the ssh destination, and also the host name of the
    machine when rung 5 found one that differs. Field mode keeps the
    address private, so it shows only the host name, through redact().

    Returns:
        For example ``cloudv@megrez-32g-1 (host rockos-eswin)``.
    """
    config = get_target()
    name = redact(_machine_name) if _machine_name else ""
    if config.mode == ALIAS_MODE:
        host = config.ssh_host.rpartition("@")[2]
        if name and name != host:
            return f"{config.ssh_host} (host {name})"
        return config.ssh_host
    return f"host {name}" if name else "the native machine"


def remote_path_expr(path: str) -> str:
    """Return a remote shell expression for a work-directory path.

    A leading ``~`` becomes the remote ``$HOME``. The rest is quoted,
    so the path reaches the remote shell as one word.

    Args:
        path: A path as ``ATESOR_REMOTE_WORKDIR`` gives it.

    Returns:
        A POSIX shell word.
    """
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def redact(text: str) -> str:
    """Hide the field-mode host and user, and any IPv4 address.

    Field mode keeps the address of the machine private, so ssh error
    text goes through this function before Atesor prints or logs it.

    Args:
        text: Text that can contain connection details.

    Returns:
        The text with those details replaced by placeholders.
    """
    config = get_target()
    if config.mode == FIELD_MODE:
        if config.hostname:
            text = re.sub(
                rf"\b{re.escape(config.hostname)}\b",
                "<ATESOR_SSH_HOSTNAME>",
                text,
            )
        if config.user:
            text = re.sub(
                rf"\b{re.escape(config.user)}\b",
                "<ATESOR_SSH_USER>",
                text,
            )
    return _IPV4_RE.sub("<address>", text)


def _load_native(env: MutableMapping[str, str]) -> TargetConfig:
    """Validate the native keys and return the native configuration."""
    errors: List[str] = []
    platform = env.get("ATESOR_PLATFORM", "").strip().lower()
    if platform not in NATIVE_PLATFORMS:
        errors.append(
            "Set ATESOR_PLATFORM to alpine, debian or ubuntu, or pass "
            "--platform. The native target does not detect it."
        )

    values = {
        key: env.get(key, "").strip()
        for key in env
        if key.startswith("ATESOR_SSH_")
    }
    known = set(FIELD_KEYS) | {ALIAS_KEY}
    for key in sorted(values):
        if values[key] and key not in known:
            errors.append(
                f"{key} is not a known key. Atesor supports key login "
                "only, so remove it from .env."
            )

    alias = values.get(ALIAS_KEY, "")
    field_set = any(values.get(key) for key in FIELD_KEYS)
    mode = ""
    hostname = user = identity = ""
    port = DEFAULT_SSH_PORT
    if alias and field_set:
        errors.append(
            "Set ATESOR_SSH_HOST (alias mode) or the ATESOR_SSH_HOSTNAME "
            "keys (field mode), not both."
        )
    elif not alias and not field_set:
        errors.append(
            "Set ATESOR_SSH_HOST (alias mode) or ATESOR_SSH_HOSTNAME "
            "(field mode) in .env."
        )
    elif alias:
        mode = ALIAS_MODE
        if not _DESTINATION_RE.match(alias):
            errors.append(
                "ATESOR_SSH_HOST must be one ssh destination: an alias, a "
                "host name or user@host, with no spaces, no leading '-' "
                "and no shell characters."
            )
    else:
        mode = FIELD_MODE
        hostname = values.get("ATESOR_SSH_HOSTNAME", "")
        user = values.get("ATESOR_SSH_USER", "")
        identity = values.get("ATESOR_SSH_IDENTITY_FILE", "")
        port, field_errors = _check_field_values(
            hostname, values.get("ATESOR_SSH_PORT", ""), user, identity
        )
        errors.extend(field_errors)

    workdir = (
        env.get("ATESOR_REMOTE_WORKDIR", "").strip() or DEFAULT_REMOTE_WORKDIR
    )
    if not _WORKDIR_RE.match(workdir):
        errors.append(
            "ATESOR_REMOTE_WORKDIR may use only letters, digits and the "
            "characters . _ / ~ -"
        )

    return TargetConfig(
        name=NATIVE,
        errors=tuple(errors),
        platform=platform,
        mode=mode,
        ssh_host=alias,
        hostname=hostname,
        port=port,
        user=user,
        identity_file=identity,
        remote_workdir=workdir,
    )


def _check_field_values(
    hostname: str, port_text: str, user: str, identity: str
) -> Tuple[int, List[str]]:
    """Validate the field-mode values and return the port and errors."""
    errors: List[str] = []
    if not hostname:
        errors.append("Field mode needs ATESOR_SSH_HOSTNAME.")
    elif not _HOSTNAME_RE.match(hostname):
        errors.append(
            "ATESOR_SSH_HOSTNAME must be a host name or an address, with "
            "no spaces and no leading '-'."
        )
    port = DEFAULT_SSH_PORT
    if port_text:
        if _PORT_RE.match(port_text) and 1 <= int(port_text) <= 65535:
            port = int(port_text)
        else:
            errors.append(
                "ATESOR_SSH_PORT must be an integer from 1 to 65535."
            )
    if user and not _USER_RE.match(user):
        errors.append("ATESOR_SSH_USER must be a plain user name.")
    if identity and not _PATH_VALUE_RE.match(identity):
        errors.append(
            "ATESOR_SSH_IDENTITY_FILE must be a path with no spaces, "
            "quotes or '%'."
        )
    return port, errors


def _require_valid_native() -> TargetConfig:
    """Return the config, or raise when it is not a valid native one."""
    config = get_target()
    if not config.is_native:
        raise TargetConfigError("The native target is not selected.")
    if not config.is_valid:
        raise TargetConfigError(
            "The native configuration is not valid: " + " ".join(config.errors)
        )
    return config


def _ensure_private_dir(path: str) -> str:
    """Create a directory with mode 0700 and return its path."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)
    return path

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

"""Native preflight: the machine checks that run before native work.

The ladder has 11 rungs. Rungs 1 to 4 stop at the first failure,
because Atesor cannot inspect the machine without a valid
configuration, the local tools and a working login. Rungs 5 to 11 all
run, and the report ends with one list of everything that the machine
still needs.

Atesor assumes that the machine is ready. It never installs anything
and never runs ``sudo``; it only prints the command that provides each
missing item. Rung 10 creates the work directory. Rung 8 runs
``podman info``, so on the first run rootless podman creates its
storage directory and starts its pause process (``catatonit -P``).
Podman keeps both after the run.
"""

import logging
import os
import posixpath
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import target

logger = logging.getLogger(__name__)

LAST_RUNG = 11
CLEANUP_LAST_RUNG = 8
# Rungs up to this number stop the ladder when they fail.
LAST_STOPPING_RUNG = 4
MIN_PODMAN_MAJOR = 4
# 10 GB each for podman storage and the work directory, in KB.
MIN_FREE_KB = 10 * 1024 * 1024
REMOTE_TIMEOUT = 60
PREFLIGHT_COMMAND = "atesor-ai --target native --preflight"

# Rung 4 sends this text through the remote login shell. It holds
# spaces, both quotes, a backslash, "$", "!" and "*", which a non-POSIX
# shell such as fish or csh changes.
LOGIN_PROBE = "atesor probe 'q' \"d\" \\b $HOME !x *"

RUNG_TITLES = {
    1: "Configuration",
    2: "Local tools",
    3: "SSH config",
    4: "Login and shell",
    5: "Architecture",
    6: "Real hardware",
    7: "Podman",
    8: "Rootless IDs",
    9: "Remote rsync",
    10: "Work directory",
    11: "Disk",
}

_SUBID_FIX = (
    "sudo usermod --add-subuids 100000-165535 "
    "--add-subgids 100000-165535 <login user>"
)
_WORKDIR_FIX = (
    "set ATESOR_REMOTE_WORKDIR in .env to a dedicated directory, "
    "for example ~/atesor-ai"
)
_DISK_FIX = (
    "free space, or put podman storage on another disk "
    "(graphroot in ~/.config/containers/storage.conf)"
)

# A paint function gives one piece of report text a console color:
# paint(text, color). main.py passes termcolor.colored.
Paint = Callable[[str, str], str]


def _no_paint(text: str, color: str) -> str:
    """Return the text without a color."""
    return text


@dataclass(frozen=True)
class Missing:
    """One thing that the machine or the setup still needs.

    Attributes:
        item: What is missing, in a few words.
        fix: The command or action that provides it, or empty when the
            item text already says what to do.
    """

    item: str
    fix: str = ""


@dataclass
class RungResult:
    """The outcome of one rung.

    Attributes:
        number: The rung number, 1 to 11.
        passed: True when the check passed.
        detail: What the rung saw, with connection details redacted.
        missing: The items that the rung found missing.
    """

    number: int
    passed: bool
    detail: str = ""
    missing: List[Missing] = field(default_factory=list)

    @property
    def title(self) -> str:
        """Return the short name of the rung."""
        return RUNG_TITLES[self.number]


@dataclass
class PreflightReport:
    """The results of one preflight run.

    Attributes:
        results: One result per rung that ran, in order.
    """

    results: List[RungResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True when every rung that ran passed."""
        return all(result.passed for result in self.results)

    @property
    def stopped_at(self) -> Optional[int]:
        """Return the rung that stopped the ladder, or None."""
        for result in self.results:
            if not result.passed and result.number <= LAST_STOPPING_RUNG:
                return result.number
        return None

    def missing_items(self) -> List[Missing]:
        """Return every missing item once, in rung order.

        Returns:
            The missing items, without duplicates.
        """
        seen = set()
        items: List[Missing] = []
        for result in self.results:
            for entry in result.missing:
                if entry.item not in seen:
                    seen.add(entry.item)
                    items.append(entry)
        return items

    def render(self, verbose: bool = False, paint: Paint = _no_paint) -> str:
        """Return the report as text for the console.

        Only failures are red: each failed rung line and the failed or
        stopped verdict. A passed rung gets a green mark, and a passed
        verdict is green. The missing items and the notes have no color.

        Args:
            verbose: If True, also list each rung and what it saw, then
                an empty line.
            paint: Gives one piece of text a color, for example
                ``termcolor.colored``. The default adds no color.

        Returns:
            The report text.
        """
        if not self.results:
            return ""
        lines: List[str] = []
        if verbose:
            lines.extend(_rung_line(result, paint) for result in self.results)
            lines.append("")
        first = self.results[0].number
        last = self.results[-1].number
        if self.passed:
            lines.append(
                paint(
                    f"Native preflight passed: rungs {first} to {last}.",
                    "green",
                )
            )
            return "\n".join(lines)
        stopped = self.stopped_at
        if stopped is not None:
            lines.append(
                paint(
                    f"Native preflight stopped at rung {stopped} "
                    f"({RUNG_TITLES[stopped]}). Fix this first:",
                    "red",
                )
            )
            lines.extend(_format_items(self.missing_items()))
            lines.append(f"Then run: {PREFLIGHT_COMMAND}")
            return "\n".join(lines)
        lines.append(
            paint(
                "Native preflight failed. Still required on the machine:",
                "red",
            )
        )
        lines.extend(_format_items(self.missing_items()))
        lines.append(
            "Atesor does not install or change these. The user or the "
            "machine provider"
        )
        lines.append(f"must fix them, then run: {PREFLIGHT_COMMAND}")
        return "\n".join(lines)


@dataclass
class _Facts:
    """Facts that one rung finds and a later rung uses."""

    podman_ok: bool = False
    graph_root: str = ""
    workdir: str = ""


def run_preflight(first: int = 1, last: int = LAST_RUNG) -> PreflightReport:
    """Run the rungs from ``first`` to ``last`` and return the report.

    Rungs 1 to 4 stop the ladder at the first failure. Rungs 5 and up
    all run. Rungs 2 and up need a valid native configuration, so the
    caller runs rung 1 first.

    Args:
        first: The first rung to run.
        last: The last rung to run: 8 for cleanup, 11 otherwise.

    Returns:
        The report, with one result per rung that ran.
    """
    report = PreflightReport()
    facts = _Facts()
    for number in range(first, last + 1):
        result = _RUNGS[number](facts)
        report.results.append(result)
        logger.info(
            "Preflight rung %d (%s): %s%s",
            number,
            result.title,
            "pass" if result.passed else "FAIL",
            f" - {result.detail}" if result.detail else "",
        )
        if not result.passed and number <= LAST_STOPPING_RUNG:
            break
    return report


def _rung_line(result: RungResult, paint: Paint) -> str:
    """Format one rung for the verbose list.

    A failed rung is red from its mark to the end of the line. A passed
    rung has a green mark and no color on its text.
    """
    text = f"{result.number:>2} {result.title}"
    if result.detail:
        text += f": {result.detail}"
    if result.passed:
        return "  " + paint("[  ok]", "green") + " " + text
    return "  " + paint("[FAIL] " + text, "red")


def _format_items(items: List[Missing]) -> List[str]:
    """Format missing items as aligned console lines."""
    width = min(max((len(entry.item) for entry in items), default=0), 40)
    lines = []
    for entry in items:
        if entry.fix:
            lines.append(f"  - {entry.item:<{width}}  fix: {entry.fix}")
        else:
            lines.append(f"  - {entry.item}")
    return lines


def _remote(
    script: str, timeout: int = REMOTE_TIMEOUT
) -> subprocess.CompletedProcess:
    """Run one shell script on the machine over ssh.

    The script reaches the remote login shell as one argv element.
    Stdin is closed, so ssh never forwards the local terminal.

    Args:
        script: A POSIX shell script.
        timeout: Seconds before the call counts as a failed connection.

    Returns:
        The completed process. A timeout gives exit status 255, like an
        ssh connection error.
    """
    argv = target.ssh_argv() + [target.ssh_alias(), script]
    try:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            argv, 255, "", "ssh: the preflight call timed out"
        )


def _last_line(text: str) -> str:
    """Return the last non-empty line of text, redacted."""
    lines = [line.strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    return target.redact(lines[-1]) if lines else ""


def _connection_lost(number: int, completed) -> RungResult:
    """Return a failed result for an ssh error after rung 4 passed."""
    return RungResult(
        number,
        False,
        detail=_last_line(completed.stderr),
        missing=[
            Missing(
                "a stable SSH connection to the machine",
                "check the network, then run the preflight again",
            )
        ],
    )


def _ssh_hint() -> str:
    """Return the ssh command that reaches the machine, redacted."""
    config = target.get_target()
    if config.mode == target.ALIAS_MODE:
        return f"ssh {config.ssh_host}"
    user = "<ATESOR_SSH_USER>@" if config.user else ""
    return f"ssh -p <ATESOR_SSH_PORT> {user}<ATESOR_SSH_HOSTNAME>"


def _copy_id_hint() -> str:
    """Return the ssh-copy-id command for the machine, redacted."""
    return _ssh_hint().replace("ssh ", "ssh-copy-id ", 1)


def _rung_configuration(facts: _Facts) -> RungResult:
    """Rung 1: validate --target and the .env keys."""
    config = target.get_target()
    if not config.is_valid:
        return RungResult(
            1,
            False,
            missing=[Missing(error) for error in config.errors],
        )
    if not config.is_native:
        return RungResult(1, True, detail="qemu target")
    # Field mode keeps the address private, so only alias mode names it.
    mode = f"{config.mode} mode"
    if config.mode == target.ALIAS_MODE:
        mode += f" ({config.ssh_host})"
    return RungResult(
        1, True, detail=f"native, {mode}, platform {config.platform}"
    )


def _rung_local_tools(facts: _Facts) -> RungResult:
    """Rung 2: ssh and rsync exist on the local computer."""
    missing = []
    if shutil.which("ssh") is None:
        missing.append(
            Missing(
                "ssh on the local computer",
                "sudo apt install openssh-client",
            )
        )
    if shutil.which("rsync") is None:
        missing.append(
            Missing("rsync on the local computer", "sudo apt install rsync")
        )
    return RungResult(2, not missing, missing=missing)


def _rung_ssh_config(facts: _Facts) -> RungResult:
    """Rung 3: the control socket path, the key file and the config."""
    config = target.get_target()
    if not target.control_path_fits():
        # ssh refuses a control socket path over the Unix limit, and its
        # error would look like an unreachable machine in rung 4.
        return RungResult(
            3,
            False,
            detail=f"{target.control_path()} is too long for a socket",
            missing=[
                Missing(
                    "a short directory for the ssh control socket",
                    "set XDG_RUNTIME_DIR, or a shorter XDG_CACHE_HOME",
                )
            ],
        )
    if config.mode == target.ALIAS_MODE:
        return RungResult(3, True, detail="alias mode; rung 4 tests login")
    if config.identity_file:
        path = os.path.expanduser(config.identity_file)
        if not os.path.isfile(path):
            return RungResult(
                3,
                False,
                missing=[
                    Missing(
                        "the private key in ATESOR_SSH_IDENTITY_FILE",
                        "create the key, or fix the path in .env",
                    )
                ],
            )
        if os.stat(path).st_mode & 0o077:
            return RungResult(
                3,
                False,
                missing=[
                    Missing(
                        "a private key with mode 0600 or stricter",
                        "chmod 600 " + shlex.quote(config.identity_file),
                    )
                ],
            )
    try:
        target.ssh_argv()
    except OSError as error:
        return RungResult(
            3,
            False,
            detail=str(error),
            missing=[
                Missing(
                    f"a writable {target.cache_dir()} directory",
                    "fix the permissions of that directory",
                )
            ],
        )
    return RungResult(3, True, detail="field mode; SSH config written")


def _rung_login(facts: _Facts) -> RungResult:
    """Rung 4: key login works and the login shell keeps quoting."""
    completed = _remote("printf %s " + shlex.quote(LOGIN_PROBE))
    if completed.returncode == 0 and completed.stdout == LOGIN_PROBE:
        return RungResult(4, True, detail="key login, POSIX login shell")
    return RungResult(
        4,
        False,
        detail=_last_line(completed.stderr),
        missing=[_login_failure(completed)],
    )


def _login_failure(completed) -> Missing:
    """Return the missing item that explains a failed login probe."""
    stderr = completed.stderr or ""
    if completed.returncode != 255:
        return Missing(
            "a POSIX login shell: bash, dash, zsh or ksh",
            "chsh -s /bin/bash, run on the machine as the login user",
        )
    if "REMOTE HOST IDENTIFICATION HAS CHANGED" in stderr:
        return Missing(
            "a matching host key for the machine",
            "ssh-keygen -R <host>, after you confirm that the machine "
            "was reinstalled",
        )
    if "Host key verification failed" in stderr:
        return Missing(
            "a known host key for the machine",
            f"connect once by hand ({_ssh_hint()}) and accept the key",
        )
    if "Permission denied" in stderr:
        return Missing("SSH key login to the machine", _copy_id_hint())
    if "Could not resolve hostname" in stderr:
        return Missing(
            "a host name that resolves",
            "fix ATESOR_SSH_HOST or ATESOR_SSH_HOSTNAME in .env",
        )
    return Missing(
        "a reachable SSH server on the machine",
        f"check the address, the port and the network ({_ssh_hint()})",
    )


def _rung_architecture(facts: _Facts) -> RungResult:
    """Rung 5: the machine reports riscv64, and its host name."""
    completed = _remote("uname -n -m")
    if completed.returncode == 255:
        return _connection_lost(5, completed)
    # uname prints the host name first and the architecture last.
    words = completed.stdout.split()
    arch = words[-1] if words else ""
    host = words[0] if len(words) > 1 else ""
    if completed.returncode == 0 and arch == "riscv64":
        if not host:
            return RungResult(5, True, detail="riscv64")
        target.set_machine_name(host)
        return RungResult(
            5, True, detail=f"riscv64, host {target.redact(host)}"
        )
    return RungResult(
        5,
        False,
        detail=f"found {arch or 'nothing'}",
        missing=[
            Missing(
                "a riscv64 machine",
                "point the SSH settings in .env at a riscv64 machine",
            )
        ],
    )


def _rung_real_hardware(facts: _Facts) -> RungResult:
    """Rung 6: the machine is not a QEMU ``virt`` VM (heuristic)."""
    completed = _remote(
        "tr '\\0' ' ' < /proc/device-tree/compatible 2>/dev/null; echo; "
        "grep -m1 '^mvendorid' /proc/cpuinfo 2>/dev/null; true"
    )
    if completed.returncode == 255:
        return _connection_lost(6, completed)
    lines = completed.stdout.splitlines()
    compatible = lines[0] if lines else ""
    vendor = 0
    for line in lines[1:]:
        if line.startswith("mvendorid"):
            try:
                vendor = int(line.split(":", 1)[-1].strip(), 16)
            except ValueError:
                vendor = 0
    detail = f"mvendorid {vendor:#x}"
    if "riscv-virtio" in compatible and vendor == 0:
        return RungResult(
            6,
            False,
            detail=detail + ", QEMU virt board",
            missing=[
                Missing(
                    "real RISC-V hardware, not a QEMU virt VM",
                    "ask the provider for a machine on real hardware",
                )
            ],
        )
    return RungResult(6, True, detail=detail)


def _rung_podman(facts: _Facts) -> RungResult:
    """Rung 7: podman 4.0 or newer is installed."""
    completed = _remote("podman --version")
    if completed.returncode == 255:
        return _connection_lost(7, completed)
    version = completed.stdout.strip()
    match = re.search(r"(\d+)\.(\d+)", version)
    if (
        completed.returncode == 0
        and match
        and int(match.group(1)) >= MIN_PODMAN_MAJOR
    ):
        facts.podman_ok = True
        return RungResult(7, True, detail=version)
    detail = "not installed" if completed.returncode else f"found {version}"
    return RungResult(
        7,
        False,
        detail=detail,
        missing=[Missing("podman 4.0 or newer", "sudo apt install podman")],
    )


def _rung_rootless_ids(facts: _Facts) -> RungResult:
    """Rung 8: newuidmap, subordinate IDs and rootless podman."""
    completed = _remote(
        "u=$(id -un); "
        'printf "newuidmap=%s\\n" "$(command -v newuidmap)"; '
        'printf "subuid=%s\\n" "$(grep -c "^$u:" /etc/subuid 2>/dev/null)"; '
        'printf "subgid=%s\\n" "$(grep -c "^$u:" /etc/subgid 2>/dev/null)"'
    )
    if completed.returncode == 255:
        return _connection_lost(8, completed)
    values = _key_values(completed.stdout)
    missing = []
    if not values.get("newuidmap"):
        missing.append(
            Missing("uidmap (newuidmap)", "sudo apt install uidmap")
        )
    if not _positive(values.get("subuid")) or not _positive(
        values.get("subgid")
    ):
        missing.append(
            Missing(
                "subordinate UID and GID ranges for the login user",
                _SUBID_FIX,
            )
        )
    detail = "podman not installed, rootless mode not checked"
    if facts.podman_ok:
        detail, rootless_missing = _check_rootless(facts, bool(missing))
        missing.extend(rootless_missing)
    return RungResult(8, not missing, detail=detail, missing=missing)


def _check_rootless(
    facts: _Facts, ids_missing: bool
) -> Tuple[str, List[Missing]]:
    """Ask podman for its mode and storage path; return detail and gaps."""
    completed = _remote(
        "podman info --format "
        "'{{.Host.Security.Rootless}} {{.Store.GraphRoot}}'"
    )
    if completed.returncode != 0:
        if ids_missing:
            # podman info fails without the ID mapping; the items above
            # already name the cause.
            return "podman info failed", []
        return "podman info failed", [
            Missing(
                "a working rootless podman",
                "run podman info on the machine and fix its error",
            )
        ]
    parts = completed.stdout.strip().split(" ", 1)
    rootless = parts[0] if parts else ""
    facts.graph_root = parts[1].strip() if len(parts) > 1 else ""
    if rootless != "true":
        return "podman is not rootless", [
            Missing(
                "rootless podman for the login user",
                "run podman as the login user, not as root",
            )
        ]
    return "rootless podman", []


def _rung_remote_rsync(facts: _Facts) -> RungResult:
    """Rung 9: rsync exists on the machine."""
    completed = _remote("command -v rsync")
    if completed.returncode == 255:
        return _connection_lost(9, completed)
    if completed.returncode == 0 and completed.stdout.strip():
        return RungResult(9, True, detail="rsync present")
    return RungResult(
        9,
        False,
        detail="not installed",
        missing=[Missing("rsync on the machine", "sudo apt install rsync")],
    )


def _rung_workdir(facts: _Facts) -> RungResult:
    """Rung 10: the work directory is safe, created and writable."""
    config = target.get_target()
    completed = _remote('printf "%s" "$HOME"')
    if completed.returncode == 255:
        return _connection_lost(10, completed)
    home = posixpath.normpath(completed.stdout.strip() or "/")
    wanted = _absolute_workdir(config.remote_workdir, home)
    problem = _workdir_problem(wanted, home)
    if problem:
        return RungResult(10, False, missing=[Missing(problem, _WORKDIR_FIX)])
    expr = target.remote_path_expr(config.remote_workdir)
    made = _remote(f"mkdir -p {expr} && cd {expr} && test -w . && pwd -P")
    if made.returncode == 255:
        return _connection_lost(10, made)
    physical = made.stdout.strip()
    if made.returncode != 0 or not physical.startswith("/"):
        return RungResult(
            10,
            False,
            detail=_last_line(made.stderr),
            missing=[
                Missing(
                    f"a writable work directory at {config.remote_workdir}",
                    "create it, or set ATESOR_REMOTE_WORKDIR to a directory "
                    "that the login user owns",
                )
            ],
        )
    problem = _workdir_problem(physical, home)
    if problem:
        return RungResult(10, False, missing=[Missing(problem, _WORKDIR_FIX)])
    facts.workdir = physical
    target.set_remote_workdir(physical)
    return RungResult(10, True, detail=config.remote_workdir)


def _rung_disk(facts: _Facts) -> RungResult:
    """Rung 11: enough free space for podman storage and the workdir."""
    config = target.get_target()
    if facts.graph_root:
        graph_expr = shlex.quote(facts.graph_root)
    else:
        graph_expr = '"$HOME"/.local/share/containers/storage'
    if facts.workdir:
        work_expr = shlex.quote(facts.workdir)
    else:
        work_expr = target.remote_path_expr(config.remote_workdir)
    completed = _remote(
        'free_kb() { p=$1; while [ ! -e "$p" ]; do p=$(dirname "$p"); '
        "done; df -Pk \"$p\" | awk 'NR==2 {print $1, $4}'; }; "
        f'printf "graph=%s\\n" "$(free_kb {graph_expr})"; '
        f'printf "work=%s\\n" "$(free_kb {work_expr})"'
    )
    if completed.returncode == 255:
        return _connection_lost(11, completed)
    values = _key_values(completed.stdout)
    graph_fs, graph_kb = _parse_free(values.get("graph", ""))
    work_fs, work_kb = _parse_free(values.get("work", ""))
    missing = []
    if graph_fs and graph_fs == work_fs:
        detail = f"{_gb(graph_kb)} GB free on one filesystem"
        if graph_kb < 2 * MIN_FREE_KB:
            missing.append(
                Missing(
                    "20 GB free on the filesystem that holds podman "
                    f"storage and the work directory (has {_gb(graph_kb)} "
                    "GB)",
                    _DISK_FIX,
                )
            )
    else:
        detail = (
            f"{_gb(graph_kb)} GB free for podman, "
            f"{_gb(work_kb)} GB in the work directory"
        )
        if graph_kb < MIN_FREE_KB:
            missing.append(
                Missing(
                    f"10 GB free for podman storage (has {_gb(graph_kb)} "
                    "GB)",
                    _DISK_FIX,
                )
            )
        if work_kb < MIN_FREE_KB:
            missing.append(
                Missing(
                    "10 GB free in the work directory (has "
                    f"{_gb(work_kb)} GB)",
                    "free space, or set ATESOR_REMOTE_WORKDIR to a larger "
                    "disk",
                )
            )
    return RungResult(11, not missing, detail=detail, missing=missing)


def _key_values(text: str) -> Dict[str, str]:
    """Parse ``key=value`` lines into a dict."""
    values = {}
    for line in (text or "").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip()
    return values


def _positive(value: Optional[str]) -> bool:
    """Return True when value is a positive integer string."""
    return bool(value) and value.isdigit() and int(value) > 0


def _parse_free(value: str) -> Tuple[str, int]:
    """Parse ``<filesystem> <free KB>`` from df into its two parts."""
    parts = value.split()
    if len(parts) != 2 or not parts[1].isdigit():
        return "", 0
    return parts[0], int(parts[1])


def _gb(kilobytes: int) -> str:
    """Format a KB count as GB with one decimal."""
    return f"{kilobytes / (1024 * 1024):.1f}"


def _absolute_workdir(path: str, home: str) -> str:
    """Return the absolute, normalized form of a work-directory path."""
    if path == "~":
        raw = home
    elif path.startswith("~/"):
        raw = home + path[1:]
    elif path.startswith("/"):
        raw = path
    else:
        # ssh starts the remote command in the home directory.
        raw = posixpath.join(home, path)
    return posixpath.normpath(raw)


def _workdir_problem(path: str, home: str) -> str:
    """Return why a work directory is unsafe, or an empty string.

    Atesor bind-mounts the work directory with SELinux relabeling, so
    it must not be ``$HOME``, a parent of it, or anything in ``.ssh``.
    """
    home = posixpath.normpath(home)
    ssh_dir = posixpath.join(home, ".ssh")
    if path == home:
        return "a work directory other than $HOME"
    if path == "/" or home.startswith(path.rstrip("/") + "/"):
        return "a work directory that is not a parent of $HOME"
    if path == ssh_dir or path.startswith(ssh_dir + "/"):
        return "a work directory outside ~/.ssh"
    return ""


_RUNGS: Dict[int, Callable[[_Facts], RungResult]] = {
    1: _rung_configuration,
    2: _rung_local_tools,
    3: _rung_ssh_config,
    4: _rung_login,
    5: _rung_architecture,
    6: _rung_real_hardware,
    7: _rung_podman,
    8: _rung_rootless_ids,
    9: _rung_remote_rsync,
    10: _rung_workdir,
    11: _rung_disk,
}

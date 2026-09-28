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

"""Local mirror of the native repositories, for host-side reads.

On the native target, the repositories live in the work directory on
the user's machine. Atesor also reads repository files on the local
computer: the scripted analysis, the evidence excerpts, the graph
checks and the packager. The mirror is a local copy of each repository
under ``WORKSPACE_ROOT/repos``, which rsync refreshes from the machine.

The mirror copies in one direction only, from the machine to the local
computer. Each native command can change the tree, so
``execute_command()`` calls ``mark_stale()`` after it. The next
host-side read calls ``pull()`` through ``config.to_host_path()``.
"""

import logging
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import threading
import time
from typing import Dict, Optional

from . import config, target
from .sandbox import SandboxUnavailableError

logger = logging.getLogger(__name__)

# The first copy of a large repository can take a while.
PULL_TIMEOUT = 900
# rsync exit statuses that are only warnings: 23 means that some files
# did not transfer, and 24 means that some source files vanished.
_WARNING_EXITS = (23, 24)
_REPOS_PREFIX = "/workspace/repos/"
# The same names that the clone step accepts, and no leading dot.
_REPO_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
# rsync reports a missing source directory like this.
_MISSING_SOURCE_RE = re.compile(
    r'change_dir "[^"]*" failed: No such file or directory'
)

_lock = threading.RLock()
_generation = 0
_pulled_at: Dict[str, int] = {}


def reset() -> None:
    """Forget the pull state. Tests call it through an autouse fixture."""
    global _generation
    with _lock:
        _generation = 0
        _pulled_at.clear()


def mark_stale() -> None:
    """Record that a native command may have changed the repositories."""
    global _generation
    with _lock:
        _generation += 1


def is_stale(repo: str) -> bool:
    """Return True when a command ran after the last pull of a repo."""
    with _lock:
        return _pulled_at.get(repo) != _generation


def repo_of(path: str) -> Optional[str]:
    """Return the repository name of a container path.

    Args:
        path: A container path, for example ``/workspace/repos/zlib/x.c``.

    Returns:
        ``zlib`` for that example, or None for a path outside
        ``/workspace/repos/<repo>``.
    """
    normal = posixpath.normpath(path)
    if not normal.startswith(_REPOS_PREFIX):
        return None
    name = normal[len(_REPOS_PREFIX) :].split("/", 1)[0]
    return name if _REPO_RE.match(name) else None


def mirror_dir(repo: str) -> str:
    """Return the local mirror directory of a repository."""
    return os.path.join(config.WORKSPACE_ROOT, "repos", repo)


def pull(repo: str, force: bool = False) -> None:
    """Refresh the mirror of one repository from the native machine.

    On the qemu target, the call does nothing, because the container
    mounts the local workspace.

    Args:
        repo: The repository name, as in ``/workspace/repos/<repo>``.
        force: If True, copy even when no command ran since the last
            copy.

    Raises:
        ValueError: If the name is not a plain repository name.
        SandboxUnavailableError: If rsync fails or runs too long.
    """
    if not target.is_native():
        return
    if not _REPO_RE.match(repo):
        raise ValueError(f"Unsafe repository name: {repo!r}")
    with _lock:
        generation = _generation
        if not force and _pulled_at.get(repo) == generation:
            return
        _copy(repo)
        _pulled_at[repo] = generation


def _copy(repo: str) -> None:
    """Run rsync from the machine into the local mirror."""
    workdir = target.remote_workdir()
    if not workdir:
        raise SandboxUnavailableError(
            "The mirror cannot copy, because preflight rung 10 did not "
            "find the work directory."
        )
    local = mirror_dir(repo)
    os.makedirs(local, exist_ok=True)
    # The trailing slashes copy the content of the directory, and
    # --delete removes the local files that the machine no longer has.
    # rsync skips a file when its size and time match, and by default it
    # compares whole seconds only. --modify-window=-1 compares the
    # nanoseconds too, so a same-size edit in the same second is copied.
    argv = ["rsync", "-a", "--delete", "--modify-window=-1"]
    argv.append("--exclude=.git")
    argv.extend(["-e", shlex.join(target.ssh_argv())])
    argv.append(f"{target.ssh_alias()}:{workdir}/repos/{repo}/")
    argv.append(local + "/")
    started = time.time()
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=PULL_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise SandboxUnavailableError(
            f"The mirror copy of {repo} stopped at the time limit "
            f"({PULL_TIMEOUT} s)."
        ) from exc
    seconds = time.time() - started
    stderr = target.redact(result.stderr.strip())
    if result.returncode == 0:
        logger.info(f"Mirror: copied {repo} in {seconds:.1f} s")
        return
    if result.returncode in _WARNING_EXITS:
        if _MISSING_SOURCE_RE.search(result.stderr):
            # The repository is gone on the machine, so the copy goes too.
            shutil.rmtree(local, ignore_errors=True)
            logger.info(
                f"Mirror: {repo} does not exist on the machine; removed "
                "the local copy"
            )
            return
        logger.warning(
            f"Mirror: rsync exit {result.returncode} for {repo}: "
            f"{stderr[:500]}"
        )
        return
    raise SandboxUnavailableError(
        f"The mirror copy of {repo} failed (rsync exit "
        f"{result.returncode}): {stderr[:300]}"
    )


__all__ = [
    "PULL_TIMEOUT",
    "is_stale",
    "mark_stale",
    "mirror_dir",
    "pull",
    "repo_of",
    "reset",
]

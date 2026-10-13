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

"""Environment-aware paths and Docker detection.

The module-level constants (``WORKSPACE_ROOT``, ``REPOS_DIR``, ...) are
computed once at import time from ``ATESOR_HOME``, the active target,
and the Docker context, so ``main.py`` loads ``.env`` and applies
``--target`` before importing this module. Every getter creates its
directory.
"""

import os
from pathlib import Path

from .target import NATIVE, is_native, target_name_from_env


def is_running_in_docker() -> bool:
    """Return True when the process runs inside a container.

    Heuristic: any of ``/.dockerenv``, a container runtime named in
    ``/proc/1/cgroup``, or the sandbox layout (``/workspace`` without
    ``/home``) counts.
    """
    if os.path.exists("/.dockerenv"):
        return True

    # /proc/1/cgroup names the runtime that started PID 1.
    try:
        with open("/proc/1/cgroup", "r") as f:
            for line in f:
                if (
                    "docker" in line
                    or "containerd" in line
                    or "kubepods" in line
                ):
                    return True
    except (FileNotFoundError, PermissionError):
        pass

    # Sandbox-image heuristic: /workspace exists and /home does not.
    if (
        os.path.abspath(os.sep) == "/"
        and os.path.exists("/workspace")
        and not os.path.exists("/home")
    ):
        return True

    return False


def get_state_home() -> str:
    """Return the writable base directory for all runtime state.

    Resolution order: the ``ATESOR_HOME`` environment variable, then
    ``/workspace`` inside the sandbox container, then the source
    checkout root when running from a writable clone, and finally
    ``$XDG_DATA_HOME/atesor-ai`` (default ``~/.local/share/atesor-ai``).
    Keeping mutable state out of the install tree lets the tool run from
    a read-only system location such as ``/opt``.
    """
    override = os.environ.get("ATESOR_HOME")
    if override:
        base = os.path.abspath(os.path.expanduser(override))
    elif is_running_in_docker():
        base = "/workspace"
    else:
        repo_root = Path(__file__).resolve().parent.parent
        is_source = (repo_root / ".git").exists() or (
            repo_root / "tests"
        ).is_dir()
        if is_source and os.access(str(repo_root), os.W_OK):
            base = str(repo_root)
        else:
            xdg = os.environ.get("XDG_DATA_HOME") or os.path.join(
                os.path.expanduser("~"), ".local", "share"
            )
            base = os.path.join(xdg, "atesor-ai")
    os.makedirs(base, exist_ok=True)
    return base


def get_workspace_root() -> str:
    """Get appropriate workspace root based on environment.

    ``ATESOR_HOME`` always wins — including inside a container — so the
    documented override contract of ``get_state_home`` holds for the
    workspace tree too. The bare ``/workspace`` default applies only
    inside the sandbox container, for the qemu target, when no override
    is set. The native target always uses ``workspace-native``, even
    from a container: its repositories live on the machine, and the
    local mirror of them must stay apart from the qemu tree that the
    container keeps directly in ``/workspace``.
    """
    native = target_name_from_env() == NATIVE
    if (
        is_running_in_docker()
        and not native
        and not os.environ.get("ATESOR_HOME")
    ):
        return "/workspace"
    name = "workspace-native" if native else "workspace"
    workspace = os.path.join(get_state_home(), name)
    os.makedirs(workspace, exist_ok=True)
    return workspace


def get_data_dir() -> str:
    """Get the writable directory for examples and the recipe cache."""
    data_dir = os.path.join(get_state_home(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_output_dir() -> str:
    """Return the output directory, creating it if missing."""
    workspace = get_workspace_root()
    output_dir = os.path.join(workspace, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_repos_dir() -> str:
    """Return the cloned-repositories directory, creating it if missing."""
    workspace = get_workspace_root()
    repos_dir = os.path.join(workspace, "repos")
    os.makedirs(repos_dir, exist_ok=True)
    return repos_dir


def get_cache_dir() -> str:
    """Return the cache directory, creating it if missing."""
    workspace = get_workspace_root()
    cache_dir = os.path.join(workspace, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_logs_dir() -> str:
    """Return the logs directory, creating it if missing."""
    workspace = get_workspace_root()
    logs_dir = os.path.join(workspace, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def get_packages_dir() -> str:
    """Return the zip-artifact directory, creating it if missing."""
    workspace = get_workspace_root()
    packages_dir = os.path.join(workspace, "packages")
    os.makedirs(packages_dir, exist_ok=True)
    return packages_dir


# Resolved once at import time; main.py applies .env and --target
# before this module is imported.
_IN_DOCKER = is_running_in_docker()
STATE_HOME = get_state_home()
WORKSPACE_ROOT = get_workspace_root()
OUTPUT_DIR = get_output_dir()
REPOS_DIR = get_repos_dir()
CACHE_DIR = get_cache_dir()
LOGS_DIR = get_logs_dir()
PACKAGES_DIR = get_packages_dir()
DATA_DIR = get_data_dir()

# Defaults matching the Alpine profile; provisioning resolves the
# active names through src.platforms instead.
CONTAINER_NAME = "atesor-ai-sandbox"
IMAGE_NAME = "atesor-ai-sandbox:latest"


def to_host_path(path: str) -> str:
    """Translate a ``/workspace`` container path to a host path.

    Canonical translator — every module that needs host-side access to
    sandbox files must use this (or delegate to it) so container/host
    path mapping stays consistent in one place.

    Args:
        path: A path that may start with the in-container ``/workspace``
            prefix.

    Returns:
        The equivalent host path, or ``path`` unchanged when it is not
        a container path, when it is already under ``WORKSPACE_ROOT``,
        or when ``/workspace`` exists locally on the qemu target (as
        inside the sandbox container).

    Raises:
        SandboxUnavailableError: On native, if the mirror copy fails.
    """
    # Match the path component, not the prefix: "/workspaces/x"
    # (Codespaces) or "/workspace-backup/x" are not container paths.
    if path != "/workspace" and not path.startswith("/workspace/"):
        return path
    # A host path under a workspace root that itself lives in
    # /workspace (native from a container) is already translated.
    if WORKSPACE_ROOT != "/workspace" and (
        path == WORKSPACE_ROOT or path.startswith(WORKSPACE_ROOT + "/")
    ):
        return path
    # Native always needs a mirror refresh and a translation to the
    # native workspace, even from a development container where
    # ``/workspace`` also exists locally: without this, host-side reads
    # would fall through to the local container tree instead of the
    # remote repository, and QEMU/native state would no longer be
    # separated. WORKSPACE_ROOT is ``/workspace`` here only when this
    # module computed it for the qemu target. Then the path stays
    # unchanged, but the mirror still refreshes.
    if is_native():
        refresh_mirror(path)
        if WORKSPACE_ROOT != "/workspace":
            return WORKSPACE_ROOT + path[len("/workspace"):]
        return path
    if not os.path.exists("/workspace"):
        return WORKSPACE_ROOT + path[len("/workspace"):]
    return path


def refresh_mirror(path: str) -> None:
    """Refresh the local copy of the repository that a path is in.

    Only the native target has this copy (the mirror), because only
    there the repositories live on another machine. On qemu, and for a
    path outside ``/workspace/repos/<repo>``, the call does nothing.

    Args:
        path: A container path, for example ``/workspace/repos/zlib/x.c``.

    Raises:
        SandboxUnavailableError: If the mirror copy fails.
    """
    if not is_native():
        return
    # Deferred import: src.mirror imports this module, so importing it
    # at module level would create a cycle.
    from . import mirror

    repo = mirror.repo_of(path)
    if repo:
        mirror.pull(repo)


__all__ = [
    "is_running_in_docker",
    "get_workspace_root",
    "get_output_dir",
    "get_repos_dir",
    "get_cache_dir",
    "get_logs_dir",
    "get_packages_dir",
    "get_state_home",
    "get_data_dir",
    "STATE_HOME",
    "DATA_DIR",
    "WORKSPACE_ROOT",
    "OUTPUT_DIR",
    "REPOS_DIR",
    "CACHE_DIR",
    "LOGS_DIR",
    "PACKAGES_DIR",
    "to_host_path",
    "refresh_mirror",
    "CONTAINER_NAME",
    "IMAGE_NAME",
    "_IN_DOCKER",
]

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

"""Environment configuration and path management.

Detects the Docker context and sets up the workspace directories used
throughout the Atesor AI system.
"""

import os
from pathlib import Path


def is_running_in_docker() -> bool:
    """Detect if running inside a Docker container."""
    # Check for .dockerenv file
    if os.path.exists("/.dockerenv"):
        return True

    # Check cgroup for docker/containerd
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

    # Check if we are at /workspace (standard for our image).
    if (
        os.path.abspath(os.sep) == "/"
        and os.path.exists("/workspace")
        and not os.path.exists("/home")
    ):
        # If /workspace exists and /home is missing, likely a container.
        return True

    return False


def get_state_home() -> str:
    """Return the writable base directory for all runtime state.

    Resolution order: the ``ATESOR_HOME`` environment variable, then
    ``/workspace`` inside the sandbox container, then the source
    checkout root when running from a clone, and finally
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
    workspace tree too (previously the in-docker shortcut silently
    ignored it, sending state to ``/workspace`` in CI/devcontainers).
    The bare ``/workspace`` default applies only inside the sandbox
    container when no override is set.
    """
    if is_running_in_docker() and not os.environ.get("ATESOR_HOME"):
        return "/workspace"
    workspace = os.path.join(get_state_home(), "workspace")
    os.makedirs(workspace, exist_ok=True)
    return workspace


def get_data_dir() -> str:
    """Get the writable directory for examples and the recipe cache."""
    data_dir = os.path.join(get_state_home(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_output_dir() -> str:
    """Get output directory path."""
    workspace = get_workspace_root()
    output_dir = os.path.join(workspace, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_repos_dir() -> str:
    """Get repositories directory path."""
    workspace = get_workspace_root()
    repos_dir = os.path.join(workspace, "repos")
    os.makedirs(repos_dir, exist_ok=True)
    return repos_dir


def get_cache_dir() -> str:
    """Get cache directory path."""
    workspace = get_workspace_root()
    cache_dir = os.path.join(workspace, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_logs_dir() -> str:
    """Get logs directory path."""
    workspace = get_workspace_root()
    logs_dir = os.path.join(workspace, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def get_packages_dir() -> str:
    """Get packages directory path (zip artifacts of successful builds)."""
    workspace = get_workspace_root()
    packages_dir = os.path.join(workspace, "packages")
    os.makedirs(packages_dir, exist_ok=True)
    return packages_dir


# Global configuration
_IN_DOCKER = is_running_in_docker()
STATE_HOME = get_state_home()
WORKSPACE_ROOT = get_workspace_root()
OUTPUT_DIR = get_output_dir()
REPOS_DIR = get_repos_dir()
CACHE_DIR = get_cache_dir()
LOGS_DIR = get_logs_dir()
PACKAGES_DIR = get_packages_dir()
DATA_DIR = get_data_dir()

# Docker Configuration
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
        a container path (or when we ARE running inside the container).
    """
    if path.startswith("/workspace") and not os.path.exists("/workspace"):
        return path.replace("/workspace", WORKSPACE_ROOT, 1)
    return path


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
    "CONTAINER_NAME",
    "IMAGE_NAME",
    "_IN_DOCKER",
]

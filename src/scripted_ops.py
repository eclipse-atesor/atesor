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

"""Deterministic, non-LLM repository analysis operations.

Handles repository cloning, build system detection, and dependency
extraction without any LLM involvement.
"""

import json
import logging
import os
import re
import shlex
import threading
from typing import Any, Dict, List, Optional

from src.state import (
    ArchSpecificCode,
    BuildSystemInfo,
    CommandResult,
    DependencyInfo,
)
from src.tools import execute_command

from .config import CACHE_DIR, REPOS_DIR, WORKSPACE_ROOT

logger = logging.getLogger(__name__)


# ============================================================================
# SCRIPTED OPERATIONS CLASS
# ============================================================================


def _walk_root_is_excluded(root: str, *names: str) -> bool:
    """True when any path COMPONENT of ``root`` equals one of ``names``.

    Substring checks like ``".git" in root`` misfire for repos whose
    own name contains the token (``user.github.io`` contains ``.git``),
    silently disabling the whole walk.
    """
    parts = root.split(os.sep)
    return any(part in names for part in parts)


class ScriptedOperations:
    """Deterministic operations that don't require LLM intelligence."""

    def __init__(self, workspace_root: str = None) -> None:
        """Initialize with environment-aware workspace."""
        if workspace_root is None:
            self.workspace_root = WORKSPACE_ROOT
            self.repos_dir = REPOS_DIR
            self.cache_dir = CACHE_DIR
        else:
            self.workspace_root = workspace_root
            self.repos_dir = os.path.join(workspace_root, "repos")
            self.cache_dir = os.path.join(workspace_root, ".cache")
            try:
                os.makedirs(self.repos_dir, exist_ok=True)
                os.makedirs(self.cache_dir, exist_ok=True)
            except PermissionError as e:
                logger.error(f"Permission denied: {e}")
                raise

        logger.info(f"ScriptedOperations initialized: {self.workspace_root}")

    # Per-clone GitHub auth overlay (see _github_auth_env), stored
    # thread-locally: graph.py shares one ScriptedOperations instance,
    # so concurrent threads cloning different URLs must not observe
    # each other's headers. Class-level so direct helper calls (tests)
    # see an empty overlay by default.
    _auth_tls = threading.local()

    def _github_auth_env(self, url: str) -> Dict[str, str]:
        """Return a per-invocation git-config env for GitHub auth.

        When a ``GIT_TOKEN``/``GH_TOKEN`` is present and ``url`` is a
        GitHub URL, returns ``GIT_CONFIG_COUNT/KEY_0/VALUE_0`` variables
        that inject the bearer header for THAT git invocation only.

        Two safety properties:
          * The config key is URL-matched
            (``http.https://github.com/.extraHeader``), so the header is
            only ever sent to github.com — a clone of an
            attacker-controlled homepage URL never receives the token.
          * Nothing is written to the container's git config, so there
            is no state for a concurrent worker to race (the previous
            set→clone→unset design could strip a sibling's header on a
            shared container) and nothing for later LLM-generated
            commands to read.
        """
        token = os.environ.get("GIT_TOKEN", "") or os.environ.get(
            "GH_TOKEN", ""
        )
        if not token or not url.startswith("https://github.com/"):
            return {}
        return {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraHeader",
            "GIT_CONFIG_VALUE_0": f"Authorization: bearer {token}",
        }

    def _git_env(self) -> Dict[str, str]:
        """Fail-fast git env merged with this thread's auth overlay."""
        return {
            **self._GIT_FAIL_FAST_ENV,
            **getattr(self._auth_tls, "env", {}),
        }

    def _build_clone_cmd(self, url: str, container_repo_path: str) -> str:
        """Return a shell-safe shallow-clone command for ``url``.

        Auth travels out-of-band via ``_git_env()`` (env-injected git
        config), never in the command string.
        """
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"Refusing non-http URL: {url!r}")

        return (
            f"git clone --depth 1 {shlex.quote(url)} "
            f"{shlex.quote(container_repo_path)}"
        )

    def _is_auth_error(self, result: CommandResult) -> bool:
        err = (result.stderr or "").lower()
        return any(
            pat in err
            for pat in [
                "could not read username",
                "could not read password",
                "authentication failed",
                "403 forbidden",
                "401 unauthorized",
                "access denied",
                "no such device or address",
            ]
        )

    # ------------------------------------------------------------------
    # Fail-fast git environment: apply on every git subprocess so a
    # missing credential / hanging SSH prompt exits immediately instead
    # of stalling for 60+ seconds. Cheap & safe to always pass.
    # ------------------------------------------------------------------
    _GIT_FAIL_FAST_ENV: Dict[str, str] = {
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/echo",
        "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o StrictHostKeyChecking=no",
    }

    # ------------------------------------------------------------------
    # Homepage → git URL heuristics. Applied as an ordered chain when
    # the raw URL cannot be cloned. Order matters: most specific first,
    # generic fallbacks last. Every substitution must yield a syntactic
    # git URL (ends with a repo name, optionally `.git`); the actual
    # existence is verified by trying to clone.
    # ------------------------------------------------------------------
    _HOMEPAGE_TO_GIT_RULES = (
        # `https://www.gnu.org/software/<name>/` → savannah
        (
            re.compile(
                r"^https?://(?:www\.)?gnu\.org/software/([A-Za-z0-9_.\-]+)/?$"
            ),
            r"https://git.savannah.gnu.org/git/\1.git",
        ),
        # `https://git.savannah.gnu.org/cgit/<name>.git/` → git/
        (
            re.compile(
                r"^https?://git\.savannah\.gnu\.org/cgit/"
                # Repo name: no dots so we don't swallow the `.git`
                # suffix into the capture; matches real savannah paths.
                r"([A-Za-z0-9_\-]+)(?:\.git)?/?$"
            ),
            r"https://git.savannah.gnu.org/git/\1.git",
        ),
        # `https://<name>.savannah.gnu.org/` → git.savannah.gnu.org/git
        (
            re.compile(r"^https?://([A-Za-z0-9_.\-]+)\.savannah\.gnu\.org/?$"),
            r"https://git.savannah.gnu.org/git/\1.git",
        ),
        # `https://dev.yorhel.nl` → code.blicky.net/yorhel/<name>
        (
            re.compile(r"^https?://dev\.yorhel\.nl/?$"),
            None,  # sentinel: dispatcher fills in `name` at call time
        ),
    )

    def _resolve_homepage_to_git_urls(self, url: str, name: str) -> List[str]:
        """Return candidate git URLs derived from a project homepage.

        The input ``url`` is often a project's *homepage* (e.g.
        ``https://www.gnu.org/software/wget/``) rather than a cloneable
        git URL. This helper applies deterministic host-specific rules
        to turn the homepage into likely-valid git URLs, so downstream
        clone recovery has something concrete to try.

        Args:
            url: The (possibly non-git) source URL.
            name: The canonical repo name; used as a fallback when the
                URL alone does not disclose it (e.g. ``dev.yorhel.nl``).

        Returns:
            A list of unique candidate git URLs (may be empty).
        """
        candidates: List[str] = []
        stripped = url.rstrip("/")
        for pattern, replacement in self._HOMEPAGE_TO_GIT_RULES:
            match = pattern.match(stripped)
            if not match:
                continue
            if replacement is None:
                # Dispatcher: dev.yorhel.nl repos live on code.blicky.net.
                cand = f"https://code.blicky.net/yorhel/{name}.git"
            else:
                cand = pattern.sub(replacement, stripped)
            if cand and cand not in candidates:
                candidates.append(cand)
        return candidates

    def _try_url_variants(
        self, url: str, name: str
    ) -> Optional[CommandResult]:
        """Attempt to clone from URL variants derived from ``url``.

        Order (broadest to most heuristic):
            1. ``/cgit/`` → ``/git/`` (cgit is a *browsing* frontend).
            2. Append ``.git`` if not already present.
            3. Homepage-to-git rules (GNU savannah, blicky, ...).

        Returns:
            The successful ``CommandResult`` if any variant clones
            cleanly, otherwise ``None``.
        """
        # Validate name — only allow safe path components
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
            raise ValueError(f"Unsafe repo name: {name!r}")

        # Validate url scheme
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"Refusing non-http URL: {url!r}")

        container_repo_path = f"/workspace/repos/{name}"
        variants: List[str] = []

        if "/cgit/" in url:
            variants.append(url.replace("/cgit/", "/git/"))
        if not url.endswith(".git"):
            variants.append(url.rstrip("/") + ".git")
        for cand in self._resolve_homepage_to_git_urls(url, name):
            if cand not in variants:
                variants.append(cand)

        for variant in variants:
            # Use list form — no shell, no injection
            execute_command(
                ["rm", "-rf", container_repo_path], use_docker=True
            )
            cmd = [
                "git",
                "clone",
                "--depth",
                "1",
                variant,
                container_repo_path,
            ]
            result = execute_command(
                cmd, use_docker=True, extra_env=self._git_env()
            )
            if result.success:
                # No f-string in logger call: cheaper when log is off
                logger.info("Clone succeeded via URL variant: %s", variant)
                return result

        return None

    def _to_host_path(self, path: str) -> str:
        """Translate container path to host path if necessary.

        Only the leading ``/workspace`` prefix is rewritten — a plain
        ``str.replace`` would also mangle any later ``workspace``
        segment inside the path.
        """
        if path.startswith("/workspace") and not os.path.exists("/workspace"):
            return self.workspace_root + path[len("/workspace") :]
        return path

    def _to_container_path(self, path: str) -> str:
        """Translate host path to container path if necessary."""
        if path.startswith(self.workspace_root):
            return "/workspace" + path[len(self.workspace_root) :]
        return path

    # ------------------------------------------------------------------
    # Container-health precheck. Detects a "poisoned" sandbox — one
    # where a previous package's LLM fix broke git or apt — and repairs
    # it in place before we try to clone anything. Observed root cause
    # for the 34-package cascade on debian shard 8 (run 28020958388):
    # one repo's fix added a broken deb source to sources.list.d, then
    # every subsequent repo failed with `git: command not found` or
    # `apt-get update` errors because git was missing / apt was wedged.
    # ------------------------------------------------------------------
    def _ensure_container_healthy(self) -> None:
        """Verify sandbox tooling is intact; repair in place if not.

        Runs at most three cheap probes:
            1. ``git --version`` — installs ``git`` via the active
               platform profile if missing (exit 127).
            2. ``apt-get update`` / ``apk update`` — if it fails, drops
               any files under ``/etc/apt/sources.list.d/`` and
               ``/etc/apk/repositories.d/`` that are not part of the
               base image (heuristic: created within the last day).
            3. Retries the failing probe once after repair.

        All failures are non-fatal (best effort); the subsequent clone
        will surface the real error via ``CommandResult``.
        """
        from src.platforms import get_active_profile

        try:
            profile = get_active_profile()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Profile lookup failed during precheck: {exc}")
            return

        # 1) Git availability
        probe = execute_command(
            ["git", "--version"],
            use_docker=True,
            validate=False,
        )
        if not probe.success and (
            "not found" in (probe.stderr or "").lower()
            or probe.exit_code == 127
        ):
            logger.warning(
                "Sandbox precheck: git missing, installing via %s",
                profile.name,
            )
            install = execute_command(
                profile.install_cmd(["git"]),
                use_docker=True,
            )
            state = "ok" if install.success else "still-broken"
            logger.info(f"Sandbox git self-repair: {state}")

        # 2) Package manager: only check on Debian/Ubuntu where the
        # cascade from a bad third-party source is worst; Alpine's
        # apk failures usually surface per-install.
        if profile.name in ("debian", "ubuntu"):
            update = execute_command(
                "apt-get update",
                use_docker=True,
            )
            if not update.success and (
                "Release file" in (update.stderr or "")
                or "not have a Release" in (update.stderr or "")
            ):
                logger.warning(
                    "Sandbox precheck: apt sources broken, removing "
                    "LLM-added source list files under "
                    "/etc/apt/sources.list.d/"
                )
                # Only remove non-base-image entries. `.list` files
                # baked into the image are typically created at image
                # build time; anything newer was added post-boot by a
                # fix command that we now know failed.
                cleanup = (
                    "find /etc/apt/sources.list.d -maxdepth 1 "
                    "-type f -mmin -1440 -delete"
                )
                execute_command(cleanup, use_docker=True)
                retry = execute_command("apt-get update", use_docker=True)
                if retry.success:
                    logger.info(
                        "Sandbox apt self-repair: recovered after "
                        "removing stale source lists"
                    )
                else:
                    logger.warning(
                        "Sandbox apt self-repair: still failing after "
                        "cleanup (%s)",
                        (retry.stderr or "")[:120],
                    )

    def _init_submodules_if_present(self, container_repo_path: str) -> None:
        """Best-effort ``git submodule update --init --recursive``.

        Runs only when the cloned repo contains a top-level
        ``.gitmodules``; downgrades any failure to a warning. Many Go
        repos (aliyun-cli, upstream vendor mirrors) resolve their
        real Go packages from a submodule, and skipping this step
        surfaces later as ``no required module provides package``.
        """
        check = execute_command(
            f"test -f {container_repo_path}/.gitmodules",
            use_docker=True,
        )
        if not check.success:
            return
        logger.info("Repository declares submodules; initializing recursively")
        result = execute_command(
            f"cd {container_repo_path} && "
            f"git submodule update --init --recursive",
            use_docker=True,
            extra_env=self._git_env(),
            timeout=300,
        )
        if not result.success:
            logger.warning(
                "Submodule init failed (continuing without): %s",
                (result.stderr or "")[:200],
            )

    def clone_or_update_repository(self, url: str, name: str) -> CommandResult:
        """Clone a repository or update it if it already exists.

        This is a zero-cost operation. Clones run inside the Docker
        container to avoid ownership issues. If ``git pull`` fails (for
        example a force-pushed branch), it falls back to a re-clone.

        GitHub auth (when a token is configured) is injected
        per-invocation via ``GIT_CONFIG_*`` environment variables — it
        is never written to the container's git config, so concurrent
        workers cannot race it and LLM-generated commands running later
        in the sandbox cannot read it.

        Args:
            url: The repository clone URL.
            name: The local directory name for the repository.

        Returns:
            The ``CommandResult`` of the clone or update operation.
        """
        self._auth_tls.env = self._github_auth_env(url)
        try:
            return self._clone_or_update_inner(url, name)
        finally:
            self._auth_tls.env = {}

    def _clone_or_update_inner(self, url: str, name: str) -> CommandResult:
        """Perform the actual clone/reset flow (see public wrapper)."""
        # Validate inputs once, up front: both are interpolated into
        # in-container shell strings below (rm -rf, git clone, cd).
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name) or name.startswith("."):
            raise ValueError(f"Unsafe repo name: {name!r}")
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"Refusing non-http URL: {url!r}")

        # Use container path
        container_repo_path = f"/workspace/repos/{name}"
        host_repo_path = os.path.join(self.repos_dir, name)

        # ------------------------------------------------------------
        # Container-health precheck: make sure the sandbox still has
        # git / working apt sources before we attempt anything else.
        # ------------------------------------------------------------
        self._ensure_container_healthy()

        # Check if already exists (check on host for speed)
        if os.path.exists(os.path.join(host_repo_path, ".git")):
            logger.info(f"Repository {name} already exists, resetting...")
            # STRATEGIC HARDENING (dasel regression, 2026-07-01): a
            # previous run's LLM may have authored broken files (e.g. a
            # syntactically-invalid Makefile or half-applied patches).
            # `git pull` alone preserves those files (they are untracked
            # or committed-locally), so every subsequent run replays the
            # same failure. Always reset to pristine upstream state
            # before touching the working tree; this keeps every run
            # idempotent and preserves the self-healing contract.
            # Reset to FETCH_HEAD rather than a command-substituted
            # `origin/<branch>`: if `git remote show origin` fails
            # (network / non-English locale) the substitution is empty
            # and `git reset --hard` silently resets to the poisoned
            # LOCAL head — exactly the state this hardening removes.
            reset_cmd = (
                f"cd {container_repo_path} && "
                f"git fetch --depth 1 origin && "
                f"git reset --hard FETCH_HEAD && "
                f"git clean -fdx"
            )
            result = execute_command(
                reset_cmd,
                use_docker=True,
                extra_env=self._git_env(),
            )

            if not result.success:
                # Reset failed — likely diverged history, force-push, or
                # a genuinely broken clone. Delete and re-clone.
                logger.warning(
                    f"fetch+reset failed for {name} "
                    f"(stderr={result.stderr[:200]!r}); "
                    f"re-cloning from scratch..."
                )
                execute_command(
                    ["rm", "-rf", container_repo_path], use_docker=True
                )
                clone_cmd = (
                    f"git clone --depth 1 {shlex.quote(url)} "
                    f"{shlex.quote(container_repo_path)}"
                )
                result = execute_command(
                    clone_cmd,
                    use_docker=True,
                    extra_env=self._git_env(),
                )
        else:
            execute_command(
                ["rm", "-rf", container_repo_path], use_docker=True
            )

            logger.info(f"Cloning repository {url}...")
            cmd = self._build_clone_cmd(url, container_repo_path)
            result = execute_command(
                cmd,
                use_docker=True,
                extra_env=self._git_env(),
            )

            if not result.success and self._is_auth_error(result):
                logger.warning(
                    f"Auth failure for {name}, retrying without token..."
                )
                execute_command(
                    ["rm", "-rf", container_repo_path], use_docker=True
                )
                cmd = (
                    f"git clone --depth 1 {shlex.quote(url)} "
                    f"{shlex.quote(container_repo_path)}"
                )
                result = execute_command(
                    cmd,
                    use_docker=True,
                    extra_env=self._GIT_FAIL_FAST_ENV,
                )

            if not result.success:
                logger.warning(
                    f"Clone failed for {name}, trying URL variants..."
                )
                variant_result = self._try_url_variants(url, name)
                if variant_result is not None:
                    result = variant_result
                    logger.info(f"Clone via URL variant succeeded for {name}")
                else:
                    logger.error(
                        f"All clone attempts failed for {name} ({url})"
                    )

        # Configure git safe.directory to prevent "dubious ownership" errors
        # This is needed because the workspace is mounted from host
        if result.success or os.path.exists(host_repo_path):
            safe_dir_cmd = (
                f"git config --global --add safe.directory "
                f"{container_repo_path}"
            )
            safe_result = execute_command(safe_dir_cmd, use_docker=True)
            if not safe_result.success:
                logger.warning(
                    f"Failed to add safe.directory: {safe_result.stderr}"
                )

        # Auto-init any git submodules the repo declares. Non-fatal.
        if result.success:
            self._init_submodules_if_present(container_repo_path)

        return result

    def get_repository_info(self, repo_path: str) -> Dict[str, str]:
        """Get basic repository information."""
        # Use container path for git operations
        container_path = self._to_container_path(repo_path)
        info = {}

        # Get current commit
        result = execute_command(
            f"cd {container_path} && git rev-parse HEAD", use_docker=True
        )
        if result.success:
            info["commit"] = result.stdout.strip()

        # Get branch
        result = execute_command(
            f"cd {container_path} && git branch --show-current",
            use_docker=True,
        )
        if result.success:
            info["branch"] = result.stdout.strip()

        # Count files
        result = execute_command(
            f"find {container_path} -type f | wc -l", use_docker=True
        )
        if result.success:
            info["file_count"] = result.stdout.strip()

        return info

    # ========== Build System Detection ==========

    def _has_npm_build_script(self, repo_path: str) -> bool:
        """Check if package.json has a 'build' script (true npm project)."""
        pkg_file = os.path.join(repo_path, "package.json")
        if not os.path.exists(pkg_file):
            return False
        try:
            with open(pkg_file, "r", encoding="utf-8", errors="ignore") as f:
                pkg = json.load(f)
            scripts = pkg.get("scripts", {})
            if isinstance(scripts, dict) and "build" in scripts:
                return True
        except Exception:
            pass
        return False

    def _score_go_subdir(self, root: str, repo_path: str) -> int:
        """Score a Go module root by main-package/cmd markers.

        Higher = more likely to be the primary build target.
        """
        score = 0
        if os.path.isfile(os.path.join(root, "main.go")):
            score += 10
        cmd_dir = os.path.join(root, "cmd")
        if os.path.isdir(cmd_dir):
            score += 8
        # `with` so the directory FD is released even though we `break`
        # out of the loop on the first match (MEM-02).
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".go"):
                    try:
                        with open(
                            entry.path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            if "package main" in f.read(2048):
                                score += 5
                                break
                    except Exception:
                        pass
        return score

    def detect_build_system(self, repo_path: str) -> BuildSystemInfo:
        """Detect the build system using file-based heuristics.

        This is a zero-cost operation.

        Args:
            repo_path: Path to the repository root.

        Returns:
            A ``BuildSystemInfo`` describing the detected build system.
        """
        repo_path = self._to_host_path(repo_path)
        build_systems = {
            "cmake": ["CMakeLists.txt"],
            "autotools": ["configure.ac", "configure.in", "configure"],
            "make": ["Makefile", "GNUmakefile"],
            "meson": ["meson.build"],
            "cargo": ["Cargo.toml"],
            "npm": ["package.json"],
            "pip": ["setup.py", "pyproject.toml"],
            "go": ["go.mod"],
            "gradle": ["build.gradle", "build.gradle.kts"],
            "maven": ["pom.xml"],
            "bazel": ["BUILD", "BUILD.bazel", "WORKSPACE"],
        }

        detected = []
        confidence_scores = {}

        for build_sys, files in build_systems.items():
            for file in files:
                filepath = os.path.join(repo_path, file)
                if os.path.exists(filepath):
                    # npm requires stricter detection: need a
                    # lockfile or a build script
                    if build_sys == "npm":
                        has_lock = os.path.exists(
                            os.path.join(repo_path, "package-lock.json")
                        )
                        has_build = self._has_npm_build_script(repo_path)
                        if not has_lock and not has_build:
                            continue
                    detected.append((build_sys, file))
                    # Never let a weaker marker clobber a stronger one
                    # (e.g. `configure` at 0.85 must not overwrite the
                    # 0.95 set by `configure.ac` a moment earlier).
                    prev = confidence_scores.get(build_sys, 0)
                    if file in [
                        "CMakeLists.txt",
                        "Cargo.toml",
                        "configure.ac",
                        "go.mod",
                    ]:
                        confidence_scores[build_sys] = max(prev, 0.95)
                    elif file == "configure":
                        confidence_scores[build_sys] = max(prev, 0.85)
                    else:
                        confidence_scores[build_sys] = min(prev + 0.3, 0.9)

        if not detected:
            go_subdir_candidates = []
            cargo_found = None
            for root, dirs, files in os.walk(repo_path):
                if _walk_root_is_excluded(root, ".git"):
                    continue
                if "go.mod" in files:
                    module_dir = root.replace(repo_path, "").lstrip("/")
                    score = self._score_go_subdir(root, repo_path)
                    go_subdir_candidates.append((root, module_dir, score))
                if "Cargo.toml" in files and cargo_found is None:
                    module_dir = root.replace(repo_path, "").lstrip("/")
                    cargo_found = (root, module_dir)
                    logger.info(
                        f"Found Cargo module in subdirectory: "
                        f"{module_dir}/Cargo.toml"
                    )

            if go_subdir_candidates:
                candidates_sorted = sorted(
                    go_subdir_candidates, key=lambda x: x[2], reverse=True
                )
                best_root, best_dir, best_score = candidates_sorted[0]
                detected.append(("go", f"{best_dir}/go.mod"))
                confidence_scores["go"] = 0.90
                logger.info(
                    f"Found Go module in subdirectory: "
                    f"{best_dir}/go.mod (score={best_score})"
                )
            elif cargo_found:
                detected.append(("cargo", f"{cargo_found[1]}/Cargo.toml"))
                confidence_scores["cargo"] = 0.90

        if not detected and os.path.exists(
            os.path.join(repo_path, "Makefile")
        ):
            detected.append(("make", "Makefile"))
            confidence_scores["make"] = 0.5
            logger.info("Only Makefile found; detected as make (fallback)")

        if not detected:
            # GOPATH-style Go repos can have a valid main package but no
            # go.mod. If we leave these as "unknown", the workflow often
            # falls back to generic `make` plans that cannot succeed.
            go_info = self.find_go_main_package(repo_path)
            if go_info.get("has_go_files"):
                main_path = go_info.get("main_path", "").strip("/")
                if main_path and main_path != ".":
                    primary = f"{main_path}/main.go"
                elif go_info.get("has_main"):
                    primary = "main.go"
                else:
                    primary = "*.go"
                detected.append(("go", primary))
                confidence_scores["go"] = 0.65
                logger.info(
                    "Detected GOPATH-style Go project (no go.mod); "
                    "using Go build-system fallback"
                )

        if not detected:
            return BuildSystemInfo(
                type="unknown",
                confidence=0.0,
                primary_file="",
                additional_files=[],
            )

        # Priority boost: prefer Go over npm when both detected
        sys_types = {s for s, _ in detected}
        if "go" in sys_types and "npm" in sys_types:
            # Demote npm below go
            if (
                "npm" in confidence_scores
                and "go" in confidence_scores
                and confidence_scores["npm"] >= confidence_scores["go"]
            ):
                confidence_scores["npm"] = confidence_scores["go"] - 0.1

        best_system = max(confidence_scores.items(), key=lambda x: x[1])

        primary_file = [f for sys, f in detected if sys == best_system[0]][0]
        additional = [
            f
            for sys, f in detected
            if sys == best_system[0] and f != primary_file
        ]

        module_dir = ""
        if best_system[0] in ["go", "cargo"] and "/" in primary_file:
            module_dir = primary_file.rsplit("/", 1)[0]

        return BuildSystemInfo(
            type=best_system[0],
            confidence=min(best_system[1], 1.0),
            primary_file=primary_file,
            additional_files=additional,
            module_dir=module_dir,
        )

    # ========== Dependency Detection ==========

    def extract_dependencies(
        self, repo_path: str, build_system: str
    ) -> DependencyInfo:
        """Extract dependencies by parsing package files.

        This is a zero-cost operation.

        Args:
            repo_path: Path to the repository root.
            build_system: The detected build system name.

        Returns:
            A ``DependencyInfo`` with the extracted dependencies.
        """
        repo_path = self._to_host_path(repo_path)
        deps = DependencyInfo()

        if build_system == "cmake":
            deps = self._extract_cmake_dependencies(repo_path)
        elif build_system == "cargo":
            deps = self._extract_cargo_dependencies(repo_path)
        elif build_system == "pip":
            deps = self._extract_python_dependencies(repo_path)
        elif build_system == "npm":
            deps = self._extract_npm_dependencies(repo_path)
        elif build_system == "go":
            deps = self._extract_go_dependencies(repo_path)
        elif build_system == "make":
            deps = self._extract_make_dependencies(repo_path)

        return deps

    def _extract_cmake_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from CMakeLists.txt."""
        deps = DependencyInfo(install_method="apk")
        cmake_file = os.path.join(repo_path, "CMakeLists.txt")

        if not os.path.exists(cmake_file):
            return deps

        with open(cmake_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Find find_package() calls
        package_pattern = r"find_package\s*\(\s*(\w+)"
        packages = re.findall(package_pattern, content, re.IGNORECASE)

        # Map common CMake packages to canonical names; PlatformProfile
        # resolves to the distro package at install time.
        from .platforms import get_active_profile

        profile = get_active_profile()
        canonical_map = {
            "Threads": "musl-dev",
            "OpenSSL": "openssl",
            "ZLIB": "zlib",
            "PNG": "libpng",
            "JPEG": "libjpeg-turbo",
            "Boost": "boost-dev",  # rare canonical, kept as Alpine name
            "Qt5": "qt5-qtbase-dev",  # rare canonical, kept as Alpine name
            "Protobuf": "protobuf",
        }

        for pkg in packages:
            if pkg in canonical_map:
                deps.system_packages.append(
                    profile.resolve(canonical_map[pkg])
                )
            deps.libraries.append(pkg)

        # Common build tools for CMake
        deps.build_tools = ["cmake", "make", "gcc", "g++"]

        return deps

    def _extract_cargo_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from Cargo.toml."""
        deps = DependencyInfo(install_method="cargo")
        cargo_file = os.path.join(repo_path, "Cargo.toml")

        if not os.path.exists(cargo_file):
            return deps

        try:
            import toml

            with open(cargo_file, "r") as f:
                cargo_config = toml.load(f)

            # Extract dependencies
            if "dependencies" in cargo_config:
                deps.libraries = list(cargo_config["dependencies"].keys())
        except Exception as e:
            logger.warning(f"Failed to parse Cargo.toml: {e}")

        deps.build_tools = ["cargo", "rustc"]
        return deps

    def _extract_python_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from requirements.txt or setup.py."""
        deps = DependencyInfo(install_method="pip")

        # Check requirements.txt
        req_file = os.path.join(repo_path, "requirements.txt")
        if os.path.exists(req_file):
            with open(req_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Extract package name (before ==, >=, etc.)
                        pkg = re.split(r"[=<>!]", line)[0].strip()
                        deps.libraries.append(pkg)

        deps.build_tools = ["python3", "pip"]
        return deps

    def _extract_npm_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from package.json."""
        deps = DependencyInfo(install_method="npm")
        package_file = os.path.join(repo_path, "package.json")

        if not os.path.exists(package_file):
            return deps

        try:
            with open(package_file, "r") as f:
                package_config = json.load(f)

            if "dependencies" in package_config:
                deps.libraries.extend(package_config["dependencies"].keys())
            if "devDependencies" in package_config:
                deps.libraries.extend(package_config["devDependencies"].keys())
        except Exception as e:
            logger.warning(f"Failed to parse package.json: {e}")

        deps.build_tools = ["node", "npm"]
        return deps

    def _extract_go_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from go.mod."""
        deps = DependencyInfo(install_method="go")
        go_mod = os.path.join(repo_path, "go.mod")

        if not os.path.exists(go_mod):
            return deps

        with open(go_mod, "r") as f:
            content = f.read()

        # go.mod uses two require forms; handle both:
        #   require example.com/pkg v1.2.3          (single line)
        #   require (                                (block)
        #       example.com/pkg v1.2.3
        #   )
        # The old single-line-only regex captured a literal "(" for
        # block form — i.e. for most real go.mod files.
        libraries: List[str] = []
        for match in re.finditer(r"require\s+\(([^)]*)\)", content, re.DOTALL):
            for line in match.group(1).splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                libraries.append(line.split()[0])
        for match in re.finditer(
            r"^require\s+([^\s(]+)", content, re.MULTILINE
        ):
            libraries.append(match.group(1))
        deps.libraries = list(dict.fromkeys(libraries))

        deps.build_tools = ["go"]
        return deps

    def find_go_main_package(self, repo_path: str) -> Dict[str, Any]:
        """Find the Go main package location.

        For repos without ``go.mod`` (old GOPATH-style), sets
        ``needs_go_init=True``.

        Args:
            repo_path: Path to the repository root.

        Returns:
            A dict describing where the main package is located.
        """
        result = {
            "has_main": False,
            "main_path": "",
            "module_dir": "",
            "build_command": "go build .",
            "needs_go_init": False,
            "has_go_mod": False,
            "has_go_files": False,
        }

        repo_path = self._to_host_path(repo_path)

        go_mod_files = []
        has_go_files = False
        for root, dirs, files in os.walk(repo_path):
            if _walk_root_is_excluded(root, ".git"):
                continue
            if "go.mod" in files:
                module_dir = root.replace(repo_path, "").lstrip("/")
                go_mod_files.append((root, module_dir))
            if any(f.endswith(".go") for f in files):
                has_go_files = True

        if not go_mod_files:
            if has_go_files:
                # GOPATH-style repo: has .go files but no go.mod
                result["needs_go_init"] = True
                result["has_go_files"] = True
                # Try to find main package anyway
                for root, dirs, files in os.walk(repo_path):
                    if _walk_root_is_excluded(root, ".git", "vendor"):
                        continue
                    for f in files:
                        if f.endswith(".go"):
                            try:
                                with open(
                                    os.path.join(root, f),
                                    "r",
                                    encoding="utf-8",
                                    errors="ignore",
                                ) as fh:
                                    if "package main" in fh.read():
                                        rel_path = root.replace(
                                            repo_path, ""
                                        ).lstrip("/")
                                        result["has_main"] = True
                                        result["main_path"] = rel_path or "."
                                        result["build_command"] = "go build ."
                                        return result
                            except Exception:
                                pass
            return result

        result["has_go_mod"] = True

        # Prefer go_mod entry whose root has main.go or cmd/, else first
        best_mod = go_mod_files[0]
        if len(go_mod_files) > 1:
            scored = [
                (root, d, self._score_go_subdir(root, repo_path))
                for root, d in go_mod_files
            ]
            scored.sort(key=lambda x: x[2], reverse=True)
            best_mod = (scored[0][0], scored[0][1])
            logger.info(
                f"Chose go.mod at '{scored[0][1]}' (score={scored[0][2]}) "
                f"out of {len(scored)} candidates"
            )

        go_mod_path, module_dir = best_mod
        result["module_dir"] = module_dir

        main_files = []
        for root, dirs, files in os.walk(go_mod_path):
            if _walk_root_is_excluded(root, ".git", "vendor"):
                continue
            for f in files:
                if f.endswith(".go"):
                    filepath = os.path.join(root, f)
                    try:
                        with open(
                            filepath, "r", encoding="utf-8", errors="ignore"
                        ) as file:
                            content = file.read()
                            if "package main" in content:
                                rel_path = root.replace(
                                    go_mod_path, ""
                                ).lstrip("/")
                                main_files.append(
                                    {
                                        "file": f,
                                        "dir": rel_path,
                                        "full_path": os.path.join(root, f),
                                    }
                                )
                    except Exception:
                        pass

        if main_files:
            result["has_main"] = True
            # Prefer: cmd/<reponame> > cmd/* without test/example >
            # root > other.
            repo_basename = os.path.basename(repo_path).lower()

            def score_main(m: dict) -> int:
                d = m["dir"].lower()
                # Hard-penalize codegen helpers, contrib utilities,
                # examples, and tests.
                if any(
                    seg in d.split("/")
                    for seg in (
                        "gen",
                        "contrib",
                        "example",
                        "examples",
                        "test",
                        "tests",
                        "internal",
                        "tools",
                        "scripts",
                        "hack",
                        "vendor",
                    )
                ):
                    if not d.startswith(f"cmd/{repo_basename}"):
                        return 0
                if d == f"cmd/{repo_basename}":
                    return 10  # perfect match
                if (
                    d.startswith("cmd/")
                    and "test" not in d
                    and "example" not in d
                    and "integration" not in d
                ):
                    return 7
                if not d:
                    return 6  # root main — prefer over arbitrary subdirs
                if d.startswith("cmd/"):
                    return 2
                if "test" in d or "example" in d or "integration" in d:
                    return 1
                return 3

            main = max(main_files, key=score_main)
            result["all_main_paths"] = [m["dir"] or "." for m in main_files]

            if main["dir"]:
                result["main_path"] = main["dir"]
                result["build_command"] = f"go build ./{main['dir']}"
            else:
                result["main_path"] = "."
                result["build_command"] = "go build ."

        return result

    def _extract_make_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from Makefile (basic parsing)."""
        deps = DependencyInfo(install_method="apk")
        makefile = os.path.join(repo_path, "Makefile")

        if not os.path.exists(makefile):
            return deps

        with open(makefile, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Look for common library flags
        lib_pattern = r"-l(\w+)"
        libs = re.findall(lib_pattern, content)

        # Map -l flags to canonical names; profile resolves to distro package
        from .platforms import get_active_profile

        profile = get_active_profile()
        canonical_lib_map = {
            "ssl": "openssl",
            "crypto": "openssl",
            "z": "zlib",
            "pthread": "musl-dev",
            "dl": "musl-dev",
            "m": "musl-dev",
        }

        for lib in libs:
            if lib in canonical_lib_map:
                deps.system_packages.append(
                    profile.resolve(canonical_lib_map[lib])
                )
            deps.libraries.append(lib)

        deps.build_tools = ["make", "gcc", "g++"]
        return deps

    # ========== Architecture-Specific Code Detection ==========

    def find_architecture_specific_code(
        self, repo_path: str
    ) -> List[ArchSpecificCode]:
        """Search for architecture-specific code patterns.

        This is a zero-cost operation using grep.

        Args:
            repo_path: Path to the repository root.

        Returns:
            A list of ``ArchSpecificCode`` findings.
        """
        repo_path = self._to_host_path(repo_path)
        arch_specific = []

        # Patterns to search for
        patterns = {
            "x86": [
                r"__x86_64__",
                r"__amd64__",
                r"__i386__",
                r"_M_X64",
                r"_M_IX86",
            ],
            "x86_simd": [
                r"__SSE\d?__",
                r"__AVX\d?__",
                r"__FMA__",
                r"_mm\w+",
                r"__builtin_ia32",
            ],
            "arm": [r"__ARM__", r"__aarch64__", r"__arm__", r"_M_ARM"],
            "arm_simd": [r"__ARM_NEON", r"vld\d", r"vst\d", r"vmul", r"vadd"],
            "inline_asm": [r"__asm__", r"asm\s*\(", r"__asm\s+volatile"],
        }

        # Use container path for grep
        target_path = self._to_container_path(repo_path)

        for arch_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                # Use grep for fast searching
                cmd = (
                    f"grep -rn -E '{pattern}' {target_path} "
                    f"--include='*.c' --include='*.cpp' "
                    f"--include='*.h' --include='*.hpp' "
                    f"2>/dev/null | head -n 20"
                )

                result = execute_command(cmd, use_docker=True)

                if result.success and result.stdout.strip():
                    for line in result.stdout.strip().split("\n"):
                        if ":" in line:
                            parts = line.split(":", 2)
                            if len(parts) >= 3:
                                file_path = parts[0]
                                line_num = parts[1]
                                code = parts[2][:100]  # Truncate long lines

                                # Determine severity
                                severity = "medium"
                                if arch_type in [
                                    "x86_simd",
                                    "arm_simd",
                                    "inline_asm",
                                ]:
                                    severity = "high"

                                arch_specific.append(
                                    ArchSpecificCode(
                                        file=file_path,
                                        line=(
                                            int(line_num)
                                            if line_num.isdigit()
                                            else 0
                                        ),
                                        code_snippet=code.strip(),
                                        arch_type=arch_type,
                                        severity=severity,
                                        suggested_fix=(
                                            self._suggest_fix_for_arch_code(
                                                arch_type
                                            )
                                        ),
                                    )
                                )

        return arch_specific

    def _suggest_fix_for_arch_code(self, arch_type: str) -> str:
        """Suggest fixes for architecture-specific code."""
        suggestions = {
            "x86": "Add RISC-V conditional compilation (#ifdef __riscv)",
            "x86_simd": (
                "Use RVV (RISC-V Vector) extension or scalar fallback"
            ),
            "arm": "Add RISC-V conditional compilation (#ifdef __riscv)",
            "arm_simd": (
                "Use RVV (RISC-V Vector) extension or scalar fallback"
            ),
            "inline_asm": (
                "Rewrite assembly in C or add RISC-V assembly variant"
            ),
        }
        return suggestions.get(arch_type, "Review and port to RISC-V")

    # ========== File Operations ==========

    def get_file_tree(self, repo_path: str, max_depth: int = 3) -> str:
        """Get a tree view of the repository."""
        target_path = self._to_container_path(repo_path)

        cmd = (
            f"tree -L {max_depth} "
            f"-I '.git|node_modules|__pycache__|.venv' {target_path}"
        )
        result = execute_command(cmd, use_docker=True)

        if result.success:
            return result.stdout
        else:
            cmd = (
                f"find {target_path} -maxdepth {max_depth} "
                f"-type f | head -n 100"
            )
            result = execute_command(cmd, use_docker=True)
            return result.stdout

    def get_optimized_tree(self, repo_path: str) -> str:
        """Build a token-efficient tree structure for agent context.

        Provides essential structural information without full verbose
        output. The compact representation shows root-level files
        (especially build configs and docs), the directory structure two
        levels deep, file counts by category, and key files highlighted.

        Args:
            repo_path: Path to the repository root.

        Returns:
            A compact textual repository tree.
        """
        target_path = self._to_container_path(repo_path)

        sections = []

        sections.append("## Repository Structure Overview\n")

        ls_cmd = f"ls -la {target_path} 2>/dev/null"
        ls_result = execute_command(ls_cmd, use_docker=True)

        find_cmd = (
            f"find {target_path} -maxdepth 2 -type d 2>/dev/null "
            f"| grep -v '.git' | sort"
        )
        find_result = execute_command(find_cmd, use_docker=True)

        if ls_result.success and ls_result.stdout.strip():
            sections.append("```\n" + ls_result.stdout.strip() + "\n```\n")

        if find_result.success and find_result.stdout.strip():
            dirs = find_result.stdout.strip().split("\n")[:20]
            if len(dirs) > 1:
                sections.append("\nSubdirectories (depth 2):\n")
                for d in dirs[:15]:
                    rel = d.replace(target_path, "").lstrip("/") or "."
                    if rel != ".":
                        sections.append(f"  {rel}/\n")

        sections.append("\n\n## Key Files Detected\n")

        key_patterns = {
            "Build Config": [
                "CMakeLists.txt",
                "Makefile",
                "Cargo.toml",
                "go.mod",
                "package.json",
                "setup.py",
                "pyproject.toml",
                "meson.build",
                "configure.ac",
                "BUILD",
                "BUILD.bazel",
                "pom.xml",
                "build.gradle",
            ],
            "Documentation": [
                "README*",
                "INSTALL*",
                "BUILDING*",
                "CONTRIBUTING*",
            ],
            "Config": [".env*", "config.*", "*.yaml", "*.yml", "*.json"],
        }

        for category, patterns in key_patterns.items():
            found_files = []
            for pattern in patterns:
                cmd = (
                    f"find {target_path} -maxdepth 2 "
                    f"-name '{pattern}' -type f 2>/dev/null | head -n 5"
                )
                result = execute_command(cmd, use_docker=True)
                if result.success and result.stdout.strip():
                    for f in result.stdout.strip().split("\n"):
                        rel_path = f.replace(target_path, "").lstrip("/")
                        if rel_path and len(found_files) < 8:
                            found_files.append(rel_path)

            if found_files:
                sections.append(
                    f"- **{category}**: {', '.join(found_files[:8])}\n"
                )

        sections.append("\n## Directory Stats\n")

        stats_cmd = (
            f"find {target_path} -maxdepth 1 -type d "
            f"! -path {target_path} 2>/dev/null | wc -l"
        )
        stats_result = execute_command(stats_cmd, use_docker=True)
        if stats_result.success:
            sections.append(
                f"- Root subdirectories: {stats_result.stdout.strip()}\n"
            )

        src_cmd = (
            f"find {target_path} -maxdepth 3 -type f -name '*.c' "
            f"-o -name '*.cpp' -o -name '*.h' 2>/dev/null | wc -l"
        )
        src_result = execute_command(src_cmd, use_docker=True)
        if src_result.success and src_result.stdout.strip() != "0":
            sections.append(
                f"- C/C++ source files: {src_result.stdout.strip()}\n"
            )

        go_cmd = (
            f"find {target_path} -maxdepth 3 -type f "
            f"-name '*.go' 2>/dev/null | wc -l"
        )
        go_result = execute_command(go_cmd, use_docker=True)
        if go_result.success and go_result.stdout.strip() != "0":
            sections.append(f"- Go source files: {go_result.stdout.strip()}\n")

        rs_cmd = (
            f"find {target_path} -maxdepth 3 -type f "
            f"-name '*.rs' 2>/dev/null | wc -l"
        )
        rs_result = execute_command(rs_cmd, use_docker=True)
        if rs_result.success and rs_result.stdout.strip() != "0":
            sections.append(
                f"- Rust source files: {rs_result.stdout.strip()}\n"
            )

        py_cmd = (
            f"find {target_path} -maxdepth 3 -type f "
            f"-name '*.py' 2>/dev/null | wc -l"
        )
        py_result = execute_command(py_cmd, use_docker=True)
        if py_result.success and py_result.stdout.strip() != "0":
            sections.append(f"- Python files: {py_result.stdout.strip()}\n")

        return "\n".join(sections)

    def read_file(self, filepath: str, max_lines: int = 1000) -> str:
        """Read file content with line limit."""
        filepath = self._to_host_path(filepath)
        if not os.path.exists(filepath):
            return f"File not found: {filepath}"

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
            return f"Error reading file: {e}"

    # ========== Documentation Search ==========

    def find_documentation(self, repo_path: str) -> List[str]:
        """Find common documentation files."""
        target_path = self._to_container_path(repo_path)
        doc_patterns = [
            "README*",
            "INSTALL*",
            "BUILDING*",
            "BUILD*",
            "CONTRIBUTING*",
            "docs/*.md",
            "doc/*.md",
            "*.md",
        ]

        found_docs = []
        for pattern in doc_patterns:
            cmd = f"find {target_path} -maxdepth 2 -iname '{pattern}' -type f"
            result = execute_command(cmd, use_docker=True)

            if result.success and result.stdout.strip():
                found_docs.extend(result.stdout.strip().split("\n"))

        # Remove duplicates and sort by likely importance
        unique_docs = list(set(found_docs))

        # Prioritize certain files
        priority = ["README", "INSTALL", "BUILDING", "BUILD"]
        sorted_docs = []

        for p in priority:
            sorted_docs.extend([d for d in unique_docs if p in d.upper()])

        # Add remaining docs
        sorted_docs.extend([d for d in unique_docs if d not in sorted_docs])

        return sorted_docs[:10]  # Limit to top 10

    def get_system_info(
        self, tools: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Get system environment information inside the container."""
        info = {}
        if tools is None:
            tools = [
                "gcc",
                "g++",
                "cmake",
                "make",
                "ninja",
                "meson",
                "automake",
                "autoconf",
                "python3",
                "go",
                "rustc",
                "cargo",
            ]

        for tool in tools:
            # Check availability INSIDE the container
            res = execute_command(f"which {tool}", use_docker=True)
            if res.success:
                info[tool] = "Available in PATH"
            else:
                info[tool] = "Not installed"

        # Architecture check
        arch_res = execute_command("uname -m", use_docker=True)
        info["architecture"] = (
            arch_res.stdout.strip() if arch_res.success else "unknown"
        )

        return info

    # ========== Helper Methods ==========

    def detect_arch_specific_build_files(
        self, repo_path: str
    ) -> Dict[str, Any]:
        """Detect architecture-specific build files in the repository.

        Args:
            repo_path: Path to the repository root.

        Returns:
            A dict describing existing arch files and whether RISC-V
            support already exists.
        """
        target_path = self._to_container_path(repo_path)
        result = {
            "has_arch_specific": False,
            "archs_found": [],
            "riscv_exists": False,
            "arch_files": {},
            "suggested_riscv_files": [],
        }

        arch_patterns = [
            (
                "x64",
                r"(cmpl_gcc_|var_gcc_|makefile\.|build_)(x64|x86_64|amd64)",
            ),
            (
                "arm64",
                r"(cmpl_gcc_|var_gcc_|makefile\.|build_)(arm64|aarch64)",
            ),
            ("x86", r"(cmpl_gcc_|var_gcc_|makefile\.|build_)(x86|i386|i686)"),
            (
                "riscv",
                r"(cmpl_gcc_|var_gcc_|makefile\.|build_)(riscv|riscv64|rv64)",
            ),
        ]

        for arch, pattern in arch_patterns:
            # Use the arch-specific REGEX (not the bare arch token —
            # a Makefile comment mentioning "riscv" must not count as
            # existing RISC-V support) and parenthesize the -name
            # alternation so -type f applies to both suffixes.
            cmd = (
                f"find {shlex.quote(target_path)} -type f "
                f"\\( -name '*.mak' -o -name '*.mk' \\) 2>/dev/null "
                f"| xargs grep -liE {shlex.quote(pattern)} "
                f"2>/dev/null | head -20"
            )
            find_result = execute_command(cmd, use_docker=True)

            if find_result.success and find_result.stdout.strip():
                files = find_result.stdout.strip().split("\n")
                result["has_arch_specific"] = True
                if arch not in result["archs_found"]:
                    result["archs_found"].append(arch)
                result["arch_files"][arch] = [f for f in files if f]

        if "riscv" in result["archs_found"]:
            result["riscv_exists"] = True
        else:
            for arch in ["x64", "arm64"]:
                if arch in result["arch_files"]:
                    for src_file in result["arch_files"][arch][:3]:
                        riscv_file = src_file.replace(arch, "riscv64")
                        riscv_file = re.sub(
                            r"x86_64|amd64", "riscv64", riscv_file
                        )
                        result["suggested_riscv_files"].append(
                            {
                                "source": src_file,
                                "target": riscv_file,
                            }
                        )

        return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def quick_analysis(repo_path: str) -> Dict[str, Any]:
    """Perform quick repository analysis using only scripted operations.

    This provides initial context without any LLM calls.

    Args:
        repo_path: Path to the repository root.

    Returns:
        A dict of analysis results.
    """
    ops = ScriptedOperations()

    analysis = {
        "repo_info": ops.get_repository_info(repo_path),
        "build_system": ops.detect_build_system(repo_path),
        "file_tree": ops.get_file_tree(repo_path, max_depth=2),
        "optimized_tree": ops.get_optimized_tree(repo_path),
        "documentation": ops.find_documentation(repo_path),
        "arch_build_files": ops.detect_arch_specific_build_files(repo_path),
    }

    go_main_info = ops.find_go_main_package(repo_path)
    if go_main_info.get("has_go_mod") or go_main_info.get("has_go_files"):
        analysis["go_main_info"] = go_main_info

    if analysis["build_system"].type != "unknown":
        analysis["dependencies"] = ops.extract_dependencies(
            repo_path, analysis["build_system"].type
        )

    analysis["arch_specific_code"] = ops.find_architecture_specific_code(
        repo_path
    )

    return analysis

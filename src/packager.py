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

"""Package a successfully ported repository as a downloadable zip.

Layout produced::

    <owner>-<repo>-<YYYYMMDD-HHMMSS>-<platform>.zip
    ├── build_recipe.md          # the porting recipe (root of zip)
    ├── manifest.json            # metadata: verification verdict,
    │                            # riscv64 artifacts with SHA-256
    ├── <repo>.log               # batch_test per-package log (optional)
    ├── agent_<repo>.log         # main.py per-package debug log (optional)
    └── <repo>/                  # source tree (excluding .git/, symlinks)
        └── ...

Used by ``main.py --package`` and the CI batch workflow to produce
artifacts that downstream consumers can download directly.
"""

import hashlib
import json
import logging
import os
import re
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Top-level entries inside the repo subtree that are excluded from packages.
# Kept conservative on purpose — users asked for "repo code", not a clone.
_EXCLUDED_DIRS = frozenset({".git"})

_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _url_path_parts(repo_url: str) -> list:
    """Return the non-empty path segments of ``repo_url``.

    Accepts ``https://host/owner/repo(.git)``, ``git@host:owner/repo``
    and bare ``owner/repo`` or ``repo`` forms.
    """
    url = (repo_url or "").strip().rstrip("/")
    if url.endswith(".git"):
        url = url[: -len(".git")]
    if "://" in url:
        rest = url.split("://", 1)[1]
        url = rest.split("/", 1)[1] if "/" in rest else ""
    elif "@" in url and ":" in url:
        url = url.split(":", 1)[1]
    return [p for p in url.split("/") if p]


def _sanitize_stem(raw: str) -> str:
    return _UNSAFE_STEM_CHARS.sub("-", raw).lstrip(".")


def package_stem(repo_url: str) -> str:
    """Return the zip filename stem for ``repo_url``: ``<owner>-<repo>``.

    The URL basename alone is not unique: the catalog holds five
    different repositories named ``cli``. The owner prefix makes the
    stem unique per repository, so release dedupe and the planners
    never confuse two packages.

    Args:
        repo_url: The upstream repository URL.

    Returns:
        A lower-case, path-safe stem such as ``"hetznercloud-cli"``.
        A URL without an owner segment gives the bare repo stem.
    """
    parts = _url_path_parts(repo_url)
    stem = _sanitize_stem("-".join(parts[-2:])).lower()
    return stem or "repo"


# A copy of src.state._AMBIGUOUS_BASENAMES, not an import: the planners
# import this module on a bare python3 before the workflows install
# requirements, and src.state imports langchain. tests/test_packager.py
# fails when the copy and the original drift apart.
_LEGACY_AMBIGUOUS_BASENAMES = frozenset(
    {"cli", "core", "src", "app", "main", "client", "server", "lib"}
)


def legacy_package_stem(repo_url: str) -> str:
    """Return the zip stem of releases before owner-prefixed names.

    Those zips were named after ``src.state.derive_repo_name``: the
    sanitized URL basename, with the owner prefixed only for generic
    basenames such as ``cli`` (``hetznercloud-cli``). This function
    mirrors that derivation exactly, so old zips keep matching. The
    planners accept a legacy stem only when it is unique in the list.

    Args:
        repo_url: The upstream repository URL.

    Returns:
        The case-preserving legacy stem, for example ``"zlib"``.
    """
    trimmed = (repo_url or "").strip().rstrip("/")
    segments = [s for s in trimmed.split("/") if s]
    basename = segments[-1].removesuffix(".git") if segments else ""
    if (
        basename.lower() in _LEGACY_AMBIGUOUS_BASENAMES
        and len(segments) >= 2
    ):
        owner = segments[-2].removesuffix(".git")
        # Skip the scheme and host segments ("https:", "github.com").
        if owner and ":" not in owner and "." not in owner:
            basename = f"{owner}-{basename}"
    return _sanitize_stem(basename) or "repo"


def _safe_zip_path(packages_dir: str, base_name: str) -> str:
    """Return an unused path under ``packages_dir`` for ``base_name``.

    On collision, append ``.1``, ``.2``, ... before the ``.zip`` extension.
    Avoids silently overwriting an existing artifact when two runs share a
    seconds-precision timestamp (rare but observed in tight CI loops).
    """
    candidate = os.path.join(packages_dir, base_name)
    if not os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(base_name)
    n = 1
    while True:
        candidate = os.path.join(packages_dir, f"{stem}.{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _add_repo_tree(
    zf: zipfile.ZipFile,
    repo_path: str,
    arc_root: str,
) -> tuple[int, int]:
    """Add ``repo_path`` to ``zf`` under ``arc_root/``.

    Excludes:
      * directories named in ``_EXCLUDED_DIRS`` (e.g. ``.git``)
      * any symlink (file or dir) — security: avoids packaging files
        outside ``repo_path`` if a malicious tree links to them.

    Returns ``(files_added, symlinks_skipped)``.
    """
    files_added = 0
    symlinks_skipped = 0

    # ``followlinks=False`` keeps os.walk from descending into symlinked dirs.
    for root, dirs, files in os.walk(repo_path, followlinks=False):
        # Skip symlinked directories (os.walk lists them in ``dirs`` but
        # would only recurse if followlinks=True; we still want to log).
        kept_dirs = []
        for d in dirs:
            full = os.path.join(root, d)
            if d in _EXCLUDED_DIRS:
                continue
            if os.path.islink(full):
                symlinks_skipped += 1
                logger.warning(
                    "Skipping symlinked directory in package: %s", full
                )
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        for fname in files:
            abs_p = os.path.join(root, fname)
            if os.path.islink(abs_p):
                symlinks_skipped += 1
                logger.warning("Skipping symlink in package: %s", abs_p)
                continue
            try:
                rel_p = os.path.relpath(abs_p, start=repo_path)
                zf.write(abs_p, arcname=os.path.join(arc_root, rel_p))
                files_added += 1
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unreadable file %s: %s", abs_p, exc)

    return files_added, symlinks_skipped


def _sha256(path: str) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_artifacts(
    verification: Optional[Dict[str, Any]],
    curated: Optional[List[Dict[str, Any]]],
    repo_path: str,
    container_repo_path: Optional[str],
) -> List[Dict[str, Any]]:
    """Return the verified riscv64 outputs for the manifest.

    Each entry keeps the scanner evidence (type, arch, ABI, size), the
    curated role when there is one, and the SHA-256 of files that are
    inside the zipped repository tree. Outputs installed outside the
    tree (``/usr/local/bin``) are listed with ``in_zip: false``.
    """
    roles = {
        (a.get("filepath") or a.get("path")): a.get("role")
        for a in (curated or [])
    }
    root = (container_repo_path or "").rstrip("/")
    entries: List[Dict[str, Any]] = []
    for art in (verification or {}).get("riscv64", []):
        path = art.get("path", "")
        entry = {
            "path": path,
            "type": art.get("type"),
            "arch": art.get("arch"),
            "abi": art.get("abi"),
            "size": art.get("size"),
            "in_zip": False,
        }
        if roles.get(path):
            entry["role"] = roles[path]
        if root and path.startswith(root + "/"):
            rel = path[len(root) + 1:]
            host_file = os.path.join(repo_path, rel)
            entry["path_in_zip"] = rel
            if os.path.isfile(host_file) and not os.path.islink(host_file):
                entry["in_zip"] = True
                entry["sha256"] = _sha256(host_file)
        entries.append(entry)
    return entries


def package_build(
    repo_name: str,
    repo_path: str,
    recipe_path: str,
    platform_name: str,
    packages_dir: str,
    repo_url: Optional[str] = None,
    agent_log_path: Optional[str] = None,
    batch_log_path: Optional[str] = None,
    verification: Optional[Dict[str, Any]] = None,
    curated_artifacts: Optional[List[Dict[str, Any]]] = None,
    container_repo_path: Optional[str] = None,
    package_tests: Optional[Dict[str, Any]] = None,
) -> str:
    """Produce a zip artifact for a successful build.

    Args:
        repo_name: The repository directory name; also the file-name
            stem when ``repo_url`` is not known.
        repo_path: Host path to the cloned repository directory.
        recipe_path: Host path to the porting recipe markdown file.
        platform_name: ``"alpine"`` / ``"debian"`` / etc — used in the
            filename and manifest.
        packages_dir: Host directory in which to write the zip.
        repo_url: Original git URL. It gives the ``<owner>-<repo>``
            file-name stem and is recorded in the manifest.
        agent_log_path: Optional host path to the per-package agent log
            (``workspace/logs/agent_<repo>.log``). Included at the zip
            root as ``agent_<repo>.log`` when the file exists.
        batch_log_path: Optional host path to the per-package batch log
            (``output/batch_logs/<repo>.log``). Included at the zip
            root as ``<repo>.log`` when the file exists.
        verification: The artifact verdict
            (``AgentState.artifact_verification``). The manifest says
            whether the riscv64 output was proven.
        curated_artifacts: Curated artifacts; their roles are copied.
        container_repo_path: The repository path in the sandbox, used
            to find verified outputs inside the zipped tree.
        package_tests: The non-gating package test result
            (``AgentState.package_tests``); its summary goes in the
            manifest.

    Returns:
        Absolute path to the created zip file.

    Raises:
        FileNotFoundError: if ``repo_path`` is not a directory or
            ``recipe_path`` is not a regular file.
        OSError: on disk / zip write failures.
    """
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Repository directory not found for packaging: {repo_path}"
        )
    if not os.path.isfile(recipe_path):
        raise FileNotFoundError(
            f"Build recipe not found for packaging: {recipe_path}"
        )

    os.makedirs(packages_dir, exist_ok=True)

    stem = package_stem(repo_url) if repo_url else repo_name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{stem}-{timestamp}-{platform_name}.zip"
    zip_path = _safe_zip_path(packages_dir, base_name)

    verdict = verification or {}
    manifest = {
        "schema_version": 2,
        "repo_name": repo_name,
        "repo_url": repo_url,
        "package_stem": stem,
        "legacy_package_stem": (
            legacy_package_stem(repo_url) if repo_url else repo_name
        ),
        "platform": platform_name,
        "created_at": datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "verification": {
            "status": verdict.get("status", "not run"),
            "verified": bool(verdict.get("verified")),
            "reason": verdict.get("reason"),
            "source_commit": verdict.get("source_commit"),
            "counts": verdict.get("counts", {}),
            "abi_warnings": verdict.get("abi_warnings", []),
            "expected_missing": verdict.get("expected_missing", []),
            "truncated": bool(verdict.get("truncated")),
            "scan_error": verdict.get("scan_error"),
        },
        "artifacts": _manifest_artifacts(
            verification, curated_artifacts, repo_path, container_repo_path
        ),
        "package_tests": {
            key: (package_tests or {}).get(key)
            for key in (
                "status",
                "framework",
                "command",
                "exit_code",
                "duration_seconds",
                "reason",
            )
        },
        "recipe_filename_in_zip": "build_recipe.md",
        "source_root_in_zip": repo_name,
        "logs_in_zip": [],
    }

    # Resolve which optional logs are actually present on disk. We log
    # (don't fail) when a caller hands us a missing path — the most
    # common case is the batch log being absent for single-package runs
    # invoked outside batch_test.
    logs_to_add: list[tuple[str, str]] = []  # (host_path, arcname)
    if batch_log_path:
        if os.path.isfile(batch_log_path):
            logs_to_add.append((batch_log_path, f"{repo_name}.log"))
        else:
            logger.warning(
                "Batch log not found, omitting from package: %s",
                batch_log_path,
            )
    if agent_log_path:
        if os.path.isfile(agent_log_path):
            logs_to_add.append((agent_log_path, f"agent_{repo_name}.log"))
        else:
            logger.warning(
                "Agent log not found, omitting from package: %s",
                agent_log_path,
            )
    manifest["logs_in_zip"] = [arc for _, arc in logs_to_add]

    logger.info("Creating package: %s", zip_path)
    # Write under a temporary name and rename at the end. A run killed
    # mid-write (a batch timeout) must not leave a truncated zip under
    # its final name: release upload and the planners match *.zip.
    part_path = zip_path + ".part"
    try:
        with zipfile.ZipFile(part_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(recipe_path, arcname="build_recipe.md")
            zf.writestr(
                "manifest.json", json.dumps(manifest, indent=2) + "\n"
            )
            for host_path, arcname in logs_to_add:
                zf.write(host_path, arcname=arcname)
            files_added, symlinks_skipped = _add_repo_tree(
                zf, repo_path, arc_root=repo_name
            )
        os.replace(part_path, zip_path)
    except BaseException:
        try:
            os.remove(part_path)
        except OSError:
            pass
        raise

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    logger.info(
        "Package built: %s (%.1f MB, %d files, %d symlinks skipped)",
        zip_path,
        size_mb,
        files_added,
        symlinks_skipped,
    )
    return zip_path


__all__ = ["legacy_package_stem", "package_build", "package_stem"]

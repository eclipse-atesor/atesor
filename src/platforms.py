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

"""Platform profiles for the RISC-V porting sandbox.

A ``PlatformProfile`` describes everything that varies between distros:
package-manager commands, canonical-name → distro package map, common
"wrong-name" corrections (Debian users guessing Alpine names and vice
versa), target triplet, libc, and the lock-file path used to serialize
concurrent installs.

Profiles are auto-detected from the active container's
``/etc/os-release`` at startup. A user can force a profile via the
``ATESOR_PLATFORM`` env var or the ``--platform`` CLI flag (handled in
main.py).

Adding a new profile = one PROFILES entry; nothing else changes.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Profile dataclass
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PlatformProfile:
    """Distro-specific knowledge needed to install packages and build code."""

    # Identity
    name: str  # "alpine", "debian", "ubuntu"
    display_name: str  # human-readable, used in prompts
    libc: str  # "musl", "glibc"
    target_triplet: str  # e.g. "riscv64-alpine-linux-musl"

    # Sandbox container identity
    dockerfile: str  # path to the Dockerfile for this profile
    image_name: str  # docker image tag
    container_name: str  # docker container name

    # Package manager command templates
    pkg_install: str  # "apk add", "apt-get install -y"
    pkg_update: str  # "apk update", "apt-get update"
    pkg_remove: str  # "apk del", "apt-get remove -y"
    pkg_lock_file: str  # path used with `flock` to serialize installs

    # Canonical name (used by scripted_ops/knowledge) → distro package name
    package_map: Dict[str, str] = field(default_factory=dict)

    # Wrong-name corrections (key is the wrong name an LLM may emit).
    # E.g. on Alpine "liblzma-dev" → "xz-dev"; on Debian, the reverse.
    name_corrections: Dict[str, str] = field(default_factory=dict)

    # Free-form notes appended to the system_knowledge prompt block
    extra_notes: List[str] = field(default_factory=list)

    # -- convenience helpers --------------------------------------------------

    def install_cmd(self, packages: List[str]) -> str:
        """Return a one-shot ``update && install`` command.

        Args:
            packages: Package names to install.

        Returns:
            A shell command that updates then installs the packages.
        """
        pkg_str = " ".join(packages)
        return f"{self.pkg_update} && {self.pkg_install} {pkg_str}"

    def resolve(self, canonical: str) -> str:
        """Map a canonical name to a distro package name.

        Args:
            canonical: Canonical tool or library name.

        Returns:
            The distro package name, or ``canonical`` if unmapped.
        """
        return self.package_map.get(canonical, canonical)


# ----------------------------------------------------------------------------
# Profile definitions
# ----------------------------------------------------------------------------

ALPINE_RISCV = PlatformProfile(
    name="alpine",
    display_name="Alpine Linux (riscv64, musl)",
    libc="musl",
    target_triplet="riscv64-alpine-linux-musl",
    dockerfile="Dockerfile",
    image_name="atesor-ai-sandbox:latest",
    container_name="atesor-ai-sandbox",
    pkg_install="apk add",
    pkg_update="apk update",
    pkg_remove="apk del",
    pkg_lock_file="/tmp/apk.lock",
    package_map={
        # build systems
        "cmake": "cmake",
        "make": "make",
        "ninja": "ninja",
        "meson": "meson",
        "autotools": "automake autoconf libtool",
        "gcc": "build-base",
        "g++": "build-base",
        "pkgconfig": "pkgconf",
        "nasm": "nasm",
        "yasm": "yasm",
        "perl": "perl",
        "fortran": "gfortran",
        # languages
        "go": "go",
        # NOTE: deliberately no "rust" entry — the sandbox bakes a modern
        # rustup toolchain into /root/.cargo (see Dockerfile
        # RUST_TOOLCHAIN). Alpine's `rust`/`cargo` packages are older and
        # would shadow it on PATH, which is what made 23 Rust packages
        # fail with "rustc <old> is not supported by the following
        # packages".
        "python3": "python3 py3-pip",
        "node": "nodejs npm",
        "java": "openjdk17",
        # core C/C++
        "zlib": "zlib-dev",
        "openssl": "openssl-dev",
        "curl": "curl-dev",
        "git": "git",
        "libatomic": "libatomic",
        # compression
        "zstd": "zstd-dev",
        "lz4": "lz4-dev",
        "brotli": "brotli-dev",
        "xz": "xz-dev",
        "bzip2": "bzip2-dev",
        "snappy": "snappy-dev",
        # images
        "libpng": "libpng-dev",
        "libjpeg": "libjpeg-turbo-dev",
        "libjpeg-turbo": "libjpeg-turbo-dev",
        "libwebp": "libwebp-dev",
        "libtiff": "tiff-dev",
        "openjpeg": "openjpeg-dev",
        "libheif": "libheif-dev",
        "lcms2": "lcms2-dev",
        "freetype": "freetype-dev",
        # audio/video
        "opus": "opus-dev",
        "flac": "flac-dev",
        "ogg": "libogg-dev",
        "vorbis": "libvorbis-dev",
        "dav1d": "dav1d-dev",
        # crypto
        "mbedtls": "mbedtls-dev",
        "libsodium": "libsodium-dev",
        # net
        "libssh2": "libssh2-dev",
        "nghttp2": "nghttp2-dev",
        "c-ares": "c-ares-dev",
        "libuv": "libuv-dev",
        "libevent": "libevent-dev",
        # data
        "libxml2": "libxml2-dev",
        "expat": "expat-dev",
        "jansson": "jansson-dev",
        "yaml": "yaml-dev",
        "pcre2": "pcre2-dev",
        # archive
        "libarchive": "libarchive-dev",
        # db
        "sqlite": "sqlite-dev",
        # C++
        "abseil-cpp": "abseil-cpp-dev",
        "absl": "abseil-cpp-dev",
        "protobuf": "protobuf-dev",
        "protoc": "protoc",
        "gtest": "gtest-dev",
        "benchmark": "benchmark-dev",
        # system
        "linux-headers": "linux-headers",
        "musl-dev": "musl-dev",
        "libexecinfo": "libexecinfo-dev",
        "libunwind": "libunwind-dev",
    },
    name_corrections={
        # Debian names → Alpine names
        "liblzma-dev": "xz-dev",
        "liblz4-dev": "lz4-dev",
        "libzdev-dev": "zlib-dev",
        "libz-dev": "zlib-dev",
        "libbz2-dev": "bzip2-dev",
        "libzstd-dev": "zstd-dev",
        "libcurl-dev": "curl-dev",
        "libssl-dev": "openssl-dev",
        "libtiff-dev": "tiff-dev",
        "libjpeg-dev": "libjpeg-turbo-dev",
        "libfreetype-dev": "freetype-dev",
        "libsqlite3-dev": "sqlite-dev",
        "liblcms2-dev": "lcms2-dev",
        "pkg-config": "pkgconf",
        "golang": "go",
        "build-essential": "build-base",
        "python3-dev": "python3-dev py3-pip",
        "zlib1g-dev": "zlib-dev",
    },
    extra_notes=[
        (
            "Alpine uses **musl libc**, not glibc — some GNU extensions"
            " are absent."
        ),
        (
            "Do NOT use cross-compilation flags when building natively"
            " inside the sandbox."
        ),
        (
            "Go is pre-installed in the sandbox — do NOT install it via"
            " the package manager."
        ),
        (
            "`GOPROXY`, `GONOSUMCHECK`, and `GOFLAGS=-buildvcs=false`"
            " are pre-set."
        ),
        (
            'If `git` is not installed ("git: command not found"),'
            " install it via the package manager first."
        ),
        (
            "For large RISC-V functions, link with"
            " `-mcmodel=medany` to avoid `relocation truncated to fit`"
            " errors."
        ),
        (
            "If the bundled Go toolchain is too old for go.mod,"
            " download a newer riscv64 Go binary from"
            " https://go.dev/dl/ as a recovery option."
        ),
    ],
)


DEBIAN_RISCV = PlatformProfile(
    name="debian",
    display_name="Debian / Ubuntu (riscv64, glibc)",
    libc="glibc",
    target_triplet="riscv64-unknown-linux-gnu",
    dockerfile="Dockerfile.debian",
    image_name="atesor-ai-sandbox-debian:latest",
    container_name="atesor-ai-sandbox-debian",
    pkg_install="apt-get install -y --no-install-recommends",
    pkg_update="apt-get update",
    pkg_remove="apt-get remove -y",
    # /var/lib/dpkg/lock-frontend only exists after the first apt run;
    # use our own lock file instead.
    pkg_lock_file="/tmp/apt.lock",
    package_map={
        # build systems
        "cmake": "cmake",
        "make": "make",
        "ninja": "ninja-build",
        "meson": "meson",
        "autotools": "autoconf automake libtool",
        "gcc": "build-essential",
        "g++": "build-essential",
        "pkgconfig": "pkg-config",
        "nasm": "nasm",
        "yasm": "yasm",
        "perl": "perl",
        "fortran": "gfortran",
        # languages
        # NOTE: deliberately no "go" entry — Ubuntu jammy's golang-go package
        # is Go 1.18, which cannot parse modern go.mod files (`toolchain`
        # directive, `go 1.24.0` 3-part version). The sandbox bakes a recent
        # Go riscv64 toolchain into /usr/local/go via Dockerfile.debian; let
        # the agent use that and never `apt-get install golang*`.
        # Likewise no "rust" entry: rustup provides a current toolchain in
        # /root/.cargo, and jammy's rustc would shadow it on PATH.
        "python3": "python3 python3-pip python3-dev",
        "node": "nodejs npm",
        "java": "openjdk-17-jdk",
        # core C/C++
        "zlib": "zlib1g-dev",
        "openssl": "libssl-dev",
        "curl": "libcurl4-openssl-dev",
        "git": "git",
        "libatomic": "libatomic1",
        # compression
        "zstd": "libzstd-dev",
        "lz4": "liblz4-dev",
        "brotli": "libbrotli-dev",
        "xz": "liblzma-dev",
        "bzip2": "libbz2-dev",
        "snappy": "libsnappy-dev",
        # images
        "libpng": "libpng-dev",
        "libjpeg": "libjpeg-dev",
        "libjpeg-turbo": "libjpeg-turbo8-dev",
        "libwebp": "libwebp-dev",
        "libtiff": "libtiff-dev",
        "openjpeg": "libopenjp2-7-dev",
        "libheif": "libheif-dev",
        "lcms2": "liblcms2-dev",
        "freetype": "libfreetype6-dev",
        # audio/video
        "opus": "libopus-dev",
        "flac": "libflac-dev",
        "ogg": "libogg-dev",
        "vorbis": "libvorbis-dev",
        "dav1d": "libdav1d-dev",
        # crypto
        "mbedtls": "libmbedtls-dev",
        "libsodium": "libsodium-dev",
        # net
        "libssh2": "libssh2-1-dev",
        "nghttp2": "libnghttp2-dev",
        "c-ares": "libc-ares-dev",
        "libuv": "libuv1-dev",
        "libevent": "libevent-dev",
        # data
        "libxml2": "libxml2-dev",
        "expat": "libexpat1-dev",
        "jansson": "libjansson-dev",
        "yaml": "libyaml-dev",
        "pcre2": "libpcre2-dev",
        # archive
        "libarchive": "libarchive-dev",
        # db
        "sqlite": "libsqlite3-dev",
        # C++
        "abseil-cpp": "libabsl-dev",
        "absl": "libabsl-dev",
        "protobuf": "libprotobuf-dev",
        "protoc": "protobuf-compiler",
        "gtest": "libgtest-dev",
        "benchmark": "libbenchmark-dev",
        # system
        "linux-headers": "linux-libc-dev",
        "musl-dev": "libc6-dev",
        # backtrace is in glibc, so no separate package is needed.
        "libexecinfo": "libc6-dev",
        "libunwind": "libunwind-dev",
    },
    name_corrections={
        # Alpine names → Debian names
        "xz-dev": "liblzma-dev",
        "lz4-dev": "liblz4-dev",
        "zlib-dev": "zlib1g-dev",
        "bzip2-dev": "libbz2-dev",
        "zstd-dev": "libzstd-dev",
        "curl-dev": "libcurl4-openssl-dev",
        "openssl-dev": "libssl-dev",
        "tiff-dev": "libtiff-dev",
        "libjpeg-turbo-dev": "libjpeg-turbo8-dev",
        "freetype-dev": "libfreetype6-dev",
        "sqlite-dev": "libsqlite3-dev",
        "lcms2-dev": "liblcms2-dev",
        "pkgconf": "pkg-config",
        "build-base": "build-essential",
        "musl-dev": "libc6-dev",
        "py3-pip": "python3-pip",
        # Common LLM-hallucinated Debian names → real Debian names.
        # These were observed in batch failures (axel, nghttp2, libarchive, …)
        # where the planner/scout invented a plausible-looking but non-existent
        # package name. Apply silent rewrites instead of failing the install.
        "libcurl4-libssl-dev": "libcurl4-openssl-dev",
        "libcurl-openssl-dev": "libcurl4-openssl-dev",
        "libcares-dev": "libc-ares-dev",
        "libnettle-dev": "nettle-dev",
        "libabseil-dev": "libabsl-dev",
        # Frequent typo/hallucination repairs seen in batch logs.
        "libz1g-dev": "zlib1g-dev",
        "libgssapi-krb5-dev": "libkrb5-dev",
        "libgssapi-krb5": "libkrb5-dev",
    },
    extra_notes=[
        (
            "Debian/Ubuntu uses **glibc**, not musl — `backtrace()`,"
            " `mallinfo()`, and most GNU extensions are available."
        ),
        (
            "Always run `apt-get update` before the first `apt-get"
            " install` in a phase."
        ),
        "Use `--no-install-recommends` to keep the image small.",
        (
            "Run apt non-interactively: prefix with"
            " `DEBIAN_FRONTEND=noninteractive` if a package may prompt."
        ),
        (
            "Go is pre-installed at /usr/local/go (riscv64) — do NOT"
            " `apt-get install golang*`. Ubuntu jammy ships Go 1.18"
            " which cannot parse modern go.mod files (toolchain"
            " directive, 3-part `go 1.X.Y` version)."
        ),
        (
            "GOTOOLCHAIN=local is set image-wide; if go.mod requires a"
            " newer Go than the bundled toolchain, treat as"
            " MISSING_TOOLS and escalate -- do NOT attempt apt golang"
            " install."
        ),
        (
            'If `git` is not installed ("git: command not found"),'
            " install it via the package manager first."
        ),
        (
            "For large RISC-V functions, link with"
            " `-mcmodel=medany` to avoid `relocation truncated to fit`"
            " errors."
        ),
        (
            "If the bundled Go toolchain is too old for go.mod,"
            " download a newer riscv64 Go binary from"
            " https://go.dev/dl/ as a recovery option."
        ),
    ],
)


PROFILES: Dict[str, PlatformProfile] = {
    "alpine": ALPINE_RISCV,
    "debian": DEBIAN_RISCV,
    # apt-based; package names identical to Debian for our purposes.
    "ubuntu": DEBIAN_RISCV,
}


# ----------------------------------------------------------------------------
# Detection & access
# ----------------------------------------------------------------------------

_DEFAULT_PROFILE = ALPINE_RISCV
_cached_profile: Optional[PlatformProfile] = None


def detect_platform(container_name: Optional[str] = None) -> PlatformProfile:
    """Read ``/etc/os-release`` from the container and pick a profile.

    Falls back to ALPINE_RISCV if detection fails (container not
    running, docker missing, unknown distro). Honors the
    ``ATESOR_PLATFORM`` env var as an override -- useful for testing and
    the ``--platform`` CLI flag.

    Args:
        container_name: Container to inspect. Defaults to the active
            container when ``None``.

    Returns:
        The detected platform profile.
    """
    override = os.environ.get("ATESOR_PLATFORM", "").strip().lower()
    if override:
        if override in PROFILES:
            logger.info(f"Platform override via ATESOR_PLATFORM={override}")
            return PROFILES[override]
        logger.warning(f"ATESOR_PLATFORM={override!r} is unknown; ignoring")

    if container_name is None:
        # Resolve the container to inspect WITHOUT calling
        # get_container_name()/get_active_profile(): those depend on the
        # very profile we are detecting, so routing through them here
        # recurses (detect_platform -> get_container_name ->
        # get_active_profile -> detect_platform) until RecursionError.
        container_name = (
            os.environ.get("ATESOR_CONTAINER", "").strip()
            or _DEFAULT_PROFILE.container_name
        )

    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "cat", "/etc/os-release"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.debug(
                f"detect_platform: docker exec returned {result.returncode}"
            )
            return _DEFAULT_PROFILE

        os_id = ""
        for line in result.stdout.splitlines():
            if line.startswith("ID="):
                os_id = line.split("=", 1)[1].strip().strip('"').lower()
                break

        if os_id in PROFILES:
            logger.info(
                "Detected sandbox platform: "
                f"{os_id} → {PROFILES[os_id].display_name}"
            )
            return PROFILES[os_id]
        logger.warning(
            f"Unknown sandbox distro ID={os_id!r}; "
            f"defaulting to {_DEFAULT_PROFILE.name}"
        )
        return _DEFAULT_PROFILE
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug(f"detect_platform failed: {exc}; using default")
        return _DEFAULT_PROFILE


def get_active_profile() -> PlatformProfile:
    """Return the cached active platform profile (detect on first call)."""
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = detect_platform()
    return _cached_profile


def get_container_name() -> str:
    """Resolve the Docker container name for the current process.

    Resolution order:
        1. ``ATESOR_CONTAINER`` env var -- explicit override. Used by
           ``batch_test.py`` to assign each worker its own container
           (the "per-worker pool" model, so dpkg/apk locks do not
           serialize across parallel agents).
        2. The active platform profile's default ``container_name``
           (single-tenant mode).

    Returning a string (not the profile object) lets every caller share
    one canonical name, keeping DockerConfig, main.py, and platform
    detection in lock-step.

    Returns:
        The container name to use.
    """
    override = os.environ.get("ATESOR_CONTAINER")
    if override:
        return override.strip()
    return get_active_profile().container_name


def set_active_profile(
    profile_or_name: Union[str, PlatformProfile],
) -> PlatformProfile:
    """Manually set the cached active profile (used by CLI override).

    Args:
        profile_or_name: A profile object, or a profile key in PROFILES.

    Returns:
        The newly active platform profile.

    Raises:
        ValueError: If a string name is not a known platform.
    """
    global _cached_profile
    if isinstance(profile_or_name, str):
        if profile_or_name not in PROFILES:
            raise ValueError(
                f"unknown platform {profile_or_name!r}; "
                f"choose from {sorted(PROFILES)}"
            )
        _cached_profile = PROFILES[profile_or_name]
    else:
        _cached_profile = profile_or_name
    logger.info(
        f"Active platform profile set to: {_cached_profile.display_name}"
    )
    return _cached_profile


__all__ = [
    "PlatformProfile",
    "ALPINE_RISCV",
    "DEBIAN_RISCV",
    "PROFILES",
    "detect_platform",
    "get_active_profile",
    "get_container_name",
    "set_active_profile",
]

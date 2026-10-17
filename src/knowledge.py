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

"""Static knowledge base for RISC-V porting.

Distro-specific bits (package names, install commands, libc notes)
live in `src.platforms.PlatformProfile`. This module covers the
distro-agnostic RISC-V architecture knowledge plus a renderer that
composes the full system_knowledge prompt block from a given profile.
`src.graph` injects the rendered block into the scout and fixer
prompts.
"""

from typing import Optional

from .platforms import (
    ALPINE_RISCV,
    PlatformProfile,
    get_active_profile,
)

# Backward-compat aliases — older code/tests may import these names.
ALPINE_TOOL_MAP = ALPINE_RISCV.package_map
ALPINE_PACKAGE_CORRECTIONS = ALPINE_RISCV.name_corrections

# RISC-V architecture knowledge

RISCV_PREPROCESSOR_MACROS = {
    "__riscv": "Defined on any RISC-V target",
    "__riscv_xlen": "Register width: 32 or 64",
    "__riscv_flen": (
        "FP register width: 32 (F ext) or 64 (D ext), absent if no FP"
    ),
    "__riscv_atomic": "Defined when A (atomic) extension is present",
    "__riscv_mul": "Defined when M (multiply/divide) extension is present",
    "__riscv_vector": "Defined when V (vector) extension is present",
    "__riscv_compressed": "Defined when C (compressed) extension is present",
}

# Failure patterns seen when porting to RISC-V. Only ``musl_libc`` is
# distro-specific; the renderer drops it for glibc profiles.
COMMON_PORTING_ISSUES = {
    "x86_intrinsics": {
        "symptoms": [
            "error: unknown type name '__m128'",
            "error: unknown type name '__m256'",
            "undefined reference to '_mm_*'",
            "undefined reference to '_mm256_*'",
            "'xmmintrin.h' file not found",
            "'immintrin.h' file not found",
        ],
        "root_cause": "x86 SSE/AVX SIMD intrinsics not available on RISC-V",
        "solutions": [
            (
                "Use SIMDe (SIMD Everywhere) header-only library for"
                " portable SIMD"
            ),
            (
                "Provide scalar C fallback under #if"
                " !defined(__x86_64__) && !defined(__aarch64__)"
            ),
            (
                "Use RISC-V Vector (RVV) intrinsics if V extension is"
                " available (#ifdef __riscv_vector)"
            ),
            (
                "Disable SIMD via build option (e.g., -DENABLE_SSE=OFF,"
                " -DWITH_SIMD=0)"
            ),
        ],
    },
    "inline_assembly": {
        "symptoms": [
            "unknown register name",
            "invalid instruction mnemonic",
            "error in asm",
        ],
        "root_cause": "x86/ARM inline assembly incompatible with RISC-V ISA",
        "solutions": [
            "Guard with #if defined(__x86_64__) / #elif defined(__riscv)",
            "Replace with portable C code or compiler builtins",
            (
                "Provide RISC-V assembly implementation for"
                " performance-critical paths"
            ),
        ],
    },
    "builtin_functions": {
        "symptoms": [
            "__builtin_ia32_",
            "__rdtsc",
            "__builtin_cpu_supports",
        ],
        "root_cause": "x86-specific GCC/Clang builtin functions",
        "solutions": [
            (
                "Replace __rdtsc with clock_gettime(CLOCK_MONOTONIC) —"
                " portable and works with GCC (the"
                " __builtin_readcyclecounter alternative is Clang-only)"
            ),
            "Replace __builtin_ia32_* with portable C equivalents",
            "Use C11 atomics instead of __sync_* builtins where possible",
        ],
    },
    "memory_ordering": {
        "symptoms": [
            "data races under RISC-V that pass on x86",
            "lock-free algorithms failing",
        ],
        "root_cause": (
            "RISC-V has a weaker memory model (RVWMO) than x86 (TSO)"
        ),
        "solutions": [
            "Use C11/C++11 atomics with explicit memory orderings",
            (
                "Add __atomic_thread_fence() where x86 relied on"
                " implicit ordering"
            ),
            "Audit lock-free data structures for RISC-V memory model",
        ],
    },
    "musl_libc": {
        "symptoms": [
            "execinfo.h: No such file or directory",
            "undefined reference to 'backtrace'",
            "undefined reference to 'backtrace_symbols'",
            "undefined reference to 'mallinfo'",
            "error: 'REG_RIP' undeclared",
            "error: 'REG_EIP' undeclared",
            "'sys/cdefs.h' file not found",
        ],
        "root_cause": (
            "Alpine uses musl libc, not glibc; some GNU extensions"
            " are absent"
        ),
        "solutions": [
            "Guard backtrace usage with #ifdef __GLIBC__",
            "Replace mallinfo with platform-independent memory tracking",
            (
                "Use libunwind for backtraces on musl (libexecinfo was"
                " REMOVED from Alpine 3.17+ — do not suggest it)"
            ),
            "Use portable signal handling instead of x86 register names",
        ],
    },
    "config_guess": {
        "symptoms": [
            "machine 'riscv64' not recognized",
            "unsupported host cpu type",
            "invalid configuration",
            "config.sub: too many arguments",
        ],
        "root_cause": (
            "Stale config.guess/config.sub that predate RISC-V support"
        ),
        "solutions": [
            (
                "Update config.guess and config.sub from the installed"
                " automake: cp /usr/share/automake-*/config.* ."
                " (or regenerate with autoreconf -fi)"
            ),
            (
                "Specify --build explicitly: ./configure"
                " --build=<native target triplet> (never --host — we"
                " compile natively)"
            ),
            "For CMake: no action needed (CMake auto-detects RISC-V)",
        ],
    },
    "atomics_linking": {
        "symptoms": [
            "undefined reference to '__atomic_load'",
            "undefined reference to '__atomic_store'",
            "undefined reference to '__atomic_compare_exchange'",
        ],
        "root_cause": "Some atomic operations require libatomic on RISC-V",
        "solutions": [
            (
                "Link with -latomic (add to LDFLAGS or CMake"
                " target_link_libraries)"
            ),
            "For CMake: target_link_libraries(target PRIVATE atomic)",
            "For Makefile: LDFLAGS += -latomic",
        ],
    },
    "code_model": {
        "symptoms": [
            "relocation truncated to fit",
            "relocation R_RISCV_HI20",
            "relocation R_RISCV_CALL",
        ],
        "root_cause": (
            "RISC-V default code model (-mcmodel=medlow) limits code to"
            " 2 GiB; large functions exceed this"
        ),
        "solutions": [
            (
                "Compile with -mcmodel=medany (medium/any) which uses"
                " PC-relative addressing up to 4 GiB"
            ),
            ("Add to CFLAGS/CXXFLAGS: -mcmodel=medany"),
            (
                "For CMake: set(CMAKE_C_FLAGS"
                ' "${CMAKE_C_FLAGS} -mcmodel=medany")'
            ),
        ],
    },
    "cache_line_size": {
        "symptoms": [
            "hardcoded CACHELINE_SIZE 64",
            "performance regression due to false sharing",
        ],
        "root_cause": (
            "RISC-V cache line size varies by implementation"
            " (32/64/128 bytes)"
        ),
        "solutions": [
            "Use runtime detection or conservative default",
            (
                "Replace hardcoded values with sysconf or macro-based"
                " detection"
            ),
        ],
    },
}


# Build systems whose own code is not C or C++: the C preprocessor
# macros and the C/C++ porting issues do not apply to them.
_NON_C_BUILD_SYSTEMS = frozenset(
    {"go", "cargo", "rust", "python", "pip", "npm", "node"}
)

# Porting issues that apply to every build system.
_GENERIC_ISSUES = frozenset({"memory_ordering", "musl_libc"})

# A platform note that names one of these is only for Go builds.
_GO_NOTE_MARKERS = ("go.mod", "Go ", "GOTOOLCHAIN", "GOPROXY", "golang")


def _note_applies(note: str, build_system: Optional[str]) -> bool:
    """Return True when a platform note concerns ``build_system``."""
    if not build_system or build_system == "go":
        return True
    return not any(marker in note for marker in _GO_NOTE_MARKERS)


def get_system_knowledge_summary(
    profile: Optional[PlatformProfile] = None,
    build_system: Optional[str] = None,
) -> str:
    """Produce a compact knowledge summary for agent prompts.

    Distro-specific text (install command, package map, libc,
    corrections) is rendered from ``profile``. With a ``build_system``,
    the block leaves out what cannot apply to it: the C macros and the
    C/C++ porting issues for a Go, Rust, Python or Node build, and the
    Go notes for a non-Go build. The text depends only on the profile
    and the build system, so it stays cacheable across packages.

    Args:
        profile: Platform profile to render distro-specific text from.
            Defaults to the active platform profile when ``None``.
        build_system: The plan's build system (``go``, ``cmake``, ...);
            ``None`` or ``unknown`` keeps the full block.

    Returns:
        The composed system-knowledge prompt block as a string.
    """
    if profile is None:
        profile = get_active_profile()
    build_system = (build_system or "").lower() or None
    if build_system == "unknown":
        build_system = None
    non_c = build_system in _NON_C_BUILD_SYSTEMS

    lines = [
        f"## RISC-V Porting Knowledge ({profile.display_name})",
        "",
        "### Package Installation",
        (
            f"Install: run `{profile.pkg_update}`, then "
            f"`{profile.pkg_install} <package>` as a separate command; "
            f"a fixer action never chains commands."
        ),
        "Canonical name → distro package:",
    ]
    for canonical, pkg in profile.package_map.items():
        lines.append(f"  - {canonical} → `{pkg}`")

    if not non_c:
        lines.append("")
        lines.append("### Architecture Detection (C/C++ preprocessor)")
        for macro, desc in RISCV_PREPROCESSOR_MACROS.items():
            lines.append(f"  - `{macro}`: {desc}")

    lines.append("")
    lines.append("### Common Porting Issues")
    for issue_key, issue in COMMON_PORTING_ISSUES.items():
        # musl-specific guidance is wrong (and actively misleading) in
        # a glibc sandbox prompt — skip it there.
        if issue_key == "musl_libc" and profile.libc != "musl":
            continue
        if non_c and issue_key not in _GENERIC_ISSUES:
            continue
        lines.append(f"  **{issue_key}**: {issue['root_cause']}")
        lines.append(f"    Fix: {issue['solutions'][0]}")
        # "Do not" warnings must reach the prompt even when they are not
        # the first solution; libexecinfo, for one, is gone from Alpine
        # and from the package map, so an LLM that suggests it fails.
        for extra in issue["solutions"][1:]:
            if "do not" in extra.lower():
                lines.append(f"    Also: {extra}")

    lines.append("")
    lines.append("### Key Facts")
    lines.append(
        f"  - libc: **{profile.libc}** — adjust GNU-extension usage"
        " accordingly"
    )
    lines.append(f"  - Native target triplet: `{profile.target_triplet}`")
    lines.append(
        "  - Do NOT use cross-compilation flags when building natively"
        " inside the sandbox"
    )
    lines.append(
        "  - RISC-V memory model (RVWMO) is weaker than x86 (TSO) —"
        " explicit fences may be needed"
    )
    lines.append(
        "  - RISC-V ISA extensions vary by hardware — detect, do not assume"
    )
    for note in profile.extra_notes:
        if _note_applies(note, build_system):
            lines.append(f"  - {note}")

    if profile.name_corrections:
        lines.append("")
        lines.append(f"### Package Name Corrections ({profile.display_name})")
        lines.append(
            "  Names from another distro: the agent rewrites the left"
            " name to the right one. Prefer the right name:"
        )
        for wrong, correct in profile.name_corrections.items():
            lines.append(f"  - `{wrong}` → use `{correct}`")

    return "\n".join(lines)

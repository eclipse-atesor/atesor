"""Tests for src/knowledge.py — system knowledge summary rendering."""

import unittest

from src import platforms
from src.knowledge import (
    ALPINE_PACKAGE_CORRECTIONS,
    ALPINE_TOOL_MAP,
    RISCV_PREPROCESSOR_MACROS,
    get_system_knowledge_summary,
)
from src.platforms import ALPINE_RISCV, DEBIAN_RISCV


class TestSystemKnowledgeSummary(unittest.TestCase):
    """Tests for get_system_knowledge_summary."""

    def test_summary_includes_alpine_display_name(self) -> None:
        out = get_system_knowledge_summary(ALPINE_RISCV)
        self.assertIn(ALPINE_RISCV.display_name, out)
        self.assertIn("apk update", out)
        self.assertIn("apk add", out)

    def test_summary_includes_debian_display_name(self) -> None:
        out = get_system_knowledge_summary(DEBIAN_RISCV)
        self.assertIn(DEBIAN_RISCV.display_name, out)
        self.assertIn("apt-get update", out)
        self.assertIn("apt-get install", out)

    def test_summary_lists_riscv_macros(self) -> None:
        out = get_system_knowledge_summary(ALPINE_RISCV)
        for macro in RISCV_PREPROCESSOR_MACROS:
            with self.subTest(macro=macro):
                self.assertIn(macro, out)

    def test_summary_includes_name_corrections_section(self) -> None:
        out = get_system_knowledge_summary(ALPINE_RISCV)
        self.assertIn("Package Name Corrections", out)
        # spot check a known correction
        self.assertIn("liblzma-dev", out)
        self.assertIn("xz-dev", out)

    def test_summary_includes_libc_marker(self) -> None:
        self.assertIn("musl", get_system_knowledge_summary(ALPINE_RISCV))
        self.assertIn("glibc", get_system_knowledge_summary(DEBIAN_RISCV))

    def test_summary_defaults_to_active_profile(self) -> None:
        platforms.set_active_profile("alpine")
        out = get_system_knowledge_summary()
        self.assertIn("Alpine", out)
        platforms.set_active_profile("debian")
        out = get_system_knowledge_summary()
        self.assertIn("Debian", out)

    def test_summary_mentions_extra_notes(self) -> None:
        out = get_system_knowledge_summary(ALPINE_RISCV)
        # ALPINE_RISCV.extra_notes includes "musl libc" mention
        self.assertIn("musl libc", out)

    def test_alpine_summary_keeps_libexecinfo_warning(self) -> None:
        """Alpine knowledge warns not to install removed libexecinfo."""
        out = get_system_knowledge_summary(ALPINE_RISCV)
        self.assertIn("libexecinfo was REMOVED", out)

    def test_install_line_is_two_commands_not_a_chain(self) -> None:
        """Fixer actions never chain, so the knowledge must not teach it."""
        for profile in (ALPINE_RISCV, DEBIAN_RISCV):
            with self.subTest(profile=profile.name):
                out = get_system_knowledge_summary(profile)
                self.assertNotIn("&&", out)
                self.assertIn("as a separate command", out)

    def test_go_build_drops_c_only_sections(self) -> None:
        out = get_system_knowledge_summary(DEBIAN_RISCV, build_system="go")
        self.assertNotIn("__riscv_xlen", out)
        self.assertNotIn("x86_intrinsics", out)
        self.assertIn("memory_ordering", out)
        self.assertIn("GOTOOLCHAIN", out)
        self.assertIn("libssl-dev", out)

    def test_c_build_drops_go_notes(self) -> None:
        out = get_system_knowledge_summary(DEBIAN_RISCV, build_system="cmake")
        self.assertIn("__riscv_xlen", out)
        self.assertIn("x86_intrinsics", out)
        self.assertNotIn("GOTOOLCHAIN", out)
        self.assertNotIn("go.mod", out)

    def test_unknown_build_system_keeps_the_full_block(self) -> None:
        self.assertEqual(
            get_system_knowledge_summary(DEBIAN_RISCV, build_system="unknown"),
            get_system_knowledge_summary(DEBIAN_RISCV),
        )

    def test_go_notes_do_not_contradict_each_other(self) -> None:
        """The builder upgrades Go itself; no note says to escalate."""
        for profile in (ALPINE_RISCV, DEBIAN_RISCV):
            with self.subTest(profile=profile.name):
                out = get_system_knowledge_summary(profile, build_system="go")
                self.assertNotIn("escalate --", out)
                self.assertNotIn("MISSING_TOOLS and escalate", out)
                self.assertIn("builder downloads that Go release", out)

    def test_summary_includes_package_map_entries(self) -> None:
        out = get_system_knowledge_summary(ALPINE_RISCV)
        # Spot-check a few canonicals
        for canonical in ["zlib", "openssl", "cmake"]:
            with self.subTest(canonical=canonical):
                self.assertIn(canonical, out)


class TestBackwardCompatAliases(unittest.TestCase):
    """Tests for the ALPINE_TOOL_MAP and ALPINE_PACKAGE_CORRECTIONS aliases."""

    def test_alpine_tool_map_aliases(self) -> None:
        self.assertIs(ALPINE_TOOL_MAP, ALPINE_RISCV.package_map)

    def test_alpine_package_corrections_aliases(self) -> None:
        self.assertIs(
            ALPINE_PACKAGE_CORRECTIONS, ALPINE_RISCV.name_corrections
        )


if __name__ == "__main__":
    unittest.main()

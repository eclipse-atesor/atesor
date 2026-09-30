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
    """Tests for SystemKnowledgeSummary."""

    def test_summary_includes_alpine_display_name(self) -> None:
        """Test summary includes alpine display name."""
        out = get_system_knowledge_summary(ALPINE_RISCV)
        self.assertIn(ALPINE_RISCV.display_name, out)
        self.assertIn("apk update", out)
        self.assertIn("apk add", out)

    def test_summary_includes_debian_display_name(self) -> None:
        """Test summary includes debian display name."""
        out = get_system_knowledge_summary(DEBIAN_RISCV)
        self.assertIn(DEBIAN_RISCV.display_name, out)
        self.assertIn("apt-get update", out)
        self.assertIn("apt-get install", out)

    def test_summary_lists_riscv_macros(self) -> None:
        """Test summary lists riscv macros."""
        out = get_system_knowledge_summary(ALPINE_RISCV)
        for macro in RISCV_PREPROCESSOR_MACROS:
            with self.subTest(macro=macro):
                self.assertIn(macro, out)

    def test_summary_includes_name_corrections_section(self) -> None:
        """Test summary includes name corrections section."""
        out = get_system_knowledge_summary(ALPINE_RISCV)
        self.assertIn("Package Name Corrections", out)
        # spot check a known correction
        self.assertIn("liblzma-dev", out)
        self.assertIn("xz-dev", out)

    def test_summary_includes_libc_marker(self) -> None:
        """Test summary includes libc marker."""
        self.assertIn("musl", get_system_knowledge_summary(ALPINE_RISCV))
        self.assertIn("glibc", get_system_knowledge_summary(DEBIAN_RISCV))

    def test_summary_defaults_to_active_profile(self) -> None:
        """Test summary defaults to active profile."""
        platforms.set_active_profile("alpine")
        out = get_system_knowledge_summary()
        self.assertIn("Alpine", out)
        platforms.set_active_profile("debian")
        out = get_system_knowledge_summary()
        self.assertIn("Debian", out)

    def test_summary_mentions_extra_notes(self) -> None:
        """Test summary mentions extra notes."""
        out = get_system_knowledge_summary(ALPINE_RISCV)
        # ALPINE_RISCV.extra_notes includes "musl libc" mention
        self.assertIn("musl libc", out)

    def test_alpine_summary_keeps_libexecinfo_warning(self) -> None:
        """Alpine knowledge warns not to install removed libexecinfo."""
        out = get_system_knowledge_summary(ALPINE_RISCV)
        self.assertIn("libexecinfo was REMOVED", out)

    def test_summary_includes_package_map_entries(self) -> None:
        """Test summary includes package map entries."""
        out = get_system_knowledge_summary(ALPINE_RISCV)
        # Spot-check a few canonicals
        for canonical in ["zlib", "openssl", "cmake"]:
            with self.subTest(canonical=canonical):
                self.assertIn(canonical, out)


class TestBackwardCompatAliases(unittest.TestCase):
    """Tests for BackwardCompatAliases."""

    def test_alpine_tool_map_aliases(self) -> None:
        """Test alpine tool map aliases."""
        self.assertIs(ALPINE_TOOL_MAP, ALPINE_RISCV.package_map)

    def test_alpine_package_corrections_aliases(self) -> None:
        """Test alpine package corrections aliases."""
        self.assertIs(
            ALPINE_PACKAGE_CORRECTIONS, ALPINE_RISCV.name_corrections
        )


if __name__ == "__main__":
    unittest.main()

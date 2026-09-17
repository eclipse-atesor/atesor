"""Tests for src/artifact_curator.py — rule-based + LLM curation."""

import unittest
from unittest import mock

from src.artifact_curator import _parse_curator_json, curate_artifacts


def _art(path, type="binary"):
    """Art."""
    return {"filepath": path, "type": type, "architecture": "RISC-V"}


class TestRuleBasedCurator(unittest.TestCase):
    """Tests for RuleBasedCurator."""

    def test_empty_input_returns_empty(self) -> None:
        """Test empty input returns empty."""
        self.assertEqual(curate_artifacts([], "foo"), [])

    def test_cmake_internals_dropped(self) -> None:
        """Test cmake internals dropped."""
        out = curate_artifacts(
            [
                _art("/b/CMakeFiles/CompilerIdC/a.out"),
                _art("/b/_deps/foo-build/x"),
                _art("/b/zlib"),
            ],
            "zlib",
        )
        paths = [a["filepath"] for a in out]
        self.assertNotIn("/b/CMakeFiles/CompilerIdC/a.out", paths)
        self.assertNotIn("/b/_deps/foo-build/x", paths)
        self.assertIn("/b/zlib", paths)

    def test_libtool_shim_dropped(self) -> None:
        """Test libtool shim dropped."""
        out = curate_artifacts([_art("/b/.libs/lt-foo")], "foo")
        self.assertEqual(out, [])

    def test_conftest_dropped(self) -> None:
        """Test conftest dropped."""
        out = curate_artifacts([_art("/b/conftest")], "foo")
        self.assertEqual(out, [])

    def test_meson_internals_dropped(self) -> None:
        """Test meson internals dropped."""
        for p in [
            "/b/meson-private/foo",
            "/b/meson-info/x",
            "/b/meson-logs/y",
        ]:
            with self.subTest(path=p):
                out = curate_artifacts([_art(p)], "foo")
                self.assertEqual(out, [])

    def test_library_always_primary(self) -> None:
        """Test library always primary."""
        out = curate_artifacts(
            [_art("/b/libfoo.so.1.2", type="library_shared")], "foo"
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "primary")

    def test_static_lib_primary(self) -> None:
        """Test static lib primary."""
        out = curate_artifacts(
            [_art("/b/libfoo.a", type="library_static")], "foo"
        )
        self.assertEqual(out[0]["role"], "primary")

    def test_repo_name_match_marks_primary(self) -> None:
        # Even in a test/ subdir, name match wins
        """Test repo name match marks primary."""
        out = curate_artifacts([_art("/b/myapp")], "myapp")
        self.assertEqual(out[0]["role"], "primary")

    def test_test_path_marks_secondary(self) -> None:
        """Test test path marks secondary."""
        out = curate_artifacts([_art("/b/tests/runner")], "myapp")
        self.assertEqual(out[0]["role"], "secondary")

    def test_examples_path_marks_secondary(self) -> None:
        """Test examples path marks secondary."""
        out = curate_artifacts([_art("/b/examples/demo")], "myapp")
        self.assertEqual(out[0]["role"], "secondary")

    def test_primary_listed_before_secondary(self) -> None:
        """Test primary listed before secondary."""
        out = curate_artifacts(
            [
                _art("/b/tests/runner"),
                _art("/b/myapp"),
                _art("/b/examples/demo"),
            ],
            "myapp",
        )
        self.assertEqual(out[0]["role"], "primary")
        self.assertEqual(out[0]["filepath"], "/b/myapp")
        self.assertEqual(out[-1]["role"], "secondary")


class TestLLMCurator(unittest.TestCase):
    """Tests for LLMCurator."""

    def _llm(self, response_text):
        """Llm."""
        m = mock.MagicMock()
        m.invoke.return_value = mock.MagicMock(content=response_text)
        return m

    def test_llm_classification_respected(self) -> None:
        """Test llm classification respected."""
        artifacts = [
            _art("/b/foo"),
            _art("/b/tests/foo-test"),
            _art("/b/aux/probe"),
        ]
        llm = self._llm('{"primary": [1], "secondary": [2], "drop": [3]}')
        out = curate_artifacts(artifacts, "foo", build_system="cmake", llm=llm)
        roles = {a["filepath"]: a["role"] for a in out}
        self.assertEqual(roles.get("/b/foo"), "primary")
        self.assertEqual(roles.get("/b/tests/foo-test"), "secondary")
        # Dropped item absent
        self.assertNotIn("/b/aux/probe", roles)

    def test_malformed_json_falls_back_to_rules(self) -> None:
        """Test malformed json falls back to rules."""
        artifacts = [_art("/b/foo")]
        llm = self._llm("garbage not json")
        out = curate_artifacts(artifacts, "foo", llm=llm)
        # rule-based keeps foo as primary
        self.assertEqual(out[0]["role"], "primary")

    def test_llm_exception_falls_back_to_rules(self) -> None:
        """Test llm exception falls back to rules."""
        artifacts = [_art("/b/foo")]
        llm = mock.MagicMock()
        llm.invoke.side_effect = RuntimeError("rate limit")
        out = curate_artifacts(artifacts, "foo", llm=llm)
        self.assertEqual(out[0]["role"], "primary")

    def test_omitted_ids_default_to_secondary(self) -> None:
        """Test omitted ids default to secondary."""
        artifacts = [_art("/b/a"), _art("/b/b"), _art("/b/c")]
        # LLM only mentions id 1 as primary; 2 and 3 are forgotten
        llm = self._llm('{"primary": [1], "secondary": [], "drop": []}')
        out = curate_artifacts(artifacts, "x", llm=llm)
        roles = {a["filepath"]: a["role"] for a in out}
        self.assertEqual(roles["/b/a"], "primary")
        self.assertEqual(roles["/b/b"], "secondary")
        self.assertEqual(roles["/b/c"], "secondary")

    def test_llm_dropping_everything_falls_back_to_rules(self) -> None:
        """Test llm dropping everything falls back to rules."""
        artifacts = [_art("/b/foo")]
        llm = self._llm('{"primary": [], "secondary": [], "drop": [1]}')
        out = curate_artifacts(artifacts, "foo", llm=llm)
        # Rules should restore foo
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "primary")

    def test_hard_noise_filtered_before_llm(self) -> None:
        """Test hard noise filtered before llm."""
        # Even when an LLM is provided, CMakeFiles paths get stripped
        # before the prompt.
        artifacts = [
            _art("/b/CMakeFiles/CompilerIdC/a.out"),
            _art("/b/foo"),
        ]
        llm = self._llm('{"primary": [1], "secondary": [], "drop": []}')
        out = curate_artifacts(artifacts, "foo", llm=llm)
        paths = [a["filepath"] for a in out]
        self.assertNotIn("/b/CMakeFiles/CompilerIdC/a.out", paths)
        # The prompt only saw one item, so id 1 == /b/foo
        self.assertIn("/b/foo", paths)


class TestCuratorEdgeCases(unittest.TestCase):
    """Cover noise drops, plain-primary fallback, and JSON parsing."""

    def test_empty_path_and_noise_are_dropped(self) -> None:
        """Test empty path and noise are dropped."""
        out = curate_artifacts(
            [
                {"filepath": "", "type": "binary"},
                _art("/b/meson-private/probe"),
            ],
            "foo",
        )
        self.assertEqual(out, [])

    def test_plain_binary_defaults_to_primary(self) -> None:
        """Test plain binary defaults to primary."""
        out = curate_artifacts([_art("/b/otherbin")], "foo")
        self.assertEqual(out[0]["role"], "primary")

    def test_parse_curator_json_edge_inputs(self) -> None:
        """Test parse curator json edge inputs."""
        self.assertIsNone(_parse_curator_json("", 3))
        self.assertIsNone(_parse_curator_json("no braces here", 3))
        self.assertIsNone(_parse_curator_json("{not valid json}", 3))
        parsed = _parse_curator_json(
            '{"primary": ["x", 2, 99], "secondary": [], "drop": []}', 3
        )
        self.assertEqual(parsed, ([2], [], []))

    def test_llm_list_content_parts_are_joined(self) -> None:
        """Test llm list content parts are joined."""
        artifacts = [_art("/b/foo"), _art("/b/tests/foo-test")]
        llm = mock.MagicMock()
        llm.invoke.return_value = mock.MagicMock(
            content=['{"primary": [1], "secondary"', ': [2], "drop": []}']
        )
        out = curate_artifacts(artifacts, "foo", llm=llm)
        roles = {a["filepath"]: a["role"] for a in out}
        self.assertEqual(roles.get("/b/foo"), "primary")
        self.assertEqual(roles.get("/b/tests/foo-test"), "secondary")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the Agent Memory System."""

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.memory import (
    MEMORY_INSTANCES,
    RECIPE_CACHE_PATH,
    AgentExample,
    AgentMemory,
    format_few_shot_examples,
    get_agent_memory,
    get_cached_recipe,
    load_recipe_cache,
    materialize_cached_recipe,
    reload_agent_memory,
    render_recipe_markdown,
)
from src.memory import save_learned_example as save_agent_learned_example
from src.memory import (
    save_to_recipe_cache,
)

LOCAL_TEST_ROOT = Path("workspace/test-memory")


def _make_test_dir(prefix: str) -> Path:
    """Create an isolated project-local test directory."""
    path = LOCAL_TEST_ROOT / f"{prefix}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_examples_file(
    examples_dir: Path,
    agent_type: str,
    examples: list[dict[str, object]],
) -> Path:
    """Write a compact examples file under a project-local test dir."""
    examples_dir.mkdir(parents=True, exist_ok=True)
    path = examples_dir / f"{agent_type}_examples.json"
    path.write_text(json.dumps({"version": "2.0", "examples": examples}))
    return path


class TestAgentExample:
    """Tests for AgentExample dataclass."""

    def test_example_creation(self) -> None:
        """Test creating an example with new fields."""
        example = AgentExample(
            id="test-001",
            name="Test Example",
            tags=["go", "test"],
            build_system="go",
            source="manual",
            repo_name="my-repo",
            reasoning="Test reasoning",
        )
        assert example.id == "test-001"
        assert example.tags == ["go", "test"]
        assert example.build_system == "go"
        assert example.source == "manual"
        assert example.repo_name == "my-repo"

    def test_example_creation_legacy(self) -> None:
        """Test creating an example with legacy context field."""
        example = AgentExample(
            id="test-002",
            name="Legacy Example",
            tags=["go"],
            context={"build_system": "go", "repo_name": "legacy"},
            reasoning="Legacy reasoning",
        )
        assert example.context["build_system"] == "go"

    def test_scout_prompt_format(self) -> None:
        """Test formatting example as Scout prompt (new format)."""
        example = AgentExample(
            id="scout-001",
            name="Test Scout",
            tags=["go"],
            build_system="go",
            trigger={"build_system": "go", "has_main": True, "main_path": "."},
            plan={
                "build_system": "go",
                "phases": [{"name": "build", "commands": ["go build ."]}],
            },
            reasoning="Because test",
        )
        text = example.to_prompt_text("scout")
        assert "Test Scout" in text
        assert "go" in text
        assert "Because test" in text
        assert "go build ." in text

    def test_scout_prompt_format_legacy(self) -> None:
        """Test formatting example as Scout prompt (legacy format)."""
        example = AgentExample(
            id="scout-002",
            name="Legacy Scout",
            tags=["go"],
            context={"build_system": "go", "repo_name": "test"},
            expected_output={
                "build_system": "go",
                "phases": [{"name": "build", "commands": ["go build ."]}],
            },
            reasoning="Legacy",
        )
        text = example.to_prompt_text("scout")
        assert "Legacy Scout" in text
        assert "go build ." in text

    def test_fixer_prompt_format(self) -> None:
        """Test formatting example as Fixer prompt (new format)."""
        example = AgentExample(
            id="fixer-001",
            name="Test Fixer",
            tags=["go", "error"],
            build_system="go",
            error_pattern="go: command not found",
            fix={
                "strategy": "Install Go",
                "actions": [{"type": "command", "command": "apk add go"}],
            },
            reasoning="Test reasoning",
            raw={
                "error_context": {
                    "category": "TEST",
                    "error_message": "go: command not found",
                }
            },
        )
        text = example.to_prompt_text("fixer")
        assert "Test Fixer" in text
        assert "Install Go" in text
        assert "apk add go" in text

    def test_builder_prompt_format(self) -> None:
        """Test formatting example as Builder prompt (new format)."""
        example = AgentExample(
            id="builder-001",
            name="Test Builder",
            tags=["go"],
            build_system="go",
            phases=[{"name": "build", "commands": ["go build ."]}],
            timeout_recommendation="60s",
            reasoning="Simple build",
        )
        text = example.to_prompt_text("builder")
        assert "Test Builder" in text
        assert "go" in text
        assert "60s" in text

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        example = AgentExample(
            id="test-003",
            name="Serialize Test",
            tags=["go"],
            build_system="go",
            source="auto",
            repo_name="test-repo",
            trigger={"build_system": "go"},
            plan={"phases": [{"name": "build", "commands": ["go build ."]}]},
            reasoning="Serializable",
        )
        d = example.to_dict()
        assert d["id"] == "test-003"
        assert d["build_system"] == "go"
        assert d["source"] == "auto"
        assert "trigger" in d
        assert "plan" in d


class TestAgentMemory:
    """Tests for AgentMemory class."""

    def test_load_scout_examples(self) -> None:
        """Test loading scout examples from file (new v2.0 format)."""
        memory = AgentMemory("scout")
        assert len(memory.examples) > 0
        assert all(isinstance(ex, AgentExample) for ex in memory.examples)
        # Check new fields are populated
        for ex in memory.examples:
            assert ex.build_system != ""

    def test_load_fixer_examples(self) -> None:
        """Test loading fixer examples from file."""
        memory = AgentMemory("fixer")
        assert len(memory.examples) > 0
        # Check error_pattern is populated for manual examples
        has_pattern = any(ex.error_pattern for ex in memory.examples)
        assert has_pattern

    def test_load_builder_examples(self) -> None:
        """Test loading builder examples from file."""
        memory = AgentMemory("builder")
        assert len(memory.examples) > 0
        has_phases = any(ex.phases for ex in memory.examples)
        assert has_phases

    def test_get_relevant_examples(self) -> None:
        """Test getting relevant examples based on context."""
        memory = AgentMemory("scout")
        context = {"build_system": "go", "has_main": True}
        examples = memory.get_relevant_examples(context, max_examples=2)
        assert len(examples) <= 2
        for ex in examples:
            assert ex.build_system == "go" or "go" in ex.tags

    def test_relevance_scoring_build_system(self) -> None:
        """Test relevance scoring prioritizes build system match."""
        memory = AgentMemory("scout")

        context_go = {"build_system": "go"}
        examples_go = memory.get_relevant_examples(context_go, max_examples=3)

        context_cmake = {"build_system": "cmake"}
        examples_cmake = memory.get_relevant_examples(
            context_cmake, max_examples=3
        )

        assert len(examples_go) > 0
        assert len(examples_cmake) > 0
        # Go examples should be first for go context
        assert examples_go[0].build_system == "go"

    def test_relevance_scoring_error_pattern(self) -> None:
        """Test regex error_pattern matching in fixer relevance scoring."""
        memory = AgentMemory("fixer")
        context = {
            "build_system": "go",
            "error_message": "go: command not found",
        }
        examples = memory.get_relevant_examples(context, max_examples=3)
        assert len(examples) > 0
        # The "Missing Go Command" example should score high
        names = [ex.name for ex in examples]
        assert any("Go" in name or "command" in name.lower() for name in names)

    def test_format_examples_for_prompt(self) -> None:
        """Test formatting examples for prompt."""
        memory = AgentMemory("scout")
        context = {"build_system": "go"}
        examples = memory.get_relevant_examples(context, max_examples=2)
        formatted = memory.format_examples_for_prompt(examples, max_chars=1000)
        assert "# Few-Shot Examples" in formatted
        assert len(formatted) <= 1200

    def test_reload(self) -> None:
        """Test reloading examples from disk."""
        memory = AgentMemory("scout")
        initial_count = len(memory.examples)
        memory.reload()
        assert len(memory.examples) == initial_count


class TestAutoLearning:
    """Tests for auto-learning (save_learned_example)."""

    def setup_method(self) -> None:
        """Create a project-local examples directory for each test."""
        self.tmp_dir = _make_test_dir("auto-learning")
        self.examples_dir = self.tmp_dir / "examples"
        self.examples_dir.mkdir()
        # Write a minimal examples file
        test_file = self.examples_dir / "scout_examples.json"
        test_file.write_text(
            json.dumps(
                {
                    "version": "2.0",
                    "examples": [
                        {
                            "id": "scout-001",
                            "name": "Existing Example",
                            "tags": ["go"],
                            "build_system": "go",
                            "source": "manual",
                            "repo_name": "existing-repo",
                            "plan": {
                                "phases": [
                                    {
                                        "name": "build",
                                        "commands": ["go build ."],
                                    }
                                ]
                            },
                            "reasoning": "Existing",
                        }
                    ],
                }
            )
        )

    def teardown_method(self) -> None:
        """Clean up temp directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_learned_example(self) -> None:
        """Test saving a new auto-learned example."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)
        assert len(memory.examples) == 1

        result = memory.save_learned_example(
            {
                "name": "Auto: new-repo (go)",
                "tags": ["go"],
                "build_system": "go",
                "repo_name": "new-repo",
                "plan": {
                    "phases": [
                        {"name": "build", "commands": ["go build ./cmd/app"]}
                    ]
                },
                "reasoning": "Auto-learned",
            }
        )
        assert result is True
        assert len(memory.examples) == 2
        new_ex = memory.examples[-1]
        assert new_ex.id == "scout-auto-001"
        assert new_ex.source == "auto"

    def test_save_duplicate_rejected(self) -> None:
        """Test that duplicate examples are rejected."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)

        result = memory.save_learned_example(
            {
                "name": "Duplicate",
                "tags": ["go"],
                "build_system": "go",
                "repo_name": "existing-repo",
                "reasoning": "Should be rejected",
            }
        )
        assert result is False
        assert len(memory.examples) == 1

    def test_save_missing_fields_rejected(self) -> None:
        """Test that examples without required fields are rejected."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)

        result = memory.save_learned_example(
            {
                "tags": ["go"],
            }
        )
        assert result is False

    def test_prune_oldest_auto(self) -> None:
        """Test pruning removes oldest auto-learned first."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)
        # Manually add many auto examples to test pruning
        data = memory._read_json_file()
        for i in range(105):
            data["examples"].append(
                {
                    "id": f"scout-auto-{i:03d}",
                    "name": f"Auto Example {i}",
                    "tags": ["go"],
                    "build_system": "go",
                    "source": "auto",
                    "repo_name": f"repo-{i}",
                    "timestamp": f"2026-01-{i % 28 + 1:02d}",
                }
            )
        pruned = memory._prune_examples_list(data["examples"])
        assert len(pruned) <= 100
        # Manual example should survive
        assert any(e.get("source", "manual") == "manual" for e in pruned)


class TestFormatFewShotExamples:
    """Tests for the convenience function."""

    def setup_method(self) -> None:
        """Clear cached instances for clean tests."""
        MEMORY_INSTANCES.clear()

    def test_format_scout_examples(self) -> None:
        """Test formatting scout examples."""
        context = {"build_system": "go", "has_main": True, "main_path": "."}
        result = format_few_shot_examples("scout", context, max_examples=2)
        assert "# Few-Shot Examples" in result
        assert "scout" in result.lower()

    def test_format_fixer_examples(self) -> None:
        """Test formatting fixer examples."""
        context = {
            "build_system": "go",
            "error_message": "go: command not found",
        }
        result = format_few_shot_examples("fixer", context, max_examples=2)
        assert "# Few-Shot Examples" in result
        assert "fixer" in result.lower()

    def test_format_builder_examples(self) -> None:
        """Test formatting builder examples."""
        context = {"build_system": "go"}
        result = format_few_shot_examples("builder", context, max_examples=2)
        assert isinstance(result, str)

    def test_empty_context(self) -> None:
        """Test with empty context."""
        result = format_few_shot_examples("scout", {}, max_examples=2)
        assert isinstance(result, str)


class TestMemoryCaching:
    """Tests for memory instance caching."""

    def setup_method(self) -> None:
        """Set up the test by clearing cached memory instances."""
        MEMORY_INSTANCES.clear()

    def test_get_agent_memory_caching(self) -> None:
        """Test that memory instances are cached."""
        memory1 = get_agent_memory("scout")
        memory2 = get_agent_memory("scout")
        assert memory1 is memory2

    def test_different_agents_different_memory(self) -> None:
        """Test that different agents have different memory."""
        scout_memory = get_agent_memory("scout")
        fixer_memory = get_agent_memory("fixer")
        assert scout_memory is not fixer_memory
        assert scout_memory.agent_type == "scout"
        assert fixer_memory.agent_type == "fixer"

    def test_reload_agent_memory(self) -> None:
        """Test force reloading memory."""
        get_agent_memory("scout")
        reload_agent_memory("scout")
        memory2 = get_agent_memory("scout")
        # After reload, should have fresh examples
        assert len(memory2.examples) > 0


class TestRecipeCache:
    """Tests for recipe cache functions."""

    def setup_method(self) -> None:
        """Back up recipe cache if it exists."""
        self.backup = None
        if RECIPE_CACHE_PATH.exists():
            self.backup = RECIPE_CACHE_PATH.read_text()

    def teardown_method(self) -> None:
        """Restore recipe cache."""
        if self.backup is not None:
            RECIPE_CACHE_PATH.write_text(self.backup)
        elif RECIPE_CACHE_PATH.exists():
            RECIPE_CACHE_PATH.write_text(
                json.dumps({"version": "1.0", "packages": {}})
            )

    def test_load_empty_cache(self) -> None:
        """Test loading when cache is empty."""
        RECIPE_CACHE_PATH.write_text(
            json.dumps({"version": "1.0", "packages": {}})
        )
        cache = load_recipe_cache()
        assert cache["version"] == "1.0"
        assert cache["packages"] == {}

    def test_save_and_get_recipe(self) -> None:
        """Test saving and retrieving a recipe."""
        RECIPE_CACHE_PATH.write_text(
            json.dumps({"version": "1.0", "packages": {}})
        )

        result = save_to_recipe_cache(
            repo_name="test-pkg",
            repo_url="https://github.com/test/test-pkg",
            build_system="go",
            build_plan={
                "phases": [{"name": "build", "commands": ["go build ."]}]
            },
            dependencies=["go"],
            patches=[],
            artifacts=[{"type": "binary", "filepath": "test-pkg"}],
            build_duration_seconds=30.5,
        )
        assert result is True

        cached = get_cached_recipe("test-pkg")
        assert cached is not None
        assert cached["build_system"] == "go"
        assert cached["architecture"] == "riscv64"
        assert cached["build_duration_seconds"] == 30.5

    def test_get_nonexistent_recipe(self) -> None:
        """Test cache miss returns None."""
        RECIPE_CACHE_PATH.write_text(
            json.dumps({"version": "1.0", "packages": {}})
        )
        cached = get_cached_recipe("nonexistent-pkg")
        assert cached is None

    def test_upsert_recipe(self) -> None:
        """Test that saving again updates the entry."""
        RECIPE_CACHE_PATH.write_text(
            json.dumps({"version": "1.0", "packages": {}})
        )

        save_to_recipe_cache(
            repo_name="update-test",
            repo_url="https://github.com/test/update-test",
            build_system="go",
            build_plan={
                "phases": [{"name": "build", "commands": ["go build ."]}]
            },
            dependencies=[],
            patches=[],
            artifacts=[],
            build_duration_seconds=10.0,
        )
        save_to_recipe_cache(
            repo_name="update-test",
            repo_url="https://github.com/test/update-test",
            build_system="go",
            build_plan={
                "phases": [{"name": "build", "commands": ["go build -v ."]}]
            },
            dependencies=["go"],
            patches=["fixed something"],
            artifacts=[],
            build_duration_seconds=20.0,
        )

        cached = get_cached_recipe("update-test")
        assert cached["build_duration_seconds"] == 20.0
        assert cached["dependencies"] == ["go"]


class TestRecipeMaterialization:
    """Tests for rendering and writing cached recipes to disk."""

    def test_render_reconstructs_from_structured_data(self) -> None:
        """Renderer builds Markdown from structured cache fields."""
        recipe = {
            "repo_url": "https://github.com/madler/zlib",
            "build_system": "cmake",
            "architecture": "riscv64",
            "sandbox": "alpine-riscv64",
            "last_built": "2026-05-22T22:57:07",
            "build_plan": {
                "phases": [{"name": "build", "commands": ["make -j$(nproc)"]}]
            },
            "dependencies": ["cmake"],
            "patches": ["disabled SIMD"],
            "artifacts": [
                {"type": "library_static", "path": "libz.a", "role": "primary"}
            ],
        }
        md = render_recipe_markdown("zlib", recipe)
        assert "# RISC-V Porting Recipe: zlib" in md
        assert "make -j$(nproc)" in md
        assert "cmake" in md
        assert "disabled SIMD" in md
        assert "libz.a" in md
        assert "Reconstructed from the recipe cache" in md

    def test_render_uses_stored_markdown_verbatim(self) -> None:
        """Stored recipe_markdown is returned without reconstruction."""
        recipe = {
            "build_system": "cmake",
            "recipe_markdown": "# Exact Guide\n\nVerbatim body.\n",
        }
        md = render_recipe_markdown("zlib", recipe)
        assert md == "# Exact Guide\n\nVerbatim body.\n"
        assert "Reconstructed" not in md

    def test_save_persists_recipe_markdown(self) -> None:
        """save_to_recipe_cache stores recipe_markdown when provided."""
        import src.memory as memory

        test_dir = _make_test_dir("recipe-markdown")
        try:
            cache_file = test_dir / "recipe_cache.json"
            cache_file.write_text(
                json.dumps({"version": "2.0", "packages": {}})
            )

            with mock.patch.object(memory, "RECIPE_CACHE_PATH", cache_file):
                memory.save_to_recipe_cache(
                    repo_name="mark-pkg",
                    repo_url="https://github.com/test/mark-pkg",
                    build_system="go",
                    build_plan={"phases": []},
                    dependencies=[],
                    patches=[],
                    artifacts=[],
                    build_duration_seconds=1.0,
                    recipe_markdown="# Stored\n",
                )
                cached = memory.get_cached_recipe("mark-pkg")

            assert cached["recipe_markdown"] == "# Stored\n"
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_materialize_writes_file(self) -> None:
        """materialize_cached_recipe writes <repo>_recipe.md to disk."""
        recipe = {
            "build_system": "cmake",
            "build_plan": {
                "phases": [{"name": "build", "commands": ["cmake .."]}]
            },
        }
        test_dir = _make_test_dir("materialize")
        try:
            out = materialize_cached_recipe("zlib", str(test_dir), recipe)
            assert out is not None
            written = test_dir / "zlib_recipe.md"
            assert written.exists()
            assert "cmake .." in written.read_text()
            assert out == str(written.resolve())
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_materialize_returns_none_without_recipe(self) -> None:
        """Materialize returns None when no recipe is available."""
        test_dir = _make_test_dir("materialize-miss")
        try:
            out = materialize_cached_recipe("absent-pkg", str(test_dir), {})
            assert out is None
            assert not (test_dir / "absent-pkg_recipe.md").exists()
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class TestMemoryEdgeCases(unittest.TestCase):
    """Tests for defensive and less common memory branches."""

    def setUp(self) -> None:
        """Create a project-local scratch tree for each test."""
        self.test_dir = _make_test_dir("memory-edges")
        self.examples_dir = self.test_dir / "examples"
        self.examples_dir.mkdir()
        self.addCleanup(shutil.rmtree, self.test_dir, True)

    def _memory_with_examples(
        self,
        agent_type: str,
        examples: list[dict[str, object]],
    ) -> AgentMemory:
        """Build an AgentMemory backed by isolated example data."""
        _write_examples_file(self.examples_dir, agent_type, examples)
        return AgentMemory(agent_type, examples_dir=self.examples_dir)

    def test_seed_bundled_data_copies_examples_and_cache(self) -> None:
        """Bundled examples and cache are copied into writable paths."""
        import src.memory as memory

        bundled_dir = self.test_dir / "bundled"
        bundled_examples = bundled_dir / "examples"
        bundled_examples.mkdir(parents=True)
        (bundled_examples / "scout_examples.json").write_text(
            json.dumps({"version": "2.0", "examples": []})
        )
        bundled_cache = bundled_dir / "recipe_cache.json"
        bundled_cache.write_text(
            json.dumps({"version": "2.0", "packages": {"zlib": {}}})
        )
        target_examples = self.test_dir / "seeded_examples"
        target_cache = self.test_dir / "seeded_cache.json"

        with mock.patch.object(memory, "_BUNDLED_DATA_DIR", bundled_dir):
            with mock.patch.object(memory, "EXAMPLES_DIR", target_examples):
                with mock.patch.object(memory, "RECIPE_CACHE_PATH",
                                       target_cache):
                    memory._seed_bundled_data()

        self.assertTrue((target_examples / "scout_examples.json").exists())
        self.assertEqual(
            json.loads(target_cache.read_text())["packages"],
            {"zlib": {}},
        )

    def test_seed_bundled_data_creates_default_cache(self) -> None:
        """A missing bundled cache creates an empty writable cache."""
        import src.memory as memory

        bundled_dir = self.test_dir / "empty_bundle"
        bundled_dir.mkdir()
        target_examples = self.test_dir / "default_examples"
        target_cache = self.test_dir / "default_cache.json"

        with mock.patch.object(memory, "_BUNDLED_DATA_DIR", bundled_dir):
            with mock.patch.object(memory, "EXAMPLES_DIR", target_examples):
                with mock.patch.object(memory, "RECIPE_CACHE_PATH",
                                       target_cache):
                    memory._seed_bundled_data()

        self.assertEqual(
            json.loads(target_cache.read_text()),
            {"version": "2.0", "packages": {}},
        )

    def test_unknown_prompt_type_returns_empty(self) -> None:
        """Unknown example prompt types return an empty string."""
        example = AgentExample(id="ex", name="Example", tags=[])
        self.assertEqual(example.to_prompt_text("supervisor"), "")

    def test_fixer_prompt_formats_create_file_and_patch(self) -> None:
        """Fixer examples include create-file and patch actions."""
        example = AgentExample(
            id="fixer-extra",
            name="Fixer Extra",
            tags=["c"],
            fix={
                "analysis": "Patch build glue",
                "actions": [
                    {"type": "create_file", "path": "config.h"},
                    {"type": "patch", "file": "src/main.c"},
                ],
            },
            reasoning="Need generated config",
        )

        text = example.to_prompt_text("fixer")

        self.assertIn("create: config.h", text)
        self.assertIn("patch: src/main.c", text)
        self.assertIn("Patch build glue", text)

    def test_to_dict_includes_all_optional_fields(self) -> None:
        """Serialization preserves every optional payload field."""
        example = AgentExample(
            id="full",
            name="Full Example",
            tags=["go"],
            build_system="go",
            trigger={"has_main": True},
            plan={"phases": []},
            error_pattern="missing",
            fix={"actions": []},
            phases=[{"name": "build", "commands": ["make"]}],
            timeout_recommendation="90s",
            reasoning="Complete",
            context={"module_dir": "cmd/app"},
            expected_output={"phases": []},
            solution={"strategy": "patch"},
            execution={"duration": 1},
        )

        data = example.to_dict()

        for key in [
            "trigger",
            "plan",
            "error_pattern",
            "fix",
            "phases",
            "timeout_recommendation",
            "reasoning",
            "context",
            "expected_output",
            "solution",
            "execution",
        ]:
            with self.subTest(key=key):
                self.assertIn(key, data)

    def test_missing_example_file_loads_empty_memory(self) -> None:
        """A missing examples file logs and leaves memory empty."""
        memory = AgentMemory("missing", examples_dir=self.examples_dir)
        self.assertEqual(memory.examples, [])
        self.assertEqual(memory._read_json_file()["examples"], [])

    def test_invalid_example_file_loads_empty_memory(self) -> None:
        """Invalid JSON is caught during example loading."""
        path = self.examples_dir / "fixer_examples.json"
        path.write_text("{not-json")

        memory = AgentMemory("fixer", examples_dir=self.examples_dir)

        self.assertEqual(memory.examples, [])

    def test_empty_memory_has_no_relevant_examples_or_prompt(self) -> None:
        """Empty memories return no matches and no prompt section."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)

        self.assertEqual(memory.get_relevant_examples({"build_system": "go"}),
                         [])
        self.assertEqual(memory.format_examples_for_prompt([]), "")

    def test_profile_failure_still_scores_examples(self) -> None:
        """Profile lookup failures fall back to sandbox-agnostic scoring."""
        self._memory_with_examples(
            "scout",
            [
                {
                    "id": "scout-001",
                    "name": "Go build",
                    "tags": ["go"],
                    "build_system": "go",
                    "trigger": {"has_main": True},
                    "plan": {"phases": []},
                }
            ],
        )
        memory = AgentMemory("scout", examples_dir=self.examples_dir)

        with mock.patch(
            "src.platforms.get_active_profile",
            side_effect=RuntimeError("no profile"),
        ):
            examples = memory.get_relevant_examples(
                {"build_system": "go"}, max_examples=1
            )

        self.assertEqual(len(examples), 1)

    def test_wrong_sandbox_returns_no_examples(self) -> None:
        """Examples from a different sandbox are hard-filtered."""
        memory = self._memory_with_examples(
            "scout",
            [
                {
                    "id": "scout-debian",
                    "name": "Debian build",
                    "tags": ["go"],
                    "build_system": "go",
                    "sandbox": "debian-riscv64",
                }
            ],
        )

        examples = memory.get_relevant_examples(
            {"build_system": "go", "sandbox": "alpine-riscv64"}
        )

        self.assertEqual(examples, [])

    def test_relevance_handles_regex_and_bonus_edges(self) -> None:
        """Relevance scoring handles regex errors and extra bonuses."""
        memory = AgentMemory("fixer", examples_dir=self.examples_dir)
        example = AgentExample(
            id="fixer",
            name="Fixer",
            tags=["cgo", "missing"],
            build_system="go",
            sandbox="debian-riscv64",
            trigger={"has_main": True, "module_dir": "cmd/app"},
            error_pattern="[",
            raw={"sandbox": "debian-riscv64"},
        )

        score = memory._calculate_relevance(
            example,
            {
                "build_system": "go",
                "error_message": "missing symbol",
                "has_main": True,
                "module_dir": "cmd/app",
                "has_cgo": True,
                "sandbox": "alpine-riscv64",
            },
        )
        partial_score = memory._calculate_relevance(
            example,
            {"module_dir": "other", "sandbox": "debian-riscv64"},
        )

        self.assertGreater(score, 0.0)
        self.assertGreater(partial_score, 0.0)

    def test_save_learned_example_skips_bad_auto_suffix(self) -> None:
        """Bad auto-id suffixes are ignored when assigning the next id."""
        memory = self._memory_with_examples(
            "scout",
            [
                {
                    "id": "scout-auto-bad",
                    "name": "Bad suffix",
                    "tags": ["go"],
                    "build_system": "go",
                    "source": "auto",
                    "repo_name": "old-repo",
                }
            ],
        )

        result = memory.save_learned_example(
            {
                "name": "New",
                "tags": ["go"],
                "build_system": "go",
                "repo_name": "new-repo",
            }
        )

        self.assertTrue(result)
        self.assertEqual(memory.examples[-1].id, "scout-auto-001")

    def test_save_learned_example_returns_false_on_lock_error(self) -> None:
        """File-lock failures make save_learned_example return False."""
        memory = self._memory_with_examples("scout", [])

        with mock.patch(
            "src.memory.filelock.FileLock",
            side_effect=RuntimeError("locked"),
        ):
            result = memory.save_learned_example(
                {"name": "New", "tags": [], "build_system": "go"}
            )

        self.assertFalse(result)

    def test_duplicate_checks_sandbox_error_pattern_and_commands(self) -> None:
        """Duplicate detection respects sandbox, patterns, and commands."""
        scout = self._memory_with_examples(
            "scout",
            [
                {
                    "id": "scout-001",
                    "name": "Scout",
                    "tags": ["go"],
                    "build_system": "go",
                    "repo_name": "same-repo",
                    "sandbox": "alpine-riscv64",
                }
            ],
        )
        self.assertFalse(
            scout._is_duplicate(
                {
                    "build_system": "go",
                    "repo_name": "same-repo",
                    "sandbox": "debian-riscv64",
                }
            )
        )

        fixer = self._memory_with_examples(
            "fixer",
            [
                {
                    "id": "fixer-001",
                    "name": "Fixer",
                    "tags": ["go"],
                    "build_system": "go",
                    "error_pattern": "missing symbol",
                    "sandbox": "alpine-riscv64",
                }
            ],
        )
        self.assertTrue(
            fixer._is_duplicate(
                {
                    "build_system": "go",
                    "error_pattern": "missing symbol",
                    "sandbox": "alpine-riscv64",
                }
            )
        )

        builder = self._memory_with_examples(
            "builder",
            [
                {
                    "id": "builder-001",
                    "name": "Builder",
                    "tags": ["go"],
                    "build_system": "go",
                    "phases": [{"name": "build", "commands": ["go build"]}],
                    "sandbox": "alpine-riscv64",
                }
            ],
        )
        self.assertTrue(
            builder._is_duplicate(
                {
                    "build_system": "go",
                    "phases": [
                        {"name": "build", "commands": ["go build"]}
                    ],
                    "sandbox": "alpine-riscv64",
                }
            )
        )

    def test_duplicate_profile_failure_falls_back(self) -> None:
        """Duplicate checks tolerate active-profile lookup failures."""
        memory = self._memory_with_examples("scout", [])

        with mock.patch(
            "src.platforms.get_active_profile",
            side_effect=RuntimeError("no profile"),
        ):
            is_duplicate = memory._is_duplicate({"build_system": "go"})

        self.assertFalse(is_duplicate)

    def test_extract_commands_reads_all_supported_shapes(self) -> None:
        """Command fingerprints include phases, plan, and legacy output."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)

        fingerprint = memory._extract_commands(
            {
                "phases": [{"commands": ["make"]}],
                "plan": {"phases": [{"commands": ["go build"]}]},
                "expected_output": {
                    "phases": [{"commands": ["cmake --build build"]}]
                },
            }
        )

        self.assertTrue(fingerprint)
        self.assertEqual(memory._extract_commands({}), "")

    def test_prune_drops_auto_when_overflow_exceeds_auto_count(self) -> None:
        """If overflow is larger than auto examples, all autos drop."""
        memory = AgentMemory("scout", examples_dir=self.examples_dir)
        examples = [
            {
                "id": f"manual-{i}",
                "name": f"Manual {i}",
                "source": "manual",
            }
            for i in range(101)
        ]
        examples.append(
            {
                "id": "auto-001",
                "name": "Auto",
                "source": "auto",
                "timestamp": "2026-01-01",
            }
        )

        pruned = memory._prune_examples_list(examples)

        self.assertFalse(any(e.get("source") == "auto" for e in pruned))

    def test_migrate_legacy_cache_skips_non_dict_and_wraps_flat(self) -> None:
        """Legacy cache migration skips junk and wraps flat entries."""
        import src.memory as memory

        cache = {
            "packages": {
                "junk": "not-a-dict",
                "zlib": {
                    "sandbox": "debian-riscv64",
                    "build_plan": {"phases": []},
                },
            }
        }

        migrated = memory._migrate_legacy_cache(cache)

        self.assertEqual(migrated["packages"]["junk"], "not-a-dict")
        self.assertIn("debian-riscv64", migrated["packages"]["zlib"])

    def test_load_recipe_cache_missing_and_invalid(self) -> None:
        """Cache loading returns defaults for missing or invalid JSON."""
        import src.memory as memory

        cache_file = self.test_dir / "recipe_cache.json"
        with mock.patch.object(memory, "RECIPE_CACHE_PATH", cache_file):
            self.assertEqual(
                memory.load_recipe_cache(),
                {"version": "2.0", "packages": {}},
            )

            cache_file.write_text("{not-json")
            self.assertEqual(
                memory.load_recipe_cache(),
                {"version": "2.0", "packages": {}},
            )

    def test_default_sandbox_falls_back_on_profile_error(self) -> None:
        """Default sandbox falls back to Alpine if profile lookup fails."""
        import src.memory as memory

        with mock.patch(
            "src.platforms.get_active_profile",
            side_effect=RuntimeError("no profile"),
        ):
            self.assertEqual(memory._default_sandbox(), "alpine-riscv64")

    def test_get_cached_recipe_respects_architecture(self) -> None:
        """Recipes for a different architecture do not match."""
        import src.memory as memory

        cache = {
            "packages": {
                "pkg": {
                    "alpine-riscv64": {
                        "architecture": "x86_64",
                        "build_plan": {"phases": []},
                    }
                }
            }
        }
        with mock.patch.object(memory, "load_recipe_cache",
                               return_value=cache):
            recipe = memory.get_cached_recipe(
                "pkg", architecture="riscv64", sandbox="alpine-riscv64"
            )

        self.assertIsNone(recipe)

    def test_save_to_recipe_cache_preserves_legacy_entry(self) -> None:
        """Saving over a flat legacy entry preserves its sandbox copy."""
        import src.memory as memory

        cache_file = self.test_dir / "legacy_cache.json"
        legacy_cache = {
            "version": "1.0",
            "packages": {
                "pkg": {
                    "sandbox": "alpine-riscv64",
                    "build_plan": {"phases": []},
                }
            },
        }

        with mock.patch.object(memory, "RECIPE_CACHE_PATH", cache_file):
            with mock.patch.object(memory, "load_recipe_cache",
                                   return_value=legacy_cache):
                saved = memory.save_to_recipe_cache(
                    repo_name="pkg",
                    repo_url="https://github.com/test/pkg",
                    build_system="go",
                    build_plan={"phases": [{"commands": ["go build"]}]},
                    dependencies=[],
                    patches=[],
                    artifacts=[],
                    build_duration_seconds=2.0,
                    sandbox="debian-riscv64",
                )

        data = json.loads(cache_file.read_text())
        self.assertTrue(saved)
        self.assertIn("alpine-riscv64", data["packages"]["pkg"])
        self.assertIn("debian-riscv64", data["packages"]["pkg"])

    def test_reload_agent_memory_creates_uncached_memory(self) -> None:
        """Reloading an uncached agent creates a new memory instance."""
        import src.memory as memory

        fake_memory = object()
        memory.MEMORY_INSTANCES.clear()
        with mock.patch.object(memory, "AgentMemory",
                               return_value=fake_memory) as mock_agent:
            memory.reload_agent_memory("fresh")

        self.assertIs(memory.MEMORY_INSTANCES["fresh"], fake_memory)
        mock_agent.assert_called_once_with("fresh")

    def test_save_to_recipe_cache_returns_false_on_lock_error(self) -> None:
        """Recipe-cache lock failures return False."""
        import src.memory as memory

        cache_file = self.test_dir / "lock_cache.json"
        cache_file.write_text(json.dumps({"version": "2.0", "packages": {}}))

        with mock.patch.object(memory, "RECIPE_CACHE_PATH", cache_file):
            with mock.patch(
                "src.memory.filelock.FileLock",
                side_effect=RuntimeError("locked"),
            ):
                saved = memory.save_to_recipe_cache(
                    repo_name="pkg",
                    repo_url="https://github.com/test/pkg",
                    build_system="go",
                    build_plan={"phases": []},
                    dependencies=[],
                    patches=[],
                    artifacts=[],
                    build_duration_seconds=1.0,
                )

        self.assertFalse(saved)

    def test_materialize_fetches_recipe_when_none(self) -> None:
        """Materialization fetches a cached recipe when omitted."""
        import src.memory as memory

        output_dir = self.test_dir / "output"
        recipe = {"build_system": "go", "build_plan": {"phases": []}}
        with mock.patch.object(memory, "get_cached_recipe",
                               return_value=recipe):
            path = memory.materialize_cached_recipe("pkg", str(output_dir))

        self.assertIsNotNone(path)
        self.assertTrue((output_dir / "pkg_recipe.md").exists())

    def test_materialize_returns_none_on_write_error(self) -> None:
        """Materialization returns None when the output write fails."""
        import src.memory as memory

        with mock.patch.object(memory.os, "makedirs",
                               side_effect=OSError("denied")):
            path = memory.materialize_cached_recipe(
                "pkg", str(self.test_dir / "blocked"), {"build_system": "go"}
            )

        self.assertIsNone(path)

    def test_save_learned_example_convenience_uses_memory(self) -> None:
        """The module-level helper delegates to the cached memory."""
        fake_memory = mock.Mock()
        fake_memory.save_learned_example.return_value = True

        with mock.patch(
            "src.memory.get_agent_memory", return_value=fake_memory
        ) as mock_get:
            saved = save_agent_learned_example(
                "scout", {"name": "New", "build_system": "go"}
            )

        self.assertTrue(saved)
        mock_get.assert_called_once_with("scout")
        fake_memory.save_learned_example.assert_called_once_with(
            {"name": "New", "build_system": "go"}
        )

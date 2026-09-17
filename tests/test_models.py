"""Tests for src.models LLM pool behavior."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from src.models import (
    _DEFAULT_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT,
    MODEL_CONFIG,
    OPENROUTER_FREE_ROUTER,
    _create_llm_with_model,
    _openrouter_fallback_ids,
    _resolve_model_name,
    check_api_keys,
    create_llm,
    create_llm_pool,
    is_free_model,
    print_model_info,
)
from src.state import AgentRole


class TestCheckApiKeys(unittest.TestCase):
    """Tests for provider API-key validation."""

    def test_openai_key_branches(self) -> None:
        """Validate missing, placeholder, and present OpenAI keys."""
        cases = [
            ({}, False, "OPENAI_API_KEY not found"),
            ({"OPENAI_API_KEY": "your_key_here"}, False, "not found"),
            ({"OPENAI_API_KEY": "sk-test"}, True, "OpenAI API key"),
        ]
        for extra_env, expected_ok, expected_msg in cases:
            env = {"LLM_PROVIDER": "openai", **extra_env}
            with self.subTest(extra_env=extra_env):
                with mock.patch.dict(os.environ, env, clear=True):
                    ok, message, provider = check_api_keys()

                self.assertEqual(ok, expected_ok)
                self.assertIn(expected_msg, message)
                self.assertEqual(provider, "openai")

    def test_gemini_key_branches(self) -> None:
        """Validate missing, placeholder, and present Gemini keys."""
        cases = [
            ({}, False, "GOOGLE_API_KEY not found"),
            ({"GOOGLE_API_KEY": "your_key_here"}, False, "not found"),
            ({"GOOGLE_API_KEY": "google-test"}, True, "Gemini API key"),
        ]
        for extra_env, expected_ok, expected_msg in cases:
            env = {"LLM_PROVIDER": "gemini", **extra_env}
            with self.subTest(extra_env=extra_env):
                with mock.patch.dict(os.environ, env, clear=True):
                    ok, message, provider = check_api_keys()

                self.assertEqual(ok, expected_ok)
                self.assertIn(expected_msg, message)
                self.assertEqual(provider, "gemini")

    def test_openrouter_key_branches(self) -> None:
        """Validate missing, placeholder, and present OpenRouter keys."""
        cases = [
            ({}, False, "OPENROUTER_API_KEY not found"),
            ({"OPENROUTER_API_KEY": "your_key_here"}, False, "not found"),
            (
                {"OPENROUTER_API_KEY": "router-test"},
                True,
                "OpenRouter API key",
            ),
        ]
        for extra_env, expected_ok, expected_msg in cases:
            env = {"LLM_PROVIDER": "openrouter", **extra_env}
            with self.subTest(extra_env=extra_env):
                with mock.patch.dict(os.environ, env, clear=True):
                    ok, message, provider = check_api_keys()

                self.assertEqual(ok, expected_ok)
                self.assertIn(expected_msg, message)
                self.assertEqual(provider, "openrouter")

    def test_unknown_provider_fails(self) -> None:
        """Unknown providers are reported without falling through."""
        with mock.patch.dict(
            os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True
        ):
            ok, message, provider = check_api_keys()

        self.assertFalse(ok)
        self.assertEqual(message, "Unknown provider: anthropic")
        self.assertEqual(provider, "anthropic")


class TestIsFreeModel(unittest.TestCase):
    """Tests for free-model slug detection."""

    def test_detects_free_tier_slugs_case_insensitively(self) -> None:
        """Free suffixes and router id are detected."""
        self.assertTrue(is_free_model("QWEN/QWEN3-CODER:FREE"))
        self.assertTrue(is_free_model(OPENROUTER_FREE_ROUTER))

    def test_false_for_empty_or_paid_model(self) -> None:
        """Empty and paid model ids are not free."""
        self.assertFalse(is_free_model(""))
        self.assertFalse(is_free_model("gpt-4o"))


class TestCreateLLM(unittest.TestCase):
    """Tests for create_llm and role-to-model resolution."""

    @mock.patch("src.models._resolve_model_name", return_value="model-id")
    @mock.patch("src.models._create_llm_with_model", return_value="llm")
    def test_create_llm_uses_resolved_model(
        self,
        mock_create_with_model: mock.MagicMock,
        mock_resolve: mock.MagicMock,
    ) -> None:
        """create_llm delegates construction to the explicit builder."""
        self.assertEqual(create_llm(AgentRole.SCOUT), "llm")
        mock_resolve.assert_called_once_with(AgentRole.SCOUT)
        mock_create_with_model.assert_called_once_with(
            AgentRole.SCOUT, "model-id"
        )

    def test_resolve_model_uses_provider_role_and_fallbacks(self) -> None:
        """Model resolution falls back for unknown providers and roles."""
        with mock.patch.dict(
            os.environ, {"LLM_PROVIDER": "openai"}, clear=True
        ):
            self.assertEqual(_resolve_model_name(AgentRole.FIXER), "gpt-4o")

        with mock.patch.dict(
            os.environ, {"LLM_PROVIDER": "unknown"}, clear=True
        ):
            self.assertEqual(
                _resolve_model_name(AgentRole.BUILDER),
                "gemini-flash-lite-latest",
            )

        with mock.patch.dict(
            os.environ, {"LLM_PROVIDER": "openai"}, clear=True
        ):
            self.assertEqual(_resolve_model_name("made-up"), "gpt-4o-mini")


class TestCreateLLMWithModel(unittest.TestCase):
    """Tests for provider-specific LLM construction without real clients."""

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True)
    @mock.patch("src.models.ChatOpenAI")
    def test_openai_constructor_arguments(
        self, mock_chat: mock.MagicMock
    ) -> None:
        """Provider clients receive role temperature and token limits."""
        _create_llm_with_model(AgentRole.BUILDER, "gpt-test")

        mock_chat.assert_called_once_with(
            model="gpt-test",
            temperature=0.0,
            request_timeout=LLM_REQUEST_TIMEOUT,
            max_tokens=_DEFAULT_MAX_TOKENS,
        )

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=True)
    @mock.patch("src.models.ChatGoogleGenerativeAI")
    def test_gemini_constructor_arguments(
        self, mock_chat: mock.MagicMock
    ) -> None:
        """Gemini clients receive timeout and max-output arguments."""
        _create_llm_with_model(AgentRole.SCOUT, "gemini-test")

        mock_chat.assert_called_once_with(
            model="gemini-test",
            temperature=0.1,
            timeout=LLM_REQUEST_TIMEOUT,
            max_output_tokens=_DEFAULT_MAX_TOKENS,
        )

    @mock.patch.dict(
        os.environ,
        {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "router-key"},
        clear=True,
    )
    @mock.patch("src.models.ChatOpenAI")
    def test_openrouter_constructor_arguments(
        self, mock_chat: mock.MagicMock
    ) -> None:
        """Router clients use base URL and fallback settings."""
        _create_llm_with_model(AgentRole.FIXER, "openai/gpt-oss-120b:free")

        kwargs = mock_chat.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-oss-120b:free")
        self.assertEqual(kwargs["openai_api_key"], "router-key")
        self.assertEqual(
            kwargs["openai_api_base"], "https://openrouter.ai/api/v1"
        )
        self.assertEqual(kwargs["extra_body"]["models"][-1], "openrouter/free")
        self.assertLessEqual(len(kwargs["extra_body"]["models"]), 3)

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=True)
    @mock.patch("src.models.ChatGoogleGenerativeAI")
    def test_unknown_provider_falls_back_to_gemini(
        self, mock_chat: mock.MagicMock
    ) -> None:
        """Unknown configured providers fall back to Gemini."""
        _create_llm_with_model(AgentRole.SCOUT, "gemini-test")
        self.assertEqual(mock_chat.call_args.kwargs["model"], "gemini-test")

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True)
    @mock.patch("src.models.ChatOpenAI")
    def test_unknown_role_falls_back_to_supervisor_temperature(
        self, mock_chat: mock.MagicMock
    ) -> None:
        """Unknown roles use the supervisor temperature."""
        _create_llm_with_model("unknown-role", "gpt-test")
        self.assertEqual(mock_chat.call_args.kwargs["temperature"], 0.0)

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "unsupported"}, clear=True)
    def test_configured_but_unsupported_provider_raises(self) -> None:
        """Providers in config but without plumbing raise ValueError."""
        fake_config = {
            "unsupported": {
                "supervisor": {
                    "model": "unsupported-model",
                    "temperature": 0.0,
                }
            }
        }
        with mock.patch.dict(MODEL_CONFIG, fake_config):
            with self.assertRaisesRegex(ValueError, "Unsupported provider"):
                _create_llm_with_model(AgentRole.SUPERVISOR, "model")


class TestCreateLLMPool(unittest.TestCase):
    """Tests for create_llm_pool."""

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=False)
    @mock.patch("src.models._create_llm_with_model")
    @mock.patch("src.models.create_llm")
    def test_openrouter_uses_default_fallbacks_when_env_missing(
        self,
        mock_create_llm: mock.MagicMock,
        mock_create_with_model: mock.MagicMock,
    ) -> None:
        """Use built-in fallback models when env var is unset.

        Health-check probing was removed (self-inflicted rate limits and
        false rejections of throttled models). The pool now includes
        every curated fallback that instantiates successfully.
        """
        with mock.patch.dict(
            os.environ, {"OPENROUTER_FALLBACK_MODELS": ""}, clear=False
        ):
            mock_create_llm.return_value = "primary"
            mock_create_with_model.side_effect = [
                "fb1",
                "fb2",
                "fb3",
                "fb4",
                "fb5",
                "fb6",
                "fb7",
            ]

            pool = create_llm_pool(AgentRole.FIXER)

        self.assertEqual(pool[0], "primary")
        self.assertEqual(len(pool), 8)
        self.assertEqual(mock_create_with_model.call_count, 7)

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=False)
    @mock.patch("src.models.create_llm")
    def test_non_openrouter_returns_primary_only(
        self, mock_create_llm: mock.MagicMock
    ) -> None:
        """Gemini/OpenAI should not build fallback pools."""
        mock_create_llm.return_value = "primary"
        pool = create_llm_pool(AgentRole.SCOUT)
        self.assertEqual(pool, ["primary"])

    @mock.patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_FALLBACK_MODELS": "primary-model, bad, good",
        },
        clear=False,
    )
    @mock.patch("src.models._create_llm_with_model")
    @mock.patch("src.models.create_llm")
    def test_openrouter_skips_primary_and_failed_fallbacks(
        self,
        mock_create_llm: mock.MagicMock,
        mock_create_with_model: mock.MagicMock,
    ) -> None:
        """Router pools skip duplicate and failed fallback models."""
        primary = SimpleNamespace(model_name="primary-model")
        mock_create_llm.return_value = primary
        mock_create_with_model.side_effect = [
            RuntimeError("boom"),
            "good",
            "router",
        ]

        pool = create_llm_pool(AgentRole.FIXER)

        self.assertEqual(pool, [primary, "good", "router"])
        self.assertEqual(
            [call.args[1] for call in mock_create_with_model.call_args_list],
            ["bad", "good", OPENROUTER_FREE_ROUTER],
        )


class TestOpenRouterFallbackIds(unittest.TestCase):
    """Tests for the shared OpenRouter fallback chain."""

    @mock.patch.dict(
        os.environ, {"OPENROUTER_FALLBACK_MODELS": ""}, clear=False
    )
    def test_free_router_is_terminal_entry(self) -> None:
        """openrouter/free must close the default chain."""
        ids = _openrouter_fallback_ids()
        self.assertEqual(ids[-1], OPENROUTER_FREE_ROUTER)

    @mock.patch.dict(
        os.environ, {"OPENROUTER_FALLBACK_MODELS": ""}, clear=False
    )
    def test_auto_router_not_in_defaults(self) -> None:
        """Keep openrouter/auto out of the default chain.

        The Auto Router is paid-only — a guaranteed 402 on a
        zero-credit account.
        """
        self.assertNotIn("openrouter/auto", _openrouter_fallback_ids())

    @mock.patch.dict(
        os.environ,
        {"OPENROUTER_FALLBACK_MODELS": "a/x:free, b/y:free"},
        clear=False,
    )
    def test_env_override_still_appends_free_router(self) -> None:
        """A custom chain still degrades to openrouter/free last."""
        self.assertEqual(
            _openrouter_fallback_ids(),
            ["a/x:free", "b/y:free", OPENROUTER_FREE_ROUTER],
        )


class TestServerSideFallback(unittest.TestCase):
    """The OpenRouter LLM must carry a server-side models array."""

    @mock.patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_FALLBACK_MODELS": "",
        },
        clear=False,
    )
    @mock.patch("src.models.ChatOpenAI")
    def test_extra_body_models_excludes_primary(
        self, mock_chat: mock.MagicMock
    ) -> None:
        """extra_body carries the fallback chain minus the primary."""
        def fake_openai(**kwargs):
            """Return constructor kwargs as a simple fake LLM."""
            return SimpleNamespace(**kwargs, model_name=kwargs["model"])

        mock_chat.side_effect = fake_openai
        llm = _create_llm_with_model(
            AgentRole.FIXER, "openai/gpt-oss-120b:free"
        )
        models = llm.extra_body["models"]
        self.assertNotIn("openai/gpt-oss-120b:free", models)
        self.assertEqual(models[-1], OPENROUTER_FREE_ROUTER)
        # OpenRouter rejects the request with HTTP 400 when the models
        # array has more than 3 entries — regression guard for the
        # 2026-07-02 zlib planner failure.
        self.assertLessEqual(len(models), 3)
        self.assertEqual(len(models), len(set(models)))


class TestCostForUsage(unittest.TestCase):
    """Tests for real token-based cost computation."""

    def test_free_slug_costs_zero(self) -> None:
        """Free-tier ``:free`` slugs bill nothing."""
        from src.models import cost_for_usage

        self.assertEqual(
            cost_for_usage("qwen/qwen3-coder:free", 100000, 50000), 0.0
        )

    def test_free_router_costs_zero(self) -> None:
        """The Free Models Router bills nothing."""
        from src.models import cost_for_usage

        self.assertEqual(cost_for_usage("openrouter/free", 1000, 1000), 0.0)

    def test_paid_model_priced_from_table(self) -> None:
        """Table-listed paid models price per million tokens."""
        from src.models import cost_for_usage

        self.assertAlmostEqual(cost_for_usage("gpt-4o", 1_000_000, 0), 2.50)
        self.assertAlmostEqual(
            cost_for_usage("gpt-4o-mini", 0, 1_000_000), 0.60
        )

    def test_unknown_paid_model_uses_conservative_default(self) -> None:
        """Unlisted paid models over-count rather than bill as free."""
        from src.models import cost_for_usage

        cost = cost_for_usage("some/unknown-model", 1_000_000, 0)
        self.assertGreater(cost, 0.0)

    def test_negative_tokens_clamped(self) -> None:
        """Bogus negative usage never produces negative cost."""
        from src.models import cost_for_usage

        self.assertEqual(cost_for_usage("gpt-4o", -5, -5), 0.0)


class TestPrintModelInfo(unittest.TestCase):
    """Tests for selected-provider reporting."""

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True)
    @mock.patch("builtins.print")
    def test_prints_provider_and_models(
        self, mock_print: mock.MagicMock
    ) -> None:
        """Known providers print the compact model summary."""
        print_model_info()

        printed = [call.args[0] for call in mock_print.call_args_list]
        self.assertEqual(printed[0], "   Provider: openai")
        self.assertIn("gpt-4o-mini", printed[1])
        self.assertIn("gpt-4o", printed[1])

    @mock.patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=True)
    @mock.patch("builtins.print")
    def test_unknown_provider_prints_provider_only(
        self, mock_print: mock.MagicMock
    ) -> None:
        """Unknown providers do not attempt a model summary."""
        print_model_info()

        mock_print.assert_called_once_with("   Provider: unknown")


if __name__ == "__main__":
    unittest.main()

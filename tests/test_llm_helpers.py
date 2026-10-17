"""Tests for src/llm_helpers.py — validated LLM call with retry + fallback."""

import unittest
from types import SimpleNamespace
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.llm_helpers import (
    LLMCallOutcome,
    ValidationResult,
    _accepts_timeout,
    _build_replacement_llm,
    _extract_affordable_tokens,
    _extract_slug_hint,
    _is_provider_error,
    _shrink_max_tokens,
    bind_output_schema,
    extract_content,
    extract_json_block,
    is_refusal,
    llm_call_with_validation,
    response_cache_usage,
    response_cost,
    response_stop_reason,
    response_usage,
    system_message,
)


class TestExtractHelpers(unittest.TestCase):
    """Tests for extract_content and extract_json_block."""

    def test_extract_content_str_passthrough(self) -> None:
        self.assertEqual(extract_content("hello"), "hello")

    def test_extract_content_list_of_dicts(self) -> None:
        out = extract_content([{"text": "a"}, {"text": "b"}])
        self.assertEqual(out, "a\nb")

    def test_extract_content_list_of_strings(self) -> None:
        out = extract_content(["x", "y"])
        self.assertEqual(out, "x\ny")

    def test_extract_content_falls_back_to_str(self) -> None:
        out = extract_content(SimpleNamespace(value=3))
        self.assertEqual(out, "namespace(value=3)")

    def test_extract_json_block_strips_prose(self) -> None:
        text = 'Some prose\n{"a": 1, "b": 2}\nmore prose'
        self.assertEqual(extract_json_block(text), '{"a": 1, "b": 2}')

    def test_extract_json_block_handles_no_braces(self) -> None:
        text = "no json here"
        self.assertEqual(extract_json_block(text), "no json here")

    def test_extract_json_block_nested(self) -> None:
        text = 'prose {"outer": {"inner": 1}} more'
        self.assertEqual(extract_json_block(text), '{"outer": {"inner": 1}}')


def _ok_validator(_data):
    return ValidationResult.good()


def _make_invoke(responses):
    """Build a fake invoke_fn that returns responses[i] on the i-th call."""
    calls = {"n": 0}

    def invoke_fn(llm, messages, timeout=120):
        i = calls["n"]
        calls["n"] += 1
        r = responses[i]
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=r)

    return invoke_fn, calls


class TestRateLimitHandling(unittest.TestCase):
    """Daily-cap fast-fail, Retry-After honoring, and timeout rotation."""

    def test_daily_cap_breaks_immediately_to_fallback(self) -> None:
        """The account-wide daily cap must not trigger rotation.

        Every extra request against a free-models-per-day-capped
        account is wasted; the helper must stop after one attempt and
        use the deterministic fallback.
        """
        err = RuntimeError(
            "Error code: 429 - Rate limit exceeded: free-models-per-day."
            " Add 10 credits to unlock 1000 free model requests per day"
        )
        invoke, calls = _make_invoke([err, '{"x": 1}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=SimpleNamespace(model_name="primary"),
                fallback_llms=[SimpleNamespace(model_name="backup")],
                prompt="p",
                validator=lambda d: ValidationResult.good(),
                fallback_factory=lambda: {"fb": True},
            )
        self.assertEqual(calls["n"], 1)  # no rotation, no retry
        self.assertTrue(out.used_fallback)
        self.assertIn("free-model quota exhausted", out.last_error)

    def test_timeout_rotates_only_when_asked(self) -> None:
        """rotate_on_timeout moves on; the last model's timeout ends it."""
        models = []

        def run(responses, **kwargs):
            invoke, calls = _make_invoke(responses)

            def recording(llm, messages, timeout=120):
                models.append((llm.model_name, timeout))
                return invoke(llm, messages, timeout)

            with mock.patch("src.llm_helpers.log_llm_call"):
                out = llm_call_with_validation(
                    invoke_fn=recording,
                    llm=SimpleNamespace(model_name="primary"),
                    fallback_llms=[SimpleNamespace(model_name="backup")],
                    prompt="p",
                    validator=lambda d: ValidationResult.good(),
                    timeout=240,
                    **kwargs,
                )
            return out, calls["n"]

        late = TimeoutError("LLM call timed out after 240s")
        out, n = run([late, '{"x": 1}'], rotate_on_timeout=True)
        self.assertEqual((out.data, n), ({"x": 1}, 2))
        self.assertEqual(models, [("primary", 240), ("backup", 240)])
        models.clear()
        out, n = run([late, late, '{"x": 1}'], rotate_on_timeout=True)
        self.assertEqual((out.data, n), (None, 2))
        self.assertIn("timed out", out.last_error)
        models.clear()
        out, n = run([late, '{"x": 1}'])
        self.assertEqual(out.data, {"x": 1})
        self.assertEqual(models, [("primary", 240), ("primary", 240)])

    def test_short_retry_after_sleeps_and_retries_same_model(self) -> None:
        err = RuntimeError(
            "Error code: 429 - temporarily rate-limited upstream,"
            " 'retry_after_seconds': 29, 'Retry-After': '29'"
        )
        invoke, calls = _make_invoke([err, '{"x": 1}'])
        sleeps: list = []
        with (
            mock.patch("src.llm_helpers.log_llm_call"),
            mock.patch(
                "src.llm_helpers.time.sleep", side_effect=sleeps.append
            ),
        ):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=SimpleNamespace(model_name="primary"),
                fallback_llms=[SimpleNamespace(model_name="backup")],
                prompt="p",
                validator=lambda d: ValidationResult.good(),
            )
        self.assertEqual(sleeps, [29])
        self.assertEqual(out.data, {"x": 1})
        self.assertEqual(calls["n"], 2)

    def test_huge_retry_after_rotates_instead_of_sleeping(self) -> None:
        err = RuntimeError(
            "Error code: 429 - Provider returned error,"
            " 'Retry-After': '45091'"
        )
        invoke, calls = _make_invoke([err, '{"x": 1}'])
        sleeps: list = []
        with (
            mock.patch("src.llm_helpers.log_llm_call"),
            mock.patch(
                "src.llm_helpers.time.sleep", side_effect=sleeps.append
            ),
        ):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=SimpleNamespace(model_name="primary"),
                fallback_llms=[SimpleNamespace(model_name="backup")],
                prompt="p",
                validator=lambda d: ValidationResult.good(),
            )
        self.assertEqual(sleeps, [])
        self.assertEqual(out.data, {"x": 1})


class TestLLMCallWithValidation(unittest.TestCase):
    """Tests for llm_call_with_validation retries and fallbacks."""

    def test_first_attempt_success(self) -> None:
        invoke, calls = _make_invoke(['{"x": 1}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
            )
        self.assertIsInstance(out, LLMCallOutcome)
        self.assertEqual(out.data, {"x": 1})
        self.assertFalse(out.used_fallback)
        self.assertEqual(out.attempts, 1)
        self.assertEqual(calls["n"], 1)

    def test_json_parse_failure_then_retry_success(self) -> None:
        invoke, calls = _make_invoke(["not json", '{"x": 2}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                max_retries=2,
            )
        self.assertEqual(out.data, {"x": 2})
        self.assertEqual(out.attempts, 2)

    def test_validation_failure_retries_with_critique(self) -> None:
        def validator(d):
            if d.get("x") == 1:
                return ValidationResult.bad("missing y")
            return ValidationResult.good()

        invoke, calls = _make_invoke(['{"x": 1}', '{"x": 2, "y": 3}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=validator,
                max_retries=2,
            )
        self.assertTrue(out.data is not None)
        self.assertEqual(out.data["y"], 3)
        self.assertEqual(out.attempts, 2)

    def test_exhausts_retries_uses_fallback(self) -> None:
        invoke, calls = _make_invoke(["bad", "still bad", "still bad"])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                fallback_factory=lambda: {"fallback": True},
                max_retries=2,
            )
        self.assertTrue(out.used_fallback)
        self.assertEqual(out.data, {"fallback": True})
        self.assertEqual(out.attempts, 3)

    def test_exhausts_retries_no_fallback_returns_none(self) -> None:
        invoke, _ = _make_invoke(["bad", "bad"])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                max_retries=1,
            )
        self.assertIsNone(out.data)
        self.assertFalse(out.used_fallback)
        self.assertTrue(out.last_error)

    def test_llm_exception_triggers_retry(self) -> None:
        invoke, _ = _make_invoke([RuntimeError("network blip"), '{"x": 1}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                max_retries=2,
            )
        self.assertEqual(out.data, {"x": 1})
        self.assertEqual(out.attempts, 2)

    def test_non_provider_exception_without_retry_returns_none(self) -> None:
        invoke, _ = _make_invoke([RuntimeError("local validation crash")])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                max_retries=0,
            )
        self.assertIsNone(out.data)
        self.assertIn("local validation crash", out.last_error)

    def test_top_level_non_dict_json_rejected(self) -> None:
        invoke, _ = _make_invoke(["[1, 2, 3]"])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                fallback_factory=lambda: {"fb": 1},
                max_retries=0,
            )
        self.assertTrue(out.used_fallback)

    def test_empty_response_without_retry_returns_none(self) -> None:
        invoke, _ = _make_invoke([""])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                max_retries=0,
            )
        self.assertIsNone(out.data)
        self.assertIn("empty response", out.last_error)

    def test_validation_failure_without_retry_returns_none(self) -> None:
        invoke, _ = _make_invoke(['{"x": 1}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=lambda _data: ValidationResult.bad("missing y"),
                max_retries=0,
            )
        self.assertIsNone(out.data)
        self.assertIn("missing y", out.last_error)

    def test_fallback_factory_exception_returns_none(self) -> None:
        invoke, _ = _make_invoke(["not json"])

        def failing_fallback():
            """Raise from a deterministic fallback factory."""
            raise RuntimeError("fallback crashed")

        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hello",
                validator=_ok_validator,
                fallback_factory=failing_fallback,
                max_retries=0,
            )
        self.assertIsNone(out.data)
        self.assertFalse(out.used_fallback)


class TestProviderErrorDetection(unittest.TestCase):
    """Tests for provider-error detection used for model rotation."""

    def test_detects_429_rate_limit(self) -> None:
        self.assertTrue(_is_provider_error(RuntimeError("Error code: 429")))

    def test_detects_rate_limit_phrase(self) -> None:
        self.assertTrue(
            _is_provider_error(
                RuntimeError("provider temporarily rate limited")
            )
        )

    def test_ignores_local_validation_error(self) -> None:
        self.assertFalse(
            _is_provider_error(RuntimeError("json parse failure"))
        )


class TestSlugHintExtraction(unittest.TestCase):
    """Tests for parsing OpenRouter's ``use this slug instead:`` hint."""

    def test_extracts_slug_from_404_body(self) -> None:
        exc = RuntimeError(
            "Error code: 404 - {'error': {'message': 'This model is "
            "unavailable for free. The paid version is available now "
            "- use this slug instead: qwen/qwen3-14b', 'code': 404}}"
        )
        self.assertEqual(_extract_slug_hint(exc), "qwen/qwen3-14b")

    def test_returns_none_when_no_hint(self) -> None:
        self.assertIsNone(_extract_slug_hint(RuntimeError("no hint here")))

    def test_extracts_slug_with_version_and_colon(self) -> None:
        exc = RuntimeError(
            "use this slug instead: google/gemini-2.0-flash-exp:free "
            "(other prose)"
        )
        self.assertEqual(
            _extract_slug_hint(exc), "google/gemini-2.0-flash-exp:free"
        )


class TestHotAddedReplacementModel(unittest.TestCase):
    """Rotation kicks in for a hinted slug, even with an empty pool."""

    def test_hinted_slug_added_and_rotated_on_next_attempt(self) -> None:
        seen_models: list = []

        def invoke(llm, messages, timeout=120):
            """Fail on the primary, succeed on the replacement."""
            seen_models.append(getattr(llm, "model_name", "primary"))
            if getattr(llm, "model_name", None) == "primary":
                raise RuntimeError(
                    "Error code: 404 - {'error': {'message': "
                    "'use this slug instead: replacement/free', "
                    "'code': 404}}"
                )
            return SimpleNamespace(content='{"ok": true}')

        primary = mock.MagicMock()
        primary.model_name = "primary"

        # `_build_replacement_llm` calls type(reference)(**kwargs); we
        # patch it so we don't need a real LangChain constructor.
        replacement = mock.MagicMock()
        replacement.model_name = "replacement/free"

        with (
            mock.patch("src.llm_helpers.log_llm_call"),
            mock.patch(
                "src.llm_helpers._build_replacement_llm",
                return_value=replacement,
            ),
        ):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=primary,
                prompt="hi",
                validator=lambda d: ValidationResult.good(),
                max_retries=2,
            )

        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(seen_models[0], "primary")
        self.assertIn("replacement/free", seen_models)


class TestEmptyResponseRotation(unittest.TestCase):
    """Empty / whitespace-only content must trigger pool rotation."""

    def test_empty_string_rotates_to_fallback(self) -> None:
        """Provider returning ``""`` should rotate, not consume retries."""
        invoke, calls = _make_invoke(["", '{"ok": true}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                fallback_llms=[mock.MagicMock()],
                prompt="hi",
                validator=_ok_validator,
                max_retries=0,
            )
        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(calls["n"], 2)

    def test_whitespace_only_rotates_to_fallback(self) -> None:
        invoke, calls = _make_invoke(["   \n\t ", '{"ok": true}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                fallback_llms=[mock.MagicMock()],
                prompt="hi",
                validator=_ok_validator,
                max_retries=0,
            )
        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(calls["n"], 2)

    def test_empty_falls_back_to_critique_when_pool_exhausted(self) -> None:
        invoke, calls = _make_invoke(["", '{"ok": true}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hi",
                validator=_ok_validator,
                max_retries=2,
            )
        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(calls["n"], 2)


class TestJSONParseRotation(unittest.TestCase):
    """Persistent unparseable output should rotate models before critique."""

    def test_bad_json_rotates_before_critique(self) -> None:
        invoke, calls = _make_invoke(["not json", '{"ok": true}'])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                fallback_llms=[mock.MagicMock()],
                prompt="hi",
                validator=_ok_validator,
                max_retries=0,
            )
        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(calls["n"], 2)


class TestTokenBudgetSelfHeal(unittest.TestCase):
    """HTTP 402 with ``can only afford N`` must shrink caps + retry."""

    def test_extract_affordable_tokens_from_402_body(self) -> None:
        exc = RuntimeError(
            "Error code: 402 - {'error': {'message': "
            "'This request requires more credits, or fewer max_tokens. "
            "You requested up to 65536 tokens, but can only afford "
            "5702. To increase, visit...'}}"
        )
        self.assertEqual(_extract_affordable_tokens(exc), 5702)

    def test_extract_affordable_tokens_returns_none_without_hint(self) -> None:
        self.assertIsNone(
            _extract_affordable_tokens(RuntimeError("connection reset"))
        )

    def test_shrink_max_tokens_updates_attribute(self) -> None:
        llm = SimpleNamespace(max_tokens=65536)
        changed = _shrink_max_tokens(llm, 5702)
        self.assertTrue(changed)
        # 10 % headroom: floor(5702 * 0.9) == 5131
        self.assertEqual(llm.max_tokens, 5131)

    def test_shrink_max_tokens_leaves_already_smaller_cap(self) -> None:
        llm = SimpleNamespace(max_tokens=1000)
        changed = _shrink_max_tokens(llm, 5702)
        self.assertFalse(changed)
        self.assertEqual(llm.max_tokens, 1000)

    def test_shrink_max_tokens_updates_model_kwargs(self) -> None:
        llm = SimpleNamespace(
            model_kwargs={"max_tokens": 8000, "max_output_tokens": 7000}
        )
        changed = _shrink_max_tokens(llm, 1000)
        self.assertTrue(changed)
        self.assertEqual(llm.model_kwargs["max_tokens"], 900)
        self.assertEqual(llm.model_kwargs["max_output_tokens"], 900)

    def test_shrink_max_tokens_ignores_setattr_errors(self) -> None:
        class ReadOnlyCap:
            """Object whose cap property refuses writes."""

            @property
            def max_tokens(self):
                """Return an oversized cap."""
                return 9000

            @max_tokens.setter
            def max_tokens(self, value):
                """Reject cap updates."""
                raise RuntimeError("read-only")

        changed = _shrink_max_tokens(ReadOnlyCap(), 1000)
        self.assertFalse(changed)

    def test_402_error_shrinks_and_retries_same_model(self) -> None:
        primary = SimpleNamespace(max_tokens=65536, model_name="p")
        fallback = SimpleNamespace(max_tokens=65536, model_name="f")

        seen_caps: list = []

        def invoke(llm, messages, timeout=120):
            """Fail with 402 while cap > 6000, then succeed."""
            seen_caps.append(getattr(llm, "max_tokens", None))
            if getattr(llm, "max_tokens", 0) > 6000:
                raise RuntimeError(
                    "Error code: 402 - {'error': {'message': "
                    "'requires more credits. You requested up to "
                    "65536 tokens, but can only afford 5702.'}}"
                )
            return SimpleNamespace(content='{"ok": true}')

        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=primary,
                fallback_llms=[fallback],
                prompt="hi",
                validator=_ok_validator,
                max_retries=1,
            )

        self.assertEqual(out.data, {"ok": True})
        # First call at 65536 → 402; shrink to 5131; second call
        # succeeds on the SAME model (no rotation needed).
        self.assertEqual(seen_caps[0], 65536)
        self.assertLess(seen_caps[1], 6000)
        # Fallback should have been shrunk too, ready for future calls.
        self.assertLess(fallback.max_tokens, 6000)


class TestInternalUtilities(unittest.TestCase):
    """Tests for small internal LLM helper utilities."""

    def test_accepts_timeout_handles_uninspectable_callable(self) -> None:
        with mock.patch("inspect.signature", side_effect=ValueError):
            self.assertFalse(_accepts_timeout(lambda _llm, _messages: None))

    def test_build_replacement_llm_clones_common_attrs(self) -> None:
        class CloneableLLM:
            """Minimal LLM constructor used by replacement cloning."""

            def __init__(self, model=None, **kwargs):
                self.model = model
                self.kwargs = kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)

        reference = CloneableLLM(
            model="old/model",
            temperature=0.2,
            openai_api_key="key",
            openai_api_base="https://example.invalid",
            request_timeout=30,
            timeout=31,
        )

        replacement = _build_replacement_llm(reference, "new/model")

        self.assertIsInstance(replacement, CloneableLLM)
        self.assertEqual(replacement.model, "new/model")
        self.assertEqual(replacement.temperature, 0.2)
        self.assertEqual(replacement.openai_api_key, "key")
        self.assertEqual(replacement.timeout, 31)


class TestUsageAccounting(unittest.TestCase):
    """Real token/cost accounting on LLMCallOutcome."""

    def _llm(self, name):
        """Fake LLM with a model name."""
        llm = mock.MagicMock()
        llm.model_name = name
        return llm

    def _invoke_with_usage(self, content, tokens_in, tokens_out):
        """Fake invoke_fn returning content plus usage metadata."""

        def invoke_fn(llm, messages, timeout=120):
            return SimpleNamespace(
                content=content,
                usage_metadata={
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                },
            )

        return invoke_fn

    def test_response_usage_reads_usage_metadata(self) -> None:
        r = SimpleNamespace(
            usage_metadata={"input_tokens": 10, "output_tokens": 4}
        )
        self.assertEqual(response_usage(r), (10, 4))

    def test_response_usage_falls_back_to_token_usage(self) -> None:
        r = SimpleNamespace(
            usage_metadata=None,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                }
            },
        )
        self.assertEqual(response_usage(r), (7, 3))

    def test_response_usage_unknown_is_zero(self) -> None:
        self.assertEqual(response_usage(SimpleNamespace()), (0, 0))

    def test_response_cost_prices_usage(self) -> None:
        response = SimpleNamespace(
            usage_metadata={"input_tokens": 1000, "output_tokens": 100}
        )
        cost = response_cost(self._llm("gpt-4o"), response)
        self.assertAlmostEqual(cost, 0.0035)

    def test_outcome_carries_free_model_zero_cost(self) -> None:
        """Free-tier models bill $0 but still report tokens."""
        outcome = llm_call_with_validation(
            invoke_fn=self._invoke_with_usage('{"a": 1}', 500, 100),
            llm=self._llm("qwen/qwen3-coder:free"),
            prompt="p",
            validator=_ok_validator,
        )
        self.assertEqual(outcome.input_tokens, 500)
        self.assertEqual(outcome.output_tokens, 100)
        self.assertEqual(outcome.cost_usd, 0.0)

    def test_outcome_prices_paid_model_from_table(self) -> None:
        outcome = llm_call_with_validation(
            invoke_fn=self._invoke_with_usage('{"a": 1}', 1000, 100),
            llm=self._llm("gpt-4o"),
            prompt="p",
            validator=_ok_validator,
        )
        # 1000 * 2.50/1M + 100 * 10.00/1M
        self.assertAlmostEqual(outcome.cost_usd, 0.0035)

    def test_usage_accumulates_across_retries(self) -> None:
        """Every attempt bills tokens — retries accumulate."""
        verdicts = iter(
            [ValidationResult.bad("nope"), ValidationResult.good()]
        )

        def flaky_validator(_data):
            """Fail once, then pass."""
            return next(verdicts)

        outcome = llm_call_with_validation(
            invoke_fn=self._invoke_with_usage('{"a": 1}', 100, 10),
            llm=self._llm("gpt-4o"),
            prompt="p",
            validator=flaky_validator,
        )
        self.assertEqual(outcome.attempts, 2)
        self.assertEqual(outcome.input_tokens, 200)
        self.assertEqual(outcome.output_tokens, 20)


class TestClaudeResponses(unittest.TestCase):
    """Tests for Claude reply shapes in the shared helpers."""

    def test_extract_content_skips_thinking_blocks(self) -> None:
        content = [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": '{"x": 1}'},
        ]
        self.assertEqual(extract_content(content), '{"x": 1}')

    def test_audit_log_names_a_model_without_model_name(self) -> None:
        invoke, _ = _make_invoke(['{"x": 1}'])
        llm = SimpleNamespace(model="claude-haiku-5-5")
        with mock.patch("src.llm_helpers.log_llm_call") as log:
            llm_call_with_validation(
                invoke_fn=invoke,
                llm=llm,
                prompt="hello",
                validator=_ok_validator,
            )
        self.assertEqual(log.call_args.kwargs["model"], "claude-haiku-5-5")


def _reply(content, stop_reason):
    """Build a fake LLM reply with a provider stop reason."""
    return SimpleNamespace(
        content=content, response_metadata={"stop_reason": stop_reason}
    )


class TestRefusalHandling(unittest.TestCase):
    """A safety refusal ends the call: no critique retry of the prompt."""

    def _invoke(self, replies):
        calls = {"n": 0}

        def invoke_fn(llm, messages, timeout=120):
            calls["n"] += 1
            return replies[calls["n"] - 1]

        return invoke_fn, calls

    def test_stop_reason_and_refusal_detection(self) -> None:
        self.assertEqual(
            response_stop_reason(_reply("x", "end_turn")), "end_turn"
        )
        self.assertTrue(is_refusal(_reply("", "refusal")))
        self.assertTrue(
            is_refusal(
                SimpleNamespace(
                    content="",
                    response_metadata={"finish_reason": "content_filter"},
                )
            )
        )
        self.assertFalse(is_refusal(_reply('{"a": 1}', "end_turn")))
        self.assertFalse(is_refusal(SimpleNamespace(content="x")))
        self.assertEqual(
            response_stop_reason(SimpleNamespace(content="x")), ""
        )

    def test_refusal_is_not_retried(self) -> None:
        invoke, calls = self._invoke(
            [_reply("", "refusal"), _reply('{"ok": true}', "end_turn")]
        )
        with mock.patch("src.llm_helpers.log_llm_call") as log:
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hi",
                validator=_ok_validator,
                max_retries=2,
            )
        self.assertEqual(calls["n"], 1)
        self.assertTrue(out.refused)
        self.assertIsNone(out.data)
        self.assertEqual(out.attempts, 1)
        self.assertIn("refused", out.last_error)
        self.assertEqual(
            log.call_args.kwargs["metadata"]["stop_reason"], "refusal"
        )

    def test_refusal_rotates_to_another_model(self) -> None:
        invoke, calls = self._invoke(
            [_reply("", "refusal"), _reply('{"ok": true}', "end_turn")]
        )
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                fallback_llms=[mock.MagicMock()],
                prompt="hi",
                validator=_ok_validator,
            )
        self.assertEqual(calls["n"], 2)
        self.assertEqual(out.data, {"ok": True})
        self.assertFalse(out.refused)

    def test_refusal_still_returns_the_fallback(self) -> None:
        invoke, _ = self._invoke([_reply("", "refusal")])
        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="hi",
                validator=_ok_validator,
                fallback_factory=lambda: {"fb": 1},
            )
        self.assertTrue(out.used_fallback)
        self.assertTrue(out.refused)
        self.assertEqual(out.data, {"fb": 1})


class TestCachedSystemPrompt(unittest.TestCase):
    """Static instructions go first, as a cacheable system message."""

    def test_claude_system_blocks_carry_cache_breakpoints(self) -> None:
        msgs = system_message(["rules", "", "knowledge"], _claude())
        self.assertEqual(len(msgs), 1)
        self.assertIsInstance(msgs[0], SystemMessage)
        self.assertEqual(
            msgs[0].content,
            [
                {
                    "type": "text",
                    "text": "rules",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "knowledge",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        )

    def test_other_models_get_plain_text(self) -> None:
        msgs = system_message("rules", SimpleNamespace(model_name="gpt-4o"))
        self.assertEqual(msgs[0].content, "rules")
        self.assertEqual(system_message(None, _claude()), [])

    def test_cache_usage_is_read_from_usage_metadata(self) -> None:
        reply = SimpleNamespace(
            content="x",
            usage_metadata={
                "input_tokens": 5000,
                "output_tokens": 10,
                "input_token_details": {
                    "cache_read": 3000,
                    "cache_creation": 0,
                    "ephemeral_5m_input_tokens": 1500,
                },
            },
        )
        self.assertEqual(response_cache_usage(reply), (3000, 1500))
        self.assertEqual(
            response_cache_usage(SimpleNamespace(content="x")), (0, 0)
        )

    def test_call_sends_system_first_and_logs_cache_tokens(self) -> None:
        seen = {}

        def invoke_fn(llm, messages, timeout=120):
            seen["messages"] = list(messages)
            return SimpleNamespace(
                content='{"ok": true}',
                response_metadata={"stop_reason": "end_turn"},
                usage_metadata={
                    "input_tokens": 4000,
                    "output_tokens": 100,
                    "input_token_details": {"cache_read": 3000},
                },
            )

        with mock.patch("src.llm_helpers.log_llm_call") as log:
            out = llm_call_with_validation(
                invoke_fn=invoke_fn,
                llm=_claude(),
                prompt="the task",
                system_prompt=["the rules"],
                validator=_ok_validator,
            )
        first, second = seen["messages"]
        self.assertIsInstance(first, SystemMessage)
        self.assertEqual(second.content, "the task")
        meta = log.call_args.kwargs["metadata"]
        self.assertEqual((meta["cache_read"], meta["cache_write"]), (3000, 0))
        # 1000 plain tokens at $2/M + 3000 read at $0.2/M + 100 out at $10/M
        self.assertAlmostEqual(out.cost_usd, (2000 + 600 + 1000) / 1e6)
        self.assertIn("the rules", log.call_args.kwargs["prompt"])


def _claude():
    """Build a fake Claude chat model."""
    return SimpleNamespace(model="claude-sonnet-5-5")


class TestInformedRetry(unittest.TestCase):
    """A retry shows the model its rejected reply before the critique."""

    def test_rejected_reply_is_kept_before_the_critique(self) -> None:
        threads = []
        replies = [
            AIMessage(content='{"x": 1}'),
            AIMessage(content='{"x": 2, "y": 3}'),
        ]

        def invoke_fn(llm, messages, timeout=120):
            threads.append(list(messages))
            return replies[len(threads) - 1]

        def validator(d):
            if "y" not in d:
                return ValidationResult.bad("missing y")
            return ValidationResult.good()

        with mock.patch("src.llm_helpers.log_llm_call"):
            out = llm_call_with_validation(
                invoke_fn=invoke_fn,
                llm=mock.MagicMock(),
                prompt="task",
                validator=validator,
            )
        second = threads[1]
        self.assertEqual(
            [type(m).__name__ for m in second],
            ["HumanMessage", "AIMessage", "HumanMessage"],
        )
        self.assertIs(second[1], replies[0])
        self.assertIn("missing y", second[2].content)
        self.assertIn("(above)", second[2].content)
        self.assertEqual(out.data, {"x": 2, "y": 3})
        # The outcome carries the thread for a follow-up round.
        self.assertEqual(len(out.messages), 4)
        self.assertIs(out.messages[-1], replies[1])

    def test_unparseable_reply_is_kept_too(self) -> None:
        threads = []
        replies = ["not json", '{"ok": true}']

        def invoke_fn(llm, messages, timeout=120):
            threads.append(list(messages))
            return SimpleNamespace(content=replies[len(threads) - 1])

        with mock.patch("src.llm_helpers.log_llm_call"):
            llm_call_with_validation(
                invoke_fn=invoke_fn,
                llm=mock.MagicMock(),
                prompt="task",
                validator=_ok_validator,
            )
        self.assertIsInstance(threads[1][1], AIMessage)
        self.assertEqual(threads[1][1].content, "not json")

    def test_history_goes_between_system_and_prompt(self) -> None:
        seen = {}

        def invoke_fn(llm, messages, timeout=120):
            seen["m"] = list(messages)
            return SimpleNamespace(content='{"ok": true}')

        earlier = [HumanMessage(content="round 0"), AIMessage(content="{}")]
        with mock.patch("src.llm_helpers.log_llm_call"):
            llm_call_with_validation(
                invoke_fn=invoke_fn,
                llm=_claude(),
                prompt="round 1",
                system_prompt="rules",
                history=earlier,
                validator=_ok_validator,
            )
        names = [type(m).__name__ for m in seen["m"]]
        self.assertEqual(
            names,
            ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"],
        )
        self.assertEqual(seen["m"][-1].content, "round 1")


class TestStructuredOutput(unittest.TestCase):
    """JSON roles ask Claude for structured output of their schema."""

    def test_claude_models_are_bound_to_the_schema(self) -> None:
        llm = mock.MagicMock()
        llm.model_name = None
        llm.model = "claude-haiku-5-5"
        schema = {"type": "object", "properties": {}}
        bound = bind_output_schema(llm, schema)
        llm.bind.assert_called_once_with(response_format=schema)
        self.assertIs(bound, llm.bind.return_value)

    def test_other_models_and_no_schema_are_unchanged(self) -> None:
        llm = SimpleNamespace(model_name="gpt-4o")
        self.assertIs(bind_output_schema(llm, {"type": "object"}), llm)
        claude = _claude()
        self.assertIs(bind_output_schema(claude, None), claude)

    def test_the_call_invokes_the_bound_model(self) -> None:
        llm = mock.MagicMock()
        llm.model_name = None
        llm.model = "claude-sonnet-5-5"
        seen = []

        def invoke_fn(model, messages, timeout=120):
            seen.append(model)
            return SimpleNamespace(content='{"ok": true}')

        with mock.patch("src.llm_helpers.log_llm_call") as log:
            llm_call_with_validation(
                invoke_fn=invoke_fn,
                llm=llm,
                prompt="task",
                output_schema={"type": "object"},
                validator=_ok_validator,
            )
        self.assertIs(seen[0], llm.bind.return_value)
        self.assertEqual(log.call_args.kwargs["model"], "claude-sonnet-5-5")


class TestRaisedAttemptsAreLogged(unittest.TestCase):
    """An attempt that raised is in the audit log, with the error."""

    def test_raised_attempt_is_logged(self) -> None:
        invoke, _ = _make_invoke([RuntimeError("socket closed"), '{"ok": 1}'])
        with mock.patch("src.llm_helpers.log_llm_call") as log:
            llm_call_with_validation(
                invoke_fn=invoke,
                llm=mock.MagicMock(),
                prompt="task",
                validator=_ok_validator,
            )
        first = log.call_args_list[0].kwargs
        self.assertEqual(first["response"], "")
        self.assertIn("socket closed", first["metadata"]["error"])
        self.assertEqual(log.call_count, 2)


if __name__ == "__main__":
    unittest.main()

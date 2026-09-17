"""Tests for src/llm_helpers.py — validated LLM call with retry + fallback."""

import unittest
from types import SimpleNamespace
from unittest import mock

from src.llm_helpers import (
    LLMCallOutcome,
    ValidationResult,
    _accepts_timeout,
    _build_replacement_llm,
    _extract_affordable_tokens,
    _extract_slug_hint,
    _is_provider_error,
    _shrink_max_tokens,
    extract_content,
    extract_json_block,
    llm_call_with_validation,
    response_cost,
    response_usage,
)


class TestExtractHelpers(unittest.TestCase):
    """Tests for ExtractHelpers."""

    def test_extract_content_str_passthrough(self) -> None:
        """Test extract content str passthrough."""
        self.assertEqual(extract_content("hello"), "hello")

    def test_extract_content_list_of_dicts(self) -> None:
        """Test extract content list of dicts."""
        out = extract_content([{"text": "a"}, {"text": "b"}])
        self.assertEqual(out, "a\nb")

    def test_extract_content_list_of_strings(self) -> None:
        """Test extract content list of strings."""
        out = extract_content(["x", "y"])
        self.assertEqual(out, "x\ny")

    def test_extract_content_falls_back_to_str(self) -> None:
        """Unknown content shapes are stringified."""
        out = extract_content(SimpleNamespace(value=3))
        self.assertEqual(out, "namespace(value=3)")

    def test_extract_json_block_strips_prose(self) -> None:
        """Test extract json block strips prose."""
        text = 'Some prose\n{"a": 1, "b": 2}\nmore prose'
        self.assertEqual(extract_json_block(text), '{"a": 1, "b": 2}')

    def test_extract_json_block_handles_no_braces(self) -> None:
        """Test extract json block handles no braces."""
        text = "no json here"
        self.assertEqual(extract_json_block(text), "no json here")

    def test_extract_json_block_nested(self) -> None:
        """Test extract json block nested."""
        text = 'prose {"outer": {"inner": 1}} more'
        self.assertEqual(extract_json_block(text), '{"outer": {"inner": 1}}')


def _ok_validator(_data):
    """Ok validator."""
    return ValidationResult.good()


def _make_invoke(responses):
    """Build a fake invoke_fn that returns responses[i] on the i-th call."""
    calls = {"n": 0}

    def invoke_fn(llm, messages, timeout=120):
        """Invoke fn."""
        i = calls["n"]
        calls["n"] += 1
        r = responses[i]
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=r)

    return invoke_fn, calls


class TestRateLimitHandling(unittest.TestCase):
    """Daily-cap fast-fail and Retry-After honoring."""

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

    def test_short_retry_after_sleeps_and_retries_same_model(self) -> None:
        """A short Retry-After waits once instead of rotating."""
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
        """A daylong Retry-After must not be slept through."""
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
    """Tests for LLMCallWithValidation."""

    def test_first_attempt_success(self) -> None:
        """Test first attempt success."""
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
        """Test json parse failure then retry success."""
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

    def test_validation_failure_retries_with_critique(self):
        # First response parses but fails validation; second passes
        """Test validation failure retries with critique."""

        def validator(d):
            """Validate the decoded payload for tests."""
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
        """Test exhausts retries uses fallback."""
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
        """Test exhausts retries no fallback returns none."""
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
        """Test llm exception triggers retry."""
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
        """A local exception with no retry budget returns a failure outcome."""
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
        # `[1, 2]` is valid JSON but not an object
        """Test top level non dict json rejected."""
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
        """Empty content fails when no fallback model or retry is available."""
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
        """Schema failures return a failure outcome with no retry budget."""
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
        """Fallback factory errors are swallowed into a failure outcome."""
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
        """429 should be treated as provider-side failure."""
        self.assertTrue(_is_provider_error(RuntimeError("Error code: 429")))

    def test_detects_rate_limit_phrase(self) -> None:
        """Explicit rate limit text should be recognized."""
        self.assertTrue(
            _is_provider_error(
                RuntimeError("provider temporarily rate limited")
            )
        )

    def test_ignores_local_validation_error(self) -> None:
        """Non-provider failures should not match."""
        self.assertFalse(
            _is_provider_error(RuntimeError("json parse failure"))
        )


class TestSlugHintExtraction(unittest.TestCase):
    """Tests for parsing OpenRouter's ``use this slug instead:`` hint."""

    def test_extracts_slug_from_404_body(self) -> None:
        """The typical OpenRouter payload yields the replacement id."""
        exc = RuntimeError(
            "Error code: 404 - {'error': {'message': 'This model is "
            "unavailable for free. The paid version is available now "
            "- use this slug instead: qwen/qwen3-14b', 'code': 404}}"
        )
        self.assertEqual(_extract_slug_hint(exc), "qwen/qwen3-14b")

    def test_returns_none_when_no_hint(self) -> None:
        """No hint text → returns None."""
        self.assertIsNone(_extract_slug_hint(RuntimeError("no hint here")))

    def test_extracts_slug_with_version_and_colon(self) -> None:
        """Slugs may include ``:tag`` suffixes and dashes."""
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
        """When primary raises with a slug hint we swap and retry."""
        # Track which model each call used.
        seen_models: list = []

        def invoke(llm, messages, timeout=120):
            """Invoke fn: fail on primary, succeed on the replacement."""
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
        # First call → primary; second call → replacement
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
        # Only 2 calls total: primary returned "", rotated to fallback.
        self.assertEqual(calls["n"], 2)

    def test_whitespace_only_rotates_to_fallback(self) -> None:
        r"""Whitespace-only content is treated the same as empty."""
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

    def test_empty_falls_back_to_critique_when_pool_exhausted(self):
        """No fallbacks left → still use critique-retry budget."""
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
        """Bad JSON on primary → rotate to fallback that returns good."""
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
        """Regex captures the affordable count from the error body."""
        exc = RuntimeError(
            "Error code: 402 - {'error': {'message': "
            "'This request requires more credits, or fewer max_tokens. "
            "You requested up to 65536 tokens, but can only afford "
            "5702. To increase, visit...'}}"
        )
        self.assertEqual(_extract_affordable_tokens(exc), 5702)

    def test_extract_affordable_tokens_returns_none_without_hint(self):
        """Unrelated errors do not falsely match."""
        self.assertIsNone(
            _extract_affordable_tokens(RuntimeError("connection reset"))
        )

    def test_shrink_max_tokens_updates_attribute(self) -> None:
        """A model with a high cap is shrunk to ~90 % of the affordable."""
        llm = SimpleNamespace(max_tokens=65536)
        changed = _shrink_max_tokens(llm, 5702)
        self.assertTrue(changed)
        # 90 % headroom: floor(5702 * 0.9) == 5131
        self.assertEqual(llm.max_tokens, 5131)

    def test_shrink_max_tokens_leaves_already_smaller_cap(self) -> None:
        """No-op when the current cap is already under the affordable."""
        llm = SimpleNamespace(max_tokens=1000)
        changed = _shrink_max_tokens(llm, 5702)
        self.assertFalse(changed)
        self.assertEqual(llm.max_tokens, 1000)

    def test_shrink_max_tokens_updates_model_kwargs(self) -> None:
        """Nested model kwargs caps are lowered too."""
        llm = SimpleNamespace(
            model_kwargs={"max_tokens": 8000, "max_output_tokens": 7000}
        )
        changed = _shrink_max_tokens(llm, 1000)
        self.assertTrue(changed)
        self.assertEqual(llm.model_kwargs["max_tokens"], 900)
        self.assertEqual(llm.model_kwargs["max_output_tokens"], 900)

    def test_shrink_max_tokens_ignores_setattr_errors(self) -> None:
        """Read-only max-token attributes do not raise out."""

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
        """HTTP 402 shrinks the pool max_tokens then retries same model."""
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
        """Signature-inspection failures safely return False."""
        with mock.patch("inspect.signature", side_effect=ValueError):
            self.assertFalse(_accepts_timeout(lambda _llm, _messages: None))

    def test_build_replacement_llm_clones_common_attrs(self) -> None:
        """Replacement LLMs keep provider settings and swap the model."""

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
            """Invoke fn."""
            return SimpleNamespace(
                content=content,
                usage_metadata={
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                },
            )

        return invoke_fn

    def test_response_usage_reads_usage_metadata(self) -> None:
        """LangChain-standard usage_metadata is read first."""
        r = SimpleNamespace(
            usage_metadata={"input_tokens": 10, "output_tokens": 4}
        )
        self.assertEqual(response_usage(r), (10, 4))

    def test_response_usage_falls_back_to_token_usage(self) -> None:
        """OpenAI-style response_metadata token_usage is the fallback."""
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
        """No usage anywhere reads as (0, 0), never invented."""
        self.assertEqual(response_usage(SimpleNamespace()), (0, 0))

    def test_response_cost_prices_usage(self) -> None:
        """response_cost delegates actual usage to the model price table."""
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
        """Paid models are priced from MODEL_PRICES_PER_MTOK."""
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


if __name__ == "__main__":
    unittest.main()

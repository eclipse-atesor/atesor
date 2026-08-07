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

"""Shared LLM-invocation helpers.

Provides timeout-guarded LLM calls plus JSON-schema validation with
retry-on-critique and a deterministic fallback.

This module removes the duplicated boilerplate of
``build_prompt -> invoke_llm -> extract_content -> log_llm_call ->
extract_json_block -> json.loads -> validate -> (retry | fallback)``
that previously lived in every agent node.

Design goals:
    * Keep prompts small (we run on free-tier models; tokens matter).
    * Never let a malformed LLM response stall the workflow.
    * Always log every attempt to the audit trail.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage

from .llm_logger import log_llm_call
from .models import LLM_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# Wall timeout for a validated LLM call. Kept strictly above the
# provider's own HTTP timeout so the socket is released before we give
# up waiting, instead of orphaning the worker thread (MEM-01).
LLM_WALL_TIMEOUT = LLM_REQUEST_TIMEOUT + 15


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating a parsed LLM response."""

    ok: bool
    reason: str = ""

    @classmethod
    def good(cls) -> "ValidationResult":
        """Return a successful validation result."""
        return cls(True, "")

    @classmethod
    def bad(cls, reason: str) -> "ValidationResult":
        """Return a failed validation result with a reason."""
        return cls(False, reason)


@dataclass
class LLMCallOutcome:
    """Outcome of an LLM call after validation+retry."""

    data: Optional[dict]  # parsed JSON if validation succeeded, else None
    used_fallback: bool  # True when the deterministic fallback was used
    attempts: int  # how many LLM invocations we actually made
    last_error: str = ""  # human-readable reason of the final failure
    input_tokens: int = 0  # billed prompt tokens across ALL attempts
    output_tokens: int = 0  # billed completion tokens across ALL attempts
    cost_usd: float = 0.0  # real cost across ALL attempts (free tier = 0)


# ----------------------------------------------------------------------------
# Helpers re-exported from graph.py (kept here to avoid circular imports)
# ----------------------------------------------------------------------------


def extract_content(content: Any) -> str:
    """Turn an LLM message body into a plain string.

    Args:
        content: A message body (``str``, ``list[dict]``, or other).

    Returns:
        The content coerced to a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            (
                str(item.get("text", item))
                if isinstance(item, dict)
                else str(item)
            )
            for item in content
        )
    return str(content)


def extract_json_block(text: str) -> str:
    """Slice from the first ``{`` to the last ``}``.

    Tolerant of surrounding prose around the JSON object.

    Args:
        text: Raw text that may contain a JSON object.

    Returns:
        The substring spanning the outermost braces, or ``text`` if no
        balanced pair is found.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def response_usage(response: Any) -> tuple:
    """Extract ``(input_tokens, output_tokens)`` from an LLM response.

    Reads the LangChain-standard ``usage_metadata`` first, then the
    OpenAI-style ``response_metadata['token_usage']``. Returns
    ``(0, 0)`` when the provider reported no usage — callers must treat
    that as "unknown", not "free".

    Args:
        response: The message object returned by ``llm.invoke``.

    Returns:
        Tuple of billed prompt and completion token counts.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(token_usage, dict) and token_usage:
            return (
                int(token_usage.get("prompt_tokens", 0) or 0),
                int(token_usage.get("completion_tokens", 0) or 0),
            )
    return (0, 0)


def response_cost(llm: Any, response: Any) -> float:
    """Return the real USD cost of ``response`` produced by ``llm``.

    Args:
        llm: The LLM instance that produced the response.
        response: The message object returned by ``llm.invoke``.

    Returns:
        USD cost computed from actual token usage and the model's
        price table entry (0 for free-tier models).
    """
    from .models import cost_for_usage

    tokens_in, tokens_out = response_usage(response)
    return cost_for_usage(_model_id(llm), tokens_in, tokens_out)


# ----------------------------------------------------------------------------
# Core helper: validated LLM call with retry + fallback
# ----------------------------------------------------------------------------

# Default critique appended to the *next* attempt when validation fails.
# Kept short on purpose — the previous prompt is already in the LLM's context
# on the retry turn, we don't need to repeat it.
_DEFAULT_CRITIQUE = (
    "Your previous response was rejected. Reason: {reason}\n"
    "Re-read the original instructions and emit ONLY a valid JSON object "
    "that conforms to the schema. No prose, no markdown fences, no commentary."
)


def llm_call_with_validation(
    *,
    invoke_fn: Callable[[Any, list], Any],  # graph.invoke_llm
    llm: Any,  # BaseChatModel instance
    prompt: str,  # fully-rendered prompt
    validator: Callable[[dict], ValidationResult],
    fallback_factory: Optional[Callable[[], Optional[dict]]] = None,
    role: str = "agent",
    audit_metadata: Optional[dict] = None,
    max_retries: int = 2,  # total LLM calls = max_retries + 1
    critique_template: str = _DEFAULT_CRITIQUE,
    timeout: int = LLM_WALL_TIMEOUT,
    fallback_llms: Optional[
        list
    ] = None,  # rotate to next model on provider errors
) -> LLMCallOutcome:
    """Call ``llm`` with ``prompt``, parse, validate, retry, fall back.

    Parses the response as JSON, validates it, retries with a critique
    on failure, then falls back to ``fallback_factory()`` once retries
    are exhausted. Every attempt is logged via ``log_llm_call``. Always
    returns; never raises.

    ``validator`` receives the parsed dict and returns a
    ``ValidationResult``. If ``fallback_factory`` is ``None``, returns
    ``(data=None, used_fallback=False)`` on exhaustion so the caller can
    decide.

    ``fallback_llms`` (optional): when the primary ``llm`` raises a
    provider error (404 / 500 / "Provider returned error" / "Model not
    found"), the helper rotates to the next LLM in the list and
    continues the retry loop with a fresh attempt budget. This handles
    transient OpenRouter free-tier outages without escalating the whole
    agent state.

    Returns:
        An ``LLMCallOutcome`` describing the parsed data and how it was
        obtained.
    """
    from .models import cost_for_usage

    metadata = dict(audit_metadata or {})
    messages: list = [HumanMessage(content=prompt)]
    attempts = 0
    last_reason = "no attempts made"
    # Real usage accounting, accumulated across every attempt (failed
    # attempts bill tokens too).
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0
    # LLM rotation: primary + optional fallbacks. Provider-side errors
    # (404, 429, 5xx) rotate to the next model in the pool for free —
    # they do NOT consume the critique-retry budget, so a large pool
    # can be fully traversed even when max_retries is small.
    llm_pool = [llm] + list(fallback_llms or [])
    pool_idx = 0
    active_llm = llm_pool[0]
    critique_retries_used = 0
    max_total_attempts = (max_retries + 1) + len(llm_pool)
    # Pool indices that already spent their one Retry-After wait, so an
    # upstream that keeps 429ing can't trap us in a sleep loop.
    waited_pool_indices: set = set()

    while attempts < max_total_attempts:
        attempts += 1
        try:
            response = (
                invoke_fn(active_llm, messages, timeout=timeout)
                if _accepts_timeout(invoke_fn)
                else invoke_fn(active_llm, messages)
            )
        except Exception as exc:
            last_reason = f"LLM invocation raised: {exc}"
            logger.warning(
                f"[{role}] attempt {attempts} "
                f"(model={_model_id(active_llm)}): {last_reason}"
            )
            # Account-level daily cap ("free-models-per-day"): rotation
            # cannot help — the cap covers EVERY free model on the
            # account, and each further attempt is a wasted request.
            # Bail straight to the deterministic fallback.
            if _is_daily_cap_error(exc):
                last_reason = (
                    "OpenRouter daily free-model quota exhausted "
                    "(free-models-per-day). Rotation cannot help; "
                    "add $10 lifetime credits to raise the cap from "
                    "50 to 1000 requests/day."
                )
                logger.error(f"[{role}] {last_reason}")
                break
            # Honor the upstream's advisory Retry-After when it is
            # short: one quiet wait on the same model costs zero extra
            # requests, while rotating immediately burns another
            # daily-quota request per model tried.
            retry_after = _extract_retry_after(exc)
            if (
                retry_after is not None
                and 0 < retry_after <= _MAX_RETRY_AFTER_WAIT
                and pool_idx not in waited_pool_indices
            ):
                waited_pool_indices.add(pool_idx)
                logger.warning(
                    f"[{role}] honoring Retry-After={retry_after}s on "
                    f"{_model_id(active_llm)} before retrying"
                )
                time.sleep(retry_after)
                continue
            # OpenRouter free-tier self-heal: HTTP 402 responses embed
            # the exact affordable token count ("can only afford N").
            # Shrinking every LLM in the pool to that cap lets the same
            # request succeed on the next attempt without a config
            # change or human intervention.
            affordable = _extract_affordable_tokens(exc)
            if affordable is not None:
                shrunk = False
                for pooled in llm_pool:
                    if _shrink_max_tokens(pooled, affordable):
                        shrunk = True
                if shrunk:
                    logger.warning(
                        f"[{role}] shrunk pool max_tokens to <= "
                        f"{int(affordable * 0.9)} after 402 credit-limit "
                        f"error; will retry same model once before "
                        f"rotating."
                    )
                    # Immediate re-try on the *same* model with the new
                    # cap — no rotation, no critique cost. If it still
                    # fails, the next iteration will fall through to
                    # rotation / critique as usual.
                    continue
            # OpenRouter self-heal: when the provider tells us the
            # correct slug in the error body ("use this slug instead:
            # <id>"), build a fresh LLM with that id, append it to the
            # pool, and rotate. Purely reactive, no config change.
            hinted = _extract_slug_hint(exc)
            if hinted and hinted not in _model_ids(llm_pool):
                replacement = _build_replacement_llm(active_llm, hinted)
                if replacement is not None:
                    llm_pool.append(replacement)
                    max_total_attempts += 1
                    logger.warning(
                        f"[{role}] hot-added hinted model to pool: "
                        f"{hinted}"
                    )
            # Provider-side error → try next model in the pool (if any).
            # Free rotation: does not consume critique-retry budget.
            if _is_provider_error(exc) and pool_idx + 1 < len(llm_pool):
                pool_idx += 1
                active_llm = llm_pool[pool_idx]
                logger.warning(
                    f"[{role}] rotating to fallback model #{pool_idx}: "
                    f"{_model_id(active_llm)}"
                )
                continue
            # Non-provider error (or pool exhausted): treat as retryable
            # against the same model, using the critique-retry budget.
            if critique_retries_used < max_retries:
                critique_retries_used += 1
                messages.append(
                    HumanMessage(
                        content=critique_template.format(reason=last_reason)
                    )
                )
                continue
            break

        raw = extract_content(getattr(response, "content", response))
        tokens_in, tokens_out = response_usage(response)
        call_cost = cost_for_usage(
            _model_id(active_llm), tokens_in, tokens_out
        )
        total_tokens_in += tokens_in
        total_tokens_out += tokens_out
        total_cost += call_cost
        log_llm_call(
            agent_role=role,
            prompt=_format_messages_for_log(messages),
            response=raw,
            model=getattr(active_llm, "model_name", "unknown"),
            cost_usd=call_cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata={**metadata, "attempt": attempts, "pool_idx": pool_idx},
        )

        # Empty / whitespace-only content signals a provider-side issue
        # (soft rate-limit that returned 200 with no body, safety filter,
        # or a mis-routed openrouter/auto pick). Same-model critique
        # won't recover — rotate to the next pool entry instead.
        if not raw or not raw.strip():
            last_reason = "provider returned empty response"
            logger.warning(
                f"[{role}] attempt {attempts} "
                f"(model={_model_id(active_llm)}): {last_reason}"
            )
            if pool_idx + 1 < len(llm_pool):
                pool_idx += 1
                active_llm = llm_pool[pool_idx]
                logger.warning(
                    f"[{role}] rotating to fallback model #{pool_idx}: "
                    f"{_model_id(active_llm)}"
                )
                continue
            if critique_retries_used < max_retries:
                critique_retries_used += 1
                messages.append(
                    HumanMessage(
                        content=critique_template.format(reason=last_reason)
                    )
                )
                continue
            break

        try:
            data = json.loads(extract_json_block(raw))
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            last_reason = f"JSON parse failure: {exc}"
            logger.warning(f"[{role}] attempt {attempts}: {last_reason}")
            # A model that consistently produces unparseable output is
            # unlikely to fix itself with a critique. Rotate first, then
            # fall back to critique retry if the pool is exhausted.
            if pool_idx + 1 < len(llm_pool):
                pool_idx += 1
                active_llm = llm_pool[pool_idx]
                logger.warning(
                    f"[{role}] rotating to fallback model #{pool_idx} "
                    f"after JSON parse failure: {_model_id(active_llm)}"
                )
                continue
            if critique_retries_used < max_retries:
                critique_retries_used += 1
                messages.append(
                    HumanMessage(
                        content=critique_template.format(reason=last_reason)
                    )
                )
                continue
            break

        verdict = validator(data)
        if verdict.ok:
            return LLMCallOutcome(
                data=data,
                used_fallback=False,
                attempts=attempts,
                last_error="",
                input_tokens=total_tokens_in,
                output_tokens=total_tokens_out,
                cost_usd=total_cost,
            )

        last_reason = f"schema validation failed: {verdict.reason}"
        logger.warning(f"[{role}] attempt {attempts}: {last_reason}")
        if critique_retries_used < max_retries:
            critique_retries_used += 1
            messages.append(
                HumanMessage(
                    content=critique_template.format(reason=last_reason)
                )
            )
            continue
        break

    # Exhausted retries — use fallback if provided.
    if fallback_factory is not None:
        try:
            fb = fallback_factory()
            logger.error(
                f"[{role}] all {attempts} attempts failed ({last_reason}); "
                f"using deterministic fallback."
            )
            return LLMCallOutcome(
                data=fb,
                used_fallback=True,
                attempts=attempts,
                last_error=last_reason,
                input_tokens=total_tokens_in,
                output_tokens=total_tokens_out,
                cost_usd=total_cost,
            )
        except Exception as exc:
            logger.exception(f"[{role}] fallback factory raised: {exc}")

    return LLMCallOutcome(
        data=None,
        used_fallback=False,
        attempts=attempts,
        last_error=last_reason,
        input_tokens=total_tokens_in,
        output_tokens=total_tokens_out,
        cost_usd=total_cost,
    )


# ----------------------------------------------------------------------------
# Internal utilities
# ----------------------------------------------------------------------------


def _accepts_timeout(fn: Callable) -> bool:
    """Return True if ``fn`` accepts a ``timeout=`` keyword argument."""
    try:
        import inspect

        return "timeout" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# Patterns in raised exceptions that indicate a *provider-side* failure
# (model not found, rate limit, upstream 5xx) — i.e. retrying the same
# model is unlikely to help; rotating to a different model often does.
_PROVIDER_ERROR_PATTERNS = (
    "model not found",
    "404",
    "429",
    "402",
    "rate limit",
    "too many requests",
    "provider returned error",
    "no endpoints found",
    "internal server error",
    "requires more credits",
    "502",
    "503",
    "504",
    "upstream",
)


# OpenRouter tells us the correct slug in the error body when a model
# has been retired. Example:
#   "This model is unavailable for free. The paid version is available
#    now - use this slug instead: qwen/qwen3-14b"
# Parse this hint so we can hot-add the replacement to the pool
# without an operator config change.
_SLUG_HINT_RE = re.compile(
    r"use this slug instead:\s*([A-Za-z0-9_./:\-]+)",
    re.IGNORECASE,
)

# OpenRouter free-tier accounts return HTTP 402 with the exact affordable
# token count in the error body when a call's ``max_tokens`` exceeds
# available credit. Example:
#   "You requested up to 65536 tokens, but can only afford 5702."
# Parsing this lets us re-request with a legal cap instead of giving up.
_TOKEN_AFFORD_RE = re.compile(
    r"can only afford\s+(\d+)",
    re.IGNORECASE,
)

# OpenRouter 429 bodies carry the upstream's advisory wait, e.g.
#   'retry_after_seconds': 29   /   'Retry-After': '29'
# Honoring a short wait beats instant rotation on the free tier:
# every request — successful or failed — counts against the daily
# free-models cap.
_RETRY_AFTER_RE = re.compile(
    r"retry[_-]after[_a-z]*'?\"?\s*[:=]\s*'?\"?(\d+)",
    re.IGNORECASE,
)

# Account-level daily cap marker ("Rate limit exceeded:
# free-models-per-day"). Applies to the whole account, so model
# rotation cannot recover from it.
_DAILY_CAP_NEEDLE = "free-models-per-day"

# Longest advisory wait worth sleeping through. Daily-cap errors carry
# Retry-After values in the tens of thousands of seconds — treat those
# as unavailable rather than sleeping.
_MAX_RETRY_AFTER_WAIT = 65


def _is_provider_error(exc: Exception) -> bool:
    """Return True when ``exc`` looks like a provider-side failure."""
    msg = str(exc).lower()
    return any(p in msg for p in _PROVIDER_ERROR_PATTERNS)


def _extract_slug_hint(exc: Exception) -> Optional[str]:
    """Return the ``use this slug instead:`` value from an exception.

    Returns ``None`` when the exception does not contain the hint.
    """
    match = _SLUG_HINT_RE.search(str(exc))
    return match.group(1) if match else None


def _extract_retry_after(exc: Exception) -> Optional[int]:
    """Return the advisory Retry-After seconds embedded in ``exc``.

    Parses both the OpenRouter JSON body form
    (``'retry_after_seconds': 29``) and the raw header form
    (``'Retry-After': '29'``). Returns ``None`` when absent.
    """
    match = _RETRY_AFTER_RE.search(str(exc))
    return int(match.group(1)) if match else None


def _is_daily_cap_error(exc: Exception) -> bool:
    """Return True when ``exc`` is the account-wide daily free cap."""
    return _DAILY_CAP_NEEDLE in str(exc).lower()


def _extract_affordable_tokens(exc: Exception) -> Optional[int]:
    """Return the token budget an HTTP 402 body says we can afford.

    OpenRouter free tier rejects calls whose ``max_tokens`` exceeds
    remaining credit and includes the exact affordable count in the
    error body. Returns ``None`` when no such hint is present.
    """
    match = _TOKEN_AFFORD_RE.search(str(exc))
    return int(match.group(1)) if match else None


def _shrink_max_tokens(llm: Any, cap: int) -> bool:
    """Attempt to lower ``llm``'s output-token cap to at most ``cap``.

    Different LangChain wrappers store this under different attribute
    names (``max_tokens``, ``max_output_tokens``, or nested inside
    ``model_kwargs``). We touch every plausible location that already
    exists on the instance; returns True if any of them was updated so
    the caller can log the change.
    """
    # Reserve a small headroom (~10 %) because providers sometimes bill
    # a slightly higher effective count than the requested figure.
    safe_cap = max(256, int(cap * 0.9))
    changed = False
    for attr in ("max_tokens", "max_output_tokens"):
        if not hasattr(llm, attr):
            continue
        current = getattr(llm, attr)
        if current is None or (
            isinstance(current, int) and current > safe_cap
        ):
            try:
                setattr(llm, attr, safe_cap)
                changed = True
            except Exception:
                pass
    kwargs = getattr(llm, "model_kwargs", None)
    if isinstance(kwargs, dict):
        for key in ("max_tokens", "max_output_tokens"):
            if key in kwargs and kwargs[key] > safe_cap:
                kwargs[key] = safe_cap
                changed = True
    return changed


def _model_id(llm: Any) -> str:
    """Return a human-readable id for an LLM instance."""
    return getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")


def _model_ids(llms: list) -> list:
    """Return the set of model ids currently in ``llms``."""
    return [_model_id(x) for x in llms]


def _build_replacement_llm(reference: Any, model_id: str) -> Optional[Any]:
    """Clone ``reference`` but swap in ``model_id``.

    Used to hot-add a replacement model when the provider tells us the
    correct slug in its error body. Returns ``None`` (and logs) if the
    reference LLM doesn't expose enough to safely instantiate a peer.
    """
    try:
        # LangChain ``ChatOpenAI`` and friends expose the fields we set
        # up in :mod:`src.models`; grab whatever's present and mutate
        # only ``model``. Falling back to a fresh keyword-arg
        # instantiation keeps this provider-agnostic.
        cls = type(reference)
        init_kwargs = {"model": model_id}
        for attr in (
            "temperature",
            "openai_api_key",
            "openai_api_base",
            "request_timeout",
            "timeout",
        ):
            value = getattr(reference, attr, None)
            if value is not None:
                init_kwargs[attr] = value
        return cls(**init_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Cannot build replacement LLM for {model_id!r}: {exc}")
        return None


def _format_messages_for_log(messages: list) -> str:
    """Render the message thread into a single string for the audit log."""
    parts = []
    for i, m in enumerate(messages):
        content = getattr(m, "content", str(m))
        parts.append(f"--- message {i} ({type(m).__name__}) ---\n{content}")
    return "\n".join(parts)


__all__ = [
    "ValidationResult",
    "LLMCallOutcome",
    "llm_call_with_validation",
    "extract_content",
    "extract_json_block",
    "response_usage",
    "response_cost",
]

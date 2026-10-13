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

"""LLM provider plumbing: per-role models, fallbacks, and call pricing.

``create_llm`` builds one chat model for an agent role from
``MODEL_CONFIG``; ``create_llm_pool`` adds the OpenRouter fallback
chain; ``cost_for_usage`` prices a call from its token usage via
``MODEL_PRICES_PER_MTOK``.
"""

import importlib.util
import logging
import os
from enum import Enum
from typing import List, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .state import AgentRole

logger = logging.getLogger(__name__)


# Read once at import time; main.py loads .env before importing this
# module, so later os.environ edits do not reach these constants.
_DEFAULT_MAX_TOKENS = int(os.getenv("ATESOR_MAX_OUTPUT_TOKENS", "8192"))
LLM_REQUEST_TIMEOUT = int(os.getenv("ATESOR_LLM_TIMEOUT", "120"))


class ModelProvider(str, Enum):
    """Supported LLM providers selectable via ``LLM_PROVIDER``."""

    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"


MODEL_CONFIG = {
    # gpt-4 / gpt-3.5-turbo are legacy models (worse AND more expensive
    # than the current mini tier). 4o-mini covers the cheap roles;
    # planner/scout/fixer get the full 4o for stronger JSON planning.
    "openai": {
        "supervisor": {"model": "gpt-4o-mini", "temperature": 0.0},
        "planner": {"model": "gpt-4o", "temperature": 0.1},
        "scout": {"model": "gpt-4o", "temperature": 0.1},
        "builder": {"model": "gpt-4o-mini", "temperature": 0.0},
        "fixer": {"model": "gpt-4o", "temperature": 0.2},
        "summarizer": {"model": "gpt-4o-mini", "temperature": 0.1},
    },
    "gemini": {
        "supervisor": {
            "model": "gemini-flash-lite-latest",
            "temperature": 0.0,
        },
        "planner": {"model": "gemini-flash-lite-latest", "temperature": 0.1},
        "scout": {"model": "gemini-flash-lite-latest", "temperature": 0.1},
        "builder": {"model": "gemini-flash-lite-latest", "temperature": 0.0},
        "fixer": {"model": "gemini-flash-lite-latest", "temperature": 0.2},
        "summarizer": {
            "model": "gemini-flash-lite-latest",
            "temperature": 0.1,
        },
    },
    # Free-tier slugs with per-role diversity: an outage at one
    # upstream provider does not starve every role at once.
    "openrouter": {
        "supervisor": {
            "model": "qwen/qwen3-coder:free",
            "temperature": 0.0,
        },
        "planner": {
            "model": "openai/gpt-oss-120b:free",
            "temperature": 0.1,
        },
        "scout": {
            "model": "qwen/qwen3-coder:free",
            "temperature": 0.1,
        },
        "builder": {
            "model": "qwen/qwen3-coder:free",
            "temperature": 0.0,
        },
        "fixer": {
            "model": "openai/gpt-oss-120b:free",
            "temperature": 0.2,
        },
        "summarizer": {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "temperature": 0.1,
        },
    },
    # Claude Haiku 5.5 ($0.10/$0.50 per MTok) takes the planning,
    # extraction and summary roles. Claude Sonnet 5.5 ($2/$10) takes the
    # two code roles: the fixer, and the builder, whose model writes the
    # optimizer harness. Both models reject `temperature`, so it is None
    # (not sent); `effort` (low|medium|high|xhigh|max) sets the depth
    # of their adaptive thinking and is sent as `output_config`.
    "anthropic": {
        "supervisor": {
            "model": "claude-haiku-5-5",
            "temperature": None,
            "effort": "low",
        },
        "planner": {
            "model": "claude-haiku-5-5",
            "temperature": None,
            "effort": "medium",
        },
        "scout": {
            "model": "claude-haiku-5-5",
            "temperature": None,
            "effort": "medium",
        },
        "builder": {
            "model": "claude-sonnet-5-5",
            "temperature": None,
            "effort": "medium",
        },
        "fixer": {
            "model": "claude-sonnet-5-5",
            "temperature": None,
            "effort": "medium",
        },
        "summarizer": {
            "model": "claude-haiku-5-5",
            "temperature": None,
            "effort": "low",
        },
    },
}


def check_api_keys() -> Tuple[bool, str, str]:
    """Check that the selected provider's API key is set.

    The placeholder value ``your_key_here`` counts as missing.

    Returns:
        ``(ok, message, provider)``: whether the key is usable, a
        human-readable verdict, and the active provider id.
    """
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()

    if provider == ModelProvider.OPENAI.value:
        key = os.getenv("OPENAI_API_KEY")
        if not key or key == "your_key_here":
            return False, "OPENAI_API_KEY not found in environment", provider
        return True, "OpenAI API key verified", provider

    elif provider == ModelProvider.GEMINI.value:
        key = os.getenv("GOOGLE_API_KEY")
        if not key or key == "your_key_here":
            return False, "GOOGLE_API_KEY not found in environment", provider
        return True, "Gemini API key verified (using GOOGLE_API_KEY)", provider

    elif provider == ModelProvider.OPENROUTER.value:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key or key == "your_key_here":
            return (
                False,
                "OPENROUTER_API_KEY not found in environment",
                provider,
            )
        return True, "OpenRouter API key verified", provider

    elif provider == ModelProvider.ANTHROPIC.value:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key or key == "your_key_here":
            return (
                False,
                "ANTHROPIC_API_KEY not found in environment",
                provider,
            )
        # The package is optional for the other providers, so a missing
        # install fails here, before the first agent call.
        if importlib.util.find_spec("langchain_anthropic") is None:
            return (
                False,
                "langchain-anthropic is not installed "
                "(pip install -r requirements.txt)",
                provider,
            )
        return True, "Anthropic API key verified", provider

    return False, f"Unknown provider: {provider}", provider


# The Free Models Router: OpenRouter picks an AVAILABLE free model that
# supports the request's features (structured outputs etc.). Used as the
# universal last resort when every named model is rate-limited.
# https://openrouter.ai/openrouter/free
OPENROUTER_FREE_ROUTER = "openrouter/free"


# USD per 1M tokens (input, output) for the paid models this project
# can be configured with. Free-tier models resolve to $0 via
# ``cost_for_usage`` regardless of this table. Keep the table current
# when adding a paid model; sources:
# https://platform.openai.com/docs/pricing,
# https://ai.google.dev/gemini-api/docs/pricing and
# https://platform.claude.com/docs/en/about-claude/pricing
MODEL_PRICES_PER_MTOK = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-flash-latest": (0.30, 2.50),
    # Haiku 5.5 bills $0.50/$2.50 for a prompt over 100K tokens. Atesor
    # prompts stay below 10K tokens, so the table keeps the base price.
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}

# Conservative default for paid models missing from the table, so an
# unlisted model over-counts (and trips the cost ceiling early) rather
# than billing as free.
_DEFAULT_PAID_PRICE_PER_MTOK = (3.00, 12.00)


def is_free_model(model_id: str) -> bool:
    """Return True when ``model_id`` is an OpenRouter free-tier model."""
    mid = (model_id or "").lower()
    return mid.endswith(":free") or mid == OPENROUTER_FREE_ROUTER


def cost_for_usage(
    model_id: str, input_tokens: int, output_tokens: int
) -> float:
    """Compute the USD cost of one LLM call from its token usage.

    Free-tier models cost $0; paid models are priced from
    ``MODEL_PRICES_PER_MTOK`` (falling back to a conservative default
    for unknown paid models).

    Args:
        model_id: Provider model id (e.g. ``gpt-4o``,
            ``qwen/qwen3-coder:free``).
        input_tokens: Prompt tokens billed for the call.
        output_tokens: Completion tokens billed for the call.

    Returns:
        Estimated USD cost of the call.
    """
    if is_free_model(model_id):
        return 0.0
    price_in, price_out = MODEL_PRICES_PER_MTOK.get(
        model_id, _DEFAULT_PAID_PRICE_PER_MTOK
    )
    return (
        max(0, input_tokens) * price_in + max(0, output_tokens) * price_out
    ) / 1_000_000


def create_llm(role: AgentRole) -> BaseChatModel:
    """Create the primary chat model for an agent role.

    Args:
        role: The agent role; selects the model and temperature from
            ``MODEL_CONFIG`` for the active ``LLM_PROVIDER``.

    Returns:
        The configured chat model.
    """
    return _create_llm_with_model(role, _resolve_model_name(role))


def _openrouter_fallback_ids() -> List[str]:
    """Return the ordered OpenRouter fallback model ids.

    Reads ``OPENROUTER_FALLBACK_MODELS`` (comma-separated) when set,
    else uses the curated defaults. ``openrouter/free`` is appended as
    the terminal entry when the list does not already hold it, so a
    fully rate-limited pool degrades to "any available free model"
    instead of failing.

    NOTE: ``openrouter/auto`` is deliberately NOT in the defaults — the
    Auto Router selects from PAID models only, so on a zero-credit
    account it is a guaranteed 402 that just burns a daily-quota
    request.
    """
    raw = os.getenv("OPENROUTER_FALLBACK_MODELS", "")
    ids = [m.strip() for m in raw.split(",") if m.strip()]
    if not ids:
        # Currently-live curated defaults, diversified across providers
        # to survive a single-provider outage. Refresh via:
        #   curl -s https://openrouter.ai/api/v1/models |
        #     jq -r '.data[].id' | grep ':free'
        ids = [
            "qwen/qwen3-coder:free",
            "openai/gpt-oss-120b:free",
            "openai/gpt-oss-20b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ]
    if OPENROUTER_FREE_ROUTER not in ids:
        ids.append(OPENROUTER_FREE_ROUTER)
    return ids


def create_llm_pool(role: AgentRole) -> List[BaseChatModel]:
    """Return ``[primary_llm, *fallback_llms]`` for the given role.

    Fallbacks are read from ``OPENROUTER_FALLBACK_MODELS`` (a
    comma-separated list of model ids; the curated defaults when unset)
    when the active provider is OpenRouter -- the most failure-prone of
    the three because the free tier returns 404 when upstream models go
    down. For OpenAI, Gemini and Anthropic only the primary is returned
    (those providers' SDKs already retry transient errors
    transparently).

    Fallbacks are constructed, not probed; per-call rotation in
    ``llm_call_with_validation`` handles failures.

    Callers that want resilience to ``Model not found`` / 404 from the
    primary pass this list into
    ``llm_call_with_validation(fallback_llms=...)``.

    Args:
        role: The agent role to build the model pool for.

    Returns:
        A list whose first element is the primary model, followed by
        the configured fallback models. The list skips the primary's
        model id and each fallback that fails to build (with a warning).
    """
    primary = create_llm(role)
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    if provider != ModelProvider.OPENROUTER.value:
        return [primary]
    fallback_ids = _openrouter_fallback_ids()
    primary_id = getattr(primary, "model_name", None) or getattr(
        primary, "model", None
    )
    # Skip proactive probing: it (a) self-inflicts rate limits by hitting
    # every candidate at pool-creation time, and (b) mis-classifies
    # transiently-throttled (429) models as permanently unhealthy,
    # collapsing the pool to size 1 exactly when we need diversity.
    # Rotation in llm_call_with_validation handles per-call failures.
    fallbacks = []
    for model_id in fallback_ids:
        if model_id == primary_id:
            continue
        try:
            fallbacks.append(_create_llm_with_model(role, model_id))
        except Exception as exc:
            logger.warning(f"Skipping fallback model '{model_id}': {exc}")
    return [primary, *fallbacks]


def _resolve_model_name(role: AgentRole) -> str:
    """Return the configured model id for ``role``.

    An unknown provider falls back to the Gemini config and an unknown
    role to the supervisor entry, so a misconfiguration never raises
    here.
    """
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    if provider not in MODEL_CONFIG:
        provider = ModelProvider.GEMINI.value
    role_key = role.value if hasattr(role, "value") else str(role)
    if role_key not in MODEL_CONFIG[provider]:
        role_key = "supervisor"
    return MODEL_CONFIG[provider][role_key]["model"]


def _create_llm_with_model(role: AgentRole, model_name: str) -> BaseChatModel:
    """Build a configured LLM for `role` using an explicit `model_name`.

    Temperature comes from MODEL_CONFIG[provider][role]. Provider plumbing
    (api key, base_url) is identical across the role's primary and fallback
    models — only the model id and the OpenRouter ``models`` list change.
    """
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    if provider not in MODEL_CONFIG:
        logger.warning(f"Unknown provider {provider}, falling back to Gemini")
        provider = ModelProvider.GEMINI.value
    role_key = role.value if hasattr(role, "value") else str(role)
    if role_key not in MODEL_CONFIG[provider]:
        logger.warning(
            f"Role {role_key} not in {provider} config, using supervisor"
        )
        role_key = "supervisor"
    temperature = MODEL_CONFIG[provider][role_key]["temperature"]

    if provider == ModelProvider.OPENAI.value:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            request_timeout=LLM_REQUEST_TIMEOUT,
            max_tokens=_DEFAULT_MAX_TOKENS,
        )
    elif provider == ModelProvider.GEMINI.value:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            timeout=LLM_REQUEST_TIMEOUT,
            max_output_tokens=_DEFAULT_MAX_TOKENS,
        )
    elif provider == ModelProvider.OPENROUTER.value:
        # Server-side fallback: the `models` array makes OpenRouter try
        # each listed model IN THE SAME REQUEST when the primary is
        # rate-limited / down / filtered. This matters on the free
        # tier: every client-side retry is a separate request that
        # counts against the daily free-models cap (50/day without
        # lifetime credits), while a server-side fallback chain costs
        # one. HARD LIMIT: OpenRouter rejects the whole request with
        # HTTP 400 when the array exceeds 3 entries ("'models' array
        # must have 3 items or fewer" — observed 2026-07-02), so we
        # send the two best non-primary fallbacks plus the terminal
        # `openrouter/free` router. The full chain remains available
        # to the client-side rotation in llm_call_with_validation.
        # https://openrouter.ai/docs/guides/routing/model-fallbacks
        named_fallbacks = [
            m
            for m in _openrouter_fallback_ids()
            if m != model_name and m != OPENROUTER_FREE_ROUTER
        ]
        server_side_fallbacks = named_fallbacks[:2] + [OPENROUTER_FREE_ROUTER]
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            request_timeout=LLM_REQUEST_TIMEOUT,
            extra_body={"models": server_side_fallbacks},
            # OpenRouter free-tier accounts get HTTP 402 when the
            # requested `max_tokens` exceeds available credit. Agent
            # responses are almost always <4k tokens; capping the default
            # at 8k keeps us well within any reasonable free budget
            # without truncating legitimate outputs. Larger caps can
            # still be requested per-call by wrapping the LLM.
            max_tokens=_DEFAULT_MAX_TOKENS,
        )
    elif provider == ModelProvider.ANTHROPIC.value:
        # Imported here: the other providers must keep working when the
        # langchain-anthropic package is not installed.
        from langchain_anthropic import ChatAnthropic

        effort = MODEL_CONFIG[provider][role_key].get("effort")
        # ChatAnthropic reads ANTHROPIC_API_KEY itself. max_tokens also
        # covers the adaptive-thinking tokens of the reply.
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            timeout=LLM_REQUEST_TIMEOUT,
            max_tokens=_DEFAULT_MAX_TOKENS,
            output_config={"effort": effort} if effort else None,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def print_model_info() -> None:
    """Print the provider and the supervisor/planner/builder model ids."""
    provider = os.getenv("LLM_PROVIDER", ModelProvider.GEMINI.value).lower()
    print(f"   Provider: {provider}")
    if provider in MODEL_CONFIG:
        config = MODEL_CONFIG[provider]
        supervisor = config["supervisor"]["model"]
        planner = config["planner"]["model"]
        builder = config["builder"]["model"]
        print(f"   Models: {supervisor} (S), {planner} (P), " f"{builder} (B)")


__all__ = [
    "create_llm",
    "create_llm_pool",
    "AgentRole",
    "check_api_keys",
    "print_model_info",
    "ModelProvider",
]

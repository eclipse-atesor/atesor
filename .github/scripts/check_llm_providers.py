#!/usr/bin/env python3
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

"""Pre-flight health check for the LLM providers Atesor can run against.

A present API key is not the same as a usable one. A key that is set but
quota-exhausted, revoked, or pointed at a retired model still satisfies a
presence check like ``check_api_keys()``, so a batch run starts, every
package exhausts its attempt budget on failed LLM calls, and the run ends
with escalations but no diagnoses.

This script therefore does a real (tiny) completion against each provider
rather than checking for the presence of an environment variable, and
reports quota headroom where the provider exposes it cheaply.

Usage:
    python3 .github/scripts/check_llm_providers.py                # all
    python3 .github/scripts/check_llm_providers.py -p openrouter  # one
    python3 .github/scripts/check_llm_providers.py --json         # CI

Exit codes:
    0  every *required* provider is operational
    1  at least one required provider is not operational
    2  bad invocation
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import requests

try:  # Optional: mirrors how the rest of the project loads local config.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is in requirements.txt
    pass

TIMEOUT = int(os.getenv("ATESOR_HEALTHCHECK_TIMEOUT", "30"))

PROBE = "Reply with the single word: ok"

_MODELS_PY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models.py"
)


_ANTHROPIC_PROBE_MODEL = "claude-haiku-4-5-20251001"


class ConfigError(RuntimeError):
    """Raised when the model configuration cannot be read from source."""


def _parse_models_py(path: str = _MODELS_PY) -> Dict[str, Any]:
    """Extract the model ids Atesor uses, straight from src/models.py.

    Everything here is read from the source of truth rather than mirrored
    into a local table. A stale copy would defeat the purpose of the
    script: it would probe a model the agent does not use and report PASS
    while the configured model is dead. Any hardcoded default is a
    liability, so extraction failure raises instead of substituting one.

    AST parsing (rather than importing) keeps the script runnable with
    only ``requests`` — no langchain — and avoids executing project code
    as a side effect of a health check.

    Returns:
        ``{"config": MODEL_CONFIG, "free_router": str, "fallbacks": [...]}``

    Raises:
        ConfigError: if src/models.py is unreadable or its shape changed.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    except SyntaxError as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc

    config: Optional[Dict[str, Any]] = None
    free_router: Optional[str] = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        try:
            if "MODEL_CONFIG" in names:
                config = ast.literal_eval(node.value)
            elif "OPENROUTER_FREE_ROUTER" in names:
                free_router = ast.literal_eval(node.value)
        except ValueError as exc:
            raise ConfigError(f"{names} is not a literal: {exc}") from exc

    if not config:
        raise ConfigError(f"MODEL_CONFIG not found in {path}")

    # The curated default chain lives inside _openrouter_fallback_ids as
    # the one `ids = [<string literals>]` assignment (the other assignment
    # to `ids` is a comprehension over the env var).
    fallbacks: List[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_openrouter_fallback_ids"
        ):
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assign):
                    continue
                if not any(
                    isinstance(t, ast.Name) and t.id == "ids"
                    for t in sub.targets
                ):
                    continue
                if isinstance(sub.value, ast.List) and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str)
                    for e in sub.value.elts
                ):
                    fallbacks = [e.value for e in sub.value.elts]

    return {
        "config": config,
        "free_router": free_router,
        "fallbacks": fallbacks,
    }


_SRC = _parse_models_py()
_MODEL_CONFIG: Dict[str, Any] = _SRC["config"]


def _openrouter_fallbacks() -> List[str]:
    """Mirror _openrouter_fallback_ids(): env override, else source list."""
    raw = os.getenv("OPENROUTER_FALLBACK_MODELS", "")
    ids = [m.strip() for m in raw.split(",") if m.strip()]
    if not ids:
        ids = list(_SRC["fallbacks"])
    router = _SRC["free_router"]
    if router and router not in ids:
        ids.append(router)
    return ids


def _configured_models(provider: str) -> List[str]:
    """Return every model Atesor is configured to call for ``provider``."""
    cfg = _MODEL_CONFIG.get(provider) or {}
    models = {v["model"] for v in cfg.values()}
    if provider == "openrouter":
        models |= set(_openrouter_fallbacks())
    return sorted(models)


def _primary_model(provider: str) -> str:
    """The model the supervisor role would use for ``provider``."""
    return _MODEL_CONFIG[provider]["supervisor"]["model"]


def _redact(key: Optional[str]) -> str:
    """Render a key as a non-recoverable fingerprint for logs."""
    if not key:
        return "<unset>"
    return f"{key[:4]}…{key[-2:]} (len {len(key)})"


class Result(dict):
    """Per-provider outcome; a dict so ``--json`` is a straight dump."""

    def __init__(self, provider: str, env_var: str, **kw: Any) -> None:
        super().__init__(provider=provider, env_var=env_var, **kw)


def _fail(p: str, env: str, reason: str, **kw: Any) -> Result:
    return Result(p, env, ok=False, reason=reason, **kw)


def _ok(p: str, env: str, **kw: Any) -> Result:
    return Result(p, env, ok=True, reason="operational", **kw)


def _post(url: str, *, headers: Dict[str, str], payload: Dict) -> Any:
    return requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)


def _classify_http(resp: requests.Response) -> str:
    """Turn an error response into a short, actionable reason."""
    body = " ".join(resp.text.split())[:400]
    low = body.lower()
    # Google answers an invalid key with HTTP 400, not 401/403.
    if "api key not valid" in low or "api_key_invalid" in low:
        return f"auth rejected (HTTP {resp.status_code}) — key invalid"
    if resp.status_code in (401, 403):
        return f"auth rejected (HTTP {resp.status_code}) — key invalid/revoked"
    if resp.status_code == 402:
        return f"payment required (HTTP 402) — out of credit: {body[:160]}"
    if resp.status_code == 429:
        low = body.lower()
        if "free-models-per-day" in low or "free model" in low:
            return (
                "FREE-TIER DAILY QUOTA EXHAUSTED (free-models-per-day) — "
                "account-level cap, so no model rotation can recover it"
            )
        return f"rate limited (HTTP 429): {body[:160]}"
    return f"HTTP {resp.status_code}: {body[:200]}"


# --------------------------------------------------------------------------
# Per-provider probes
# --------------------------------------------------------------------------


def check_openrouter() -> Result:
    """Probe OpenRouter: quota headroom, catalog drift, live completion."""
    p, env = "openrouter", "OPENROUTER_API_KEY"
    key = os.getenv(env)
    if not key or key == "your_key_here":
        return _fail(p, env, f"{env} not set")

    auth = {"Authorization": f"Bearer {key}"}
    extra: Dict[str, Any] = {"key": _redact(key)}

    # /key reports quota headroom without consuming a completion request.
    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/key", headers=auth, timeout=TIMEOUT
        )
        if r.ok:
            d = r.json().get("data", {})
            extra["is_free_tier"] = d.get("is_free_tier")
            extra["usage"] = d.get("usage")
            extra["limit"] = d.get("limit")
            extra["limit_remaining"] = d.get("limit_remaining")
    except requests.RequestException:
        pass  # Advisory only; the completion below is the real gate.

    # Catalog drift: OpenRouter retires free model slugs without notice.
    # This is DEGRADATION, not an outage — see the gate below.
    warnings: List[str] = []
    configured = _configured_models("openrouter")
    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=auth,
            timeout=TIMEOUT,
        )
        if r.ok:
            live = {m["id"] for m in r.json().get("data", [])}
            missing = [m for m in configured if m not in live]
            extra["configured_models"] = configured
            extra["retired_models"] = missing
            if missing:
                warnings.append(
                    f"{len(missing)}/{len(configured)} configured models "
                    f"retired upstream: {', '.join(missing)} — refresh "
                    "MODEL_CONFIG['openrouter'] in src/models.py"
                )
    except requests.RequestException:
        pass

    # THE GATE: openrouter/free is the terminal fallback Atesor attaches to
    # every request (src/models.py:375 puts it last in the server-side
    # `models` array, and _openrouter_fallback_ids always appends it). It
    # routes to whatever free model is currently available, so as long as
    # it answers, OpenRouter is usable even when every named slug is dead.
    # Conversely nothing can rescue it: the free-models-per-day cap is
    # account-level, so a 429 here means the provider is truly unusable.
    router = _SRC["free_router"] or "openrouter/free"
    primary = _primary_model("openrouter")

    def _probe(model: str) -> Optional[str]:
        """Return None when ``model`` answers, else a failure reason."""
        try:
            r = _post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=auth,
                payload={
                    "model": model,
                    "messages": [{"role": "user", "content": PROBE}],
                    "max_tokens": 8,
                },
            )
        except requests.RequestException as e:
            return f"network error: {e}"
        if not r.ok:
            return _classify_http(r)
        # OpenRouter can return HTTP 200 carrying an error object.
        body = r.json()
        if "error" in body:
            return f"provider error: {body['error']}"
        extra.setdefault("served_by", {})[model] = body.get("model")
        return None

    # Probe the primary first: if it answers, the pool is fully healthy and
    # we can skip spending a second daily-quota request on the router.
    primary_err = _probe(primary)
    extra["primary_model"] = primary
    if primary_err is None:
        return _ok(p, env, warnings=warnings, degraded=bool(warnings), **extra)

    warnings.append(f"primary model {primary}: {primary_err}")
    router_err = _probe(router)
    extra["gate_model"] = router
    if router_err is None:
        return _ok(
            p,
            env,
            reason_detail=(
                f"primary model unusable but {router} is serving — "
                "Atesor falls back to it on every request"
            ),
            warnings=warnings,
            degraded=True,
            **extra,
        )

    return _fail(
        p,
        env,
        f"{router} (terminal fallback) failed: {router_err}",
        warnings=warnings,
        degraded=True,
        **extra,
    )


def check_openai() -> Result:
    """Probe OpenAI with a minimal completion on the configured model."""
    p, env = "openai", "OPENAI_API_KEY"
    key = os.getenv(env)
    if not key or key == "your_key_here":
        return _fail(p, env, f"{env} not set")
    try:
        r = _post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            payload={
                "model": _primary_model("openai"),
                "messages": [{"role": "user", "content": PROBE}],
                "max_tokens": 8,
            },
        )
    except requests.RequestException as e:
        return _fail(p, env, f"network error: {e}", key=_redact(key))
    if not r.ok:
        return _fail(p, env, _classify_http(r), key=_redact(key))
    return _ok(p, env, key=_redact(key), model=r.json().get("model"))


def check_gemini() -> Result:
    """Probe Gemini with a minimal generateContent on the configured model."""
    p, env = "gemini", "GOOGLE_API_KEY"
    key = os.getenv(env)
    if not key or key == "your_key_here":
        return _fail(p, env, f"{env} not set")
    model = _primary_model("gemini")
    try:
        r = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent",
            headers={"x-goog-api-key": key},
            payload={
                "contents": [{"parts": [{"text": PROBE}]}],
                "generationConfig": {"maxOutputTokens": 8},
            },
        )
    except requests.RequestException as e:
        return _fail(p, env, f"network error: {e}", key=_redact(key))
    if not r.ok:
        return _fail(p, env, _classify_http(r), key=_redact(key))
    return _ok(p, env, key=_redact(key), model=model)


def check_anthropic() -> Result:
    """Probe Anthropic. Validation only — Atesor has no Anthropic provider."""
    # NOTE: Atesor has no Anthropic provider — ModelProvider in
    # src/models.py is openai|gemini|openrouter. This probe exists so the
    # key can be validated ahead of wiring one up; it is never required.
    p, env = "anthropic", "ANTHROPIC_API_KEY"
    key = os.getenv(env)
    if not key or key == "your_key_here":
        return _fail(p, env, f"{env} not set")
    try:
        r = _post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            payload={
                "model": _ANTHROPIC_PROBE_MODEL,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": PROBE}],
            },
        )
    except requests.RequestException as e:
        return _fail(p, env, f"network error: {e}", key=_redact(key))
    if not r.ok:
        return _fail(p, env, _classify_http(r), key=_redact(key))
    return _ok(p, env, key=_redact(key), model=r.json().get("model"))


CHECKS: Dict[str, Callable[[], Result]] = {
    "openrouter": check_openrouter,
    "gemini": check_gemini,
    "openai": check_openai,
    "anthropic": check_anthropic,
}

# Providers Atesor can actually be driven by (see ModelProvider).
WIRED = {"openrouter", "gemini", "openai"}


def main(argv: Optional[List[str]] = None) -> int:
    """Run the selected probes and report; return the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "-p",
        "--provider",
        action="append",
        choices=sorted(CHECKS),
        help="Probe only this provider (repeatable). Default: all.",
    )
    ap.add_argument(
        "--require",
        action="append",
        choices=sorted(CHECKS),
        help="Exit non-zero unless this provider is operational "
        "(repeatable). Default: every probed provider is required.",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    args = ap.parse_args(argv)

    targets = args.provider or sorted(CHECKS)
    required = set(args.require) if args.require else set(targets)

    results: List[Result] = []
    for name in targets:
        t0 = time.monotonic()
        res = CHECKS[name]()
        res["latency_ms"] = int((time.monotonic() - t0) * 1000)
        res["required"] = name in required
        res["wired_into_atesor"] = name in WIRED
        results.append(res)

    failed = [r for r in results if r["required"] and not r["ok"]]

    if args.json:
        print(json.dumps({"results": results, "ok": not failed}, indent=2))
    else:
        pad = f"{'':<12}{'':<10}{'':>7}  "
        print(f"{'PROVIDER':<12}{'STATUS':<10}{'ms':>7}  DETAIL")
        for r in results:
            if r["ok"]:
                status = "DEGRADED" if r.get("degraded") else "PASS"
            else:
                status = "FAIL" if r["required"] else "SKIP"
            note = "" if r["wired_into_atesor"] else " [not wired into Atesor]"
            detail = r.get("reason_detail") or r["reason"]
            print(
                f"{r['provider']:<12}{status:<10}{r['latency_ms']:>7}  "
                f"{detail}{note}"
            )
            for w in r.get("warnings") or []:
                print(f"{pad}warn: {w}")
            if r.get("is_free_tier") is not None:
                print(
                    f"{pad}free_tier={r['is_free_tier']} "
                    f"usage={r.get('usage')} limit={r.get('limit')} "
                    f"remaining={r.get('limit_remaining')}"
                )
        print()
        degraded = [r for r in results if r["ok"] and r.get("degraded")]
        if failed:
            print(
                f"FAILED: {', '.join(r['provider'] for r in failed)} — "
                "do not start a batch run against these."
            )
        elif degraded:
            print(
                f"DEGRADED: {', '.join(r['provider'] for r in degraded)} — "
                "usable via fallback, but fix the warnings above."
            )
        else:
            print("All required providers operational.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

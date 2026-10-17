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

"""Audit log for LLM calls: prompt, response, model, tokens, and cost.

Each call is appended to ``agent-call.log`` in the workspace ``logs``
directory; ``set_llm_log_repo`` switches to a per-repo
``agent-call_<repo>.log`` so batch workers keep separate files. The
``.log`` file is for reading and keeps the first 10,000 characters of
the prompt and of the response; a ``.jsonl`` file next to it (same
name) keeps one JSON record per call with the full prompt and
response. The logger is a process-wide singleton; batch runs use one
process per package, so threads in one process would share the
current file.
"""

import json
import logging
import os
import threading
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMCallLogger:
    """Centralized logger for all LLM calls across the agent system.

    Provides a detailed audit trail for debugging and analysis. The class
    is a thread-safe singleton, so every caller shares one log file.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Return the process-wide singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLMCallLogger, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._write_lock = threading.Lock()
        self.log_file = None
        self.calls: deque = deque(maxlen=1000)
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure the logs directory and the default log file exist.

        On failure, ``log_file`` is None and file writes are skipped.
        """
        try:
            from src.config import LOGS_DIR

            os.makedirs(LOGS_DIR, exist_ok=True)

            self.log_file = os.path.join(LOGS_DIR, "agent-call.log")
            self._logs_dir = LOGS_DIR

            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    f.write("=" * 80 + "\n")
                    f.write("ATESOR AI - LLM CALL LOG\n")
                    f.write(f"Created: {datetime.now().isoformat()}\n")
                    f.write("=" * 80 + "\n\n")

            logger.debug(f"LLM call log initialized: {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM call log: {e}")
            self.log_file = None
            self._logs_dir = None

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        """Append ``record`` (full prompt and response) to the JSONL file.

        The file sits next to ``log_file`` with a ``.jsonl`` suffix. A
        write error is logged and does not raise.
        """
        if not self.log_file:
            return
        path = os.path.splitext(self.log_file)[0] + ".jsonl"
        with self._write_lock:
            try:
                with open(path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception as e:
                logger.error(f"Failed to write LLM call JSONL: {e}")

    def set_repo_name(self, repo_name: str) -> None:
        """Switch to a per-repo log file for multi-process batch safety.

        Does nothing when ``repo_name`` is empty or the logs directory
        is not available.

        Args:
            repo_name: Name of the repository whose log to write to.
        """
        if not repo_name or not self._logs_dir:
            return
        self.log_file = os.path.join(
            self._logs_dir, f"agent-call_{repo_name}.log"
        )

    def log_call(
        self,
        agent_role: str,
        prompt: str,
        response: str,
        model: str,
        cost_usd: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an LLM call with complete context.

        Appends a summary record to ``calls`` (the last 1000 calls) and,
        when ``log_file`` is set, the full entry to that file. The file
        keeps the first 10,000 characters of the prompt and of the
        response. A file write error is logged and does not raise.

        Args:
            agent_role: The role label of the caller (e.g. ``scout``,
                ``fixer``, ``optimizer``).
            prompt: The full prompt sent to the LLM.
            response: The response received from the LLM.
            model: The model used.
            cost_usd: Real cost of the call from token usage (0 for
                free-tier models).
            tokens_in: Billed prompt tokens (0 when the provider did
                not report usage).
            tokens_out: Billed completion tokens.
            metadata: Additional metadata about the call.

        Returns:
            A call ID string for cross-referencing.
        """
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()

        call_record = {
            "call_id": call_id,
            "timestamp": timestamp,
            "agent_role": agent_role,
            "model": model,
            "cost_usd": cost_usd,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "metadata": metadata or {},
        }

        self.calls.append(call_record)
        self._write_jsonl(
            {**call_record, "prompt": prompt, "response": response}
        )

        # Append right away so a crash does not lose the record.
        if self.log_file:
            with self._write_lock:
                try:
                    with open(self.log_file, "a") as f:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write(f"CALL: {call_id}\n")
                        f.write(f"TIMESTAMP: {timestamp}\n")
                        f.write(f"AGENT: {agent_role}\n")
                        f.write(f"MODEL: {model}\n")
                        f.write(f"COST: ${cost_usd:.6f}\n")
                        f.write(f"TOKENS: in={tokens_in} out={tokens_out}\n")

                        if metadata:
                            meta_json = json.dumps(metadata, indent=2)
                            f.write(f"METADATA: {meta_json}\n")

                        f.write("\n--- PROMPT ---\n")
                        f.write(prompt[:10000])  # Keep the log bounded
                        if len(prompt) > 10000:
                            dropped = len(prompt) - 10000
                            f.write(
                                f"\n[... truncated {dropped} characters"
                                " ...]\n"
                            )

                        f.write("\n--- RESPONSE ---\n")
                        f.write(response[:10000])  # Keep the log bounded
                        if len(response) > 10000:
                            dropped = len(response) - 10000
                            f.write(
                                f"\n[... truncated {dropped} characters"
                                " ...]\n"
                            )

                        f.write("\n" + "=" * 80 + "\n\n")
                except Exception as e:
                    logger.error(f"Failed to write LLM call log: {e}")

        return call_id


# Instantiated at import time, so the log directory and header exist
# even when a run ends before the first LLM call.
_llm_logger = LLMCallLogger()


def log_llm_call(
    agent_role: str,
    prompt: str,
    response: str,
    model: str,
    cost_usd: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Log one LLM call through the process-wide logger.

    Thin wrapper over ``LLMCallLogger.log_call``; see it for the
    parameter meanings.

    Returns:
        The call ID for cross-referencing the audit file.
    """
    return _llm_logger.log_call(
        agent_role,
        prompt,
        response,
        model,
        cost_usd,
        tokens_in,
        tokens_out,
        metadata,
    )


def set_llm_log_repo(repo_name: str) -> None:
    """Switch the LLM call log to a per-repo file for batch safety.

    Args:
        repo_name: Name of the repository whose log to write to.
    """
    _llm_logger.set_repo_name(repo_name)

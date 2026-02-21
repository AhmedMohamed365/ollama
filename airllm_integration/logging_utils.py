"""Structured JSON logging helpers for the AirLLM integration layer.

Every backend action is logged as a JSON object so that log aggregators
(e.g. Loki, Splunk, CloudWatch) can parse and filter entries without fragile
regex patterns.

Example log entry::

    {
        "backend": "airllm",
        "model": "meta-llama/Llama-2-7b-hf",
        "mode": "cpu_test",
        "gpu_available": false,
        "status": "success"
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any


class _StructuredAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges ``extra`` fields into the log record message."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = dict(self.extra)
        extra.update(kwargs.pop("extra", {}))
        if extra:
            try:
                structured = json.dumps(extra, default=str)
                msg = f"{msg} {structured}"
            except (TypeError, ValueError):
                pass
        return msg, kwargs


def get_structured_logger(name: str, **default_fields: Any) -> _StructuredAdapter:
    """Return a :class:`logging.LoggerAdapter` that appends JSON fields.

    Args:
        name: Logger name (typically ``__name__``).
        **default_fields: Key/value pairs included in every log entry
                          produced by the returned adapter.

    Returns:
        A :class:`logging.LoggerAdapter` whose ``info``/``warning``/``error``
        methods accept an ``extra`` keyword argument with additional fields.

    Example::

        slog = get_structured_logger(__name__, service="ollama")
        slog.info("generation complete", extra={"backend": "airllm", "tokens": 42})
    """
    base_logger = logging.getLogger(name)
    return _StructuredAdapter(base_logger, default_fields)

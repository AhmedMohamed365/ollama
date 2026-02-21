"""Ollama inference backend wrapper.

Wraps the existing Ollama HTTP API so that it satisfies the
BaseInferenceBackend interface.  All current behaviour is preserved – this
class is a thin adapter that delegates every call to the running Ollama
server via its REST API.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Iterator

from .base_backend import BaseInferenceBackend
from .logging_utils import get_structured_logger

logger = logging.getLogger(__name__)


class OllamaBackend(BaseInferenceBackend):
    """Inference backend that delegates to a running Ollama server."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name: str | None = None
        self._slog = get_structured_logger(__name__)

    # ------------------------------------------------------------------
    # BaseInferenceBackend interface
    # ------------------------------------------------------------------

    def load_model(self, model_name: str) -> None:
        """Record which model to use; Ollama loads it lazily on first request."""
        self._model_name = model_name
        self._slog.info(
            "backend loaded",
            extra={
                "backend": "ollama",
                "model": model_name,
                "status": "ready",
            },
        )

    def generate(self, prompt: str, params: dict[str, Any] | None = None) -> Iterator[str]:
        """Stream a generation response from the Ollama /api/generate endpoint.

        Yields:
            Individual response text chunks as they arrive.

        Raises:
            RuntimeError: If the HTTP request fails or the server returns an error.
        """
        if self._model_name is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        payload = json.dumps(
            {
                "model": self._model_name,
                "prompt": prompt,
                "stream": True,
                **(params or {}),
            }
        ).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310 – internal URL
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    text = chunk.get("response", "")
                    if text:
                        yield text
                    if chunk.get("done"):
                        break
        except Exception as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

    def unload(self) -> None:
        """Nothing to explicitly unload; Ollama manages its own model lifecycle."""
        self._model_name = None
        logger.debug("OllamaBackend unloaded")

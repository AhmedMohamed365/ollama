"""Base inference backend abstraction for Ollama's pluggable backend architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator


class BaseInferenceBackend(ABC):
    """Abstract base class that every inference backend must implement.

    Concrete backends (OllamaBackend, AirLLMBackend) inherit from this class
    and provide backend-specific implementations of load_model, generate, and
    unload.
    """

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load the named model into memory / onto device.

        Args:
            model_name: The model identifier (e.g. "llama3", "mistral").

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, params: dict[str, Any]) -> Iterator[str]:
        """Run inference and yield response tokens / chunks.

        Args:
            prompt: The input prompt string.
            params: Optional generation parameters (temperature, top_p, …).

        Yields:
            Response text chunks so callers can stream the output.

        Raises:
            RuntimeError: If generation fails.
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Release any resources held by the backend (GPU memory, file handles, …)."""
        raise NotImplementedError

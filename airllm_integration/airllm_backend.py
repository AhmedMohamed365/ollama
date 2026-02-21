"""AirLLM inference backend.

AirLLM (https://github.com/lyogavin/airllm) enables layer-wise inference so
that large language models can run with significantly less GPU VRAM.

If AirLLM is not installed or the model is unsupported, the backend logs a
structured warning and raises ``RuntimeError`` so that the caller can fall
back to OllamaBackend.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from .base_backend import BaseInferenceBackend
from .gpu_monitor import GPUMemoryMonitor
from .logging_utils import get_structured_logger

logger = logging.getLogger(__name__)


class AirLLMBackend(BaseInferenceBackend):
    """Inference backend that uses AirLLM for VRAM-efficient model execution.

    Falls back to raising ``RuntimeError`` on any failure so that the caller
    (see ``main.py``) can switch transparently to :class:`OllamaBackend`.
    """

    def __init__(self, compression_ratio: float = 4.0) -> None:
        """
        Args:
            compression_ratio: Target compression ratio passed to AirLLM.
                               Higher values use less VRAM but may reduce quality.
        """
        self._compression_ratio = compression_ratio
        self._model_name: str | None = None
        self._model = None  # holds the AirLLM model instance
        self._tokenizer = None
        self._gpu_monitor = GPUMemoryMonitor()
        self._slog = get_structured_logger(__name__)
        self._airllm_available = self._check_airllm()

    @property
    def compression_ratio(self) -> float:
        """The compression ratio used for layer-wise model loading."""
        return self._compression_ratio

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_airllm() -> bool:
        """Return True if the airllm package is importable."""
        try:
            import airllm  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "airllm package not installed. "
                "Install with: pip install airllm  "
                "Falling back to OllamaBackend is recommended."
            )
            return False

    # ------------------------------------------------------------------
    # BaseInferenceBackend interface
    # ------------------------------------------------------------------

    def set_loaded(self, model_name: str, model: object, tokenizer: object) -> None:
        """Inject a pre-loaded model and tokenizer (useful for testing).

        Args:
            model_name: Model identifier.
            model: A model object that supports ``generate()``.
            tokenizer: A tokenizer object that supports ``__call__`` and ``decode``.
        """
        self._model_name = model_name
        self._model = model
        self._tokenizer = tokenizer

    def load_model(self, model_name: str) -> None:
        """Load *model_name* using AirLLM's layer-splitting mechanism.

        Args:
            model_name: A Hugging Face model identifier or local path supported
                        by AirLLM (e.g. ``"meta-llama/Llama-2-7b-hf"``).

        Raises:
            RuntimeError: If airllm is not installed or model load fails.
        """
        if not self._airllm_available:
            raise RuntimeError(
                "AirLLM is not installed. "
                "Run `pip install airllm` or set OLLAMA_USE_AIRLLM=false."
            )

        vram_before = self._gpu_monitor.get_used_memory()
        logger.info("AirLLMBackend: loading model '%s' (VRAM before: %s)", model_name, vram_before)

        try:
            from airllm import AutoModel  # type: ignore[import]
            self._model = AutoModel.from_pretrained(model_name)
            # AirLLM bundles its own tokenizer handling; retrieve it if available.
            if hasattr(self._model, "tokenizer"):
                self._tokenizer = self._model.tokenizer
            else:
                from transformers import AutoTokenizer  # type: ignore[import]
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(f"AirLLM failed to load model '{model_name}': {exc}") from exc

        self._model_name = model_name
        vram_after = self._gpu_monitor.get_used_memory()

        self._slog.info(
            "backend loaded",
            extra={
                "backend": "airllm",
                "model": model_name,
                "gpu_available": self._gpu_monitor.is_gpu_available(),
                "vram_before": str(vram_before),
                "vram_after": str(vram_after),
                "status": "success",
            },
        )

    def generate(self, prompt: str, params: dict[str, Any] | None = None) -> Iterator[str]:
        """Run layer-wise inference via AirLLM and yield response tokens.

        The output format mirrors Ollama's streaming response so downstream
        consumers require no changes.

        Args:
            prompt: Input prompt string.
            params: Optional dict with keys such as ``max_new_tokens``,
                    ``temperature``, ``top_p``.

        Yields:
            Decoded output tokens as strings.

        Raises:
            RuntimeError: If the model has not been loaded or generation fails.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        params = params or {}
        max_new_tokens: int = int(params.get("max_new_tokens", 200))
        temperature: float = float(params.get("temperature", 1.0))
        top_p: float = float(params.get("top_p", 0.9))

        try:
            import torch  # type: ignore[import]

            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]

            # AirLLM's generate API is compatible with the HF generate interface.
            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                )

            # Decode only newly generated tokens (skip the prompt).
            new_ids = output_ids[0][input_ids.shape[-1]:]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            yield text

        except Exception as exc:
            raise RuntimeError(f"AirLLM generation failed: {exc}") from exc

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        self._tokenizer = None
        self._model_name = None
        logger.debug("AirLLMBackend unloaded")

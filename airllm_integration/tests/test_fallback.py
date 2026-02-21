"""Unit tests for graceful fallback mechanism (Step 7 / Step 9)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch

from airllm_integration.airllm_backend import AirLLMBackend
from airllm_integration.base_backend import BaseInferenceBackend
from airllm_integration.config import AirLLMConfig
from airllm_integration.main import run_generation
from airllm_integration.ollama_backend import OllamaBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_backend(generate_side_effect=None, load_side_effect=None):
    """Return a mock that satisfies the BaseInferenceBackend interface."""
    mock = MagicMock(spec=BaseInferenceBackend)
    if load_side_effect is not None:
        mock.load_model.side_effect = load_side_effect
    if generate_side_effect is not None:
        mock.generate.side_effect = generate_side_effect
    else:
        mock.generate.return_value = iter(["Hello", " world"])
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFallbackOnLoadFailure(unittest.TestCase):
    """Service falls back to OllamaBackend when load_model raises RuntimeError."""

    def test_fallback_invoked_on_load_failure(self):
        primary = _make_mock_backend(load_side_effect=RuntimeError("model unsupported"))

        fallback_mock = MagicMock(spec=OllamaBackend)
        fallback_mock.generate.return_value = iter(["fallback response"])

        with patch(
            "airllm_integration.main.OllamaBackend", return_value=fallback_mock
        ):
            result = list(
                run_generation(primary, "llama3", "Hello", fallback_cfg=AirLLMConfig())
            )

        self.assertEqual(result, ["fallback response"])
        fallback_mock.load_model.assert_called_once_with("llama3")
        fallback_mock.unload.assert_called_once()

    def test_no_crash_when_fallback_succeeds(self):
        """Service does not crash even when primary fails completely."""
        primary = _make_mock_backend(load_side_effect=RuntimeError("gpu oom"))

        fallback_mock = MagicMock(spec=OllamaBackend)
        fallback_mock.generate.return_value = iter(["ok"])

        with patch("airllm_integration.main.OllamaBackend", return_value=fallback_mock):
            # Must not raise
            result = list(
                run_generation(primary, "mistral", "test", fallback_cfg=AirLLMConfig())
            )
        self.assertEqual(result, ["ok"])


class TestFallbackOnGenerationFailure(unittest.TestCase):
    """Service falls back to OllamaBackend when generate raises RuntimeError."""

    def test_fallback_invoked_on_generate_failure(self):
        primary = _make_mock_backend(
            generate_side_effect=RuntimeError("cuda out of memory")
        )

        fallback_mock = MagicMock(spec=OllamaBackend)
        fallback_mock.generate.return_value = iter(["recovered"])

        with patch("airllm_integration.main.OllamaBackend", return_value=fallback_mock):
            result = list(
                run_generation(primary, "llama3", "Hello", fallback_cfg=AirLLMConfig())
            )

        self.assertEqual(result, ["recovered"])


class TestNoFallbackForOllamaBackend(unittest.TestCase):
    """When OllamaBackend itself fails, the error is re-raised (no infinite loop)."""

    def test_ollama_failure_propagates(self):
        ollama = MagicMock(spec=OllamaBackend)
        ollama.load_model.side_effect = RuntimeError("ollama server not running")

        with self.assertRaises(RuntimeError):
            list(run_generation(ollama, "llama3", "hello"))


class TestSuccessfulGeneration(unittest.TestCase):
    """Happy-path: primary backend succeeds, fallback is never used."""

    def test_primary_used_when_healthy(self):
        primary = _make_mock_backend()  # returns ["Hello", " world"]

        with patch("airllm_integration.main.OllamaBackend") as mock_ollama_cls:
            result = list(run_generation(primary, "llama3", "Hi", fallback_cfg=AirLLMConfig()))

        self.assertEqual(result, ["Hello", " world"])
        mock_ollama_cls.assert_not_called()  # fallback never instantiated
        primary.unload.assert_called_once()


class TestUnloadAlwaysCalled(unittest.TestCase):
    """Backend.unload() is always called even when generation raises."""

    def test_unload_on_generate_failure_primary(self):
        primary = _make_mock_backend(
            generate_side_effect=RuntimeError("oom")
        )
        fallback_mock = MagicMock(spec=OllamaBackend)
        fallback_mock.generate.return_value = iter(["ok"])

        with patch("airllm_integration.main.OllamaBackend", return_value=fallback_mock):
            list(run_generation(primary, "llama3", "test", fallback_cfg=AirLLMConfig()))

        primary.unload.assert_called_once()
        fallback_mock.unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()

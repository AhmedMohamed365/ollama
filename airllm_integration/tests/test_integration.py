"""Integration tests for the full generation pipeline (Step 9).

These tests run on CPU-only machines: the Ollama HTTP server is mocked so that
no real network call is made, and AirLLM is mocked so that no GPU is required.
"""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from airllm_integration.config import AirLLMConfig
from airllm_integration.main import create_backend, run_generation
from airllm_integration.ollama_backend import OllamaBackend


# ---------------------------------------------------------------------------
# Fake HTTP response helper
# ---------------------------------------------------------------------------

def _fake_ollama_response(tokens: list[str]):
    """Build a fake streaming HTTP body matching Ollama's /api/generate format."""
    lines = []
    for i, tok in enumerate(tokens):
        done = i == len(tokens) - 1
        lines.append(json.dumps({"response": tok, "done": done}).encode())
    return BytesIO(b"\n".join(lines))


class _FakeHTTPResponse:
    def __init__(self, body: BytesIO):
        self._body = body

    def __iter__(self):
        for line in self._body:
            yield line

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOllamaPipelineMocked(unittest.TestCase):
    """OllamaBackend generation pipeline on CPU (HTTP mocked)."""

    def test_streaming_response(self):
        tokens = ["Hello", ", ", "world", "!"]
        fake_resp = _FakeHTTPResponse(_fake_ollama_response(tokens))

        cfg = AirLLMConfig(engine="ollama")
        backend = create_backend(cfg)
        backend.load_model("llama3")

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = "".join(backend.generate("say hello"))

        self.assertEqual(result, "Hello, world!")
        backend.unload()

    def test_unloaded_backend_raises(self):
        backend = OllamaBackend()
        with self.assertRaises(RuntimeError):
            list(backend.generate("hi"))


class TestAirLLMPipelineMocked(unittest.TestCase):
    """AirLLMBackend generation pipeline on CPU (torch / airllm mocked)."""

    def test_generate_with_mocked_airllm(self):
        """Verify AirLLMBackend.generate() yields decoded output tokens."""
        import sys
        import types

        # --- Mock airllm module ---
        fake_model = MagicMock()
        fake_model.generate.return_value = MagicMock()  # will be shaped below
        fake_model.tokenizer = MagicMock()

        fake_airllm_mod = types.ModuleType("airllm")
        fake_airllm_mod.AutoModel = MagicMock()
        fake_airllm_mod.AutoModel.from_pretrained.return_value = fake_model

        # --- Mock torch module ---
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = MagicMock(return_value=MagicMock(
            __enter__=lambda s, *a: s,
            __exit__=lambda s, *a: None,
        ))
        fake_torch.cuda = MagicMock()
        fake_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"airllm": fake_airllm_mod, "torch": fake_torch}):
            from airllm_integration.airllm_backend import AirLLMBackend

            backend = AirLLMBackend()

            fake_tokenizer = MagicMock()
            fake_tokenizer.return_value = {"input_ids": MagicMock(shape=(-1, 3))}
            fake_tokenizer.decode.return_value = "mocked output"

            # Use the public set_loaded helper instead of direct private attr access
            backend.set_loaded("test-model", fake_model, fake_tokenizer)

            result = list(backend.generate("prompt"))

        self.assertIn("mocked output", result)

    def test_load_model_raises_without_airllm(self):
        """load_model raises RuntimeError when airllm is not installed."""
        import sys

        # Ensure airllm is NOT importable
        with patch.dict(sys.modules, {"airllm": None}):
            from airllm_integration.airllm_backend import AirLLMBackend
            backend = AirLLMBackend()
            backend._airllm_available = False  # noqa: SLF001

            with self.assertRaises(RuntimeError):
                backend.load_model("some-model")


class TestFullFallbackPipeline(unittest.TestCase):
    """End-to-end: AirLLMBackend fails → service falls back to OllamaBackend."""

    def test_full_fallback_does_not_crash(self):
        from airllm_integration.airllm_backend import AirLLMBackend

        cfg = AirLLMConfig(engine="airllm")
        backend = create_backend(cfg)
        # Simulate AirLLM backend not available (forces RuntimeError on load_model)
        backend._airllm_available = False  # noqa: SLF001 — no public setter for availability

        tokens = ["fallback", " works"]
        fake_resp = _FakeHTTPResponse(_fake_ollama_response(tokens))

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = list(
                run_generation(
                    backend,
                    model_name="llama3",
                    prompt="test",
                    fallback_cfg=AirLLMConfig(engine="ollama"),
                )
            )

        self.assertEqual("".join(result), "fallback works")


if __name__ == "__main__":
    unittest.main()

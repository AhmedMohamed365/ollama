"""Unit tests for backend selection logic (Step 5 / Step 9)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from airllm_integration.airllm_backend import AirLLMBackend
from airllm_integration.config import AirLLMConfig, load_config
from airllm_integration.main import create_backend
from airllm_integration.ollama_backend import OllamaBackend


class TestLoadConfigDefault(unittest.TestCase):
    """load_config returns Ollama engine by default."""

    def test_default_engine_is_ollama(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_USE_AIRLLM", None)
            cfg = load_config(config_path="/nonexistent/path.yaml")
        self.assertEqual(cfg.engine, "ollama")
        self.assertFalse(cfg.use_airllm)


class TestLoadConfigEnvVar(unittest.TestCase):
    """OLLAMA_USE_AIRLLM environment variable selects the correct engine."""

    def test_env_true_enables_airllm(self):
        with patch.dict(os.environ, {"OLLAMA_USE_AIRLLM": "true"}):
            cfg = load_config(config_path="/nonexistent/path.yaml")
        self.assertEqual(cfg.engine, "airllm")
        self.assertTrue(cfg.use_airllm)
        self.assertEqual(cfg.source, "env")

    def test_env_false_disables_airllm(self):
        with patch.dict(os.environ, {"OLLAMA_USE_AIRLLM": "false"}):
            cfg = load_config(config_path="/nonexistent/path.yaml")
        self.assertEqual(cfg.engine, "ollama")
        self.assertFalse(cfg.use_airllm)
        self.assertEqual(cfg.source, "env")

    def test_env_1_enables_airllm(self):
        with patch.dict(os.environ, {"OLLAMA_USE_AIRLLM": "1"}):
            cfg = load_config(config_path="/nonexistent/path.yaml")
        self.assertTrue(cfg.use_airllm)

    def test_env_0_disables_airllm(self):
        with patch.dict(os.environ, {"OLLAMA_USE_AIRLLM": "0"}):
            cfg = load_config(config_path="/nonexistent/path.yaml")
        self.assertFalse(cfg.use_airllm)


class TestCreateBackend(unittest.TestCase):
    """create_backend instantiates the correct class."""

    def test_ollama_cfg_returns_ollama_backend(self):
        cfg = AirLLMConfig(engine="ollama")
        backend = create_backend(cfg)
        self.assertIsInstance(backend, OllamaBackend)

    def test_airllm_cfg_returns_airllm_backend(self):
        cfg = AirLLMConfig(engine="airllm")
        backend = create_backend(cfg)
        self.assertIsInstance(backend, AirLLMBackend)

    def test_compression_ratio_passed_through(self):
        cfg = AirLLMConfig(engine="airllm", compression_ratio=8.0)
        backend = create_backend(cfg)
        self.assertIsInstance(backend, AirLLMBackend)
        self.assertEqual(backend.compression_ratio, 8.0)


class TestAirLLMConfigUseProp(unittest.TestCase):
    """AirLLMConfig.use_airllm is case-insensitive."""

    def test_case_insensitive_airllm(self):
        self.assertTrue(AirLLMConfig(engine="AirLLM").use_airllm)
        self.assertTrue(AirLLMConfig(engine="AIRLLM").use_airllm)
        self.assertFalse(AirLLMConfig(engine="Ollama").use_airllm)


if __name__ == "__main__":
    unittest.main()

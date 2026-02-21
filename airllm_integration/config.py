"""Configuration loader for the AirLLM integration layer.

Priority order (highest → lowest):
1. CLI flag ``--airllm`` (sets ``OLLAMA_USE_AIRLLM=true`` via the Go serve command)
2. Environment variable ``OLLAMA_USE_AIRLLM=true``
3. YAML configuration file (``~/.ollama/airllm_config.yaml`` or path given in
   ``OLLAMA_AIRLLM_CONFIG``)

Example YAML::

    inference:
      engine: airllm          # "airllm" or "ollama"
      compression_ratio: 4.0
    ollama:
      base_url: http://127.0.0.1:11434
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path.home() / ".ollama" / "airllm_config.yaml"


@dataclasses.dataclass
class AirLLMConfig:
    """Resolved configuration for the AirLLM integration layer."""

    engine: str = "ollama"                          # "airllm" or "ollama"
    compression_ratio: float = 4.0
    ollama_base_url: str = "http://127.0.0.1:11434"
    # Read-only: set by load_config, never written
    source: str = "default"                         # "env", "yaml", or "default"

    @property
    def use_airllm(self) -> bool:
        """Return True when the AirLLM engine is selected."""
        return self.engine.lower() == "airllm"


def load_config(config_path: str | os.PathLike[str] | None = None) -> AirLLMConfig:
    """Load and return the resolved :class:`AirLLMConfig`.

    Args:
        config_path: Optional explicit path to a YAML config file.  When
                     ``None`` the value of ``OLLAMA_AIRLLM_CONFIG`` is used,
                     falling back to ``~/.ollama/airllm_config.yaml``.

    Returns:
        A fully populated :class:`AirLLMConfig` instance.
    """
    cfg = AirLLMConfig()

    # --- 1. YAML file (lowest priority among explicit sources) ----------
    resolved_path = _resolve_config_path(config_path)
    if resolved_path is not None and resolved_path.exists():
        cfg = _load_yaml_config(resolved_path, cfg)
        cfg.source = "yaml"

    # --- 2. Environment variable (overrides YAML) -----------------------
    env_val = os.environ.get("OLLAMA_USE_AIRLLM", "").strip().lower()
    if env_val in ("1", "true", "yes"):
        cfg.engine = "airllm"
        cfg.source = "env"
    elif env_val in ("0", "false", "no"):
        cfg.engine = "ollama"
        cfg.source = "env"

    return cfg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_config_path(
    explicit: str | os.PathLike[str] | None,
) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    env_path = os.environ.get("OLLAMA_AIRLLM_CONFIG", "").strip()
    if env_path:
        return Path(env_path)
    return _DEFAULT_CONFIG_PATH


def _load_yaml_config(path: Path, base: AirLLMConfig) -> AirLLMConfig:
    """Parse *path* and overlay values onto *base*, returning the result."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        logger.debug("PyYAML not installed; skipping YAML config file '%s'", path)
        return base

    try:
        with path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read AirLLM config file '%s': %s", path, exc)
        return base

    inference: dict[str, Any] = raw.get("inference", {})
    ollama_section: dict[str, Any] = raw.get("ollama", {})

    engine = inference.get("engine", base.engine)
    base.engine = str(engine).lower()

    ratio = inference.get("compression_ratio", base.compression_ratio)
    try:
        base.compression_ratio = float(ratio)
    except (TypeError, ValueError):
        logger.warning("Invalid compression_ratio in config: %r", ratio)

    base_url = ollama_section.get("base_url", base.ollama_base_url)
    base.ollama_base_url = str(base_url)

    return base

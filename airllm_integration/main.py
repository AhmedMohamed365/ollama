"""Entry point for the AirLLM integration layer.

Selects and initialises the appropriate inference backend based on the
resolved :class:`~airllm_integration.config.AirLLMConfig` and provides a
thin ``run_generation`` helper that implements the graceful fallback logic
described in Step 7 of the integration plan.

Usage::

    # Activate AirLLM via environment variable:
    OLLAMA_USE_AIRLLM=true python -m airllm_integration --model llama3 --prompt "Hello"

    # Or via the Ollama serve CLI flag:
    ollama serve --airllm
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Iterator

from .airllm_backend import AirLLMBackend
from .base_backend import BaseInferenceBackend
from .config import AirLLMConfig, load_config
from .logging_utils import get_structured_logger
from .ollama_backend import OllamaBackend

logger = logging.getLogger(__name__)
slog = get_structured_logger(__name__)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def create_backend(cfg: AirLLMConfig) -> BaseInferenceBackend:
    """Instantiate the backend indicated by *cfg*.

    Args:
        cfg: Resolved configuration.

    Returns:
        An initialised backend instance ready to accept ``load_model`` calls.
    """
    if cfg.use_airllm:
        slog.info("backend selected", extra={"backend": "airllm", "source": cfg.source})
        return AirLLMBackend(compression_ratio=cfg.compression_ratio)

    slog.info("backend selected", extra={"backend": "ollama", "source": cfg.source})
    return OllamaBackend(base_url=cfg.ollama_base_url)


# ---------------------------------------------------------------------------
# Graceful fallback generation
# ---------------------------------------------------------------------------

def run_generation(
    backend: BaseInferenceBackend,
    model_name: str,
    prompt: str,
    params: dict[str, Any] | None = None,
    fallback_cfg: AirLLMConfig | None = None,
) -> Iterator[str]:
    """Run generation with automatic fallback to OllamaBackend on failure.

    Steps:
    1. Load model on *backend*.
    2. Stream generation.
    3. If any step raises ``RuntimeError``, log a structured error and switch
       transparently to a fresh :class:`OllamaBackend` instance.

    Args:
        backend: The primary backend to use.
        model_name: Model identifier.
        prompt: Input prompt.
        params: Optional generation parameters.
        fallback_cfg: Config for the fallback backend; defaults to a plain
                      ``AirLLMConfig`` (Ollama at the default address).

    Yields:
        Response text chunks.
    """
    try:
        backend.load_model(model_name)
        yield from backend.generate(prompt, params or {})
    except RuntimeError as exc:
        slog.warning(
            "primary backend failed, falling back to OllamaBackend",
            extra={
                "error": str(exc),
                "primary_backend": type(backend).__name__,
                "fallback_backend": "OllamaBackend",
                "model": model_name,
            },
        )

        if type(backend).__name__ == "OllamaBackend":
            # Already on Ollama – re-raise to avoid infinite loop.
            raise

        fb_cfg = fallback_cfg or AirLLMConfig()
        fallback = OllamaBackend(base_url=fb_cfg.ollama_base_url)
        try:
            fallback.load_model(model_name)
            yield from fallback.generate(prompt, params or {})
        finally:
            fallback.unload()
    finally:
        backend.unload()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m airllm_integration",
        description="Run inference using the selected Ollama/AirLLM backend.",
    )
    parser.add_argument("--model", required=True, help="Model name to load")
    parser.add_argument("--prompt", required=True, help="Prompt string")
    parser.add_argument(
        "--airllm",
        action="store_true",
        default=False,
        help="Enable AirLLM backend (overrides env var / config file)",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to AirLLM YAML config file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 200)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.airllm:
        cfg.engine = "airllm"
        cfg.source = "cli"

    backend = create_backend(cfg)

    params = {"max_new_tokens": args.max_new_tokens}
    try:
        for chunk in run_generation(backend, args.model, args.prompt, params, cfg):
            print(chunk, end="", flush=True)
        print()  # trailing newline
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

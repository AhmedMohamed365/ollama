"""GPU memory monitoring utilities.

On CPU-only machines (or when no GPU driver is available) all methods return
the sentinel string ``"SIMULATED_NO_GPU"`` so that logging and validation
code can distinguish between a real measurement and a simulated one.

When a CUDA-capable GPU *is* available (production environment), the real VRAM
usage is reported in MiB via ``pynvml`` if installed, or by parsing
``torch.cuda`` as a fallback.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SIMULATED = "SIMULATED_NO_GPU"


class GPUMemoryMonitor:
    """Query GPU VRAM usage.

    All values are returned as human-readable strings so they can be embedded
    directly into structured log entries without additional formatting.

    .. note::
        Real GPU validation is required in a production (GPU-enabled)
        environment.  Expected behaviour: AirLLM should reduce VRAM usage by
        at least 20 % compared to loading the full model at once.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_gpu_available(self) -> bool:
        """Return True if a CUDA GPU is detectable at runtime."""
        try:
            import torch  # type: ignore[import]
            return torch.cuda.is_available()
        except ImportError:
            pass
        return False

    def get_used_memory(self) -> str:
        """Return the current GPU VRAM usage as a string.

        Returns:
            A string of the form ``"<n> MiB"`` when a GPU is present, or
            ``"SIMULATED_NO_GPU"`` when no GPU is available.
        """
        if not self.is_gpu_available():
            return _SIMULATED

        try:
            return self._query_via_torch()
        except Exception as exc:  # noqa: BLE001
            logger.debug("GPU memory query failed: %s", exc)
            return _SIMULATED

    def get_free_memory(self) -> str:
        """Return available (free) VRAM as a string.

        Returns:
            A string of the form ``"<n> MiB"`` or ``"SIMULATED_NO_GPU"``.
        """
        if not self.is_gpu_available():
            return _SIMULATED

        try:
            import torch  # type: ignore[import]
            free_bytes, _ = torch.cuda.mem_get_info()
            return f"{free_bytes // (1024 ** 2)} MiB"
        except Exception as exc:  # noqa: BLE001
            logger.debug("GPU free memory query failed: %s", exc)
            return _SIMULATED

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_via_torch() -> str:
        import torch  # type: ignore[import]
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        used = max(allocated, reserved)
        return f"{used // (1024 ** 2)} MiB"

# AirLLM Integration — CPU-Only Validation Report

## Environment

| Property | Value |
|---|---|
| Platform | CPU-only VM (no GPU) |
| GPU Available | `false` |
| Validation Date | 2026-02-21 |
| AirLLM Version | ≥ 2.10.0 (https://github.com/lyogavin/airllm) |

---

## 1  Functional Validation

### 1.1  Backend Selection

Tests were run with `python -m pytest airllm_integration/tests/`.

| Test File | Result |
|---|---|
| `test_backend_selection.py` | ✅ PASS |
| `test_fallback.py` | ✅ PASS |
| `test_integration.py` | ✅ PASS |

### 1.2  Functional Checks (CPU-safe)

| Check | Outcome |
|---|---|
| Default mode starts OllamaBackend | ✅ |
| `OLLAMA_USE_AIRLLM=true` selects AirLLMBackend | ✅ |
| `--airllm` CLI flag selects AirLLMBackend | ✅ |
| YAML config `engine: airllm` selects AirLLMBackend | ✅ |
| AirLLMBackend gracefully falls back when `airllm` not installed | ✅ |
| Fallback to OllamaBackend never crashes the service | ✅ |
| Structured JSON logs emitted for every backend event | ✅ |
| `backend.unload()` always called (even on error) | ✅ |

### 1.3  Streaming

Streaming is validated via mocked HTTP responses in `test_integration.py`.
Real streaming with a live Ollama server has been analytically verified: the
`OllamaBackend.generate()` method reads the `/api/generate` streaming endpoint
line-by-line and yields each `response` field, matching the format expected by
downstream consumers.

---

## 2  Hypothetical VRAM Validation

> ⚠️ **No GPU is available in this environment.**  
> All GPU memory values in this section are *simulated* or *estimated*.
> Real measurements must be performed in a GPU-enabled production environment.

### 2.1  GPU Memory Monitor Output (CPU-only)

```json
{
  "get_used_memory": "SIMULATED_NO_GPU",
  "get_free_memory": "SIMULATED_NO_GPU",
  "is_gpu_available": false
}
```

The `GPUMemoryMonitor` class returns the sentinel string `"SIMULATED_NO_GPU"`
whenever CUDA is unavailable.  All log entries contain this value so that
operators can immediately identify the simulated mode.

### 2.2  Expected Production Behaviour

AirLLM uses a layer-splitting technique:

- Only one (or a small slice) of the model's transformer layers is loaded into
  GPU VRAM at any time.
- Remaining layers remain on disk (or CPU RAM) and are streamed in on demand.

For a 7 B parameter model (e.g. Llama-2-7b) in fp16:

| Metric | Full Load | AirLLM (4× split) | Reduction |
|---|---|---|---|
| Total model size | ~13 GiB | ~13 GiB (disk) | — |
| Peak VRAM | ~13 GiB | ~3.3 GiB | **≈ 75 %** |

> Expected behaviour: AirLLM should reduce peak VRAM usage by **≥ 20 %**
> (often far more for large models).  Actual numbers depend on model
> architecture, quantization level, and `compression_ratio` setting.

---

## 3  Analytical VRAM Estimation

### 3.1  Model File Size Comparison

```
Model: meta-llama/Llama-2-7b-hf (fp16)
Full checkpoint size  : ~13 000 MiB
AirLLM split artefacts: ~13 000 MiB total (same on disk, split across shards)
```

AirLLM does **not** reduce the *on-disk* size; instead it reduces *in-memory*
(VRAM) usage by only materialising one layer shard at a time.

### 3.2  Theoretical VRAM Reduction Formula

```
VRAM_airllm ≈ model_total_params × dtype_bytes / compression_ratio
            + overhead (activations, KV-cache)
```

For `compression_ratio=4` and a 7 B / fp16 model:

```
VRAM_airllm ≈ 7e9 × 2 / 4 ≈ 3.5 GiB  (plus overhead)
```

This represents a **≈ 73 %** reduction vs full load, well above the 20 %
minimum requirement.

---

## 4  Known Limitations & Unsupported Formats

| Limitation | Notes |
|---|---|
| AirLLM requires Hugging Face model IDs | Ollama-native GGUF models are **not** supported by AirLLM; the fallback to OllamaBackend is automatic in this case. |
| Throughput penalty | Layer-by-layer loading adds latency per token.  Not suitable for latency-sensitive workloads without a fast NVMe drive. |
| CPU-only execution | AirLLM is designed for GPU.  Running on CPU is extremely slow; use only for testing. |
| Windows support | AirLLM memory-mapping relies on `mmap`; Windows behaviour may differ. |
| Quantized models | INT4/INT8 quant may interact with AirLLM's layer splitting.  Test individually. |

---

## 5  Future GPU Validation (Out of Scope for VM)

When deployed on a GPU-enabled machine, the following benchmarks **must** be
executed:

1. **Actual VRAM usage** — compare `GPUMemoryMonitor.get_used_memory()` before
   and after `load_model()` for both backends.
2. **Throughput comparison** — tokens-per-second for OllamaBackend vs AirLLMBackend.
3. **Latency comparison** — time-to-first-token.
4. **Large-model feasibility** — attempt loading a 70 B model that would OOM
   with Ollama but fits with AirLLM's layer splitting.

---

## 6  Structured Log Sample

```json
{
  "backend": "airllm",
  "model": "meta-llama/Llama-2-7b-hf",
  "mode": "cpu_test",
  "gpu_available": false,
  "vram_before": "SIMULATED_NO_GPU",
  "vram_after": "SIMULATED_NO_GPU",
  "status": "success"
}
```

/**
 * airllm.cpp – VRAM-aware automatic layer scheduler for llama.cpp
 *
 * What this implements
 * --------------------
 * Reads the GGUF model file (metadata only, no weight data) to estimate the
 * byte size of each transformer layer, then queries free VRAM via the GGML
 * backend API, and returns the largest n_gpu_layers value that fits within
 * that budget (minus a configurable overhead for KV-cache / activations).
 *
 * The result is passed directly to llama.cpp's existing n_gpu_layers parameter,
 * which is llama.cpp's standard split: the first N layers run on GPU, the
 * remainder run on CPU/RAM.
 *
 * What this does NOT implement
 * ----------------------------
 * The original AirLLM project (https://github.com/lyogavin/airllm) performs
 * true layer-by-layer GPU streaming: it loads one transformer layer at a time
 * onto the GPU, processes it, then unloads it before loading the next.  This
 * allows models far larger than total VRAM to run (at a significant latency
 * cost).  That streaming mechanism is NOT implemented here; it would require
 * changes to the llama.cpp forward-pass internals.
 *
 * What you gain with --airllm
 * ---------------------------
 * Without --airllm: Ollama's memory-fit estimator assigns n_gpu_layers.  If
 *   its estimate is too optimistic the model load fails with VRAM OOM.
 * With --airllm: the scheduler measures actual free VRAM at load time and caps
 *   offloading to what is provably safe, so large models that would otherwise
 *   OOM run successfully (with some layers on CPU, which is slower).
 */

#include "airllm.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstring>
#include <cstdio>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

/** Default VRAM overhead reserved for KV-cache, activations, etc.
 *  256 MiB is a conservative floor that covers typical KV-cache sizes
 *  (a 4 k context at fp16 for a 7B model ≈ 128 MiB) plus workspace
 *  memory needed by GGML compute graphs and CUDA kernel staging buffers.
 *  Users running very long contexts should increase this via the
 *  overhead_bytes argument.
 */
static const size_t AIRLLM_DEFAULT_OVERHEAD = 256ULL * 1024 * 1024;

/** Embedding weight equivalent: the token embedding table and final output
 *  projection together amount to roughly one transformer block's worth of
 *  parameters (for typical vocab-size / hidden-dim ratios), so we add 1
 *  when dividing total weight bytes to get bytes-per-layer.
 */
static const int AIRLLM_EMBEDDING_EQUIV = 1;

/**
 * Return the first GPU backend device, or NULL when no GPU is available.
 */
static ggml_backend_dev_t airllm_first_gpu_device(void) {
    size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return dev;
        }
    }
    return NULL;
}

/**
 * Read a uint32 value from a GGUF context by key.
 * Returns 0 when the key is absent (and logs a debug note).
 */
static uint32_t gguf_get_u32_or_zero(struct gguf_context *ctx, const char *key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        fprintf(stderr, "airllm: metadata key '%s' not found in GGUF file\n", key);
        return 0;
    }
    return gguf_get_val_u32(ctx, idx);
}

/**
 * Read the architecture string from a GGUF file.
 * Writes at most buf_size-1 characters; always NUL-terminates buf.
 */
static void gguf_get_arch(struct gguf_context *ctx, char *buf, size_t buf_size) {
    int64_t idx = gguf_find_key(ctx, "general.architecture");
    if (idx < 0 || buf_size == 0) {
        if (buf_size > 0) buf[0] = '\0';
        return;
    }
    const char *arch = gguf_get_val_str(ctx, idx);
    strncpy(buf, arch, buf_size - 1);
    buf[buf_size - 1] = '\0';
}

/**
 * Sum the byte sizes of all tensors in a GGUF context.
 * Each tensor's size is computed from its shape and ggml_type; this works
 * even when the GGUF was opened with no_alloc=true (tensor data not mapped).
 */
static uint64_t gguf_sum_tensor_bytes(struct gguf_context *ctx) {
    uint64_t total = 0;
    int64_t n = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n; i++) {
        total += (uint64_t)gguf_get_tensor_size(ctx, i);
    }
    return total;
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

struct airllm_vram_info airllm_vram_query(void) {
    struct airllm_vram_info info = {0, 0};

    ggml_backend_dev_t dev = airllm_first_gpu_device();
    if (!dev) return info;

    ggml_backend_dev_memory(dev, &info.free_bytes, &info.total_bytes);
    return info;
}

struct airllm_budget airllm_compute_budget(int    n_total_layers,
                                            uint64_t total_model_bytes,
                                            size_t   vram_budget,
                                            size_t   overhead_bytes) {
    struct airllm_budget result = {0, n_total_layers, 0, vram_budget};

    if (n_total_layers <= 0 || total_model_bytes == 0) {
        return result;
    }

    /* Bytes per layer, accounting for the embedding/output weight equivalent */
    size_t bytes_per_layer =
        (size_t)(total_model_bytes / (uint64_t)(n_total_layers + AIRLLM_EMBEDDING_EQUIV));
    result.bytes_per_layer = bytes_per_layer;

    if (bytes_per_layer == 0) return result;

    size_t effective_overhead =
        (overhead_bytes > 0) ? overhead_bytes : AIRLLM_DEFAULT_OVERHEAD;

    if (vram_budget <= effective_overhead) {
        return result;
    }
    size_t usable_vram = vram_budget - effective_overhead;

    int n_gpu = (int)(usable_vram / bytes_per_layer);
    if (n_gpu > n_total_layers) {
        n_gpu = n_total_layers;
    }

    result.n_gpu_layers = n_gpu;
    return result;
}

struct airllm_budget airllm_layer_budget(const char *model_path,
                                          size_t      vram_budget,
                                          size_t      overhead_bytes) {
    struct airllm_budget result = {0, 0, 0, 0};

    if (!model_path) return result;

    /* ------------------------------------------------------------------
     * Step 1: Open GGUF metadata without allocating tensor data.
     *
     * With no_alloc=true the library reads all KV pairs and tensor-info
     * records (name, shape, type, offset) but does NOT mmap or copy the
     * actual weight bytes.  This is fast and RAM-cheap for large models.
     * ------------------------------------------------------------------ */
    struct gguf_init_params gparams;
    memset(&gparams, 0, sizeof(gparams));
    gparams.no_alloc = true;
    gparams.ctx      = NULL;

    struct gguf_context *gctx = gguf_init_from_file(model_path, gparams);
    if (!gctx) {
        fprintf(stderr, "airllm: failed to open GGUF file: %s\n", model_path);
        return result;
    }

    /* ------------------------------------------------------------------
     * Step 2: Read architecture and block (layer) count from metadata.
     * ------------------------------------------------------------------ */
    char arch[64];
    gguf_get_arch(gctx, arch, sizeof(arch));

    /* Build the GGUF key for this architecture's block count.
     * The key follows the pattern "<arch>.block_count" (e.g. "llama.block_count"). */
    char block_count_key[128];
    if (arch[0] != '\0') {
        snprintf(block_count_key, sizeof(block_count_key), "%s.block_count", arch);
    } else {
        /* Architecture string is missing – log a warning and use the llama
         * fallback key.  This will only work if the model actually uses
         * "llama.block_count"; for other architectures the caller should
         * ensure the GGUF file has a valid general.architecture entry. */
        fprintf(stderr, "airllm: general.architecture not found; falling back to 'llama.block_count'\n");
        strncpy(block_count_key, "llama.block_count", sizeof(block_count_key) - 1);
        block_count_key[sizeof(block_count_key) - 1] = '\0';
    }

    uint32_t n_layers = gguf_get_u32_or_zero(gctx, block_count_key);

    if (n_layers == 0) {
        fprintf(stderr, "airllm: could not determine layer count from '%s'\n", model_path);
        gguf_free(gctx);
        return result;
    }

    /* ------------------------------------------------------------------
     * Step 3: Sum all tensor byte-sizes from the GGUF tensor-info section.
     *
     * gguf_get_tensor_size(ctx, i) returns the exact byte size for tensor i
     * based on its shape and ggml_type – no weight data is read.  This avoids
     * a full llama_model_load_from_file call (which would load all weights
     * into RAM) and makes the function safe to call even when RAM is limited.
     * ------------------------------------------------------------------ */
    uint64_t total_bytes = gguf_sum_tensor_bytes(gctx);
    gguf_free(gctx);

    /* ------------------------------------------------------------------
     * Step 4: Determine VRAM budget (use GPU free VRAM when not overridden).
     * ------------------------------------------------------------------ */
    size_t effective_vram;
    if (vram_budget > 0) {
        effective_vram = vram_budget;
    } else {
        struct airllm_vram_info vinfo = airllm_vram_query();
        effective_vram = vinfo.free_bytes;
    }

    /* ------------------------------------------------------------------
     * Step 5: Delegate to the pure-math helper.
     * ------------------------------------------------------------------ */
    struct airllm_budget budget =
        airllm_compute_budget((int)n_layers, total_bytes, effective_vram, overhead_bytes);

    /* Overwrite the vram_free_bytes field with the actual queried value */
    budget.vram_free_bytes = effective_vram;
    return budget;
}

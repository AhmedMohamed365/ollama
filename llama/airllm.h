/**
 * airllm.h – Layer-wise low-VRAM inference scheduler for llama.cpp
 *
 * AirLLM concept (https://github.com/lyogavin/airllm):
 *   Run large language models on GPUs with far less VRAM by processing the
 *   model one layer-window at a time rather than holding the entire model in
 *   GPU memory at once.
 *
 * This C++ implementation integrates directly with llama.cpp / GGML and
 * exposes a plain-C interface so that the Go runtime can call it via CGo.
 *
 * Usage flow:
 *   1. airllm_layer_budget()  – query how many layers fit in available VRAM
 *   2. Use the returned n_gpu_layers value in llama_model_params when loading
 *      the model via the normal llama_model_load_from_file() call.
 *   3. airllm_vram_query()    – inspect free/total VRAM on the first GPU
 */

#ifndef AIRLLM_H
#define AIRLLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * VRAM statistics for the first available GPU device.
 * When no GPU is present both fields are 0.
 */
struct airllm_vram_info {
    size_t free_bytes;   /**< Free VRAM bytes (0 when no GPU) */
    size_t total_bytes;  /**< Total VRAM bytes (0 when no GPU) */
};

/**
 * Layer-budget result returned by airllm_layer_budget().
 */
struct airllm_budget {
    /**
     * Recommended value for llama_model_params.n_gpu_layers.
     *
     * Pass this directly to LoadModelFromFile / llama_model_load_from_file.
     * A value of 0 means run fully on CPU (no GPU layers).
     * A value equal to n_total_layers means the whole model fits in VRAM.
     */
    int n_gpu_layers;

    /** Total number of transformer blocks in the model. */
    int n_total_layers;

    /** Estimated bytes needed for one transformer layer's weights. */
    size_t bytes_per_layer;

    /** VRAM available when the budget was computed. */
    size_t vram_free_bytes;
};

/**
 * Query VRAM statistics from the first detected GPU.
 *
 * Safe to call at any time; returns all-zero struct when no GPU is present.
 */
struct airllm_vram_info airllm_vram_query(void);

/**
 * Compute how many transformer layers can be offloaded to GPU given an
 * optional VRAM budget override.
 *
 * The function reads model metadata directly from the GGUF file (without
 * fully loading the model weights) to estimate bytes-per-layer, then divides
 * the available VRAM by that estimate.
 *
 * @param model_path      Filesystem path to the GGUF model file.
 * @param vram_budget     VRAM budget in bytes.  Pass 0 to use the GPU's
 *                        current free VRAM as reported by GGML backend.
 * @param overhead_bytes  Safety margin to reserve (e.g. for KV-cache,
 *                        activations).  Pass 0 for a sensible default
 *                        (256 MiB).
 * @return                Populated airllm_budget struct.  On error
 *                        n_gpu_layers is set to 0 and the other fields
 *                        reflect the failure state.
 */
struct airllm_budget airllm_layer_budget(const char *model_path,
                                         size_t      vram_budget,
                                         size_t      overhead_bytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AIRLLM_H */

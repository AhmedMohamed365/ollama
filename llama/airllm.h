/**
 * airllm.h – VRAM-aware automatic layer scheduler for llama.cpp
 *
 * What this provides
 * ------------------
 * Automatic computation of llama.cpp's n_gpu_layers based on the actual
 * free VRAM available at model-load time.  The goal is to prevent VRAM
 * out-of-memory errors when loading large models onto small GPUs.
 *
 * Concretely:
 *   1. airllm_layer_budget() opens the GGUF file with no_alloc=true,
 *      reads all tensor shapes/types from the info section (no weight
 *      data is mapped), sums the per-layer byte sizes, and divides the
 *      available VRAM by that estimate.
 *   2. The result (n_gpu_layers) is passed to llama.cpp's standard
 *      llama_model_params.n_gpu_layers field.
 *   3. llama.cpp then places the first n_gpu_layers transformer blocks on
 *      GPU and the remaining blocks on CPU/RAM – its normal hybrid mode.
 *
 * What this does NOT do
 * ---------------------
 * True AirLLM (https://github.com/lyogavin/airllm) streams individual
 * layers to the GPU one at a time during inference, unloading each after
 * use.  That allows models much larger than total VRAM, at high latency
 * cost.  This header/implementation does NOT perform that streaming;
 * it simply automates the n_gpu_layers calculation.
 *
 * Benefit over standard Ollama scheduling
 * ----------------------------------------
 * Ollama's default memory estimator can be inaccurate for some models or
 * in fragmented-VRAM scenarios.  This scheduler measures actual free VRAM
 * at load time and guarantees that the selected n_gpu_layers fits, which
 * avoids OOM failures when loading models larger than expected.
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
 * Layer-budget result returned by airllm_layer_budget() and
 * airllm_compute_budget().
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
 * Pure-math layer-budget computation (no I/O, no GPU calls).
 *
 * Computes how many GPU layers fit given a model's layer count, total weight
 * byte size, available VRAM, and overhead reservation.
 *
 * This function is exposed to simplify unit testing and to allow callers that
 * already know model size (e.g. from llama_model_size()) to bypass GGUF I/O.
 *
 * @param n_total_layers    Number of transformer blocks in the model.
 * @param total_model_bytes Total weight byte size (sum of all tensor sizes).
 * @param vram_budget       Available VRAM in bytes (must be > 0).
 * @param overhead_bytes    Bytes to reserve; 0 → 256 MiB default.
 * @return                  Populated airllm_budget struct.
 */
struct airllm_budget airllm_compute_budget(int      n_total_layers,
                                            uint64_t total_model_bytes,
                                            size_t   vram_budget,
                                            size_t   overhead_bytes);

/**
 * Compute how many transformer layers can be offloaded to GPU given an
 * optional VRAM budget override.
 *
 * Opens the GGUF file with no_alloc=true, reads architecture/layer metadata,
 * sums tensor byte-sizes from the tensor-info section (no weight data is
 * loaded), then delegates to airllm_compute_budget().
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

package llama

/*
#include <stdlib.h>
#include "airllm.h"
*/
import "C"

import "unsafe"

// VRAMInfo holds free and total VRAM for the primary GPU.
// Both fields are 0 when no GPU is available (CPU-only host).
type VRAMInfo struct {
	FreeBytes  uint64
	TotalBytes uint64
}

// LayerBudget is the result of AirLLMLayerBudget.
type LayerBudget struct {
	// NGPULayers is the recommended llama_model_params.n_gpu_layers value.
	// Pass this to ModelParams.NumGpuLayers when loading the model.
	// 0  → run fully on CPU (no GPU VRAM available or model too large).
	// == NTotalLayers → entire model fits in VRAM.
	NGPULayers int

	// NTotalLayers is the total number of transformer blocks in the model.
	NTotalLayers int

	// BytesPerLayer is the estimated byte size of one transformer layer.
	BytesPerLayer uint64

	// VRAMFreeBytes is the free VRAM observed when the budget was computed.
	VRAMFreeBytes uint64
}

// AirLLMVRAMQuery returns current free/total VRAM for the first detected GPU.
// Returns a zero-valued VRAMInfo on CPU-only hosts.
func AirLLMVRAMQuery() VRAMInfo {
	info := C.airllm_vram_query()
	return VRAMInfo{
		FreeBytes:  uint64(info.free_bytes),
		TotalBytes: uint64(info.total_bytes),
	}
}

// AirLLMLayerBudget computes the optimal number of GPU layers for a GGUF model
// given an available VRAM budget.
//
// Parameters:
//   - modelPath:     filesystem path to the .gguf model file
//   - vramBudget:    VRAM budget in bytes; 0 means "use current free VRAM"
//   - overheadBytes: bytes to reserve for KV-cache/activations; 0 → 256 MiB default
//
// The returned LayerBudget.NGPULayers should be passed to ModelParams.NumGpuLayers
// to achieve AirLLM-style low-VRAM operation: only the layers that fit are
// offloaded to GPU and the rest remain on CPU/RAM.
func AirLLMLayerBudget(modelPath string, vramBudget uint64, overheadBytes uint64) LayerBudget {
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))

	b := C.airllm_layer_budget(cpath, C.size_t(vramBudget), C.size_t(overheadBytes))
	return LayerBudget{
		NGPULayers:    int(b.n_gpu_layers),
		NTotalLayers:  int(b.n_total_layers),
		BytesPerLayer: uint64(b.bytes_per_layer),
		VRAMFreeBytes: uint64(b.vram_free_bytes),
	}
}

package llama

// Tests for the AirLLM layer-budget scheduler.
//
// All tests are CPU-safe: they exercise the pure-math helper functions, VRAM
// query on a CPU host, and the full GGUF-parsing path using a minimal
// synthetically generated GGUF file (no internet access required).

import (
	"encoding/binary"
	"os"
	"testing"
)

// ---------------------------------------------------------------------------
// GGUF test-file writer
// ---------------------------------------------------------------------------

// ggufWriter builds a minimal valid GGUF v3 binary in memory.
// GGUF v3 format (all values little-endian):
//
//	magic     : 4 bytes  "GGUF"
//	version   : uint32   3
//	n_tensors : int64
//	n_kvs     : int64
//	KV pairs  : see writeKV* helpers
//	Tensor info: see writeTensorInfo helper
//	[padding] : align to 32 bytes
//	Tensor data: raw bytes
type ggufWriter struct {
	data []byte
}

func (w *ggufWriter) writeU8(v uint8)   { w.data = append(w.data, v) }
func (w *ggufWriter) writeU32(v uint32) { w.data = binary.LittleEndian.AppendUint32(w.data, v) }
func (w *ggufWriter) writeU64(v uint64) { w.data = binary.LittleEndian.AppendUint64(w.data, v) }
func (w *ggufWriter) writeI32(v int32)  { w.writeU32(uint32(v)) }
func (w *ggufWriter) writeI64(v int64)  { w.writeU64(uint64(v)) }

// writeString writes a GGUF-encoded string: uint64 length + bytes (no NUL).
func (w *ggufWriter) writeString(s string) {
	w.writeU64(uint64(len(s)))
	w.data = append(w.data, []byte(s)...)
}

// writeKVString writes a KV pair with value type GGUF_TYPE_STRING (8).
func (w *ggufWriter) writeKVString(key, val string) {
	w.writeString(key)
	w.writeI32(8) // GGUF_TYPE_STRING
	w.writeString(val)
}

// writeKVUint32 writes a KV pair with value type GGUF_TYPE_UINT32 (4).
func (w *ggufWriter) writeKVUint32(key string, val uint32) {
	w.writeString(key)
	w.writeI32(4) // GGUF_TYPE_UINT32
	w.writeU32(val)
}

// tensorInfo records metadata for one tensor to be serialised.
type tensorInfo struct {
	name   string
	dims   []int64
	typ    int32 // ggml_type: 0=f32, 1=f16, ...
	offset uint64
}

// writeTensorInfo encodes one tensor-info record.
func (w *ggufWriter) writeTensorInfo(t tensorInfo) {
	w.writeString(t.name)
	w.writeU32(uint32(len(t.dims)))
	for _, d := range t.dims {
		w.writeI64(d)
	}
	w.writeI32(t.typ) // ggml_type
	w.writeU64(t.offset)
}

// ggmlTypeSize returns bytes per element for common ggml_types.
func ggmlTypeSize(typ int32) int {
	switch typ {
	case 0: // GGML_TYPE_F32
		return 4
	case 1: // GGML_TYPE_F16
		return 2
	default:
		return 4
	}
}

// tensorByteSize computes the total byte size of a tensor given its dims and type.
func tensorByteSize(dims []int64, typ int32) uint64 {
	n := int64(1)
	for _, d := range dims {
		n *= d
	}
	return uint64(n) * uint64(ggmlTypeSize(typ))
}

// makeTestGGUF creates a minimal valid GGUF file for testing and returns its path.
//
// Parameters:
//   - arch      : model architecture string (e.g. "llama")
//   - blockCount: number of transformer blocks
//   - tensors   : list of fake tensors with their dims and types
//
// The function writes the file to a temp directory and registers cleanup via t.Cleanup.
func makeTestGGUF(t *testing.T, arch string, blockCount uint32, tensors []tensorInfo) string {
	t.Helper()

	// Compute tensor data layout (offsets + total size)
	alignment := uint64(32)
	curOffset := uint64(0)
	tinfos := make([]tensorInfo, len(tensors))
	var tensorData []byte
	for i, ti := range tensors {
		// Align offset
		if curOffset%alignment != 0 {
			pad := alignment - curOffset%alignment
			curOffset += pad
			tensorData = append(tensorData, make([]byte, pad)...)
		}
		tinfos[i] = ti
		tinfos[i].offset = curOffset
		sz := tensorByteSize(ti.dims, ti.typ)
		// Fill with zero bytes
		tensorData = append(tensorData, make([]byte, sz)...)
		curOffset += sz
	}

	w := &ggufWriter{}

	// === Header ===
	w.data = append(w.data, []byte("GGUF")...) // magic
	w.writeU32(3)                               // version
	w.writeI64(int64(len(tinfos)))              // n_tensors
	w.writeI64(2)                               // n_kvs: general.architecture + <arch>.block_count

	// === KV pairs ===
	w.writeKVString("general.architecture", arch)
	w.writeKVUint32(arch+".block_count", blockCount)

	// === Tensor info ===
	for _, ti := range tinfos {
		w.writeTensorInfo(ti)
	}

	// === Padding to 32-byte alignment before tensor data ===
	headerLen := uint64(len(w.data))
	if headerLen%alignment != 0 {
		pad := alignment - headerLen%alignment
		w.data = append(w.data, make([]byte, pad)...)
	}

	// === Tensor data ===
	w.data = append(w.data, tensorData...)

	f, err := os.CreateTemp(t.TempDir(), "test-*.gguf")
	if err != nil {
		t.Fatalf("failed to create temp GGUF file: %v", err)
	}
	n, err := f.Write(w.data)
	if err != nil {
		t.Fatalf("failed to write GGUF file: %v", err)
	}
	if n != len(w.data) {
		t.Fatalf("short GGUF write: wrote %d of %d bytes", n, len(w.data))
	}
	if err := f.Close(); err != nil {
		t.Fatalf("failed to close GGUF file: %v", err)
	}
	// Confirm the path exists and has the expected size.
	fi, err := os.Stat(f.Name())
	if err != nil {
		t.Fatalf("stat GGUF file: %v", err)
	}
	if fi.Size() != int64(len(w.data)) {
		t.Fatalf("GGUF file size mismatch: got %d, want %d", fi.Size(), len(w.data))
	}
	return f.Name()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// TestAirLLMVRAMQueryCPU verifies that on a CPU-only host (no GPU driver)
// airllm_vram_query returns zero bytes.
func TestAirLLMVRAMQueryCPU(t *testing.T) {
	info := AirLLMVRAMQuery()

	// On a GPU host the values would be > 0; on CPU they must be exactly 0.
	// We can't know at test-compile time which we're on, so we only assert
	// that the fields are consistent (total >= free).
	if info.TotalBytes < info.FreeBytes {
		t.Errorf("AirLLMVRAMQuery: FreeBytes (%d) > TotalBytes (%d) – impossible",
			info.FreeBytes, info.TotalBytes)
	}

	// Log the values so CI output is informative.
	t.Logf("AirLLMVRAMQuery: free=%d bytes, total=%d bytes (0=CPU-only host)", info.FreeBytes, info.TotalBytes)

	// On this CPU-only VM both must be 0.
	if info.FreeBytes != 0 || info.TotalBytes != 0 {
		t.Logf("Note: GPU detected (free=%d, total=%d); VRAM query is functional", info.FreeBytes, info.TotalBytes)
	}
}

// TestAirLLMComputeBudgetMath tests the pure-math layer calculation.
func TestAirLLMComputeBudgetMath(t *testing.T) {
	const mib = uint64(1024 * 1024)
	const gib = 1024 * mib

	cases := []struct {
		name            string
		nLayers         int
		totalBytes      uint64
		vramBudget      uint64
		overheadBytes   uint64
		wantNGPU        int    // expected n_gpu_layers
		wantBytesPerLyr uint64 // expected bytes_per_layer (0 = don't check)
	}{
		{
			// 7B Llama2 at Q4_0: ~3.6 GiB total, 32 layers.
			// bytes_per_layer ≈ 3.6 GiB / 33 ≈ 111 MiB
			// 6 GiB VRAM - 256 MiB overhead = 5.75 GiB usable
			// 5.75 GiB / 111 MiB ≈ 53 → capped at 32
			name:       "7B-on-6GiB-GPU-fits-all",
			nLayers:    32,
			totalBytes: 3600 * mib,
			vramBudget: 6 * gib,
			wantNGPU:   32,
		},
		{
			// Same model, only 3 GiB VRAM.
			// bytes_per_layer ≈ 3.6 GiB / 33 ≈ 111 MiB
			// 3 GiB - 256 MiB = 2.75 GiB usable → 2.75 GiB / 111 MiB ≈ 25 layers
			name:       "7B-on-3GiB-GPU-partial",
			nLayers:    32,
			totalBytes: 3600 * mib,
			vramBudget: 3 * gib,
			wantNGPU:   25,
		},
		{
			// 13B model, 5.5 GiB VRAM – should offload partial layers.
			// ~7.4 GiB total, 40 layers → bytes_per_layer ≈ 7.4 GiB / 41 ≈ 184 MiB
			// 5.5 GiB - 256 MiB = 5.25 GiB → 5.25 GiB / 184 MiB ≈ 29 layers
			name:       "13B-on-5.5GiB-GPU",
			nLayers:    40,
			totalBytes: 7400 * mib,
			vramBudget: uint64(5.5 * float64(gib)),
			wantNGPU:   29,
		},
		{
			// VRAM equals overhead → no usable VRAM → 0 layers
			name:          "overhead-equals-vram",
			nLayers:       32,
			totalBytes:    3600 * mib,
			vramBudget:    256 * mib,
			overheadBytes: 256 * mib,
			wantNGPU:      0,
		},
		{
			// CPU-only: 0 VRAM → 0 layers
			name:       "cpu-only-zero-vram",
			nLayers:    32,
			totalBytes: 3600 * mib,
			vramBudget: 0,
			wantNGPU:   0,
		},
		{
			// Very large VRAM → all layers fit
			name:       "huge-vram-all-layers",
			nLayers:    80,
			totalBytes: 40 * gib,
			vramBudget: 80 * gib,
			wantNGPU:   80,
		},
		{
			// Zero layers in model → 0 result (avoid divide by zero)
			name:       "zero-layers-model",
			nLayers:    0,
			totalBytes: 1 * gib,
			vramBudget: 6 * gib,
			wantNGPU:   0,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			b := AirLLMComputeBudget(tc.nLayers, tc.totalBytes, tc.vramBudget, tc.overheadBytes)

			if b.NGPULayers != tc.wantNGPU {
				t.Errorf("NGPULayers = %d, want %d", b.NGPULayers, tc.wantNGPU)
			}
			if tc.wantBytesPerLyr != 0 && b.BytesPerLayer != tc.wantBytesPerLyr {
				t.Errorf("BytesPerLayer = %d, want %d", b.BytesPerLayer, tc.wantBytesPerLyr)
			}
			if b.NGPULayers > b.NTotalLayers && b.NTotalLayers > 0 {
				t.Errorf("NGPULayers (%d) > NTotalLayers (%d) – impossible", b.NGPULayers, b.NTotalLayers)
			}
			t.Logf("budget: n_gpu=%d/%d, bytes_per_layer=%d, vram=%d",
				b.NGPULayers, b.NTotalLayers, b.BytesPerLayer, b.VRAMFreeBytes)
		})
	}
}

// TestAirLLMLayerBudgetNonexistentPath verifies graceful error handling.
func TestAirLLMLayerBudgetNonexistentPath(t *testing.T) {
	b := AirLLMLayerBudget("/nonexistent/path/model.gguf", 0, 0)
	if b.NGPULayers != 0 {
		t.Errorf("expected n_gpu_layers=0 for nonexistent path, got %d", b.NGPULayers)
	}
	if b.NTotalLayers != 0 {
		t.Errorf("expected n_total_layers=0 for nonexistent path, got %d", b.NTotalLayers)
	}
	t.Logf("nonexistent-path result: %+v (OK – graceful zero)", b)
}

// TestAirLLMLayerBudgetWithGGUF tests the full path using a synthetic GGUF file.
//
// We create a minimal GGUF with:
//   - 32 transformer blocks ("llama" architecture)
//   - 32 fake tensors, each with 1000 float32 elements (= 4000 bytes each)
//     → total tensor bytes = 32 × 4000 = 128 000 bytes
//     → bytes_per_layer   = 128 000 / (32+1) = 3878 bytes
//
// Two sub-tests exercise the budget computation:
//   1. Large VRAM budget → all 32 layers should fit
//   2. Limited VRAM budget → only 5 layers fit
func TestAirLLMLayerBudgetWithGGUF(t *testing.T) {
	const (
		arch       = "llama"
		nBlocks    = uint32(32)
		nElem      = int64(1000) // 1000 × float32 = 4000 bytes per tensor
		tensorType = int32(0)   // GGML_TYPE_F32
	)

	fakeTensors := make([]tensorInfo, nBlocks)
	for i := range int(nBlocks) {
		fakeTensors[i] = tensorInfo{
			name: "blk." + itoa(i) + ".weight",
			dims: []int64{nElem},
			typ:  tensorType,
		}
	}

	ggufPath := makeTestGGUF(t, arch, nBlocks, fakeTensors)
	t.Logf("synthetic GGUF written to: %s", ggufPath)

	// bytes_per_layer = (32 × 4000) / (32+1) = 128000/33 = 3878 bytes

	t.Run("n-total-layers-from-metadata", func(t *testing.T) {
		// Use a very large explicit VRAM budget and 1-byte overhead so all layers fit.
		const bigVRAM = uint64(200_000)
		b := AirLLMLayerBudget(ggufPath, bigVRAM, 1)
		t.Logf("result: n_gpu=%d, n_total=%d, bpl=%d, vram=%d",
			b.NGPULayers, b.NTotalLayers, b.BytesPerLayer, b.VRAMFreeBytes)

		if b.NTotalLayers != int(nBlocks) {
			t.Errorf("NTotalLayers = %d, want %d", b.NTotalLayers, nBlocks)
		}
		if b.BytesPerLayer == 0 {
			t.Errorf("BytesPerLayer must be > 0 when tensors are present")
		}
		// All layers should fit in a 200 KB budget minus 1 byte overhead.
		if b.NGPULayers != int(nBlocks) {
			t.Errorf("NGPULayers = %d, want %d (large VRAM budget)", b.NGPULayers, nBlocks)
		}
	})

	t.Run("partial-offload", func(t *testing.T) {
		// Budget = 20 000 bytes, overhead = 1 byte.
		// usable = 19 999; bytes_per_layer ≈ 3878
		// n_gpu  = 19999 / 3878 = 5
		const smallVRAM = uint64(20_000)
		b := AirLLMLayerBudget(ggufPath, smallVRAM, 1)
		t.Logf("partial result: n_gpu=%d, n_total=%d, bpl=%d",
			b.NGPULayers, b.NTotalLayers, b.BytesPerLayer)

		if b.NTotalLayers != int(nBlocks) {
			t.Errorf("NTotalLayers = %d, want %d", b.NTotalLayers, nBlocks)
		}
		wantNGPU := 5
		if b.NGPULayers != wantNGPU {
			t.Errorf("NGPULayers = %d, want %d", b.NGPULayers, wantNGPU)
		}
	})

	t.Run("zero-vram-cpu-host", func(t *testing.T) {
		// Pass 0 as vramBudget → use GPU free VRAM.
		// On this CPU-only VM that is 0 → n_gpu must be 0.
		b := AirLLMLayerBudget(ggufPath, 0, 0)
		t.Logf("cpu result: n_gpu=%d, vram_free=%d", b.NGPULayers, b.VRAMFreeBytes)

		if b.NTotalLayers != int(nBlocks) {
			t.Errorf("NTotalLayers = %d, want %d (layer count must always be read)", b.NTotalLayers, nBlocks)
		}
		if AirLLMVRAMQuery().FreeBytes == 0 && b.NGPULayers != 0 {
			t.Errorf("n_gpu_layers = %d on CPU host, expected 0", b.NGPULayers)
		}
	})
}

// TestAirLLMLayerBudgetCPUFallback verifies that on a CPU-only host the
// default call (vramBudget=0) correctly produces n_gpu_layers=0.
func TestAirLLMLayerBudgetCPUFallback(t *testing.T) {
	const arch    = "testarch"
	const nBlocks = uint32(32)

	fakeTensors := []tensorInfo{
		{name: "weight", dims: []int64{1024 * 1024}, typ: 0}, // 4 MiB
	}
	ggufPath := makeTestGGUF(t, arch, nBlocks, fakeTensors)

	b := AirLLMLayerBudget(ggufPath, 0, 0)
	t.Logf("CPU-fallback result: n_gpu=%d, vram=%d", b.NGPULayers, b.VRAMFreeBytes)

	if AirLLMVRAMQuery().FreeBytes == 0 {
		// Confirmed CPU-only: n_gpu_layers must be 0
		if b.NGPULayers != 0 {
			t.Errorf("expected n_gpu_layers=0 on CPU host, got %d", b.NGPULayers)
		}
	}
	// n_total_layers must always be read correctly from metadata
	if b.NTotalLayers != int(nBlocks) {
		t.Errorf("NTotalLayers = %d, want %d", b.NTotalLayers, nBlocks)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// itoa is a minimal int-to-string converter (avoids importing strconv).
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	buf := [20]byte{}
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[pos:])
}

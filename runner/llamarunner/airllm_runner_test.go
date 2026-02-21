// Package llamarunner_test provides an end-to-end integration test for the
// AirLLM layer-budget mode.
//
// The test:
//  1. Generates a minimal but fully loadable LLaMA GGUF model file in a temp
//     directory.  The model has tiny dimensions (vocab=32, embd=64, 1 layer)
//     and zero-valued weights, so it produces garbage tokens – but it DOES run.
//  2. Launches the llamarunner subprocess with --airllm.
//  3. Sends a /load request (LoadOperationCommit).
//  4. Sends a /completion request with a short prompt.
//  5. Reads the NDJSON streaming response and asserts at least one token was
//     generated (content may be garbage – that is expected).
package llamarunner_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/runner/llamarunner"
)

// ---------------------------------------------------------------------------
// Minimal GGUF model generator
// ---------------------------------------------------------------------------

// Tiny model hyper-parameters – small enough to be fast on CPU, big enough
// that llama.cpp accepts the architecture without error.
const (
	mVocab   = 32  // vocabulary size
	mEmbd    = 64  // embedding / hidden dimension (must be divisible by mHeads)
	mHeads   = 2   // number of attention heads
	mKVHeads = 2   // number of KV heads (GQA; here same as mHeads)
	mFF      = 128 // feed-forward intermediate size
	mLayers  = 1   // number of transformer blocks
	mCtx     = 128 // max context length
	mHeadDim = mEmbd / mHeads // 32 – RoPE dim
)

// ggufBuilder is a binary serialiser for the GGUF v3 format.
// All integer fields are little-endian.
type ggufBuilder struct{ buf []byte }

func (b *ggufBuilder) u8(v uint8)   { b.buf = append(b.buf, v) }
func (b *ggufBuilder) u32(v uint32) { b.buf = binary.LittleEndian.AppendUint32(b.buf, v) }
func (b *ggufBuilder) u64(v uint64) { b.buf = binary.LittleEndian.AppendUint64(b.buf, v) }
func (b *ggufBuilder) i32(v int32)  { b.u32(uint32(v)) }
func (b *ggufBuilder) i64(v int64)  { b.u64(uint64(v)) }

// f32 encodes a float32 as four little-endian bytes.
func (b *ggufBuilder) f32(v float32) {
	bits := math.Float32bits(v)
	b.buf = binary.LittleEndian.AppendUint32(b.buf, bits)
}

// str encodes a GGUF string: uint64 length (no NUL) + bytes.
func (b *ggufBuilder) str(s string) {
	b.u64(uint64(len(s)))
	b.buf = append(b.buf, s...)
}

// kvStr writes a KV pair with a string value (type 8 = GGUF_TYPE_STRING).
func (b *ggufBuilder) kvStr(key, val string) { b.str(key); b.i32(8); b.str(val) }

// kvU32 writes a KV pair with a uint32 value (type 4 = GGUF_TYPE_UINT32).
func (b *ggufBuilder) kvU32(key string, val uint32) { b.str(key); b.i32(4); b.u32(val) }

// kvF32 writes a KV pair with a float32 value (type 6 = GGUF_TYPE_FLOAT32).
func (b *ggufBuilder) kvF32(key string, val float32) { b.str(key); b.i32(6); b.f32(val) }

// kvArrStr writes a KV pair whose value is an array of strings (type 9, elem 8).
func (b *ggufBuilder) kvArrStr(key string, vals []string) {
	b.str(key)
	b.i32(9)     // GGUF_TYPE_ARRAY
	b.i32(8)     // element type = GGUF_TYPE_STRING
	b.u64(uint64(len(vals)))
	for _, v := range vals {
		b.str(v)
	}
}

// kvArrF32 writes a KV pair whose value is an array of float32.
func (b *ggufBuilder) kvArrF32(key string, vals []float32) {
	b.str(key)
	b.i32(9)  // GGUF_TYPE_ARRAY
	b.i32(6)  // element type = GGUF_TYPE_FLOAT32
	b.u64(uint64(len(vals)))
	for _, v := range vals {
		b.f32(v)
	}
}

// kvArrI32 writes a KV pair whose value is an array of int32.
func (b *ggufBuilder) kvArrI32(key string, vals []int32) {
	b.str(key)
	b.i32(9)  // GGUF_TYPE_ARRAY
	b.i32(5)  // element type = GGUF_TYPE_INT32
	b.u64(uint64(len(vals)))
	for _, v := range vals {
		b.i32(v)
	}
}

// tensorInfo describes one tensor to be written.
type tInfo struct {
	name   string
	dims   []int64 // in GGML "ne" order: dims[0] is the fastest (innermost)
	ggmlT  int32   // ggml_type: 0=F32, 1=F16
	offset uint64  // byte offset into the tensor data blob
}

// tensorBytes computes the byte size of a tensor from its dims and type.
func tensorBytes(t tInfo) uint64 {
	n := int64(1)
	for _, d := range t.dims {
		n *= d
	}
	switch t.ggmlT {
	case 1: // F16
		return uint64(n) * 2
	default: // F32
		return uint64(n) * 4
	}
}

// writeTensorInfo encodes a single tensor-info record.
func (b *ggufBuilder) writeTensorInfo(t tInfo) {
	b.str(t.name)
	b.u32(uint32(len(t.dims)))
	for _, d := range t.dims {
		b.i64(d)
	}
	b.i32(t.ggmlT)
	b.u64(t.offset)
}

const ggufAlignment = uint64(32)

// makeMinimalLlamaModel creates a minimal valid GGUF file representing a tiny
// LLaMA model (1 layer, 64-dim embeddings, 32-token vocabulary).
//
// All weights are zero – the model will produce garbage output but it will
// load and run without errors, which is all we need for this test.
func makeMinimalLlamaModel(t *testing.T) string {
	t.Helper()

	// ------------------------------------------------------------------
	// 1. Define all required tensors and compute their offsets.
	// ------------------------------------------------------------------
	// The llama arch in llama.cpp requires (from llama-model.cpp):
	//   token_embd.weight      {n_embd, n_vocab}
	//   output_norm.weight     {n_embd}
	//   blk.0.attn_norm.weight {n_embd}
	//   blk.0.attn_q.weight    {n_embd, n_embd_head_k * n_head}
	//   blk.0.attn_k.weight    {n_embd, n_embd_head_k * n_kv_head}
	//   blk.0.attn_v.weight    {n_embd, n_embd_head_v * n_kv_head}
	//   blk.0.attn_output.weight {n_embd_head_k * n_head, n_embd}
	//   blk.0.ffn_norm.weight  {n_embd}
	//   blk.0.ffn_gate.weight  {n_embd, n_ff}
	//   blk.0.ffn_down.weight  {n_ff, n_embd}
	//   blk.0.ffn_up.weight    {n_embd, n_ff}
	//   output.weight is TENSOR_NOT_REQUIRED – llama reuses token_embd if absent
	raw := []tInfo{
		{name: "token_embd.weight",       dims: []int64{mEmbd, mVocab},   ggmlT: 0},
		{name: "output_norm.weight",       dims: []int64{mEmbd},           ggmlT: 0},
		{name: "blk.0.attn_norm.weight",   dims: []int64{mEmbd},           ggmlT: 0},
		{name: "blk.0.attn_q.weight",      dims: []int64{mEmbd, mHeads * mHeadDim},   ggmlT: 0},
		{name: "blk.0.attn_k.weight",      dims: []int64{mEmbd, mKVHeads * mHeadDim}, ggmlT: 0},
		{name: "blk.0.attn_v.weight",      dims: []int64{mEmbd, mKVHeads * mHeadDim}, ggmlT: 0},
		{name: "blk.0.attn_output.weight", dims: []int64{mHeads * mHeadDim, mEmbd},   ggmlT: 0},
		{name: "blk.0.ffn_norm.weight",    dims: []int64{mEmbd},           ggmlT: 0},
		{name: "blk.0.ffn_gate.weight",    dims: []int64{mEmbd, mFF},      ggmlT: 0},
		{name: "blk.0.ffn_down.weight",    dims: []int64{mFF, mEmbd},      ggmlT: 0},
		{name: "blk.0.ffn_up.weight",      dims: []int64{mEmbd, mFF},      ggmlT: 0},
	}

	// Assign offsets (with 32-byte alignment between tensors).
	infos := make([]tInfo, len(raw))
	curOff := uint64(0)
	var dataBlob []byte
	for i, ti := range raw {
		// Pad to alignment
		if curOff%ggufAlignment != 0 {
			pad := ggufAlignment - curOff%ggufAlignment
			dataBlob = append(dataBlob, make([]byte, pad)...)
			curOff += pad
		}
		infos[i] = ti
		infos[i].offset = curOff
		sz := tensorBytes(ti)
		dataBlob = append(dataBlob, make([]byte, sz)...) // zero weights
		curOff += sz
	}

	// ------------------------------------------------------------------
	// 2. Build tokenizer data (32 tokens: <unk>, <s>, </s>, then a-z, 0-3).
	// ------------------------------------------------------------------
	tokens := make([]string, mVocab)
	scores := make([]float32, mVocab)
	ttypes := make([]int32, mVocab)

	tokens[0] = "<unk>"; scores[0] = -1000; ttypes[0] = 2 // UNKNOWN
	tokens[1] = "<s>";   scores[1] = -1000; ttypes[1] = 3 // CONTROL (BOS)
	tokens[2] = "</s>";  scores[2] = -1000; ttypes[2] = 3 // CONTROL (EOS)
	for i := 3; i < mVocab; i++ {
		ch := 'a' + (i - 3)
		if ch > 'z' {
			ch = '0' + (ch - 'z' - 1)
		}
		tokens[i] = string(rune(ch))
		scores[i] = float32(-i)
		ttypes[i] = 1 // NORMAL
	}

	// ------------------------------------------------------------------
	// 3. Count all KV pairs we will write.
	// ------------------------------------------------------------------
	// Architecture KV:  general.architecture, general.name,
	//                   llama.block_count, llama.context_length,
	//                   llama.embedding_length, llama.feed_forward_length,
	//                   llama.attention.head_count, llama.attention.head_count_kv,
	//                   llama.rope.dimension_count,
	//                   llama.attention.layer_norm_rms_epsilon
	// Tokenizer KV:     tokenizer.ggml.model, tokenizer.ggml.tokens,
	//                   tokenizer.ggml.scores, tokenizer.ggml.token_type,
	//                   tokenizer.ggml.bos_token_id, tokenizer.ggml.eos_token_id
	const nKV = 16

	// ------------------------------------------------------------------
	// 4. Serialise the GGUF file.
	// ------------------------------------------------------------------
	b := &ggufBuilder{}

	// Header
	b.buf = append(b.buf, []byte("GGUF")...) // magic
	b.u32(3)                                  // version
	b.i64(int64(len(infos)))                  // n_tensors
	b.i64(nKV)                                // n_kv

	// Architecture metadata
	b.kvStr("general.architecture", "llama")
	b.kvStr("general.name", "tinyllama-airllm-test")
	b.kvU32("llama.block_count",                     mLayers)
	b.kvU32("llama.context_length",                  mCtx)
	b.kvU32("llama.embedding_length",                mEmbd)
	b.kvU32("llama.feed_forward_length",             mFF)
	b.kvU32("llama.attention.head_count",            mHeads)
	b.kvU32("llama.attention.head_count_kv",         mKVHeads)
	b.kvU32("llama.rope.dimension_count",            mHeadDim)
	b.kvF32("llama.attention.layer_norm_rms_epsilon", 1e-5)

	// Tokenizer metadata
	b.kvStr("tokenizer.ggml.model", "llama")
	b.kvArrStr("tokenizer.ggml.tokens",     tokens)
	b.kvArrF32("tokenizer.ggml.scores",     scores)
	b.kvArrI32("tokenizer.ggml.token_type", ttypes)
	b.kvU32("tokenizer.ggml.bos_token_id",  1)
	b.kvU32("tokenizer.ggml.eos_token_id",  2)

	// Tensor info
	for _, ti := range infos {
		b.writeTensorInfo(ti)
	}

	// Pad header to alignment boundary before tensor data
	if uint64(len(b.buf))%ggufAlignment != 0 {
		pad := ggufAlignment - uint64(len(b.buf))%ggufAlignment
		b.buf = append(b.buf, make([]byte, pad)...)
	}

	// Tensor data
	b.buf = append(b.buf, dataBlob...)

	// Write to temp file
	f, err := os.CreateTemp(t.TempDir(), "test-minimal-*.gguf")
	if err != nil {
		t.Fatalf("create temp model: %v", err)
	}
	if _, err := f.Write(b.buf); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close model: %v", err)
	}
	t.Logf("tiny model written: %s (%.1f KiB)", f.Name(), float64(len(b.buf))/1024)
	return f.Name()
}

// ---------------------------------------------------------------------------
// Helpers: free port, poll health, HTTP send helpers
// ---------------------------------------------------------------------------

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port
}

func pollHealth(t *testing.T, base string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := http.Get(base + "/health")
		if err == nil {
			var sr llm.ServerStatusResponse
			if json.NewDecoder(resp.Body).Decode(&sr) == nil {
				resp.Body.Close()
				if sr.Status == llm.ServerStatusReady {
					return
				}
			} else {
				resp.Body.Close()
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("runner did not become ready within %v", timeout)
}

func postJSON(t *testing.T, url string, body any) *http.Response {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	return resp
}

// ---------------------------------------------------------------------------
// End-to-end test: AirLLM runner with a real (tiny) model
// ---------------------------------------------------------------------------

// TestAirLLMRunnerEndToEnd starts the llamarunner with --airllm, loads a real
// tiny LLaMA GGUF model, sends a completion prompt, and asserts that at least
// one token is streamed back.  The output will be garbage (zero-weight model)
// but the test proves the full code path works on a CPU-only host.
func TestAirLLMRunnerEndToEnd(t *testing.T) {
	modelPath := makeMinimalLlamaModel(t)
	port := freePort(t)
	base := "http://127.0.0.1:" + strconv.Itoa(port)

	// Start the runner in a goroutine; it will exit when the test binary ends.
	runnerErrCh := make(chan error, 1)
	go func() {
		runnerErrCh <- llamarunner.Execute([]string{
			"--model", modelPath,
			"--port", strconv.Itoa(port),
			"--airllm",
		})
	}()

	// Wait until the runner process is listening.
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 200*time.Millisecond)
		if err == nil {
			conn.Close()
			break
		}
		select {
		case err := <-runnerErrCh:
			t.Fatalf("runner exited early: %v", err)
		default:
		}
		time.Sleep(50 * time.Millisecond)
	}

	// ------------------------------------------------------------------
	// Step 1: /load – commit the model with minimal parameters.
	// ------------------------------------------------------------------
	t.Log("sending /load request …")
	loadReq := llm.LoadRequest{
		Operation:  llm.LoadOperationCommit,
		Parallel:   1,
		BatchSize:  512,
		KvSize:     mCtx,
		NumThreads: 2,
		UseMmap:    true,
		FlashAttention: ml.FlashAttentionDisabled,
	}
	loadResp := postJSON(t, base+"/load", loadReq)
	defer loadResp.Body.Close()

	var lr llm.LoadResponse
	if err := json.NewDecoder(loadResp.Body).Decode(&lr); err != nil {
		t.Fatalf("decode /load response: %v", err)
	}
	if loadResp.StatusCode != http.StatusOK {
		t.Fatalf("/load HTTP %d", loadResp.StatusCode)
	}

	// ------------------------------------------------------------------
	// Step 2: poll /health until ServerStatusReady (model fully loaded).
	// ------------------------------------------------------------------
	t.Log("waiting for model to finish loading …")
	pollHealth(t, base, 120*time.Second)
	t.Log("model loaded – runner is ready")

	// ------------------------------------------------------------------
	// Step 3: /completion – send an empty prompt.
	//
	// With an empty string, the SentencePiece tokenizer in our minimal
	// test model produces only the BOS token [1] (no BPE merge-rule
	// lookups are needed).  The model auto-regresses from BOS and emits
	// a few tokens of garbage output – which is exactly what we need to
	// prove the full AirLLM inference code path works on CPU.
	// ------------------------------------------------------------------
	completionReq := llm.CompletionRequest{
		Prompt: "", // empty → BOS only; no character-lookup merge rules required
		Options: &api.Options{
			NumPredict: 3, // keep test fast; model produces garbage output anyway
		},
	}
	type fullReq struct {
		llm.CompletionRequest
	}
	t.Log("sending /completion request …")
	compResp := postJSON(t, base+"/completion", fullReq{completionReq})
	defer compResp.Body.Close()

	if compResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(compResp.Body)
		t.Fatalf("/completion HTTP %d: %s", compResp.StatusCode, body)
	}

	// ------------------------------------------------------------------
	// Step 4: read NDJSON stream – look for at least one token.
	// ------------------------------------------------------------------
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	type streamResp struct {
		Content    string `json:"content"`
		Done       bool   `json:"done"`
		DoneReason int    `json:"done_reason"` // llm.DoneReason is an int
	}

	scanner := bufio.NewScanner(compResp.Body)
	totalTokens := 0
	var doneResp *streamResp
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var cr streamResp
		if err := json.Unmarshal([]byte(line), &cr); err != nil {
			t.Logf("warn: could not decode line %q: %v", line, err)
			continue
		}
		if cr.Content != "" {
			totalTokens++
			t.Logf("token #%d: %q", totalTokens, cr.Content)
		}
		if cr.Done {
			doneResp = &cr
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for completion stream")
		default:
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scanner error: %v", err)
	}

	// ------------------------------------------------------------------
	// Step 5: assertions
	// ------------------------------------------------------------------
	if totalTokens == 0 {
		t.Errorf("expected at least 1 token in completion response; got 0")
	} else {
		t.Logf("✓ got %d token(s) from the tiny model (garbage output is expected)", totalTokens)
	}

	if doneResp == nil {
		t.Error("completion stream did not send a done=true response")
	} else {
		t.Logf("✓ stream finished with done_reason=%q", doneResp.DoneReason)
	}
}

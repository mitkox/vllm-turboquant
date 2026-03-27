# TurboQuant Implementation Guide

## What is TurboQuant?

TurboQuant is a **KV cache compression algorithm** from Google that reduces the memory footprint of the KV cache during LLM inference from 16 bits/element down to **2.5 or 3.5 bits/element**. It uses a two-stage approach: **MSE codebook quantization** of the unit vector + **QJL (Johnson-Lindenstrauss) 1-bit hashing** of the residual, both operating in a structured Hadamard-rotated space.

This implementation integrates TurboQuant into vLLM's v1 engine, targeting **NVIDIA GB10 (Blackwell SM121) GPUs**.

---

## File Map: What Was Built (5,219 lines of new code)

### NEW FILES (Core Algorithm) - Read in this order:

| # | File | Lines | Purpose |
|---|------|-------|---------|
| 1 | `vllm/v1/attention/ops/turboquant_kv_cache.py` | 831 | **Core algorithm**: quantization math, codebook, transforms, pack/unpack |
| 2 | `vllm/v1/attention/ops/turboquant_metadata.py` | 359 | Per-layer metadata: which dimensions are "outliers" (high-precision group) |
| 3 | `vllm/v1/attention/ops/triton_turboquant_kv_update.py` | 672 | **GPU kernel**: fused quantize-and-store into paged KV cache |
| 4 | `vllm/v1/attention/ops/triton_turboquant_decode.py` | 849 | **GPU kernel**: fused decode attention directly on compressed cache |
| 5 | `tests/quantization/test_turboquant.py` | 1896 | Test suite |
| 6 | `benchmarks/generate_turboquant_metadata.py` | 558 | Calibration tool to generate per-layer metadata |
| 7 | `benchmarks/run_turboquant_gb10_compare.sh` | 54 | Benchmark script (FP8 vs TurboQuant) |

### MODIFIED FILES (Integration into vLLM):

| File | What Changed |
|------|-------------|
| `vllm/config/cache.py` | Added `enable_turboquant`, `turboquant_metadata_path`, `"turboquant25"`/`"turboquant35"` cache dtypes |
| `vllm/engine/arg_utils.py` | CLI args: `--enable-turboquant`, `--turboquant-metadata-path` |
| `vllm/v1/attention/selector.py` | Validates TurboQuant flag + GB10 GPU requirement |
| `vllm/v1/kv_cache_interface.py` | TurboQuant-aware memory size calculations (packed_dim instead of head_size * dtype_size) |
| `vllm/v1/attention/backends/triton_attn.py` | ~300 lines added: TurboQuant dispatch in `TritonAttentionImpl` (the main integration point) |
| `vllm/v1/attention/ops/triton_prefill_attention.py` | 18 lines: fast prefill path bypass for pure prompt batches |
| `vllm/v1/worker/gpu_worker.py` | Minor: worker-level TurboQuant awareness |
| `docs/features/quantization/quantized_kvcache.md` | Documentation |

---

## Algorithm Deep Dive (Reading Order)

### Step 1: Understand the Two-Group Split

**File:** `turboquant_kv_cache.py:241-263`

Each KV head vector (e.g., 128 dims) is split into two groups:
- **Group 0 (outliers)**: Top 25-50% of dimensions by activation magnitude -> higher precision
- **Group 1 (regular)**: Remaining dimensions -> lower precision

```
turboquant25: group0 = 32 dims @ 3 bits, group1 = 96 dims @ 2 bits  -> avg 2.5 bits
turboquant35: group0 = 64 dims @ 4 bits, group1 = 64 dims @ 3 bits  -> avg 3.5 bits
```

Which dimensions are "outliers" is determined per-layer, per-head via **calibration** (see `generate_turboquant_metadata.py`). The metadata JSON stores sorted indices of the high-precision dimensions for each head.

### Step 2: Understand the Quantization Pipeline

**File:** `turboquant_kv_cache.py:684-741` (`quantize_turboquant_vectors`)

For each group, given input vector `v` of dimension `d`:

```
1. vector_norm = ||v||                          # Store as float16 (2 bytes)
2. unit = v / vector_norm                       # Normalize to unit sphere
3. rotated = MSE_Hadamard_Transform(unit)       # Structured Hadamard rotation
4. indices = nearest_centroid(rotated)           # Lloyd-Max codebook lookup
5. rotated_hat = centroids[indices]              # Quantized approximation in rotated space
6. mse_hat = MSE_Inverse_Transform(rotated_hat) # Back to original space
7. residual = unit - mse_hat                    # What the codebook missed
8. residual_norm = ||residual||                 # Store as float16 (2 bytes)
9. qjl_bits = sign(QJL_Transform(residual))     # 1-bit JL hash of residual direction
```

**Packed storage per group:**
```
[MSE centroid indices packed at N bits] [QJL sign bits packed at 1 bit] [vector_norm fp16] [residual_norm fp16]
```

### Step 3: Understand the Structured Hadamard Transform

**File:** `turboquant_kv_cache.py:100-187`

The "rotation" is NOT a dense random matrix. It's a **structured Hadamard transform**:
1. Element-wise multiply by random signs (+1/-1), seeded deterministically
2. Apply Fast Walsh-Hadamard Transform (FWHT) on power-of-2 blocks
3. If dimension isn't power-of-2, decompose into largest-power-of-2 blocks

This gives O(d log d) transforms instead of O(d^2) for dense matrices, but with similar randomization properties. The MSE transform is normalized (divide by sqrt(block_size)), QJL is unnormalized.

**Why rotate at all?** The Hadamard rotation "spreads" the information across all coordinates, so that scalar quantization per-coordinate loses roughly equal information everywhere, rather than catastrophically losing a high-variance coordinate.

### Step 4: Understand the Lloyd-Max Codebook

**File:** `turboquant_kv_cache.py:190-238` (`_dimension_aware_codebook`)

The codebook is NOT uniform quantization. It's a **dimension-aware Lloyd-Max codebook**:
- Uses the Beta coordinate PDF `p(x) ~ (1-x^2)^((d-3)/2)` which models how coordinates of random unit vectors in d dimensions are distributed
- Runs 200 iterations of Lloyd's algorithm (k-means on 1D) weighted by this PDF
- Higher dimensions -> more probability mass near 0 -> codebook centroids bunch near center

This is computed once at startup (cached by `@cache`).

### Step 5: Understand the Decode Attention Kernel

**File:** `triton_turboquant_decode.py:117-528` (`_turboquant_decode_kernel`)

The key insight: **attention is computed directly on compressed representations without full decompression**.

For the **key** side (computing logits = q * k):
```
logit = sum over groups:
    vector_norm * (q_rot . centroids[indices])           # MSE component
  + vector_norm * residual_norm * (q_qjl . sign_bits)   # QJL component
```

The query is pre-transformed: `q_rot = MSE_Transform(q)` and `q_qjl = QJL_Transform(q) * scale / dim`.

**Why this works mathematically:**
```
q . k = q . (norm * (mse_hat + residual))
      = norm * (q . mse_hat) + norm * (q . residual)

In rotated space:
      = norm * (F(q) . centroids) + norm * res_norm * (G(q) . sign(G(residual/res_norm)))
```

where F is the MSE Hadamard transform and G is the QJL Hadamard transform. The QJL term uses the Johnson-Lindenstrauss property: `sign(G(x)) . G(y) ~ sqrt(pi/2) * (x . y) / dim`.

For the **value** side (computing weighted sum):
- Accumulate in rotated space: `acc_mse += softmax_weight * vector_norm * centroids[i]`
- Accumulate QJL: `acc_qjl += softmax_weight * vector_norm * residual_norm * scale * sign_bits`
- **Postprocess kernel** (`_turboquant_decode_postprocess_group_kernel`, line 532): multiply by inverse transform matrices and scatter back to original dimension order

### Step 6: Understand the KV Update Kernel

**File:** `triton_turboquant_kv_update.py:214-566` (`_quantize_group` + `_turboquant_quantize_store_kernel`)

Fused kernel that reads raw fp16 KV vectors and writes packed uint8 directly into the paged cache:
1. One Triton program per (token, head) pair
2. Processes both groups sequentially
3. For each group: compute norm -> project through MSE matrix -> find nearest centroids -> compute residual via MSE-to-QJL composed matrix -> pack bits -> store

The composed matrix `mse_to_qjl = MSE_Inverse @ QJL_Forward` (line 458-493 in `turboquant_kv_cache.py`) avoids materializing the intermediate vector. Instead of:
```
residual = unit - MSE_Inverse(centroid_hat)
qjl_projected = QJL_Forward(residual)
```
It computes:
```
qjl_projected = QJL_Forward(unit) - (MSE_Inverse @ QJL_Forward) @ centroid_hat
```

### Step 7: Integration Point

**File:** `triton_attn.py` (diff against base vLLM)

`TritonAttentionImpl.__init__` gains ~50 lines of TurboQuant setup:
- Loads metadata, computes transform matrices, centroids, group indices
- Caches everything per (layer_name, device) in instance-level dicts

`TritonAttentionImpl.forward` dispatches:
- **Prefill**: If pure prompt batch, uses standard Triton prefill kernel (fast path bypass)
- **Decode with TurboQuant**: Calls `turboquant_write_packed_kv()` to store, then `turboquant_decode_attention_fwd()` to attend
- **Mixed batches**: TurboQuant decode for generation tokens, standard Triton for encoder tokens

---

## Key Constants to Know

```python
TURBOQUANT_SEED = 20250428              # Deterministic seed for all random signs
TURBOQUANT_QJL_SEED_OFFSET = 10_000     # QJL uses different signs than MSE
TURBOQUANT_QJL_SCALE = sqrt(pi/2)       # ~1.2533, corrects QJL sign estimator bias
TURBOQUANT_CODEBOOK_GRID_POINTS = 32768 # Resolution for Lloyd-Max optimization
TURBOQUANT_GROUP_ALIGNMENT = 16         # Group dimensions must be multiples of 16
```

Seed offsets for different transform matrices (from `triton_attn.py`):
- Key MSE rotation: seed_offset=0
- Key QJL matrix: seed_offset=0
- Value MSE rotation: seed_offset=101
- Value QJL matrix: seed_offset=307

---

## Suggested Reading Path

1. **Start with the paper**: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
2. **`turboquant_kv_cache.py`**: Read top-to-bottom. Constants -> layout -> transforms -> codebook -> quantize/dequantize
3. **`turboquant_metadata.py`**: Understand the per-layer calibration artifact format
4. **`triton_turboquant_kv_update.py`**: The write path (encode). Read the docstring first (lines 1-37), then `_quantize_group`
5. **`triton_turboquant_decode.py`**: The read path (decode attention). Start from `turboquant_decode_attention_fwd` at line 607, then trace into the kernel
6. **`triton_attn.py` diff**: See how it plugs into vLLM's attention backend
7. **`test_turboquant.py`**: The tests serve as executable documentation of expected behavior

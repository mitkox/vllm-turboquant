# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.ops.turboquant_kv_cache import (
    dequantize_turboquant_vectors,
    get_turboquant_bits,
    get_turboquant_centroids,
    get_turboquant_packed_dim,
    get_turboquant_rotation,
    is_turboquant_kv_cache,
    pack_turboquant_indices,
    quantize_turboquant_vectors,
    unpack_turboquant_indices,
)


def test_turboquant_dtype_registry():
    for dtype, bits in (
        ("turboquant1", 1),
        ("turboquant2", 2),
        ("turboquant3", 3),
        ("turboquant4", 4),
    ):
        assert is_turboquant_kv_cache(dtype)
        assert get_turboquant_bits(dtype) == bits


@torch.inference_mode()
def test_turboquant_pack_unpack_roundtrip():
    head_size = 128
    for bits in (1, 2, 3, 4):
        indices = torch.randint(0, 1 << bits, (4, head_size), dtype=torch.uint8)
        packed = pack_turboquant_indices(indices, bits)
        unpacked = unpack_turboquant_indices(packed, head_size, bits)
        assert packed.shape[-1] == get_turboquant_packed_dim(head_size, bits) - 2
        assert torch.equal(indices, unpacked)


@torch.inference_mode()
def test_turboquant_quantize_dequantize_shapes_and_error():
    head_size = 128
    x = torch.randn(6, 3, head_size, dtype=torch.float32)

    for bits in (1, 2, 3, 4):
        rotation = get_turboquant_rotation(torch.device("cpu"), head_size)
        centroids = get_turboquant_centroids(torch.device("cpu"), head_size, bits)
        packed = quantize_turboquant_vectors(x, bits, rotation, centroids)
        restored = dequantize_turboquant_vectors(
            packed,
            bits,
            head_size,
            rotation,
            centroids,
            x.dtype,
        )
        assert packed.shape == (6, 3, get_turboquant_packed_dim(head_size, bits))
        assert restored.shape == x.shape
        assert torch.isfinite(restored).all()

    rotation = get_turboquant_rotation(torch.device("cpu"), head_size)
    mse = {}
    for bits in (1, 2, 3, 4):
        centroids = get_turboquant_centroids(torch.device("cpu"), head_size, bits)
        packed = quantize_turboquant_vectors(x, bits, rotation, centroids)
        restored = dequantize_turboquant_vectors(
            packed,
            bits,
            head_size,
            rotation,
            centroids,
            x.dtype,
        )
        mse[bits] = torch.mean((x - restored) ** 2).item()

    assert mse[4] < mse[3] < mse[2] < mse[1]


def test_turboquant_attention_spec_page_size_matches_triton_shape():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    block_size = 2096
    num_kv_heads = 2
    head_size = 256
    cache_dtype = "turboquant3"

    kv_cache_shape = TritonAttentionBackend.get_kv_cache_shape(
        num_blocks=1,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype_str=cache_dtype,
    )
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.uint8,
        cache_dtype_str=cache_dtype,
    )

    assert spec.page_size_bytes == math.prod(kv_cache_shape)


def test_turboquant_full_attention_spec_merge_preserves_cache_dtype():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    specs = [
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            cache_dtype_str="turboquant3",
        ),
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            cache_dtype_str="turboquant3",
        ),
    ]

    merged = FullAttentionSpec.merge(specs)

    assert merged.cache_dtype_str == "turboquant3"


def test_turboquant_disables_cudagraph_support_for_triton_builder():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.uint8,
        cache_dtype_str="turboquant3",
    )

    assert (
        TritonAttentionMetadataBuilder.get_cudagraph_support(None, spec)  # type: ignore[arg-type]
        == AttentionCGSupport.NEVER
    )

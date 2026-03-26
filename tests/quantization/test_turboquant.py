# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.config.cache import CacheConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    get_turboquant_norm_lut,
    turboquant_decode_attention_fwd,
)
from vllm.v1.attention.ops.turboquant_kv_cache import (
    _standard_gaussian_codebook,
    build_turboquant_outlier_masks,
    dequantize_turboquant_vectors,
    get_turboquant_bits,
    get_turboquant_centroids,
    get_turboquant_group_dims,
    get_turboquant_layout,
    get_turboquant_packed_dim,
    get_turboquant_qjl_matrix,
    get_turboquant_rotation,
    is_turboquant_kv_cache,
    pack_turboquant_indices,
    quantize_turboquant_vectors,
    unpack_turboquant_indices,
)


def _get_turboquant_tables(
    kv_cache_dtype: str,
    head_size: int,
    device: torch.device,
):
    layout = get_turboquant_layout(kv_cache_dtype, head_size)
    rotations = (
        get_turboquant_rotation(device, layout.groups[0].dim, seed_offset=101),
        get_turboquant_rotation(device, layout.groups[1].dim, seed_offset=211),
    )
    qjl_matrices = (
        get_turboquant_qjl_matrix(device, layout.groups[0].dim, seed_offset=307),
        get_turboquant_qjl_matrix(device, layout.groups[1].dim, seed_offset=401),
    )
    centroids = {
        group.mse_bits: get_turboquant_centroids(device, group.dim, group.mse_bits)
        for group in layout.groups
        if group.mse_bits > 0
    }
    return rotations, qjl_matrices, centroids, layout


@torch.inference_mode()
def _reference_turboquant_decode(
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_dtype: str,
    key_tables,
    value_tables,
    key_masks,
    value_masks,
    scale: float,
) -> torch.Tensor:
    outputs = []
    block_size = key_cache.shape[1]
    query_lens = query_start_loc[1:] - query_start_loc[:-1]

    for seq_idx, seq_len in enumerate(seq_lens.tolist()):
        q_start = int(query_start_loc[seq_idx].item())
        q_len = int(query_lens[seq_idx].item())
        q_end = q_start + q_len
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = block_table[seq_idx, :num_blocks].to(torch.int64)
        seq_key_cache = key_cache.index_select(0, block_ids).reshape(
            num_blocks * block_size, key_cache.shape[2], -1
        )[:seq_len]
        seq_value_cache = value_cache.index_select(0, block_ids).reshape(
            num_blocks * block_size, value_cache.shape[2], -1
        )[:seq_len]

        seq_key = dequantize_turboquant_vectors(
            seq_key_cache,
            kv_cache_dtype,
            query.shape[-1],
            key_tables[0],
            key_tables[1],
            key_tables[2],
            key_masks,
            query.dtype,
        )
        seq_value = dequantize_turboquant_vectors(
            seq_value_cache,
            kv_cache_dtype,
            query.shape[-1],
            value_tables[0],
            value_tables[1],
            value_tables[2],
            value_masks,
            query.dtype,
        )
        seq_query = query[q_start:q_end]
        q_states = seq_query.permute(1, 0, 2).unsqueeze(0)
        k_states = seq_key.permute(1, 0, 2).unsqueeze(0)
        v_states = seq_value.permute(1, 0, 2).unsqueeze(0)
        query_positions = torch.arange(
            seq_len - q_len, seq_len, device=query.device, dtype=torch.int32
        )
        key_positions = torch.arange(seq_len, device=query.device, dtype=torch.int32)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        seq_output = F.scaled_dot_product_attention(
            q_states,
            k_states,
            v_states,
            attn_mask=causal_mask.view(1, 1, q_len, seq_len),
            dropout_p=0.0,
            enable_gqa=query.shape[1] > key_cache.shape[2],
            scale=scale,
        )
        outputs.append(seq_output.squeeze(0).permute(1, 0, 2))

    return torch.cat(outputs, dim=0)


def test_turboquant_dtype_registry():
    for dtype, bits in (
        ("turboquant25", 2.5),
        ("turboquant35", 3.5),
    ):
        assert is_turboquant_kv_cache(dtype)
        assert get_turboquant_bits(dtype) == bits

    for dtype in ("turboquant1", "turboquant2", "turboquant3", "turboquant4"):
        assert not is_turboquant_kv_cache(dtype)
        with pytest.raises(ValueError, match="Unsupported TurboQuant"):
            get_turboquant_bits(dtype)


def test_turboquant_layout_matches_presets():
    assert get_turboquant_group_dims(128, "turboquant25") == (32, 96)
    assert get_turboquant_group_dims(128, "turboquant35") == (64, 64)
    assert get_turboquant_packed_dim(128, "turboquant25") == get_turboquant_packed_dim(
        128, 2.5
    )


def test_turboquant_codebook_is_built_on_cpu():
    codebook = _standard_gaussian_codebook(3)

    assert codebook.device.type == "cpu"
    assert codebook.dtype == torch.float32

    if torch.cuda.is_available():
        centroids = get_turboquant_centroids(torch.device("cuda"), 32, 3)
        assert centroids.device.type == "cuda"


@torch.inference_mode()
def test_turboquant_pack_unpack_roundtrip():
    head_size = 128
    for bits in (1, 2, 3, 4):
        indices = torch.randint(0, 1 << bits, (4, head_size), dtype=torch.uint8)
        packed = pack_turboquant_indices(indices, bits)
        unpacked = unpack_turboquant_indices(packed, head_size, bits)
        assert packed.shape[-1] == (head_size * bits + 7) // 8
        assert torch.equal(indices, unpacked)


@torch.inference_mode()
def test_turboquant_outlier_masks_are_deterministic():
    x = torch.randn(32, 4, 128, dtype=torch.float32)
    first = build_turboquant_outlier_masks(x, "turboquant25")
    second = build_turboquant_outlier_masks(x, "turboquant25")
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[0].shape == (4, 32)
    assert first[1].shape == (4, 96)


@torch.inference_mode()
def test_turboquant_quantize_dequantize_shapes_and_error():
    head_size = 128
    x = torch.randn(6, 3, head_size, dtype=torch.float32)
    cache_dtypes = ("turboquant25", "turboquant35")

    mse = {}
    for cache_dtype in cache_dtypes:
        tables = _get_turboquant_tables(cache_dtype, head_size, torch.device("cpu"))
        masks = build_turboquant_outlier_masks(x, cache_dtype)
        packed = quantize_turboquant_vectors(
            x,
            cache_dtype,
            tables[0],
            tables[1],
            tables[2],
            masks,
        )
        restored = dequantize_turboquant_vectors(
            packed,
            cache_dtype,
            head_size,
            tables[0],
            tables[1],
            tables[2],
            masks,
            x.dtype,
        )
        assert packed.shape == (6, 3, get_turboquant_packed_dim(head_size, cache_dtype))
        assert restored.shape == x.shape
        assert torch.isfinite(restored).all()
        mse[cache_dtype] = torch.mean((x - restored) ** 2).item()

    assert mse["turboquant35"] < mse["turboquant25"]


@torch.inference_mode()
def test_turboquant_qjl_residual_has_small_inner_product_bias():
    head_size = 128
    samples = 1024
    x = torch.randn(samples, 4, head_size, dtype=torch.float32)
    y = torch.randn(samples, 4, head_size, dtype=torch.float32)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)

    for cache_dtype in ("turboquant25", "turboquant35"):
        tables = _get_turboquant_tables(cache_dtype, head_size, torch.device("cpu"))
        masks = build_turboquant_outlier_masks(x, cache_dtype)
        packed = quantize_turboquant_vectors(
            x,
            cache_dtype,
            tables[0],
            tables[1],
            tables[2],
            masks,
        )
        restored = dequantize_turboquant_vectors(
            packed,
            cache_dtype,
            head_size,
            tables[0],
            tables[1],
            tables[2],
            masks,
            x.dtype,
        )
        inner_error = (restored * y).sum(dim=-1) - (x * y).sum(dim=-1)
        assert abs(inner_error.mean().item()) < 0.05


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 12
    or torch.cuda.get_device_capability()[1] != 1,
    reason="GB10 / SM121 CUDA device required",
)
@pytest.mark.parametrize("cache_dtype", ["turboquant25", "turboquant35"])
@torch.inference_mode()
def test_turboquant_triton_decode_matches_reference(cache_dtype: str):
    device = torch.device("cuda")
    head_size = 128
    num_heads = 4
    num_kv_heads = 2
    block_size = 16
    seq_lens = torch.tensor([11, 15], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(head_size)

    key_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    value_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    norm_lut = get_turboquant_norm_lut(device)

    query = torch.randn(5, num_heads, head_size, dtype=torch.float16, device=device)
    key_vectors = torch.randn(
        int(seq_lens.sum().item()),
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device=device,
    )
    value_vectors = torch.randn_like(key_vectors)
    key_masks = build_turboquant_outlier_masks(key_vectors, cache_dtype)
    value_masks = build_turboquant_outlier_masks(value_vectors, cache_dtype)

    packed_dim = get_turboquant_packed_dim(head_size, cache_dtype)
    key_cache = torch.zeros(
        (2, block_size, num_kv_heads, packed_dim),
        dtype=torch.uint8,
        device=device,
    )
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32, device=device)

    cursor = 0
    for seq_idx, seq_len in enumerate(seq_lens.tolist()):
        seq_key = quantize_turboquant_vectors(
            key_vectors[cursor : cursor + seq_len],
            cache_dtype,
            key_tables[0],
            key_tables[1],
            key_tables[2],
            key_masks,
        )
        seq_value = quantize_turboquant_vectors(
            value_vectors[cursor : cursor + seq_len],
            cache_dtype,
            value_tables[0],
            value_tables[1],
            value_tables[2],
            value_masks,
        )
        num_blocks = (seq_len + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, seq_len)
            physical_block = block_table[seq_idx, block_idx].item()
            key_cache[physical_block, : end - start] = seq_key[start:end]
            value_cache[physical_block, : end - start] = seq_value[start:end]
        cursor += seq_len

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    token_seq_ids = torch.repeat_interleave(
        torch.arange(seq_lens.shape[0], device=device, dtype=torch.int32),
        query_lens,
    )
    token_offsets = (
        torch.arange(query.shape[0], device=device, dtype=torch.int32)
        - query_start_loc.index_select(0, token_seq_ids.to(torch.int64))
    )
    token_seq_lens = (
        seq_lens.index_select(0, token_seq_ids.to(torch.int64))
        - query_lens.index_select(0, token_seq_ids.to(torch.int64))
        + token_offsets
        + 1
    ).to(torch.int32)
    kv_group_num = num_heads // num_kv_heads
    kv_head_for_query_head = (
        torch.arange(num_heads, device=device, dtype=torch.int64) // kv_group_num
    )
    key_query_group_indices = tuple(
        group.index_select(0, kv_head_for_query_head) for group in key_masks
    )
    value_query_group_indices = tuple(
        group.index_select(0, kv_head_for_query_head) for group in value_masks
    )
    fast_output = torch.empty_like(query)
    fast_output = turboquant_decode_attention_fwd(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        key_group_indices=key_masks,
        value_group_indices=value_masks,
        key_rotations=key_tables[0],
        key_qjl_matrices=key_tables[1],
        value_rotations=value_tables[0],
        value_qjl_matrices=value_tables[1],
        centroids=key_tables[2],
        norm_lut=norm_lut,
        softmax_scale=scale,
        kv_cache_dtype=cache_dtype,
        token_seq_ids=token_seq_ids,
        token_seq_lens=token_seq_lens,
        kv_head_for_query_head=kv_head_for_query_head,
        key_query_group_indices=key_query_group_indices,
        value_query_group_indices=value_query_group_indices,
        out=fast_output,
    )
    ref_output = _reference_turboquant_decode(
        query=query,
        query_start_loc=query_start_loc,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        kv_cache_dtype=cache_dtype,
        key_tables=key_tables,
        value_tables=value_tables,
        key_masks=key_masks,
        value_masks=value_masks,
        scale=scale,
    )

    torch.testing.assert_close(fast_output, ref_output, atol=8e-2, rtol=8e-2)


@pytest.mark.parametrize("cache_dtype", ["turboquant25", "turboquant35"])
def test_turboquant_attention_spec_page_size_matches_triton_shape(cache_dtype):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    block_size = 2096
    num_kv_heads = 2
    head_size = 256

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
            cache_dtype_str="turboquant25",
        ),
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            cache_dtype_str="turboquant25",
        ),
    ]

    merged = FullAttentionSpec.merge(specs)

    assert merged.cache_dtype_str == "turboquant25"


def test_turboquant_disables_cudagraph_support_for_triton_builder():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.uint8,
        cache_dtype_str="turboquant25",
    )

    assert (
        TritonAttentionMetadataBuilder.get_cudagraph_support(None, spec)  # type: ignore[arg-type]
        == AttentionCGSupport.NEVER
    )


def test_turboquant_validate_configuration_rejects_non_gb10():
    reasons = TritonAttentionBackend.validate_configuration(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="turboquant25",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type=AttentionType.DECODER,
    )

    assert "TurboQuant KV cache requires NVIDIA GB10 / SM121" in reasons


def test_turboquant_validate_configuration_rejects_sinks():
    reasons = TritonAttentionBackend.validate_configuration(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="turboquant25",
        block_size=16,
        use_mla=False,
        has_sink=True,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(12, 1),
        attn_type=AttentionType.DECODER,
    )

    assert "TurboQuant KV cache does not support attention sinks" in reasons


def test_cache_config_requires_feature_gate(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(12, 1),
    )

    with pytest.raises(ValueError, match="enable_turboquant=True"):
        CacheConfig(cache_dtype="turboquant25")

    CacheConfig(cache_dtype="turboquant25", enable_turboquant=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"attn_type": AttentionType.ENCODER_DECODER}, "cross-attention"),
        ({"sliding_window": 128}, "sliding window attention"),
        ({"logits_soft_cap": 1.0}, "logits soft capping"),
        ({"sinks": torch.zeros(8, dtype=torch.float32)}, "attention sinks"),
    ],
)
def test_turboquant_impl_rejects_unsupported_modes_at_init(
    monkeypatch, kwargs, match
):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(12, 1),
    )

    with pytest.raises(NotImplementedError, match=match):
        TritonAttentionImpl(
            num_heads=8,
            head_size=128,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=kwargs.get("sliding_window"),
            kv_cache_dtype="turboquant25",
            logits_soft_cap=kwargs.get("logits_soft_cap"),
            attn_type=kwargs.get("attn_type", AttentionType.DECODER),
            sinks=kwargs.get("sinks"),
        )


def test_turboquant_impl_allows_encoder_only_layers(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(12, 1),
    )

    impl = TritonAttentionImpl(
        num_heads=8,
        head_size=128,
        scale=1.0,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="turboquant25",
        attn_type=AttentionType.ENCODER_ONLY,
    )

    assert impl.attn_type == AttentionType.ENCODER_ONLY


@torch.inference_mode()
def test_turboquant_prefill_large_head_uses_safe_gpu_fallback(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(12, 1),
    )

    impl = TritonAttentionImpl(
        num_heads=4,
        head_size=256,
        scale=1.0 / math.sqrt(256),
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="turboquant35",
        attn_type=AttentionType.DECODER,
    )

    query = torch.randn(6, 4, 256, dtype=torch.float16)
    key = torch.randn(6, 2, 256, dtype=torch.float16)
    value = torch.randn_like(key)
    output = torch.empty_like(query)

    attn_metadata = TritonAttentionMetadata(
        num_actual_tokens=6,
        max_query_len=6,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 6], dtype=torch.int32),
        max_seq_len=6,
        seq_lens=torch.tensor([6], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([6], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(6, dtype=torch.int32),
        seq_threshold_3D=0,
        num_par_softmax_segments=0,
        softmax_segm_output=torch.empty(0),
        softmax_segm_max=torch.empty(0),
        softmax_segm_expsum=torch.empty(0),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )

    result = impl._forward_turboquant(
        query=query,
        key=key,
        value=value,
        key_cache=torch.empty(0, dtype=torch.uint8),
        value_cache=torch.empty(0, dtype=torch.uint8),
        output=output,
        attn_metadata=attn_metadata,
    )

    expected = F.scaled_dot_product_attention(
        query.permute(1, 0, 2).unsqueeze(0),
        key.permute(1, 0, 2).unsqueeze(0),
        value.permute(1, 0, 2).unsqueeze(0),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=True,
        scale=impl.scale,
    ).squeeze(0).permute(1, 0, 2)

    torch.testing.assert_close(result, expected, atol=5e-3, rtol=5e-3)

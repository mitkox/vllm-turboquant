# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import vllm.v1.attention.backends.triton_attn as triton_attn_module
from vllm.config.cache import CacheConfig
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
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
    _apply_mse_inverse_transform,
    _apply_mse_transform,
    _dimension_aware_codebook,
    build_turboquant_outlier_masks,
    dequantize_turboquant_vectors,
    get_turboquant_bits,
    get_turboquant_centroids,
    get_turboquant_group_dims,
    get_turboquant_layout,
    get_turboquant_mse_inverse_transform_matrix,
    get_turboquant_packed_dim,
    get_turboquant_platform_requirement,
    get_turboquant_qjl_inverse_transform_matrix,
    get_turboquant_qjl_matrix,
    get_turboquant_rotation,
    is_turboquant_kv_cache,
    pack_turboquant_indices,
    quantize_turboquant_vectors,
    supports_turboquant_cuda,
    unpack_turboquant_indices,
)
from vllm.v1.attention.ops.turboquant_metadata import (
    TurboQuantCalibrationMetadata,
    TurboQuantLayerMetadata,
    TurboQuantTensorMetadata,
    build_default_turboquant_metadata,
    discover_turboquant_metadata_path,
    load_turboquant_metadata,
    save_turboquant_metadata,
    slice_turboquant_layer_metadata_for_tp,
)

TEST_TURBOQUANT_LAYER = "model.layers.0.self_attn"
TURBOQUANT_SUPPORTED_CAPABILITY = DeviceCapability(8, 6)
TURBOQUANT_ALT_SUPPORTED_CAPABILITY = DeviceCapability(12, 1)
TURBOQUANT_TEST_SKIP_REASON = get_turboquant_platform_requirement()


def _has_supported_turboquant_cuda() -> bool:
    return torch.cuda.is_available() and supports_turboquant_cuda(
        torch.cuda.get_device_capability()
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


def _get_test_turboquant_metadata(
    kv_cache_dtype: str,
    head_size: int,
    num_kv_heads: int,
    layer_name: str = TEST_TURBOQUANT_LAYER,
):
    return build_default_turboquant_metadata(
        recipe=kv_cache_dtype,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        layer_names=[layer_name],
        model_name="tests/turboquant",
    )


def test_turboquant_metadata_resolves_legacy_qwen_layer_aliases():
    metadata = _get_test_turboquant_metadata(
        "turboquant35",
        128,
        8,
        layer_name="model.layers.3.self_attn",
    )

    layer = metadata.get_layer("language_model.model.layers.3.self_attn.attn")

    assert layer == metadata.layers["model.layers.3.self_attn"]


def _load_metadata_generator_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "generate_turboquant_metadata.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_turboquant_metadata",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_turboquant_generator_rejects_quantized_target_without_override():
    module = _load_metadata_generator_module()

    with pytest.raises(ValueError, match="should not run directly on a quantized"):
        module._validate_calibration_model_choice(
            target_model="quantized/model",
            calibration_model="quantized/model",
            quantization_config={"quant_method": "compressed-tensors"},
        )


def test_turboquant_generator_allows_separate_calibration_model():
    module = _load_metadata_generator_module()

    module._validate_calibration_model_choice(
        target_model="quantized/model",
        calibration_model="base/model",
        quantization_config={"quant_method": "compressed-tensors"},
    )


def _make_group_indices(
    head_size: int,
    outlier_count: int,
    num_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    all_idx = torch.arange(head_size, dtype=torch.int64)
    high_groups = []
    low_groups = []
    for _ in range(num_heads):
        high = torch.sort(
            torch.randperm(head_size, generator=generator)[:outlier_count]
        ).values
        high_mask = torch.ones(head_size, dtype=torch.bool)
        high_mask[high] = False
        low = all_idx[high_mask]
        high_groups.append(high)
        low_groups.append(low)
    return (
        torch.stack(high_groups, dim=0).to(device=device),
        torch.stack(low_groups, dim=0).to(device=device),
    )


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
    *,
    attn_type: AttentionType = AttentionType.DECODER,
    sliding_window: tuple[int, int] = (-1, -1),
    sinks: torch.Tensor | None = None,
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None,
    logits_soft_cap: float = 0.0,
) -> torch.Tensor:
    outputs = []
    block_size = key_cache.shape[1]
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    causal = attn_type == AttentionType.DECODER

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
        context_len = max(seq_len - q_len, 0) if causal else 0
        key_positions = torch.arange(seq_len, device=query.device, dtype=torch.int32)
        if causal:
            query_positions = torch.arange(
                context_len,
                context_len + q_len,
                device=query.device,
                dtype=torch.int32,
            )
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            left_window, right_window = sliding_window
            if left_window != -1:
                allowed &= key_positions.unsqueeze(0) >= (
                    query_positions.unsqueeze(1) - left_window
                )
            if right_window != -1:
                allowed &= key_positions.unsqueeze(0) <= (
                    query_positions.unsqueeze(1) + right_window
                )
            if mm_prefix_range is not None:
                for range_start, range_end in mm_prefix_range.get(seq_idx, []):
                    q_in_range = (query_positions >= range_start) & (
                        query_positions <= range_end
                    )
                    k_in_range = (key_positions >= range_start) & (
                        key_positions <= range_end
                    )
                    allowed |= q_in_range[:, None] & k_in_range[None, :]
        else:
            allowed = torch.ones(
                (q_len, seq_len), dtype=torch.bool, device=query.device
            )

        kv_group_num = query.shape[1] // seq_key.shape[1]
        kv_head_for_query_head = (
            torch.arange(query.shape[1], device=query.device, dtype=torch.int64)
            // kv_group_num
        )
        q_states = seq_query.permute(1, 0, 2).to(torch.float32)
        k_states = (
            seq_key.permute(1, 0, 2)
            .index_select(0, kv_head_for_query_head)
            .to(torch.float32)
        )
        v_states = (
            seq_value.permute(1, 0, 2)
            .index_select(0, kv_head_for_query_head)
            .to(torch.float32)
        )
        attn_bias = torch.zeros(
            (query.shape[1], q_len, seq_len),
            dtype=torch.float32,
            device=query.device,
        )
        attn_bias.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
        logits = torch.einsum("hqd,hkd->hqk", q_states, k_states) * scale
        logits = logits + attn_bias
        if logits_soft_cap > 0:
            logits = logits_soft_cap * torch.tanh(logits / logits_soft_cap)
        if sinks is not None:
            sink_logits = sinks[:, None, None].to(torch.float32).expand(-1, q_len, 1)
            zero_value = torch.zeros(
                (query.shape[1], 1, query.shape[2]),
                dtype=torch.float32,
                device=query.device,
            )
            v_states = torch.cat((v_states, zero_value), dim=1)
            logits = torch.cat((logits, sink_logits), dim=-1)
        probs = torch.softmax(logits, dim=-1)
        seq_output = torch.einsum("hqk,hkd->hqd", probs, v_states)
        outputs.append(seq_output.permute(1, 0, 2).to(query.dtype))

    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def _reference_turboquant_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    key_tables,
    value_tables,
    key_masks,
    value_masks,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_key_cache = key_cache.clone()
    ref_value_cache = value_cache.clone()
    valid = slot_mapping >= 0
    if not valid.any():
        return ref_key_cache, ref_value_cache

    packed_key = quantize_turboquant_vectors(
        key[valid],
        kv_cache_dtype,
        key_tables[0],
        key_tables[1],
        key_tables[2],
        key_masks,
    )
    packed_value = quantize_turboquant_vectors(
        value[valid],
        kv_cache_dtype,
        value_tables[0],
        value_tables[1],
        value_tables[2],
        value_masks,
    )
    valid_slots = slot_mapping[valid]
    block_size = ref_key_cache.shape[1]
    block_idx = torch.div(valid_slots, block_size, rounding_mode="floor")
    block_offset = valid_slots % block_size
    ref_key_cache[block_idx, block_offset] = packed_key
    ref_value_cache[block_idx, block_offset] = packed_value
    return ref_key_cache, ref_value_cache


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
    codebook = _dimension_aware_codebook(32, 3)

    assert codebook.device.type == "cpu"
    assert codebook.dtype == torch.float32

    if torch.cuda.is_available():
        centroids = get_turboquant_centroids(torch.device("cuda"), 32, 3)
        assert centroids.device.type == "cuda"


@torch.inference_mode()
def test_turboquant_mse_transform_roundtrip():
    x = torch.randn(32, 4, 128, dtype=torch.float32)
    rotation = get_turboquant_rotation(torch.device("cpu"), 128, seed_offset=101)
    restored = _apply_mse_inverse_transform(_apply_mse_transform(x, rotation), rotation)
    assert torch.allclose(restored, x, atol=1e-5, rtol=1e-5)


def test_turboquant_metadata_roundtrip(tmp_path):
    metadata = _get_test_turboquant_metadata("turboquant25", 128, 2)
    metadata_path = tmp_path / "turboquant_kv.json"
    save_turboquant_metadata(metadata, metadata_path)
    loaded = load_turboquant_metadata(str(metadata_path))

    assert loaded.recipe == "turboquant25"
    assert loaded.head_size == 128
    assert loaded.get_layer(TEST_TURBOQUANT_LAYER).key.high_precision_indices == (
        tuple(range(32)),
        tuple(range(32)),
    )


def test_turboquant_metadata_roundtrip_with_calibration(tmp_path):
    metadata = _get_test_turboquant_metadata("turboquant35", 128, 2)
    metadata = type(metadata)(
        recipe=metadata.recipe,
        head_size=metadata.head_size,
        model_name=metadata.model_name,
        layers=metadata.layers,
        calibration=TurboQuantCalibrationMetadata(
            method="activation_energy_v1",
            objective="sum_squared_activation",
            num_prompts=8,
            max_seq_len=1024,
            batch_size=2,
            num_observed_tokens=4096,
            dtype="bfloat16",
            device="cuda:0",
            prompts_sha256="abc123",
        ),
    )
    metadata_path = tmp_path / "turboquant_kv.json"
    save_turboquant_metadata(metadata, metadata_path)
    loaded = load_turboquant_metadata(str(metadata_path))

    assert loaded.calibration is not None
    assert loaded.calibration.method == "activation_energy_v1"
    assert loaded.calibration.num_observed_tokens == 4096
    assert loaded.calibration.prompts_sha256 == "abc123"


def test_turboquant_metadata_loads_legacy_json_without_calibration(tmp_path):
    metadata_path = tmp_path / "turboquant_kv.json"
    metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "recipe": "turboquant25",
                "head_size": 128,
                "model_name": "tests/turboquant",
                "transform_version": "structured_hadamard_v1",
                "codebook_version": "lloyd_beta_v1",
                "layers": {
                    TEST_TURBOQUANT_LAYER: {
                        "key_high_precision_indices": [list(range(32))] * 2,
                        "value_high_precision_indices": [list(range(32))] * 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_turboquant_metadata(str(metadata_path))

    assert loaded.calibration is None
    assert loaded.recipe == "turboquant25"


def test_turboquant_metadata_auto_discovery(tmp_path):
    model_dir = tmp_path / "model"
    metadata_path = model_dir / "turboquant_kv.json"
    save_turboquant_metadata(
        _get_test_turboquant_metadata("turboquant35", 128, 2),
        metadata_path,
    )

    assert discover_turboquant_metadata_path(str(model_dir), None) == str(metadata_path)


def test_turboquant_metadata_tp_slice_for_partitioned_heads():
    layer_metadata = TurboQuantLayerMetadata(
        key=TurboQuantTensorMetadata(
            tuple(tuple(range(i, i + 32)) for i in (0, 32, 64, 96))
        ),
        value=TurboQuantTensorMetadata(
            tuple(tuple(range(i, i + 32)) for i in (100, 132, 164, 196))
        ),
    )

    sliced = slice_turboquant_layer_metadata_for_tp(
        layer_metadata,
        num_kv_heads=2,
        tp_rank=1,
        tp_size=2,
    )

    assert sliced.key.high_precision_indices == (
        tuple(range(64, 96)),
        tuple(range(96, 128)),
    )
    assert sliced.value.high_precision_indices == (
        tuple(range(164, 196)),
        tuple(range(196, 228)),
    )


def test_turboquant_metadata_tp_slice_for_replicated_heads():
    layer_metadata = TurboQuantLayerMetadata(
        key=TurboQuantTensorMetadata((tuple(range(32)), tuple(range(32, 64)))),
        value=TurboQuantTensorMetadata((tuple(range(64, 96)), tuple(range(96, 128)))),
    )

    sliced = slice_turboquant_layer_metadata_for_tp(
        layer_metadata,
        num_kv_heads=1,
        tp_rank=2,
        tp_size=4,
    )

    assert sliced.key.high_precision_indices == (tuple(range(32, 64)),)
    assert sliced.value.high_precision_indices == (tuple(range(96, 128)),)


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
        assert abs(inner_error.mean().item()) < 0.05, cache_dtype


def test_turboquant_calibration_selects_sorted_unique_indices():
    generator_module = _load_metadata_generator_module()
    outlier_count = 32
    scores = torch.arange(256, dtype=torch.float32).reshape(2, 128)

    metadata = generator_module._build_tensor_metadata_from_scores(
        scores,
        outlier_count,
    )

    assert len(metadata.high_precision_indices) == 2
    for indices in metadata.high_precision_indices:
        assert len(indices) == outlier_count
        assert tuple(sorted(indices)) == indices
        assert len(set(indices)) == outlier_count


def test_turboquant_calibration_layer_resolution_skips_non_full_attention():
    generator_module = _load_metadata_generator_module()

    selected = generator_module._resolve_layer_indices(
        4,
        ("full_attention", "sliding_attention", "full_attention", "mamba"),
    )

    assert selected == [0, 2]


def test_turboquant_calibration_projection_discovery_fails_closed():
    generator_module = _load_metadata_generator_module()

    class _FakeSelfAttn(torch.nn.Module):
        def __init__(self, include_value: bool):
            super().__init__()
            self.k_proj = torch.nn.Linear(8, 8)
            if include_value:
                self.v_proj = torch.nn.Linear(8, 8)

    class _FakeLayer(torch.nn.Module):
        def __init__(self, include_value: bool):
            super().__init__()
            self.self_attn = _FakeSelfAttn(include_value=include_value)

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [_FakeLayer(include_value=True), _FakeLayer(include_value=False)]
            )

    with pytest.raises(ValueError, match="layer 1 value"):
        generator_module._discover_projection_modules(_FakeModel(), [0, 1])


def test_turboquant_calibration_metadata_builder_skips_non_attention_layers():
    generator_module = _load_metadata_generator_module()

    metadata = generator_module._build_calibrated_metadata(
        recipe="turboquant25",
        head_size=128,
        model_name="tests/turboquant",
        num_hidden_layers=3,
        layer_types=("full_attention", "mamba", "full_attention"),
        layer_pattern="model.layers.{i}.self_attn",
        num_kv_heads=2,
        calibration_scores={
            (0, "key"): torch.arange(256, dtype=torch.float32).reshape(2, 128),
            (0, "value"): torch.arange(256, dtype=torch.float32).reshape(2, 128),
            (2, "key"): torch.arange(256, dtype=torch.float32).reshape(2, 128),
            (2, "value"): torch.arange(256, dtype=torch.float32).reshape(2, 128),
        },
        calibration_metadata=TurboQuantCalibrationMetadata(
            method="activation_energy_v1",
            objective="sum_squared_activation",
            num_prompts=4,
            max_seq_len=256,
            batch_size=2,
            num_observed_tokens=512,
            dtype="float16",
            device="cpu",
            prompts_sha256="feedface",
        ),
    )

    assert sorted(metadata.layers) == [
        "model.layers.0.self_attn",
        "model.layers.2.self_attn",
    ]
    assert metadata.calibration is not None
    assert metadata.calibration.prompts_sha256 == "feedface"


@pytest.mark.skipif(
    not _has_supported_turboquant_cuda(), reason=TURBOQUANT_TEST_SKIP_REASON
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
    outlier_count = get_turboquant_group_dims(head_size, cache_dtype)[0]
    key_masks = _make_group_indices(head_size, outlier_count, num_kv_heads, device)
    value_masks = _make_group_indices(head_size, outlier_count, num_kv_heads, device)

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
    token_offsets = torch.arange(
        query.shape[0], device=device, dtype=torch.int32
    ) - query_start_loc.index_select(0, token_seq_ids.to(torch.int64))
    token_kv_lens = seq_lens.index_select(0, token_seq_ids.to(torch.int64)).to(
        torch.int32
    )
    token_query_positions = (
        seq_lens.index_select(0, token_seq_ids.to(torch.int64))
        - query_lens.index_select(0, token_seq_ids.to(torch.int64))
        + token_offsets
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
    value_mse_inverse_matrices = (
        get_turboquant_mse_inverse_transform_matrix(
            device, value_masks[0].shape[1], 101
        ),
        get_turboquant_mse_inverse_transform_matrix(
            device, value_masks[1].shape[1], 211
        ),
    )
    value_qjl_inverse_matrices = (
        get_turboquant_qjl_inverse_transform_matrix(
            device, value_masks[0].shape[1], 307
        ),
        get_turboquant_qjl_inverse_transform_matrix(
            device, value_masks[1].shape[1], 401
        ),
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
        token_kv_lens=token_kv_lens,
        token_query_positions=token_query_positions,
        kv_head_for_query_head=kv_head_for_query_head,
        key_query_group_indices=key_query_group_indices,
        value_query_group_indices=value_query_group_indices,
        value_mse_inverse_matrices=value_mse_inverse_matrices,
        value_qjl_inverse_matrices=value_qjl_inverse_matrices,
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

    support = TritonAttentionMetadataBuilder.get_cudagraph_support(
        None,
        spec,
    )  # type: ignore[arg-type]
    assert support == AttentionCGSupport.NEVER


def test_turboquant_validate_configuration_rejects_unsupported_cuda_device():
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

    assert get_turboquant_platform_requirement() in reasons


@pytest.mark.parametrize(
    "device_capability",
    [TURBOQUANT_SUPPORTED_CAPABILITY, TURBOQUANT_ALT_SUPPORTED_CAPABILITY],
)
def test_turboquant_validate_configuration_allows_sinks(device_capability):
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
        device_capability=device_capability,
        attn_type=AttentionType.DECODER,
    )

    assert "TurboQuant KV cache does not support attention sinks" not in reasons


@pytest.mark.parametrize(
    "device_capability",
    [TURBOQUANT_SUPPORTED_CAPABILITY, TURBOQUANT_ALT_SUPPORTED_CAPABILITY],
)
def test_turboquant_validate_configuration_allows_mm_prefix(device_capability):
    reasons = TritonAttentionBackend.validate_configuration(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="turboquant25",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=True,
        use_per_head_quant_scales=False,
        device_capability=device_capability,
        attn_type=AttentionType.DECODER,
    )

    assert "partial multimodal token full attention not supported" not in reasons


def test_cache_config_requires_feature_gate(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: TURBOQUANT_SUPPORTED_CAPABILITY,
    )

    with pytest.raises(ValueError, match="enable_turboquant=True"):
        CacheConfig(cache_dtype="turboquant25")

    CacheConfig(cache_dtype="turboquant25", enable_turboquant=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"logits_soft_cap": 1.0},
        {"attn_type": AttentionType.ENCODER_DECODER},
        {"sliding_window": 128},
        {"sinks": torch.zeros(8, dtype=torch.float32)},
    ],
)
def test_turboquant_impl_allows_extended_modes_at_init(monkeypatch, kwargs):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: TURBOQUANT_SUPPORTED_CAPABILITY,
    )

    impl = TritonAttentionImpl(
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
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata("turboquant25", 128, 8),
    )

    assert impl.attn_type == kwargs.get("attn_type", AttentionType.DECODER)


def test_turboquant_impl_rejects_metadata_kv_head_count_mismatch(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: TURBOQUANT_SUPPORTED_CAPABILITY,
    )

    with pytest.raises(ValueError, match="KV head count does not match layer"):
        TritonAttentionImpl(
            num_heads=8,
            head_size=128,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="turboquant25",
            attn_type=AttentionType.DECODER,
            turboquant_layer_name=TEST_TURBOQUANT_LAYER,
            turboquant_metadata=_get_test_turboquant_metadata("turboquant25", 128, 4),
        )


def test_turboquant_impl_accepts_global_metadata_for_replicated_kv_heads(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: TURBOQUANT_SUPPORTED_CAPABILITY,
    )
    monkeypatch.setattr(
        triton_attn_module,
        "get_tensor_model_parallel_rank",
        lambda: 2,
    )
    monkeypatch.setattr(
        triton_attn_module,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )

    impl = TritonAttentionImpl(
        num_heads=8,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="turboquant25",
        attn_type=AttentionType.DECODER,
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata("turboquant25", 128, 2),
    )

    assert impl._turboquant_layer_metadata is not None
    assert len(impl._turboquant_layer_metadata.key.high_precision_indices) == 1


def test_turboquant_impl_allows_encoder_only_layers(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: TURBOQUANT_SUPPORTED_CAPABILITY,
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
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata("turboquant25", 128, 8),
    )

    assert impl.attn_type == AttentionType.ENCODER_ONLY


def test_turboquant_impl_disables_rope_kvcache_fusion(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: TURBOQUANT_SUPPORTED_CAPABILITY,
    )

    impl = TritonAttentionImpl(
        num_heads=8,
        head_size=128,
        scale=1.0,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="turboquant25",
        attn_type=AttentionType.DECODER,
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata("turboquant25", 128, 8),
    )

    assert not impl.fused_rope_kvcache_supported()


@pytest.mark.parametrize(
    ("sliding_window", "mm_prefix_range", "match"),
    [
        (128, None, "does not support sliding window attention yet"),
        (None, {0: [(1, 3)]}, "does not support mm-prefix ranges yet"),
    ],
)
@torch.inference_mode()
def test_turboquant_cascade_rejects_unsupported_feature_combinations(
    monkeypatch,
    sliding_window: int | None,
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None,
    match: str,
):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: TURBOQUANT_SUPPORTED_CAPABILITY,
    )

    impl = TritonAttentionImpl(
        num_heads=4,
        head_size=128,
        scale=1.0 / math.sqrt(128),
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="turboquant25",
        attn_type=AttentionType.DECODER,
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata("turboquant25", 128, 2),
    )
    monkeypatch.setattr(impl, "_validate_turboquant_device", lambda device: None)

    query_len = 2
    block_size = 16
    packed_dim = get_turboquant_packed_dim(128, "turboquant25")
    query = torch.randn(query_len, 4, 128, dtype=torch.float16)
    key = torch.randn(query_len, 2, 128, dtype=torch.float16)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    key_cache = torch.zeros((2, block_size, 2, packed_dim), dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)

    attn_metadata = TritonAttentionMetadata(
        num_actual_tokens=query_len,
        max_query_len=query_len,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, query_len], dtype=torch.int32),
        max_seq_len=block_size + query_len,
        seq_lens=torch.tensor([block_size + query_len], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([block_size + query_len], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.arange(query_len, dtype=torch.int32),
        seq_threshold_3D=0,
        num_par_softmax_segments=0,
        softmax_segm_output=torch.empty(0),
        softmax_segm_max=torch.empty(0),
        softmax_segm_expsum=torch.empty(0),
        use_cascade=True,
        common_prefix_len=block_size,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=torch.tensor([query_len], dtype=torch.int32),
        mm_prefix_range=mm_prefix_range,
    )

    with pytest.raises(NotImplementedError, match=match):
        impl._forward_turboquant(
            query=query,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            attn_metadata=attn_metadata,
        )


@torch.inference_mode()
def test_turboquant_prefill_with_allocated_cache_uses_triton_prefill(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: TURBOQUANT_SUPPORTED_CAPABILITY,
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
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata("turboquant35", 256, 2),
    )

    query = torch.randn(6, 4, 256, dtype=torch.float16)
    key = torch.randn(6, 2, 256, dtype=torch.float16)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    packed_dim = get_turboquant_packed_dim(256, "turboquant35")
    key_cache = torch.zeros((1, 16, 2, packed_dim), dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)

    attn_metadata = TritonAttentionMetadata(
        num_actual_tokens=6,
        max_query_len=6,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 6], dtype=torch.int32),
        max_seq_len=6,
        seq_lens=torch.tensor([6], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([6], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.arange(6, dtype=torch.int32),
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

    called = {"context": False}

    def fake_context_attention_fwd(**kwargs):
        called["context"] = True
        kwargs["o"].copy_(
            F.scaled_dot_product_attention(
                kwargs["q"].permute(1, 0, 2).unsqueeze(0),
                kwargs["k"].permute(1, 0, 2).unsqueeze(0),
                kwargs["v"].permute(1, 0, 2).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=True,
                scale=impl.scale,
            )
            .squeeze(0)
            .permute(1, 0, 2)
        )

    monkeypatch.setattr(
        "vllm.v1.attention.backends.triton_attn.context_attention_fwd",
        fake_context_attention_fwd,
    )
    monkeypatch.setattr(
        impl,
        "_fallback_turboquant_attention",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("pure prefill should not use the Python fallback")
        ),
    )
    monkeypatch.setattr(
        impl,
        "_get_turboquant_tables",
        lambda device: (_ for _ in ()).throw(
            AssertionError("pure prefill should not touch TurboQuant decode tables")
        ),
    )

    result = impl._forward_turboquant(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        attn_metadata=attn_metadata,
    )

    expected = (
        F.scaled_dot_product_attention(
            query.permute(1, 0, 2).unsqueeze(0),
            key.permute(1, 0, 2).unsqueeze(0),
            value.permute(1, 0, 2).unsqueeze(0),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
            scale=impl.scale,
        )
        .squeeze(0)
        .permute(1, 0, 2)
    )

    assert called["context"]
    torch.testing.assert_close(result, expected, atol=5e-3, rtol=5e-3)


@torch.inference_mode()
def test_turboquant_prefill_large_head_uses_triton_prefill(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: TURBOQUANT_SUPPORTED_CAPABILITY,
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
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata("turboquant35", 256, 2),
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

    called = {"context": False, "fallback": False}

    def fake_context_attention_fwd(**kwargs):
        called["context"] = True
        kwargs["o"].copy_(
            F.scaled_dot_product_attention(
                kwargs["q"].permute(1, 0, 2).unsqueeze(0),
                kwargs["k"].permute(1, 0, 2).unsqueeze(0),
                kwargs["v"].permute(1, 0, 2).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=True,
                scale=impl.scale,
            )
            .squeeze(0)
            .permute(1, 0, 2)
        )

    def fake_fallback_turboquant_attention(**kwargs):
        called["fallback"] = True
        raise AssertionError("prefill should not use the Python fallback for D=256")

    monkeypatch.setattr(
        "vllm.v1.attention.backends.triton_attn.context_attention_fwd",
        fake_context_attention_fwd,
    )
    monkeypatch.setattr(
        impl,
        "_fallback_turboquant_attention",
        fake_fallback_turboquant_attention,
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

    expected = (
        F.scaled_dot_product_attention(
            query.permute(1, 0, 2).unsqueeze(0),
            key.permute(1, 0, 2).unsqueeze(0),
            value.permute(1, 0, 2).unsqueeze(0),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
            scale=impl.scale,
        )
        .squeeze(0)
        .permute(1, 0, 2)
    )

    assert called["context"]
    assert not called["fallback"]
    torch.testing.assert_close(result, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(
    not _has_supported_turboquant_cuda(), reason=TURBOQUANT_TEST_SKIP_REASON
)
@pytest.mark.parametrize("cache_dtype", ["turboquant25", "turboquant35"])
@torch.inference_mode()
def test_turboquant_fused_cache_update_matches_reference(cache_dtype: str):
    device = torch.device("cuda")
    head_size = 128
    num_heads = 2
    block_size = 16
    packed_dim = get_turboquant_packed_dim(head_size, cache_dtype)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    key = torch.randn(9, num_heads, head_size, dtype=torch.float16, device=device)
    value = torch.randn_like(key)
    slot_mapping = torch.tensor(
        [0, 1, -1, 17, 18, 31, 32, -1, 47],
        dtype=torch.int32,
        device=device,
    )
    kv_cache = torch.zeros(
        (4, 2, block_size, num_heads, packed_dim),
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache = kv_cache.unbind(1)

    impl = TritonAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0 / math.sqrt(head_size),
        num_kv_heads=num_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=cache_dtype,
        attn_type=AttentionType.DECODER,
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata(
            cache_dtype, head_size, num_heads
        ),
    )
    key_masks, value_masks = impl._ensure_turboquant_masks(device)
    key_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    value_tables = _get_turboquant_tables(cache_dtype, head_size, device)

    ref_key_cache, ref_value_cache = _reference_turboquant_cache_update(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype=cache_dtype,
        key_tables=key_tables,
        value_tables=value_tables,
        key_masks=key_masks,
        value_masks=value_masks,
    )
    impl.do_kv_cache_update(
        layer=None,  # type: ignore[arg-type]
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
    )

    assert torch.equal(key_cache, ref_key_cache)
    assert torch.equal(value_cache, ref_value_cache)


@pytest.mark.skipif(
    not _has_supported_turboquant_cuda(), reason=TURBOQUANT_TEST_SKIP_REASON
)
@pytest.mark.parametrize("cache_dtype", ["turboquant25", "turboquant35"])
@torch.inference_mode()
def test_turboquant_prefill_reads_quantized_cache(cache_dtype: str):
    device = torch.device("cuda")
    head_size = 128
    num_heads = 4
    num_kv_heads = 2
    block_size = 16
    seq_len = 6
    packed_dim = get_turboquant_packed_dim(head_size, cache_dtype)

    impl = TritonAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0 / math.sqrt(head_size),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=cache_dtype,
        attn_type=AttentionType.DECODER,
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata(
            cache_dtype, head_size, num_kv_heads
        ),
    )

    query = torch.randn(
        seq_len, num_heads, head_size, dtype=torch.float16, device=device
    )
    key = torch.randn(
        seq_len, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    value = torch.randn_like(key)
    kv_cache = torch.zeros(
        (1, 2, block_size, num_kv_heads, packed_dim),
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache = kv_cache.unbind(1)
    slot_mapping = torch.arange(seq_len, dtype=torch.int32, device=device)
    impl.do_kv_cache_update(
        layer=None,  # type: ignore[arg-type]
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
    )

    attn_metadata = TritonAttentionMetadata(
        num_actual_tokens=seq_len,
        max_query_len=seq_len,
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, seq_len], dtype=torch.int32),
        max_seq_len=seq_len,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
        slot_mapping=slot_mapping,
        seq_threshold_3D=0,
        num_par_softmax_segments=0,
        softmax_segm_output=torch.empty(0, device=device),
        softmax_segm_max=torch.empty(0, device=device),
        softmax_segm_expsum=torch.empty(0, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )

    output = torch.empty_like(query)
    # If prefill still consumed dense K/V, this perturbation would change the result.
    key_after_cache = key + 3.0
    value_after_cache = value - 2.0
    result = impl._forward_turboquant(
        query=query,
        key=key_after_cache,
        value=value_after_cache,
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        attn_metadata=attn_metadata,
    )

    key_masks, value_masks = impl._turboquant_masks[
        (query.device.type, query.device.index)
    ]
    key_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    value_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    expected = _reference_turboquant_decode(
        query=query,
        query_start_loc=attn_metadata.query_start_loc,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=attn_metadata.block_table,
        seq_lens=attn_metadata.seq_lens,
        kv_cache_dtype=cache_dtype,
        key_tables=key_tables,
        value_tables=value_tables,
        key_masks=key_masks,
        value_masks=value_masks,
        scale=impl.scale,
    )

    torch.testing.assert_close(result, expected, atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(
    not _has_supported_turboquant_cuda(), reason=TURBOQUANT_TEST_SKIP_REASON
)
@pytest.mark.parametrize(
    ("feature", "sliding_window", "logits_soft_cap"),
    [
        ("mm_prefix", None, 0.0),
        ("sliding_window", 4, 0.0),
        ("sinks", None, 0.0),
        ("softcap", None, 0.75),
    ],
)
@pytest.mark.parametrize("cache_dtype", ["turboquant25", "turboquant35"])
@torch.inference_mode()
def test_turboquant_extended_cached_attention_matches_reference(
    cache_dtype: str,
    feature: str,
    sliding_window: int | None,
    logits_soft_cap: float,
):
    device = torch.device("cuda")
    head_size = 128
    num_heads = 4
    num_kv_heads = 2
    block_size = 16
    seq_len = 8
    q_len = 4
    packed_dim = get_turboquant_packed_dim(head_size, cache_dtype)

    impl = TritonAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0 / math.sqrt(head_size),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype=cache_dtype,
        logits_soft_cap=logits_soft_cap,
        attn_type=AttentionType.DECODER,
        sinks=(
            torch.linspace(-0.3, 0.2, num_heads, dtype=torch.float32, device=device)
            if feature == "sinks"
            else None
        ),
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata(
            cache_dtype, head_size, num_kv_heads
        ),
    )

    key = torch.randn(
        seq_len, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    value = torch.randn_like(key)
    query = torch.randn(q_len, num_heads, head_size, dtype=torch.float16, device=device)
    kv_cache = torch.zeros(
        (1, 2, block_size, num_kv_heads, packed_dim),
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache = kv_cache.unbind(1)
    slot_mapping = torch.arange(seq_len, dtype=torch.int32, device=device)
    impl.do_kv_cache_update(
        layer=None,  # type: ignore[arg-type]
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
    )

    attn_metadata = TritonAttentionMetadata(
        num_actual_tokens=q_len,
        max_query_len=q_len,
        query_start_loc=torch.tensor([0, q_len], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, q_len], dtype=torch.int32),
        max_seq_len=seq_len,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
        slot_mapping=torch.arange(q_len, dtype=torch.int32, device=device),
        seq_threshold_3D=0,
        num_par_softmax_segments=0,
        softmax_segm_output=torch.empty(0, device=device),
        softmax_segm_max=torch.empty(0, device=device),
        softmax_segm_expsum=torch.empty(0, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        mm_prefix_range={0: [(2, 6)]} if feature == "mm_prefix" else None,
    )

    output = torch.empty_like(query)
    result = impl._forward_turboquant(
        query=query,
        key=torch.empty_like(key[:q_len]),
        value=torch.empty_like(value[:q_len]),
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        attn_metadata=attn_metadata,
    )
    key_masks, value_masks = impl._ensure_turboquant_masks(query.device)
    key_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    value_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    expected = _reference_turboquant_decode(
        query=query,
        query_start_loc=attn_metadata.query_start_loc,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=attn_metadata.block_table,
        seq_lens=attn_metadata.seq_lens,
        kv_cache_dtype=cache_dtype,
        key_tables=key_tables,
        value_tables=value_tables,
        key_masks=key_masks,
        value_masks=value_masks,
        scale=impl.scale,
        sliding_window=impl.sliding_window,
        sinks=impl.sinks,
        mm_prefix_range=attn_metadata.mm_prefix_range,
        logits_soft_cap=impl.logits_soft_cap,
    )

    atol, rtol = (7e-1, 7e-1) if feature == "softcap" else (8e-2, 8e-2)
    torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(
    not _has_supported_turboquant_cuda(), reason=TURBOQUANT_TEST_SKIP_REASON
)
@pytest.mark.parametrize("cache_dtype", ["turboquant25", "turboquant35"])
@torch.inference_mode()
def test_turboquant_cross_attention_matches_reference(cache_dtype: str):
    device = torch.device("cuda")
    head_size = 128
    num_heads = 4
    num_kv_heads = 2
    block_size = 16
    encoder_seq_len = 7
    query_len = 3
    packed_dim = get_turboquant_packed_dim(head_size, cache_dtype)

    impl = TritonAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0 / math.sqrt(head_size),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=cache_dtype,
        attn_type=AttentionType.ENCODER_DECODER,
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata(
            cache_dtype, head_size, num_kv_heads
        ),
    )

    query = torch.randn(
        query_len, num_heads, head_size, dtype=torch.float16, device=device
    )
    key = torch.randn(
        encoder_seq_len, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    value = torch.randn_like(key)
    kv_cache = torch.zeros(
        (1, 2, block_size, num_kv_heads, packed_dim),
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache = kv_cache.unbind(1)
    slot_mapping = torch.arange(encoder_seq_len, dtype=torch.int32, device=device)
    impl.do_kv_cache_update(
        layer=None,  # type: ignore[arg-type]
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
    )

    attn_metadata = TritonAttentionMetadata(
        num_actual_tokens=query_len,
        max_query_len=query_len,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, query_len], dtype=torch.int32),
        max_seq_len=encoder_seq_len,
        seq_lens=torch.tensor([encoder_seq_len], dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor([encoder_seq_len], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
        slot_mapping=torch.arange(query_len, dtype=torch.int32, device=device),
        seq_threshold_3D=0,
        num_par_softmax_segments=0,
        softmax_segm_output=torch.empty(0, device=device),
        softmax_segm_max=torch.empty(0, device=device),
        softmax_segm_expsum=torch.empty(0, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        encoder_seq_lens=torch.tensor(
            [encoder_seq_len], dtype=torch.int32, device=device
        ),
        encoder_seq_lens_cpu=torch.tensor([encoder_seq_len], dtype=torch.int32),
        causal=False,
    )

    output = torch.empty_like(query)
    result = impl._forward_turboquant(
        query=query,
        key=torch.empty_like(key[:query_len]),
        value=torch.empty_like(value[:query_len]),
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        attn_metadata=attn_metadata,
    )
    key_masks, value_masks = impl._ensure_turboquant_masks(query.device)
    key_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    value_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    expected = _reference_turboquant_decode(
        query=query,
        query_start_loc=attn_metadata.query_start_loc,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=attn_metadata.block_table,
        seq_lens=attn_metadata.seq_lens,
        kv_cache_dtype=cache_dtype,
        key_tables=key_tables,
        value_tables=value_tables,
        key_masks=key_masks,
        value_masks=value_masks,
        scale=impl.scale,
        attn_type=AttentionType.ENCODER_DECODER,
    )

    torch.testing.assert_close(result, expected, atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(
    not _has_supported_turboquant_cuda(), reason=TURBOQUANT_TEST_SKIP_REASON
)
@pytest.mark.parametrize("cache_dtype", ["turboquant25", "turboquant35"])
@torch.inference_mode()
def test_turboquant_rope_cache_update_matches_reference(
    cache_dtype: str, default_vllm_config
):
    device = torch.device("cuda")
    head_size = 128
    num_heads = 2
    seq_len = 9
    block_size = 16
    packed_dim = get_turboquant_packed_dim(head_size, cache_dtype)

    impl = TritonAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0 / math.sqrt(head_size),
        num_kv_heads=num_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=cache_dtype,
        attn_type=AttentionType.DECODER,
        turboquant_layer_name=TEST_TURBOQUANT_LAYER,
        turboquant_metadata=_get_test_turboquant_metadata(
            cache_dtype, head_size, num_heads
        ),
    )

    query = torch.randn(
        seq_len, num_heads, head_size, dtype=torch.float16, device=device
    )
    key = torch.randn_like(query)
    value = torch.randn_like(key)
    positions = torch.arange(seq_len, dtype=torch.int64, device=device)
    rotary = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=head_size,
        max_position_embeddings=64,
        base=10000.0,
        is_neox_style=True,
        dtype=torch.float16,
    )
    cos_sin_cache = rotary.cos_sin_cache.to(device=device, dtype=torch.float16)
    kv_cache = torch.zeros(
        (1, 2, block_size, num_heads, packed_dim),
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache = kv_cache.unbind(1)
    slot_mapping = torch.arange(seq_len, dtype=torch.int32, device=device)

    ref_query, ref_key = RotaryEmbedding.forward_static(
        positions=positions,
        query=query.clone(),
        key=key.clone(),
        head_size=head_size,
        rotary_dim=cos_sin_cache.shape[-1],
        cos_sin_cache=cos_sin_cache,
        is_neox_style=True,
    )
    assert ref_key is not None
    key_masks, value_masks = impl._ensure_turboquant_masks(device)
    key_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    value_tables = _get_turboquant_tables(cache_dtype, head_size, device)
    ref_key_cache, ref_value_cache = _reference_turboquant_cache_update(
        key=ref_key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype=cache_dtype,
        key_tables=key_tables,
        value_tables=value_tables,
        key_masks=key_masks,
        value_masks=value_masks,
    )

    impl.do_rope_and_kv_cache_update(
        layer=None,  # type: ignore[arg-type]
        query=query,
        key=key,
        value=value,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        kv_cache=kv_cache,
        layer_slot_mapping=slot_mapping,
    )

    torch.testing.assert_close(query, ref_query, atol=1e-3, rtol=1e-3)
    assert torch.equal(key_cache, ref_key_cache)
    assert torch.equal(value_cache, ref_value_cache)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_kv_cache import (
    TURBOQUANT_QJL_SCALE,
    TurboQuantLayout,
    apply_turboquant_query_transforms,
    get_turboquant_layout,
    scatter_turboquant_output,
)


@cache
def _norm_lut(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    values = torch.arange(1 << 16, dtype=torch.int32, device=device)
    return values.to(torch.int16).view(torch.float16).to(torch.float32)


def get_turboquant_norm_lut(device: torch.device) -> torch.Tensor:
    return _norm_lut(device.type, device.index)


def _require_gb10_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("TurboQuant Triton decode requires CUDA tensors.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 1):
        raise ValueError(
            "TurboQuant KV cache requires NVIDIA GB10 / SM121."
        )


@triton.jit
def _load_half_from_bytes(
    cache_ptr,
    token_base,
    lut_ptr,
    byte_offset: tl.constexpr,
    stride_cache_d: tl.constexpr,
):
    lo = tl.load(
        cache_ptr + token_base + byte_offset * stride_cache_d,
        mask=True,
        other=0,
    ).to(tl.int32)
    hi = tl.load(
        cache_ptr + token_base + (byte_offset + 1) * stride_cache_d,
        mask=True,
        other=0,
    ).to(tl.int32)
    return tl.load(lut_ptr + lo + (hi << 8), mask=True, other=0.0)


@triton.jit
def _unpack_fixed_indices(
    cache_ptr,
    token_base,
    offs_d,
    stride_cache_d: tl.constexpr,
    bits: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    base_offset: tl.constexpr,
    mask_d,
):
    indices = tl.zeros([BLOCK_N, HEAD_SIZE_PADDED], dtype=tl.int32)
    if bits == 0:
        return indices

    for bit in range(bits):
        bit_position = offs_d[None, :] * bits + bit
        byte_offset = base_offset + bit_position // 8
        bit_offset = bit_position % 8
        byte = tl.load(
            cache_ptr + token_base[:, None] + byte_offset * stride_cache_d,
            mask=mask_d,
            other=0,
        ).to(tl.int32)
        indices += ((byte >> bit_offset) & 1) << bit
    return indices


@triton.jit
def _unpack_signs(
    cache_ptr,
    token_base,
    offs_d,
    stride_cache_d: tl.constexpr,
    byte_offset: tl.constexpr,
    mask_d,
):
    bit_position = offs_d[None, :]
    qjl_byte_offset = byte_offset + bit_position // 8
    qjl_bit_offset = bit_position % 8
    byte = tl.load(
        cache_ptr + token_base[:, None] + qjl_byte_offset * stride_cache_d,
        mask=mask_d,
        other=0,
    ).to(tl.int32)
    bits = (byte >> qjl_bit_offset) & 1
    return bits.to(tl.float32) * 2.0 - 1.0


@triton.jit
def _turboquant_decode_kernel(
    q_rot_0_ptr,
    q_qjl_0_ptr,
    q_rot_1_ptr,
    q_qjl_1_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    token_seq_ids_ptr,
    token_seq_lens_ptr,
    acc_mse_0_ptr,
    acc_qjl_0_ptr,
    acc_mse_1_ptr,
    acc_qjl_1_ptr,
    norm_lut_ptr,
    centroids_0_ptr,
    centroids_1_ptr,
    softmax_scale,
    q0_stride_0,
    q0_stride_1,
    q1_stride_0,
    q1_stride_1,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3,
    block_table_stride,
    acc0_stride_0,
    acc0_stride_1,
    acc1_stride_0,
    acc1_stride_1,
    kv_group_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    G0_DIM: tl.constexpr,
    G0_PADDED: tl.constexpr,
    G0_MSE_BITS: tl.constexpr,
    G0_GROUP_OFFSET: tl.constexpr,
    G0_QJL_OFFSET: tl.constexpr,
    G0_VECTOR_NORM_OFFSET: tl.constexpr,
    G0_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G0_QJL_SCALE: tl.constexpr,
    G1_DIM: tl.constexpr,
    G1_PADDED: tl.constexpr,
    G1_MSE_BITS: tl.constexpr,
    G1_GROUP_OFFSET: tl.constexpr,
    G1_QJL_OFFSET: tl.constexpr,
    G1_VECTOR_NORM_OFFSET: tl.constexpr,
    G1_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G1_QJL_SCALE: tl.constexpr,
):
    cur_token = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    seq_idx = tl.load(token_seq_ids_ptr + cur_token)
    seq_len = tl.load(token_seq_lens_ptr + cur_token)

    offs_d0 = tl.arange(0, G0_PADDED)
    mask_d0 = offs_d0 < G0_DIM
    offs_d1 = tl.arange(0, G1_PADDED)
    mask_d1 = offs_d1 < G1_DIM

    q_rot_0 = tl.load(
        q_rot_0_ptr + cur_token * q0_stride_0 + cur_head * q0_stride_1 + offs_d0,
        mask=mask_d0,
        other=0.0,
    )
    q_qjl_0 = tl.load(
        q_qjl_0_ptr + cur_token * q0_stride_0 + cur_head * q0_stride_1 + offs_d0,
        mask=mask_d0,
        other=0.0,
    )
    q_rot_1 = tl.load(
        q_rot_1_ptr + cur_token * q1_stride_0 + cur_head * q1_stride_1 + offs_d1,
        mask=mask_d1,
        other=0.0,
    )
    q_qjl_1 = tl.load(
        q_qjl_1_ptr + cur_token * q1_stride_0 + cur_head * q1_stride_1 + offs_d1,
        mask=mask_d1,
        other=0.0,
    )

    e_max = -float("inf")
    e_sum = 0.0
    acc_mse_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    acc_qjl_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    acc_mse_1 = tl.zeros([G1_PADDED], dtype=tl.float32)
    acc_qjl_1 = tl.zeros([G1_PADDED], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        block_ids = tl.load(
            block_table_ptr + seq_idx * block_table_stride + offs_n // BLOCK_SIZE,
            mask=mask_n,
            other=0,
        )
        block_offsets = offs_n % BLOCK_SIZE

        key_token_base = (
            block_ids * stride_k_cache_0
            + block_offsets * stride_k_cache_1
            + cur_kv_head * stride_k_cache_2
        )
        value_token_base = (
            block_ids * stride_v_cache_0
            + block_offsets * stride_v_cache_1
            + cur_kv_head * stride_v_cache_2
        )

        key_qjl_signs_0 = _unpack_signs(
            key_cache_ptr,
            key_token_base,
            offs_d0,
            stride_k_cache_3,
            G0_QJL_OFFSET,
            mask_n[:, None] & mask_d0[None, :],
        )
        key_vector_norm_0 = _load_half_from_bytes(
            key_cache_ptr,
            key_token_base,
            norm_lut_ptr,
            G0_VECTOR_NORM_OFFSET,
            stride_k_cache_3,
        )
        key_residual_norm_0 = _load_half_from_bytes(
            key_cache_ptr,
            key_token_base,
            norm_lut_ptr,
            G0_RESIDUAL_NORM_OFFSET,
            stride_k_cache_3,
        )
        key_logits = key_vector_norm_0 * key_residual_norm_0 * tl.sum(
            key_qjl_signs_0 * q_qjl_0[None, :],
            axis=1,
        )
        if G0_MSE_BITS > 0:
            key_indices_0 = _unpack_fixed_indices(
                key_cache_ptr,
                key_token_base,
                offs_d0,
                stride_k_cache_3,
                G0_MSE_BITS,
                BLOCK_N,
                G0_PADDED,
                G0_GROUP_OFFSET,
                mask_n[:, None] & mask_d0[None, :],
            )
            key_centroids_0 = tl.load(
                centroids_0_ptr + key_indices_0,
                mask=mask_n[:, None] & mask_d0[None, :],
                other=0.0,
            )
            key_logits += key_vector_norm_0 * tl.sum(
                key_centroids_0 * q_rot_0[None, :],
                axis=1,
            )

        key_qjl_signs_1 = _unpack_signs(
            key_cache_ptr,
            key_token_base,
            offs_d1,
            stride_k_cache_3,
            G1_QJL_OFFSET,
            mask_n[:, None] & mask_d1[None, :],
        )
        key_vector_norm_1 = _load_half_from_bytes(
            key_cache_ptr,
            key_token_base,
            norm_lut_ptr,
            G1_VECTOR_NORM_OFFSET,
            stride_k_cache_3,
        )
        key_residual_norm_1 = _load_half_from_bytes(
            key_cache_ptr,
            key_token_base,
            norm_lut_ptr,
            G1_RESIDUAL_NORM_OFFSET,
            stride_k_cache_3,
        )
        key_logits += key_vector_norm_1 * key_residual_norm_1 * tl.sum(
            key_qjl_signs_1 * q_qjl_1[None, :],
            axis=1,
        )
        if G1_MSE_BITS > 0:
            key_indices_1 = _unpack_fixed_indices(
                key_cache_ptr,
                key_token_base,
                offs_d1,
                stride_k_cache_3,
                G1_MSE_BITS,
                BLOCK_N,
                G1_PADDED,
                G1_GROUP_OFFSET,
                mask_n[:, None] & mask_d1[None, :],
            )
            key_centroids_1 = tl.load(
                centroids_1_ptr + key_indices_1,
                mask=mask_n[:, None] & mask_d1[None, :],
                other=0.0,
            )
            key_logits += key_vector_norm_1 * tl.sum(
                key_centroids_1 * q_rot_1[None, :],
                axis=1,
            )

        logits = key_logits * softmax_scale
        logits = tl.where(mask_n, logits, float("-inf"))

        n_e_max = tl.maximum(tl.max(logits, axis=0), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(logits - n_e_max)

        value_vector_norm_0 = _load_half_from_bytes(
            value_cache_ptr,
            value_token_base,
            norm_lut_ptr,
            G0_VECTOR_NORM_OFFSET,
            stride_v_cache_3,
        )
        value_residual_norm_0 = _load_half_from_bytes(
            value_cache_ptr,
            value_token_base,
            norm_lut_ptr,
            G0_RESIDUAL_NORM_OFFSET,
            stride_v_cache_3,
        )
        value_vector_norm_1 = _load_half_from_bytes(
            value_cache_ptr,
            value_token_base,
            norm_lut_ptr,
            G1_VECTOR_NORM_OFFSET,
            stride_v_cache_3,
        )
        value_residual_norm_1 = _load_half_from_bytes(
            value_cache_ptr,
            value_token_base,
            norm_lut_ptr,
            G1_RESIDUAL_NORM_OFFSET,
            stride_v_cache_3,
        )

        acc_mse_0 *= re_scale
        acc_qjl_0 *= re_scale
        acc_mse_1 *= re_scale
        acc_qjl_1 *= re_scale

        if G0_MSE_BITS > 0:
            value_indices_0 = _unpack_fixed_indices(
                value_cache_ptr,
                value_token_base,
                offs_d0,
                stride_v_cache_3,
                G0_MSE_BITS,
                BLOCK_N,
                G0_PADDED,
                G0_GROUP_OFFSET,
                mask_n[:, None] & mask_d0[None, :],
            )
            value_centroids_0 = tl.load(
                centroids_0_ptr + value_indices_0,
                mask=mask_n[:, None] & mask_d0[None, :],
                other=0.0,
            )
            acc_mse_0 += tl.sum(
                (p * value_vector_norm_0)[:, None] * value_centroids_0,
                axis=0,
            )

        value_qjl_signs_0 = _unpack_signs(
            value_cache_ptr,
            value_token_base,
            offs_d0,
            stride_v_cache_3,
            G0_QJL_OFFSET,
            mask_n[:, None] & mask_d0[None, :],
        )
        acc_qjl_0 += tl.sum(
            (
                p
                * value_vector_norm_0
                * value_residual_norm_0
                * G0_QJL_SCALE
            )[:, None]
            * value_qjl_signs_0,
            axis=0,
        )

        if G1_MSE_BITS > 0:
            value_indices_1 = _unpack_fixed_indices(
                value_cache_ptr,
                value_token_base,
                offs_d1,
                stride_v_cache_3,
                G1_MSE_BITS,
                BLOCK_N,
                G1_PADDED,
                G1_GROUP_OFFSET,
                mask_n[:, None] & mask_d1[None, :],
            )
            value_centroids_1 = tl.load(
                centroids_1_ptr + value_indices_1,
                mask=mask_n[:, None] & mask_d1[None, :],
                other=0.0,
            )
            acc_mse_1 += tl.sum(
                (p * value_vector_norm_1)[:, None] * value_centroids_1,
                axis=0,
            )

        value_qjl_signs_1 = _unpack_signs(
            value_cache_ptr,
            value_token_base,
            offs_d1,
            stride_v_cache_3,
            G1_QJL_OFFSET,
            mask_n[:, None] & mask_d1[None, :],
        )
        acc_qjl_1 += tl.sum(
            (
                p
                * value_vector_norm_1
                * value_residual_norm_1
                * G1_QJL_SCALE
            )[:, None]
            * value_qjl_signs_1,
            axis=0,
        )

        e_sum = e_sum * re_scale + tl.sum(p, axis=0)
        e_max = n_e_max

    acc_mse_0 = acc_mse_0 / e_sum
    acc_qjl_0 = acc_qjl_0 / e_sum
    acc_mse_1 = acc_mse_1 / e_sum
    acc_qjl_1 = acc_qjl_1 / e_sum

    tl.store(
        acc_mse_0_ptr + cur_token * acc0_stride_0 + cur_head * acc0_stride_1 + offs_d0,
        acc_mse_0,
        mask=mask_d0,
    )
    tl.store(
        acc_qjl_0_ptr + cur_token * acc0_stride_0 + cur_head * acc0_stride_1 + offs_d0,
        acc_qjl_0,
        mask=mask_d0,
    )
    tl.store(
        acc_mse_1_ptr + cur_token * acc1_stride_0 + cur_head * acc1_stride_1 + offs_d1,
        acc_mse_1,
        mask=mask_d1,
    )
    tl.store(
        acc_qjl_1_ptr + cur_token * acc1_stride_0 + cur_head * acc1_stride_1 + offs_d1,
        acc_qjl_1,
        mask=mask_d1,
    )


def turboquant_decode_attention_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    key_group_indices: tuple[torch.Tensor, torch.Tensor],
    value_group_indices: tuple[torch.Tensor, torch.Tensor],
    key_rotations: tuple[torch.Tensor, torch.Tensor],
    key_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    value_rotations: tuple[torch.Tensor, torch.Tensor],
    value_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    norm_lut: torch.Tensor,
    softmax_scale: float,
    kv_cache_dtype: str,
    token_seq_ids: torch.Tensor | None = None,
    token_seq_lens: torch.Tensor | None = None,
    kv_head_for_query_head: torch.Tensor | None = None,
    key_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if query.ndim != 3:
        raise ValueError(f"Expected query shape [T, H, D], got {query.shape}")
    _require_gb10_cuda(query.device)

    layout = get_turboquant_layout(kv_cache_dtype, query.shape[-1])
    kv_group_num = query.shape[1] // key_cache.shape[2]
    if kv_head_for_query_head is None:
        kv_head_for_query_head = (
            torch.arange(query.shape[1], device=query.device, dtype=torch.int64)
            // kv_group_num
        )

    q_rot, q_qjl = apply_turboquant_query_transforms(
        query=query,
        group_indices=key_group_indices,
        rotations=key_rotations,
        qjl_matrices=key_qjl_matrices,
        kv_head_for_query_head=kv_head_for_query_head,
        per_query_group_indices=key_query_group_indices,
    )

    if token_seq_ids is None or token_seq_lens is None:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        token_seq_ids = torch.repeat_interleave(
            torch.arange(seq_lens.shape[0], device=query.device, dtype=torch.int32),
            query_lens,
        )
        token_offsets = (
            torch.arange(query.shape[0], device=query.device, dtype=torch.int32)
            - query_start_loc.index_select(0, token_seq_ids.to(torch.int64))
        )
        token_seq_lens = (
            seq_lens.index_select(0, token_seq_ids.to(torch.int64))
            - query_lens.index_select(0, token_seq_ids.to(torch.int64))
            + token_offsets
            + 1
        ).to(torch.int32)

    group0 = layout.groups[0]
    group1 = layout.groups[1]
    g0_pad = triton.next_power_of_2(group0.dim)
    g1_pad = triton.next_power_of_2(group1.dim)
    block_n = 8 if query.shape[-1] >= 256 else 16

    acc_mse_0 = torch.empty_like(q_rot[0], dtype=torch.float32)
    acc_qjl_0 = torch.empty_like(q_rot[0], dtype=torch.float32)
    acc_mse_1 = torch.empty_like(q_rot[1], dtype=torch.float32)
    acc_qjl_1 = torch.empty_like(q_rot[1], dtype=torch.float32)

    _turboquant_decode_kernel[(query.shape[0], query.shape[1])](
        q_rot_0_ptr=q_rot[0],
        q_qjl_0_ptr=q_qjl[0],
        q_rot_1_ptr=q_rot[1],
        q_qjl_1_ptr=q_qjl[1],
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_table_ptr=block_table,
        token_seq_ids_ptr=token_seq_ids,
        token_seq_lens_ptr=token_seq_lens,
        acc_mse_0_ptr=acc_mse_0,
        acc_qjl_0_ptr=acc_qjl_0,
        acc_mse_1_ptr=acc_mse_1,
        acc_qjl_1_ptr=acc_qjl_1,
        norm_lut_ptr=norm_lut,
        centroids_0_ptr=centroids[group0.mse_bits],
        centroids_1_ptr=centroids[group1.mse_bits],
        softmax_scale=softmax_scale,
        q0_stride_0=q_rot[0].stride(0),
        q0_stride_1=q_rot[0].stride(1),
        q1_stride_0=q_rot[1].stride(0),
        q1_stride_1=q_rot[1].stride(1),
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        block_table_stride=block_table.stride(0),
        acc0_stride_0=acc_mse_0.stride(0),
        acc0_stride_1=acc_mse_0.stride(1),
        acc1_stride_0=acc_mse_1.stride(0),
        acc1_stride_1=acc_mse_1.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_SIZE=key_cache.shape[1],
        BLOCK_N=block_n,
        G0_DIM=group0.dim,
        G0_PADDED=g0_pad,
        G0_MSE_BITS=group0.mse_bits,
        G0_GROUP_OFFSET=0,
        G0_QJL_OFFSET=group0.qjl_offset,
        G0_VECTOR_NORM_OFFSET=group0.vector_norm_offset,
        G0_RESIDUAL_NORM_OFFSET=group0.residual_norm_offset,
        G0_QJL_SCALE=TURBOQUANT_QJL_SCALE / group0.dim,
        G1_DIM=group1.dim,
        G1_PADDED=g1_pad,
        G1_MSE_BITS=group1.mse_bits,
        G1_GROUP_OFFSET=group0.packed_bytes,
        G1_QJL_OFFSET=group1.qjl_offset,
        G1_VECTOR_NORM_OFFSET=group1.vector_norm_offset,
        G1_RESIDUAL_NORM_OFFSET=group1.residual_norm_offset,
        G1_QJL_SCALE=TURBOQUANT_QJL_SCALE / group1.dim,
        num_warps=4,
        num_stages=2,
    )

    group_outputs = []
    for acc_mse, acc_qjl, rotation, qjl_matrix in zip(
        (acc_mse_0, acc_mse_1),
        (acc_qjl_0, acc_qjl_1),
        value_rotations,
        value_qjl_matrices,
        strict=True,
    ):
        group_outputs.append(
            torch.matmul(acc_mse, rotation.transpose(0, 1))
            + torch.matmul(acc_qjl, qjl_matrix.transpose(0, 1))
        )

    return scatter_turboquant_output(
        head_size=query.shape[-1],
        dtype=query.dtype,
        group_outputs=(group_outputs[0], group_outputs[1]),
        group_indices=value_group_indices,
        kv_head_for_query_head=kv_head_for_query_head,
        per_query_indices=value_query_group_indices,
        out=out,
    )

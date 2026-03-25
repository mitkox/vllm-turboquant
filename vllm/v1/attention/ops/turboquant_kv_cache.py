# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from functools import lru_cache

import torch

TURBOQUANT_KV_CACHE_BITS = {
    "turboquant1": 1,
    "turboquant2": 2,
    "turboquant3": 3,
    "turboquant4": 4,
}
TURBOQUANT_NORM_BYTES = 2
TURBOQUANT_SEED = 20250428


def is_turboquant_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in TURBOQUANT_KV_CACHE_BITS


def get_turboquant_bits(kv_cache_dtype: str) -> int:
    try:
        return TURBOQUANT_KV_CACHE_BITS[kv_cache_dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported TurboQuant KV cache dtype: {kv_cache_dtype}") from e


def get_turboquant_packed_dim(head_size: int, bits: int) -> int:
    return (head_size * bits + 7) // 8 + TURBOQUANT_NORM_BYTES


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@lru_cache(maxsize=None)
def _standard_gaussian_codebook(bits: int) -> tuple[float, ...]:
    levels = 1 << bits
    probs = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
    centroids = math.sqrt(2.0) * torch.erfinv(2.0 * probs - 1.0)

    for _ in range(200):
        bounds = torch.empty(levels + 1, dtype=torch.float64)
        bounds[0] = -torch.inf
        bounds[-1] = torch.inf
        bounds[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        bounds_lo = bounds[:-1]
        bounds_hi = bounds[1:]

        pdf_lo = torch.where(
            torch.isfinite(bounds_lo), _normal_pdf(bounds_lo), torch.zeros_like(bounds_lo)
        )
        pdf_hi = torch.where(
            torch.isfinite(bounds_hi), _normal_pdf(bounds_hi), torch.zeros_like(bounds_hi)
        )
        mass = _normal_cdf(bounds_hi) - _normal_cdf(bounds_lo)
        new_centroids = (pdf_lo - pdf_hi) / mass.clamp_min(1e-12)

        if torch.max(torch.abs(new_centroids - centroids)) < 1e-8:
            centroids = new_centroids
            break
        centroids = new_centroids

    return tuple(float(x) for x in centroids.tolist())


@lru_cache(maxsize=None)
def _rotation_matrix_cpu(head_size: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TURBOQUANT_SEED + head_size)
    gaussian = torch.randn((head_size, head_size), generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs


def get_turboquant_centroids(
    device: torch.device,
    head_size: int,
    bits: int,
) -> torch.Tensor:
    base = torch.tensor(
        _standard_gaussian_codebook(bits),
        dtype=torch.float32,
        device=device,
    )
    return base / math.sqrt(head_size)


def get_turboquant_rotation(
    device: torch.device,
    head_size: int,
) -> torch.Tensor:
    return _rotation_matrix_cpu(head_size).to(device=device, dtype=torch.float32)


def _bit_layout(
    head_size: int,
    bits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bit_offsets = torch.arange(bits, dtype=torch.int64, device=device)
    bit_offsets = bit_offsets.expand(head_size, bits).reshape(-1)
    bit_positions = torch.arange(head_size, dtype=torch.int64, device=device).unsqueeze(-1)
    bit_positions = (bit_positions * bits).expand(head_size, bits) + torch.arange(
        bits, dtype=torch.int64, device=device
    )
    flat_positions = bit_positions.reshape(-1)
    return bit_offsets, flat_positions // 8, flat_positions % 8


def pack_turboquant_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    head_size = indices.shape[-1]
    packed_bytes = (head_size * bits + 7) // 8
    bit_offsets, byte_index, bit_index = _bit_layout(head_size, bits, indices.device)
    bit_offsets = bit_offsets.reshape(*((1,) * (indices.ndim - 1)), head_size, bits)
    bit_index = bit_index.reshape(*((1,) * (indices.ndim - 1)), head_size, bits)
    contrib = (
        ((indices.unsqueeze(-1).to(torch.int32) >> bit_offsets) & 1) << bit_index
    ).to(torch.int32)
    contrib = contrib.reshape(*indices.shape[:-1], head_size * bits)
    packed = torch.zeros((*indices.shape[:-1], packed_bytes), dtype=torch.int32, device=indices.device)
    scatter_index = byte_index.reshape(
        *((1,) * (indices.ndim - 1)), head_size * bits
    ).expand_as(contrib)
    packed.scatter_add_(-1, scatter_index, contrib)
    return packed.to(torch.uint8)


def unpack_turboquant_indices(
    packed: torch.Tensor,
    head_size: int,
    bits: int,
) -> torch.Tensor:
    bit_offsets, byte_index, bit_index = _bit_layout(head_size, bits, packed.device)
    gather_index = byte_index.reshape(
        *((1,) * (packed.ndim - 1)), head_size * bits
    ).expand(*packed.shape[:-1], head_size * bits)
    selected = packed.to(torch.int32).gather(-1, gather_index)
    bits_view = ((selected >> bit_index) & 1).reshape(*packed.shape[:-1], head_size, bits)
    bit_offsets = bit_offsets.reshape(*((1,) * (packed.ndim - 1)), head_size, bits)
    values = (bits_view << bit_offsets).sum(dim=-1)
    return values.to(torch.uint8)


def _norms_to_bytes(norms: torch.Tensor) -> torch.Tensor:
    norm_half = norms.to(torch.float16).contiguous()
    return norm_half.reshape(-1).view(torch.uint8).reshape(*norm_half.shape, TURBOQUANT_NORM_BYTES)


def _bytes_to_norms(norm_bytes: torch.Tensor) -> torch.Tensor:
    raw = norm_bytes.contiguous().reshape(-1, TURBOQUANT_NORM_BYTES).view(torch.float16)
    return raw.reshape(*norm_bytes.shape[:-1], 1).to(torch.float32)


def quantize_turboquant_vectors(
    x: torch.Tensor,
    bits: int,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    norms = x_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = x_fp32 / norms
    rotated = torch.matmul(unit, rotation)
    distances = torch.abs(rotated.unsqueeze(-1) - centroids)
    indices = distances.argmin(dim=-1).to(torch.uint8)
    packed = pack_turboquant_indices(indices, bits)
    return torch.cat((packed, _norms_to_bytes(norms.squeeze(-1))), dim=-1)


def dequantize_turboquant_vectors(
    packed: torch.Tensor,
    bits: int,
    head_size: int,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    packed_bytes = (head_size * bits + 7) // 8
    indices = unpack_turboquant_indices(packed[..., :packed_bytes], head_size, bits)
    norms = _bytes_to_norms(packed[..., packed_bytes : packed_bytes + TURBOQUANT_NORM_BYTES])
    rotated = centroids[indices.long()] * norms
    return torch.matmul(rotated, rotation.transpose(0, 1)).to(dtype=dtype)
